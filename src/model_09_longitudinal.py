"""HistGradientBoosting model with advanced longitudinal & PKPD features (09_longitudinal).

Sub-task: Modele longitudinal et pharmacodynamique (Sous-tache 7)
Project: Bobathon ESILV (Probabl x IBM)

Innovations majeures :
1. Pharmacodynamie Levodopa (Courbe de Bateman pour time_since_intake_on) :
   - C(t) = A * (exp(-ke * t) - exp(-ka * t)) avec ka=0.571, ke=0.187, A=45.53
   - Estimation de l'effet therapeutique et calcul de on_adjusted = on + effet
   - Construction d'un substitut continu de OFF (surrogate = off.fillna(on_adjusted))
2. Trajectoire intra-patient et interpolation temporelle par age :
   - Interpolation lineaire intra-patient de off et on selon l'age exact
   - Ajustement de pente intra-patient (patient_slope et patient_linear_surrogate)
   - Lags longitudinaux par patient (off_prev, off_next)
   - Residus et ecarts a la trajectoire du patient (diff_from_linear_surrogate, diff_from_interp_off)
3. Encodage categoriel natif des 50 evaluateurs (rater_id) et des cohortes/genes.
4. Validation croisee groupee par patient (GroupKFold 5 plis).
5. Entrainement final et generation de submissions/09_longitudinal.csv avec bornage [0.0, 132.0].
6. Publication du rapport sur Skore Hub sous la cle '09_longitudinal'.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "09_longitudinal"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials

# Parametres de la cinetique pharmacodynamique de Bateman (ajustes sur les visites avec ON et OFF)
BATEMAN_A = 45.5346
BATEMAN_KE = 0.1870
BATEMAN_KA = 0.5708


def prepare_longitudinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare les variables explicatives enrichies de la dynamique longitudinale et PKPD."""
    data = df.copy()
    orig_index = data.index

    # Tri par patient et par age pour garantir la chronologie
    data = data.sort_values(["patient_id", "age"])

    # 1. Progression temporelle et suivi
    data["time_since_diagnosis"] = data["age"] - data["age_at_diagnosis"]
    data["visit_number"] = data.groupby("patient_id")["age"].rank(method="first").astype(int)
    data["n_visits_patient"] = data.groupby("patient_id")["age"].transform("count")
    data["visit_ratio"] = data["visit_number"] / data["n_visits_patient"]
    min_age = data.groupby("patient_id")["age"].transform("min")
    data["time_since_first_visit"] = data["age"] - min_age

    # 2. Pharmacodynamie Levodopa (Courbe de Bateman)
    t_on = data["time_since_intake_on"].fillna(2.5).clip(lower=0.0)
    data["bateman_effect_on"] = BATEMAN_A * (np.exp(-BATEMAN_KE * t_on) - np.exp(-BATEMAN_KA * t_on))
    data["on_adjusted"] = data["on"] + data["bateman_effect_on"]
    data["surrogate"] = data["off"].fillna(data["on_adjusted"])

    # 3. Statistiques de base par patient
    data["mean_off_patient"] = data.groupby("patient_id")["off"].transform("mean")
    data["mean_on_patient"] = data.groupby("patient_id")["on"].transform("mean")
    data["mean_surrogate_patient"] = data.groupby("patient_id")["surrogate"].transform("mean")
    data["std_off_patient"] = data.groupby("patient_id")["off"].transform("std")
    data["std_on_patient"] = data.groupby("patient_id")["on"].transform("std")
    data["mean_ledd_patient"] = data.groupby("patient_id")["ledd"].transform("mean")

    # 4. Trajectoires et interpolations par patient
    def _compute_patient_trajectories(g: pd.DataFrame) -> pd.DataFrame:
        x = g["age"].values
        y_surr = g["surrogate"].values

        # Pente et projection lineaire de la progression
        if len(x) >= 2:
            poly = np.polyfit(x, y_surr, 1)
            g["patient_slope"] = poly[0]
            g["patient_linear_surrogate"] = poly[0] * x + poly[1]
        else:
            g["patient_slope"] = 2.47
            g["patient_linear_surrogate"] = y_surr[0]

        # Interpolation de off selon l'age
        val_off = g["off"].notnull()
        n_off = val_off.sum()
        if n_off >= 2:
            g["off_age_interp"] = np.interp(g["age"], g.loc[val_off, "age"], g.loc[val_off, "off"])
        elif n_off == 1:
            g["off_age_interp"] = g.loc[val_off, "off"].iloc[0]
        else:
            g["off_age_interp"] = np.nan

        # Interpolation de on selon l'age
        val_on = g["on"].notnull()
        n_on = val_on.sum()
        if n_on >= 2:
            g["on_age_interp"] = np.interp(g["age"], g.loc[val_on, "age"], g.loc[val_on, "on"])
        elif n_on == 1:
            g["on_age_interp"] = g.loc[val_on, "on"].iloc[0]
        else:
            g["on_age_interp"] = np.nan

        # Lags sequentiels
        g["off_prev"] = g["off"].ffill().shift(1)
        g["off_next"] = g["off"].bfill().shift(-1)
        return g

    processed = data.groupby("patient_id", group_keys=False).apply(_compute_patient_trajectories, include_groups=False)
    for col in ["patient_slope", "patient_linear_surrogate", "off_age_interp", "on_age_interp", "off_prev", "off_next"]:
        data[col] = processed[col]

    # 5. Residus et ecarts a la trajectoire attendue
    data["off_minus_on"] = data["off"] - data["on"]
    data["diff_from_mean_off"] = data["off"] - data["mean_off_patient"]
    data["diff_from_mean_on"] = data["on"] - data["mean_on_patient"]
    data["diff_from_interp_off"] = data["off"] - data["off_age_interp"]
    data["diff_from_interp_on"] = data["on"] - data["on_age_interp"]
    data["diff_from_linear_surrogate"] = data["surrogate"] - data["patient_linear_surrogate"]

    # 6. Typage categoriel
    for col in ["cohort", "gene", "rater_id"]:
        if col in data.columns:
            data[col] = data[col].astype("category")

    # 7. Remise en ordre conforme a l'index initial et exclusion des identifiants
    data = data.loc[orig_index]
    drop_cols = [c for c in ["Index", "patient_id", "target"] if c in data.columns]
    return data.drop(columns=drop_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Modele HGBR longitudinal et pharmacodynamique (09_longitudinal)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MODELE HGBR LONGITUDINAL & PHARMACODYNAMIQUE (09_longitudinal)", flush=True)
    print("Sous-tache 7 - Reconstruction de trajectoire et cinetique Bateman", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement des donnees
    print("\n[1/5] Chargement des donnees et extraction des caracteristiques...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")
    patient_ids = visits["patient_id"]
    y = visits["target"]

    X = prepare_longitudinal_features(visits)
    X_test = prepare_longitudinal_features(X_test_raw)

    print(f"  Shape X: {X.shape}, Shape X_test: {X_test.shape}", flush=True)
    print(f"  Nombre de variables : {X.shape[1]}", flush=True)

    # 2. Validation croisee groupee par patient
    print("\n[2/5] Configuration de GroupKFold (5 plis par patient)...", flush=True)
    cv_grouped = GroupKFold(n_splits=5)
    cv_splits = list(cv_grouped.split(X, y, groups=patient_ids))

    # 3. Pipeline et evaluation skore
    print("\n[3/5] Evaluation avec skore.evaluate...", flush=True)
    model = tabular_pipeline(
        HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.03,
            min_samples_leaf=10,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    report = evaluate(model, X, y, splitter=cv_splits)

    try:
        rmse_summary = report.metrics.rmse()
        print(f"  Score RMSE moyen en GroupKFold :\n{rmse_summary}", flush=True)
    except Exception as exc:
        print(f"  Rapport genere (erreur acces metrique: {exc})", flush=True)

    # 4. Entrainement final et generation de la soumission
    print("\n[4/5] Entrainement final sur tout le train et prediction sur X_test...", flush=True)
    model.fit(X, y)
    raw_preds = model.predict(X_test)
    predictions = np.clip(raw_preds, 0.0, 132.0)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "09_longitudinal.csv"
    submission_df = pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions})
    submission_df.to_csv(sub_path, index=False)
    print(f"  Fichier de soumission genere : {sub_path} ({len(submission_df)} lignes)", flush=True)
    print(f"  Statistiques des predictions : min={predictions.min():.2f}, mean={predictions.mean():.2f}, max={predictions.max():.2f}", flush=True)

    # 5. Publication sur Skore Hub
    if args.no_hub:
        print("\n[5/5] Publication Skore Hub ignoree (--no-hub).", flush=True)
        print("=" * 70, flush=True)
        return

    print(f"\n[5/5] Publication du rapport sur Skore Hub sous la cle '{HUB_KEY}'...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    workspace = config.get("workspace")
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=workspace)
    project.put(HUB_KEY, report)
    print("  Publication sur Skore Hub terminee avec succes !", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
