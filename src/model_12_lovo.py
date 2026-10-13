"""HistGradientBoosting model with Leave-One-Visit-Out (LOVO) Trajectories (12_lovo).

Sub-task: Modele LOVO anti-surapprentissage (Sous-tache 10)
Project: Bobathon ESILV (Probabl x IBM)

Innovations :
1. Calcul LOVO (Leave-One-Visit-Out) intra-patient :
   Pour chaque visite i d'un patient, la trajectoire (moyenne et droite de regression de severite)
   est ajustee EXCLUSIVEMENT sur les autres visites {j != i} du patient.
   Cela elimine toute fuite d'information et empeche l'arbre de memoriser sa propre projection.
2. Pharmacodynamie Levodopa de Bateman integree dans le substitut continu.
3. Elagage strict des variables parasites et redondantes selon les diagnostics Skore (SKD012/SKD008).
4. Validation croisee groupee par patient (GroupKFold 5 plis).
5. Generation de submissions/12_lovo.csv et publication du rapport sur Skore Hub ('12_lovo').
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
HUB_KEY = "12_lovo"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials

BATEMAN_A = 45.5346
BATEMAN_KE = 0.1870
BATEMAN_KA = 0.5708


def prepare_lovo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare les variables explicatives en appliquant le schema LOVO intra-patient."""
    data = df.copy()
    orig_index = data.index
    data = data.sort_values(["patient_id", "age"])

    # 1. Progression temporelle et suivi
    data["time_since_diagnosis"] = data["age"] - data["age_at_diagnosis"]
    data["visit_number"] = data.groupby("patient_id")["age"].rank(method="first").astype(int)
    data["n_visits_patient"] = data.groupby("patient_id")["age"].transform("count")
    data["visit_ratio"] = data["visit_number"] / data["n_visits_patient"]

    # 2. Pharmacodynamie Levodopa de Bateman
    t_on = data["time_since_intake_on"].fillna(2.5).clip(lower=0.0)
    data["bateman_on"] = BATEMAN_A * (np.exp(-BATEMAN_KE * t_on) - np.exp(-BATEMAN_KA * t_on))
    data["on_adjusted"] = data["on"] + data["bateman_on"]
    data["surrogate"] = data["off"].fillna(data["on_adjusted"])

    # 3. Statistiques standards par patient
    data["mean_off_patient"] = data.groupby("patient_id")["off"].transform("mean")
    data["mean_on_patient"] = data.groupby("patient_id")["on"].transform("mean")
    data["mean_surrogate_patient"] = data.groupby("patient_id")["surrogate"].transform("mean")
    data["std_off_patient"] = data.groupby("patient_id")["off"].transform("std")
    data["std_on_patient"] = data.groupby("patient_id")["on"].transform("std")
    data["mean_ledd_patient"] = data.groupby("patient_id")["ledd"].transform("mean")

    # 4. Trajectoires Leave-One-Visit-Out (LOVO) par patient
    def _compute_patient_lovo(g: pd.DataFrame) -> pd.DataFrame:
        ages = g["age"].values
        surrs = g["surrogate"].values
        n = len(ages)

        lovo_means = np.empty(n)
        lovo_slopes = np.empty(n)
        lovo_preds = np.empty(n)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            a = ages[mask]
            s = surrs[mask]
            lovo_means[i] = np.mean(s)
            if len(a) >= 2:
                da = a - np.mean(a)
                ds = s - np.mean(s)
                var_a = np.sum(da**2)
                if var_a > 1e-6:
                    slope = np.sum(da * ds) / var_a
                    intercept = np.mean(s) - slope * np.mean(a)
                else:
                    slope = 2.47
                    intercept = np.mean(s) - 2.47 * np.mean(a)
                lovo_slopes[i] = slope
                lovo_preds[i] = slope * ages[i] + intercept
            else:
                lovo_slopes[i] = 2.47
                lovo_preds[i] = s[0]

        g["lovo_mean_surrogate"] = lovo_means
        g["lovo_trajectory_slope"] = lovo_slopes
        g["lovo_trajectory_pred"] = lovo_preds
        g["diff_from_lovo_pred"] = g["surrogate"] - lovo_preds

        # Interpolation de off selon l'age
        val_off = g["off"].notnull()
        n_off = val_off.sum()
        if n_off >= 2:
            g["off_age_interp"] = np.interp(g["age"], g.loc[val_off, "age"], g.loc[val_off, "off"])
        elif n_off == 1:
            g["off_age_interp"] = g.loc[val_off, "off"].iloc[0]
        else:
            g["off_age_interp"] = np.nan

        # Lags sequentiels
        g["off_prev"] = g["off"].ffill().shift(1)
        g["off_next"] = g["off"].bfill().shift(-1)
        return g

    processed = data.groupby("patient_id", group_keys=False).apply(_compute_patient_lovo, include_groups=False)
    for col in ["lovo_mean_surrogate", "lovo_trajectory_slope", "lovo_trajectory_pred", "diff_from_lovo_pred", "off_age_interp", "off_prev", "off_next"]:
        data[col] = processed[col]

    # 5. Ecarts et differences
    data["off_minus_on"] = data["off"] - data["on"]
    data["diff_from_mean_off"] = data["off"] - data["mean_off_patient"]
    data["diff_from_mean_on"] = data["on"] - data["mean_on_patient"]
    data["diff_from_interp_off"] = data["off"] - data["off_age_interp"]

    # 6. Typage categoriel de rater_id
    data["rater_id"] = data["rater_id"].astype("category")

    # 7. Remise en ordre conforme a l'index initial et exclusion des colonnes parasites (SKD012/SKD008)
    data = data.loc[orig_index]
    drop_cols = [
        c for c in [
            "Index", "patient_id", "target", "cohort", "gene", "sexM",
            "time_since_first_visit", "time_since_intake_off"
        ] if c in data.columns
    ]
    return data.drop(columns=drop_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Modele HGBR LOVO Trajectories (12_lovo)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MODELE HGBR AVEC TRAJECTOIRES LOVO (12_lovo)", flush=True)
    print("Sous-tache 10 - Leave-One-Visit-Out sans fuite intra-patient", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement et preparation des donnees
    print("\n[1/5] Chargement et extraction des features LOVO...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")
    patient_ids = visits["patient_id"]
    y = visits["target"]

    X = prepare_lovo_features(visits)
    X_test = prepare_lovo_features(X_test_raw)

    print(f"  Shape X: {X.shape}, Shape X_test: {X_test.shape}", flush=True)
    print(f"  Nombre de variables : {X.shape[1]}", flush=True)

    # 2. Validation croisee groupee par patient
    print("\n[2/5] Configuration de GroupKFold (5 plis par patient)...", flush=True)
    cv_grouped = GroupKFold(n_splits=5)
    cv_splits = list(cv_grouped.split(X, y, groups=patient_ids))

    # 3. Pipeline et evaluation skore
    print("\n[3/5] Evaluation skore.evaluate...", flush=True)
    model = tabular_pipeline(
        HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.03,
            min_samples_leaf=15,
            max_leaf_nodes=31,
            l2_regularization=1.5,
            random_state=42,
        )
    )

    report = evaluate(model, X, y, splitter=cv_splits)

    try:
        rmse_summary = report.metrics.rmse()
        print(f"  Score RMSE moyen en GroupKFold :\n{rmse_summary}", flush=True)
    except Exception as exc:
        print(f"  Rapport genere (erreur acces metrique: {exc})", flush=True)

    try:
        checks_summary = report.checks.summarize()
        print(f"  Resume des controles Skore :\n{checks_summary}", flush=True)
    except Exception as exc:
        print(f"  Erreur controles: {exc}", flush=True)

    # 4. Entrainement final et generation de la soumission
    print("\n[4/5] Entrainement final sur tout le train et prediction sur X_test...", flush=True)
    model.fit(X, y)
    raw_preds = model.predict(X_test)
    predictions = np.clip(raw_preds, 0.0, 132.0)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "12_lovo.csv"
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
