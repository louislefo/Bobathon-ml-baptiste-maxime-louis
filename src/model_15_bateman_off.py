"""HistGradientBoosting model with Bateman OFF correction (15_bateman_off).

Sub-task: Correction Bateman OFF + time_since_intake_off (Sous-tache 15)
Project: Bobathon ESILV (Probabl x IBM)

Innovations vs model_12_lovo :
1. Correction pharmacodynamique du signal OFF via la cinetique de Bateman.
   Pour les visites disposant de time_since_intake_off (21.2% du train), on calcule :
     bateman_off = A * (exp(-ke * t_off) - exp(-ka * t_off))
     off_corrected = off - bateman_off  (retire l'effet residuel de levodopa)
   Le surrogate LOVO est reconstruit en priorisant off_corrected > off > on_adjusted.
2. Inclusion de bateman_off, off_corrected et time_since_intake_off comme features.
   Les NaN de time_since_intake_off sont laisses intacts (signal de manquance + valeur quantitative).
3. Toute la logique LOVO (Leave-One-Visit-Out) est recalculee sur le nouveau surrogate.
4. Tous les anti-patterns du plan sont respectes (clipping [0,132], GroupKFold, pas de patient_id).
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
HUB_KEY = "15_bateman_off"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials

# Parametres de la cinetique pharmacodynamique de Bateman
# (ajustes sur les visites avec ON et OFF mesures simultanement dans model_09)
BATEMAN_A = 45.5346
BATEMAN_KE = 0.1870
BATEMAN_KA = 0.5708


def prepare_lovo_features_bateman_off(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare les variables LOVO enrichies de la correction Bateman OFF.

    Differences cles vs model_12_lovo.prepare_lovo_features :
    - bateman_off : effet levodopa residuel au moment de la mesure OFF
    - off_corrected : score OFF corrige de l'effet levodopa (= OFF vrai physiologique)
    - surrogate construit sur off_corrected > off > on_adjusted
    - time_since_intake_off inclus comme feature directe (NaN traites par HGBR)
    - Statistiques LOVO recalculees sur le nouveau surrogate
    """
    data = df.copy()
    orig_index = data.index
    data = data.sort_values(["patient_id", "age"])

    # 1. Progression temporelle et suivi
    data["time_since_diagnosis"] = data["age"] - data["age_at_diagnosis"]
    data["visit_number"] = data.groupby("patient_id")["age"].rank(method="first").astype(int)
    data["n_visits_patient"] = data.groupby("patient_id")["age"].transform("count")
    data["visit_ratio"] = data["visit_number"] / data["n_visits_patient"]

    # 2. Pharmacodynamie Levodopa de Bateman — signal ON (idem model_12)
    t_on = data["time_since_intake_on"].fillna(2.5).clip(lower=0.0)
    data["bateman_on"] = BATEMAN_A * (np.exp(-BATEMAN_KE * t_on) - np.exp(-BATEMAN_KA * t_on))
    data["on_adjusted"] = data["on"] + data["bateman_on"]

    # 3. Correction Bateman OFF — NOUVEAU vs model_12
    # t_off = 0 signifierait juste apres la prise : OFF tres bas (sous effet max)
    # t_off > 10h : la levodopa est presque eliminee, bateman_off -> 0
    # Moyenne observee : 14.2h -> effet residuel minimal mais non nul
    t_off = data["time_since_intake_off"].clip(lower=0.0)  # NaN restent NaN
    data["bateman_off"] = BATEMAN_A * (np.exp(-BATEMAN_KE * t_off) - np.exp(-BATEMAN_KA * t_off))

    # off_corrected = score OFF mesure - effet residuel levodopa au moment de la mesure
    # Quand t_off est grand (>10h), bateman_off -> 0 et off_corrected ~ off
    # Quand t_off est petit (<5h), la correction est importante
    data["off_corrected"] = data["off"] - data["bateman_off"]

    # 4. Surrogate continu : off_corrected > off brut > on_adjusted
    # - off_corrected quand time_since_intake_off disponible ET off disponible
    # - off brut sinon (quand off dispo mais pas time_since_intake_off)
    # - on_adjusted en dernier recours
    data["surrogate"] = (
        data["off_corrected"]
        .fillna(data["off"])
        .fillna(data["on_adjusted"])
    )

    # 5. Statistiques standards par patient
    data["mean_off_patient"] = data.groupby("patient_id")["off"].transform("mean")
    data["mean_on_patient"] = data.groupby("patient_id")["on"].transform("mean")
    data["mean_surrogate_patient"] = data.groupby("patient_id")["surrogate"].transform("mean")
    data["std_off_patient"] = data.groupby("patient_id")["off"].transform("std")
    data["std_on_patient"] = data.groupby("patient_id")["on"].transform("std")
    data["mean_ledd_patient"] = data.groupby("patient_id")["ledd"].transform("mean")

    # 6. Trajectoires Leave-One-Visit-Out (LOVO) sur le nouveau surrogate
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

        # Interpolation de off_corrected selon l'age (utilise off_corrected en priorite)
        val_off_corr = g["off_corrected"].notnull()
        n_off_corr = val_off_corr.sum()
        val_off = g["off"].notnull()
        n_off = val_off.sum()

        if n_off_corr >= 2:
            g["off_age_interp"] = np.interp(
                g["age"],
                g.loc[val_off_corr, "age"],
                g.loc[val_off_corr, "off_corrected"]
            )
        elif n_off >= 2:
            g["off_age_interp"] = np.interp(g["age"], g.loc[val_off, "age"], g.loc[val_off, "off"])
        elif n_off == 1:
            g["off_age_interp"] = g.loc[val_off, "off"].iloc[0]
        else:
            g["off_age_interp"] = np.nan

        # Lags sequentiels sur le surrogate
        g["off_prev"] = g["off"].ffill().shift(1)
        g["off_next"] = g["off"].bfill().shift(-1)
        return g

    processed = data.groupby("patient_id", group_keys=False).apply(
        _compute_patient_lovo, include_groups=False
    )
    for col in [
        "lovo_mean_surrogate", "lovo_trajectory_slope", "lovo_trajectory_pred",
        "diff_from_lovo_pred", "off_age_interp", "off_prev", "off_next",
    ]:
        data[col] = processed[col]

    # 7. Ecarts et differences
    data["off_minus_on"] = data["off"] - data["on"]
    data["diff_from_mean_off"] = data["off"] - data["mean_off_patient"]
    data["diff_from_mean_on"] = data["on"] - data["mean_on_patient"]
    data["diff_from_interp_off"] = data["off_corrected"].fillna(data["off"]) - data["off_age_interp"]

    # 8. Typage categoriel de rater_id
    data["rater_id"] = data["rater_id"].astype("category")

    # 9. Remise en ordre conforme a l'index initial
    # Note : time_since_intake_off est CONSERVE (signal de manquance + valeur quantitative)
    #        bateman_off et off_corrected sont AJOUTES
    #        Les colonnes exclues sont celles sans signal ou avec leakage
    data = data.loc[orig_index]
    drop_cols = [
        c for c in [
            "Index", "patient_id", "target",
            "cohort", "gene", "sexM",
            "time_since_first_visit",
        ] if c in data.columns
    ]
    return data.drop(columns=drop_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="HGBR LOVO avec correction Bateman OFF (15_bateman_off)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MODELE HGBR LOVO + CORRECTION BATEMAN OFF (15_bateman_off)", flush=True)
    print("Sous-tache 15 - Pharmacodynamie OFF + time_since_intake_off", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement et preparation
    print("\n[1/5] Chargement et extraction des features LOVO + Bateman OFF...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")
    patient_ids = visits["patient_id"]
    y = visits["target"]

    X = prepare_lovo_features_bateman_off(visits)
    X_test = prepare_lovo_features_bateman_off(X_test_raw)

    print(f"  Shape X: {X.shape}, Shape X_test: {X_test.shape}", flush=True)
    print(f"  Nombre de variables : {X.shape[1]}", flush=True)

    # Verification : bateman_off et off_corrected bien presents
    new_cols = [c for c in ["bateman_off", "off_corrected", "time_since_intake_off"] if c in X.columns]
    print(f"  Nouvelles features OFF : {new_cols}", flush=True)
    if "bateman_off" in X.columns:
        print(f"  bateman_off non-null : {X['bateman_off'].notna().sum()} ({X['bateman_off'].notna().mean()*100:.1f}%)", flush=True)
    if "off_corrected" in X.columns:
        print(f"  off_corrected non-null : {X['off_corrected'].notna().sum()} ({X['off_corrected'].notna().mean()*100:.1f}%)", flush=True)

    # 2. Validation croisee groupee
    print("\n[2/5] Configuration de GroupKFold (5 plis par patient)...", flush=True)
    cv_grouped = GroupKFold(n_splits=5)
    cv_splits = list(cv_grouped.split(X, y, groups=patient_ids))

    # 3. Pipeline et evaluation skore (parametres identiques a 12_lovo pour comparaison equitable)
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
    sub_path = SUBMISSIONS_DIR / "15_bateman_off.csv"
    submission_df = pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions})
    submission_df.to_csv(sub_path, index=False)
    print(f"  Fichier de soumission genere : {sub_path} ({len(submission_df)} lignes)", flush=True)
    print(
        f"  Statistiques des predictions : "
        f"min={predictions.min():.2f}, mean={predictions.mean():.2f}, max={predictions.max():.2f}",
        flush=True,
    )

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
