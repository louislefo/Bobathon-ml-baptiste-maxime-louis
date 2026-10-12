"""HistGradientBoosting model with temporal & intra-patient feature engineering (04_hgbr_features).

Author: Maxime Laurent (Trinome Bobathon ESILV)
Sub-task: Sous-tâche 2 - Feature engineering temporel intra-patient

Key features added:
1. Temporal visit rank per patient: visit_number (chronological order via age).
2. Follow-up progression: time_since_first_visit (age - min(age) per patient).
3. Patient-level aggregated baseline:
   - mean_off_patient, mean_on_patient
   - std_off_patient, std_on_patient
   - mean_ledd_patient
   - n_visits_patient
4. Clinical response dynamics:
   - off_minus_on (treatment effect amplitude)
   - diff_from_mean_off, diff_from_mean_on (visit deviation from patient baseline)
5. Clinical background features:
   - time_since_diagnosis = age - age_at_diagnosis
6. Native missingness handling via HistGradientBoostingRegressor.
7. GroupKFold(5) grouped by patient_id.
8. Evaluation reporting on Skore Hub under key '04_hgbr_features'.
9. Generation of submissions/04_hgbr_features.csv.
"""

from __future__ import annotations

import json
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

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare les variables explicatives avec l'ingenierie temporelle et intra-patient."""
    data = df.copy()

    # 1. Progression de la maladie
    data["time_since_diagnosis"] = data["age"] - data["age_at_diagnosis"]

    # 2. Caracteristiques temporelles et de suivi par patient
    data["visit_number"] = data.groupby("patient_id")["age"].rank(method="first").astype(int)
    data["n_visits_patient"] = data.groupby("patient_id")["age"].transform("count")
    data["time_since_first_visit"] = data["age"] - data.groupby("patient_id")["age"].transform("min")

    # 3. Statistiques intra-patient (profil de base de severite motrice)
    mean_off = data.groupby("patient_id")["off"].transform("mean")
    mean_on = data.groupby("patient_id")["on"].transform("mean")
    data["mean_off_patient"] = mean_off
    data["mean_on_patient"] = mean_on
    data["std_off_patient"] = data.groupby("patient_id")["off"].transform("std")
    data["std_on_patient"] = data.groupby("patient_id")["on"].transform("std")
    data["mean_ledd_patient"] = data.groupby("patient_id")["ledd"].transform("mean")

    # 4. Dynamique therapeutique et ecarts a la moyenne du patient
    data["off_minus_on"] = data["off"] - data["on"]
    data["diff_from_mean_off"] = data["off"] - mean_off
    data["diff_from_mean_on"] = data["on"] - mean_on

    # 5. Typage des colonnes categorielles
    for col in ["cohort", "gene", "rater_id"]:
        if col in data.columns:
            data[col] = data[col].astype("category")

    # 6. Exclusion des identifiants et de la cible
    drop_cols = [c for c in ["Index", "patient_id", "target"] if c in data.columns]
    return data.drop(columns=drop_cols)


def main() -> None:
    print("=" * 70, flush=True)
    print("ENTRAINEMENT HGBR AVEC FEATURES INTRA-PATIENT & TEMPORELLES (04_hgbr_features)", flush=True)
    print("Auteur : Maxime Laurent (Trinome Bobathon ESILV)", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement des donnees
    print("\n[1/5] Chargement et enrichissement des donnees...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")

    groups = visits["patient_id"]
    y = visits["target"]

    print("  Calcul des features temporelles et intra-patient...", flush=True)
    X = prepare_features(visits)
    X_test = prepare_features(X_test_raw)

    print(f"  Nombre de features : {X.shape[1]}")
    print(f"  Colonnes : {list(X.columns)}", flush=True)

    # 2. GroupKFold par patient
    print("\n[2/5] Preparation de GroupKFold (5 plis par patient_id)...", flush=True)
    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

    # 3. Modele HGBR via skrub tabular_pipeline
    print("\n[3/5] Evaluation du modele avec skore.evaluate...", flush=True)
    model = tabular_pipeline(
        HistGradientBoostingRegressor(
            random_state=42,
            max_iter=150,
            learning_rate=0.08,
            min_samples_leaf=20,
        )
    )

    report = evaluate(model, X, y, splitter=cv_splits)

    try:
        rmse = report.metrics.rmse()
        print(f"  -> Score RMSE moyen en GroupKFold : {rmse:.4f}", flush=True)
        print("  (Pour rappel : 01_dummy = ~16.48, 03_hgbr = ~7.45)", flush=True)
    except Exception as exc:
        print(f"  Rapport genere (calcul direct rmse via metrics indisponible: {exc})", flush=True)

    # 4. Entrainement final et generation du fichier de soumission
    print("\n[4/5] Entrainement sur l'ensemble du jeu d'entrainement et generation des predictions test...", flush=True)
    model.fit(X, y)
    predictions = np.clip(model.predict(X_test), 0.0, 132.0)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "04_hgbr_features.csv"
    submission_df = pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions})
    submission_df.to_csv(sub_path, index=False)
    print(f"  Fichier de soumission genere : {sub_path} ({len(submission_df)} lignes)", flush=True)

    # 5. Publication sur Skore Hub
    print("\n[5/5] Publication du rapport sur Skore Hub sous la cle '04_hgbr_features'...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    workspace = config.get("workspace")
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=workspace)
    project.put("04_hgbr_features", report)
    print("  Publication sur Skore Hub terminee avec succes !", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
