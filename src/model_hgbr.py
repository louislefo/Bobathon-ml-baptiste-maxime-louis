"""HistGradientBoosting model (03_hgbr) for Bobathon ESILV.

Best-performing model leveraging:
1. Feature engineering (time_since_diagnosis = age - age_at_diagnosis).
2. Native missingness handling (treating NaNs as clinical signals).
3. Encodage categoriel pour cohort et gene.
4. Validation croisee groupee par patient (GroupKFold sur patient_id).
5. Generation de submissions/03_hgbr.csv et reporting Skore Hub.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
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
    """Prepare les variables explicatives et effectue le feature engineering clinique."""
    data = df.copy()

    # Duree depuis le diagnostic clinique (feature cle recommandee dans la documentation)
    data["time_since_diagnosis"] = data["age"] - data["age_at_diagnosis"]

    # Conversion des variables categorielles
    for col in ["cohort", "gene"]:
        if col in data.columns:
            data[col] = data[col].astype("category")

    # Colonnes a exclure de l'apprentissage (identifiants ou cible)
    drop_cols = [c for c in ["Index", "patient_id", "target"] if c in data.columns]
    return data.drop(columns=drop_cols)


def main() -> None:
    print("=" * 60, flush=True)
    print("ENTRAINEMENT DU MODELE HGBR AVEC GROUPKFOLD (03_hgbr)", flush=True)
    print("=" * 60, flush=True)

    # 1. Chargement des donnees
    print("\n[1/5] Chargement et enrichissement des donnees...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")

    groups = visits["patient_id"]
    y = visits["target"]
    X = prepare_features(visits)
    X_test = prepare_features(X_test_raw)

    print(f"  Nombre de features utilisees : {X.shape[1]} {list(X.columns)}", flush=True)

    # 2. Preparation de la validation croisee groupee par patient (GroupKFold)
    print("\n[2/5] Preparation de GroupKFold (5 plis par patient)...", flush=True)
    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

    # 3. Modele HistGradientBoosting via pipeline skrub
    print("\n[3/5] Evaluation du modele avec skore (gestion native des NaNs)...", flush=True)
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
        print(f"  Score RMSE moyen en GroupKFold : {rmse:.4f}", flush=True)
    except Exception:
        print("  Rapport d'evaluation genere.", flush=True)

    # 4. Entrainement final et generation de la soumission
    print("\n[4/5] Entrainement final sur tout le jeu d'entrainement...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "03_hgbr.csv"
    submission_df = pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions})
    submission_df.to_csv(sub_path, index=False)
    print(f"  Fichier de soumission genere : {sub_path} ({len(submission_df)} lignes)", flush=True)

    # 5. Publication sur Skore Hub
    print("\n[5/5] Publication du rapport sur Skore Hub sous la cle '03_hgbr'...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    workspace = config.get("workspace")
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=workspace)
    project.put("03_hgbr", report)
    print("  Publication terminee avec succes !", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
