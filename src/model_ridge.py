"""Model Ridge (02_ridge) for Bobathon ESILV.

Trains a regularized linear model (Ridge) with median imputation,
evaluates it with skore, generates submissions/02_ridge.csv,
and pushes the report to Skore Hub.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials


def main() -> None:
    print("=" * 60, flush=True)
    print("ENTRAINEMENT DU MODELE RIDGE (02_ridge)", flush=True)
    print("=" * 60, flush=True)

    # 1. Chargement des donnees
    print("\n[1/4] Chargement des donnees...", flush=True)
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train.merge(y_train, on="Index")

    # Feature engineering de base
    feature_cols = [
        "sexM",
        "age_at_diagnosis",
        "age",
        "ledd",
        "time_since_intake_on",
        "time_since_intake_off",
        "on",
        "off",
    ]
    X = visits[feature_cols]
    y = visits["target"]

    # 2. Pipeline Ridge avec Imputation
    print("\n[2/4] Construction de la pipeline et evaluation via skore...", flush=True)
    ridge_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=10.0),
    )

    report = evaluate(ridge_pipeline, X, y)
    rmse = report.metrics.rmse()
    print(f"  Score RMSE (holdout standard) : {rmse:.4f}", flush=True)
    print("  (A titre de comparaison, le dummy baseline etait a ~16.48)", flush=True)

    # 3. Fit complet et generation du fichier de soumission
    print("\n[3/4] Entrainement sur l'integralite du train set et generation des predictions...", flush=True)
    ridge_pipeline.fit(X, y)
    test_preds = ridge_pipeline.predict(X_test[feature_cols])

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "02_ridge.csv"
    submission_df = pd.DataFrame({"Index": X_test["Index"], "target": test_preds})
    submission_df.to_csv(sub_path, index=False)
    print(f"  Fichier genere : {sub_path} ({len(submission_df)} lignes)", flush=True)

    # 4. Publication sur Skore Hub
    print("\n[4/4] Publication du rapport '02_ridge' sur Skore Hub...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    workspace = config.get("workspace")
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=workspace)
    project.put("02_ridge", report)
    print("  Publication terminee !", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
