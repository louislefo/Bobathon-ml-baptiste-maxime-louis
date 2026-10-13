"""Weighted meta-ensemble combining diverse HGBR regressors and linear Ridge (11_meta_ensemble).

Sub-task: Meta-ensemble hybride arbre + lineaire (Sous-tache 9)
Project: Bobathon ESILV (Probabl x IBM)

Strategie :
1. Exploite toutes les 37 variables longitudinales et PKPD de 09_longitudinal.
2. Combine par VotingRegressor pondere :
   - 4 modeles HistGradientBoostingRegressor (graines 0, 42, 123, 456, architectures variees, poids 0.22 chacun)
   - 1 modele Ridge regularise (alpha=50.0 via tabular_pipeline, poids 0.12)
3. L'ajout du signal lineaire Ridge etend la portee des arbres pour les scores moteurs extremes
   (au-dela de 90 points) tout en preservant la precision intra-patient du gradient boosting.
4. Evaluation avec skore.evaluate sur GroupKFold (5 plis groupes par patient_id).
5. Entrainement final et generation de submissions/11_meta_ensemble.csv avec bornage strict [0.0, 132.0].
6. Publication du rapport sur Skore Hub sous la cle '11_meta_ensemble'.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "11_meta_ensemble"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_09_longitudinal import prepare_longitudinal_features


def build_meta_ensemble() -> VotingRegressor:
    """Construit un VotingRegressor pondere alliant HGBR et Ridge."""
    estimators = [
        (
            "hgbr_seed_0",
            tabular_pipeline(
                HistGradientBoostingRegressor(
                    random_state=0,
                    max_iter=500,
                    learning_rate=0.030,
                    min_samples_leaf=10,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                )
            ),
        ),
        (
            "hgbr_seed_42",
            tabular_pipeline(
                HistGradientBoostingRegressor(
                    random_state=42,
                    max_iter=500,
                    learning_rate=0.030,
                    min_samples_leaf=10,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                )
            ),
        ),
        (
            "hgbr_seed_123",
            tabular_pipeline(
                HistGradientBoostingRegressor(
                    random_state=123,
                    max_iter=600,
                    learning_rate=0.025,
                    min_samples_leaf=15,
                    max_leaf_nodes=45,
                    l2_regularization=1.5,
                )
            ),
        ),
        (
            "hgbr_seed_456",
            tabular_pipeline(
                HistGradientBoostingRegressor(
                    random_state=456,
                    max_iter=450,
                    learning_rate=0.035,
                    min_samples_leaf=8,
                    max_leaf_nodes=25,
                    l2_regularization=0.5,
                )
            ),
        ),
        (
            "ridge_linear",
            tabular_pipeline(Ridge(alpha=50.0)),
        ),
    ]

    weights = [0.22, 0.22, 0.22, 0.22, 0.12]
    return VotingRegressor(estimators=estimators, weights=weights, n_jobs=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-ensemble pondere HGBR + Ridge (11_meta_ensemble)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("META-ENSEMBLE HYBRIDE HGBR + RIDGE (11_meta_ensemble)", flush=True)
    print("Sous-tache 9 - Assemblage pondere multi-architectures sur variables 09", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement et extraction des variables
    print("\n[1/5] Chargement et extraction des variables longitudinales...", flush=True)
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

    # 3. Evaluation skore
    print("\n[3/5] Evaluation du meta-ensemble avec skore.evaluate...", flush=True)
    ensemble = build_meta_ensemble()
    report = evaluate(ensemble, X, y, splitter=cv_splits)

    try:
        rmse_summary = report.metrics.rmse()
        print(f"  Score RMSE moyen en GroupKFold :\n{rmse_summary}", flush=True)
    except Exception as exc:
        print(f"  Rapport genere (erreur acces metrique: {exc})", flush=True)

    # 4. Entrainement final et generation de la soumission
    print("\n[4/5] Entrainement final de l'ensemble et prediction sur X_test...", flush=True)
    ensemble.fit(X, y)
    raw_preds = ensemble.predict(X_test)
    predictions = np.clip(raw_preds, 0.0, 132.0)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "11_meta_ensemble.csv"
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
