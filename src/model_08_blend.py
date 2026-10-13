"""Multi-seed ensemble blend of tuned HistGradientBoostingRegressor (08_hgbr_blend).

Sub-task: Sous-tache 6 du plan d'optimisation
Project: Bobathon ESILV (Probabl x IBM)

Strategie :
1. Reprend l'ingenierie de variables de 06_hgbr_tuned (features patient + rater_id).
2. Construit un ensemble multi-graines (VotingRegressor sur 5 graines differentes : 0, 42, 123, 456, 789).
3. Evalue le modele avec skore.evaluate et GroupKFold(5) groupe par patient_id.
4. Entraine l'ensemble final sur 100% du jeu d'entrainement.
5. Predite sur X_test avec bornage strict [0.0, 132.0].
6. Sauvegarde submissions/08_hgbr_blend.csv.
7. Publie le rapport sur Skore Hub sous la cle '08_hgbr_blend'.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "08_hgbr_blend"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_06_tuned import prepare_features

SEEDS = [0, 42, 123, 456, 789]


def build_ensemble() -> VotingRegressor:
    """Construit un VotingRegressor multi-graines avec les hyperparametres optimaux."""
    estimators = [
        (
            f"hgbr_seed_{seed}",
            tabular_pipeline(
                HistGradientBoostingRegressor(
                    max_iter=500,
                    learning_rate=0.03,
                    min_samples_leaf=10,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=seed,
                )
            ),
        )
        for seed in SEEDS
    ]
    return VotingRegressor(estimators=estimators, n_jobs=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed HGBR blend (08_hgbr_blend)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("ENSEMBLE MULTI-GRAINES HGBR (08_hgbr_blend)", flush=True)
    print(f"Graines utilisees : {SEEDS}", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement et preparation des donnees
    print("\n[1/5] Chargement des donnees et calcul des features...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")
    patient_ids = visits["patient_id"]
    y = visits["target"]

    X = prepare_features(visits)
    X_test = prepare_features(X_test_raw)

    print(f"  Shape X: {X.shape}, Shape X_test: {X_test.shape}", flush=True)

    # 2. Validation croisee groupee par patient
    print("\n[2/5] Preparation de GroupKFold (5 plis par patient)...", flush=True)
    cv_grouped = GroupKFold(n_splits=5)
    cv_splits = list(cv_grouped.split(X, y, groups=patient_ids))

    # 3. Evaluation avec skore
    print("\n[3/5] Evaluation du blend multi-graines avec skore.evaluate...", flush=True)
    ensemble = build_ensemble()
    report = evaluate(ensemble, X, y, splitter=cv_splits)

    try:
        rmse_summary = report.metrics.rmse()
        print(f"  Score RMSE moyen en GroupKFold :\n{rmse_summary}", flush=True)
    except Exception as exc:
        print(f"  Rapport genere (erreur acces metrique: {exc})", flush=True)

    # 4. Entrainement final et generation de la soumission
    print("\n[4/5] Entrainement final sur l'ensemble du train et prediction test...", flush=True)
    ensemble.fit(X, y)
    raw_preds = ensemble.predict(X_test)
    predictions = np.clip(raw_preds, 0.0, 132.0)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "08_hgbr_blend.csv"
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
