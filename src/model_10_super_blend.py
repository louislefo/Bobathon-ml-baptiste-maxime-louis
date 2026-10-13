"""Diverse multi-seed ensemble blend of longitudinal models (10_super_blend).

Sub-task: Ensemble diversifie et multi-graines longitudinal (Sous-tache 8)
Project: Bobathon ESILV (Probabl x IBM)

Strategie :
1. Exploite toutes les variables longitudinales et PKPD de 09_longitudinal.
2. Combine par VotingRegressor 5 modeles HGBR avec des configurations et graines diversifiees :
   - Modele 1 (graine 0)   : max_iter 500, lr 0.030, min_leaf 10, nodes 31
   - Modele 2 (graine 42)  : max_iter 500, lr 0.030, min_leaf 10, nodes 31
   - Modele 3 (graine 123) : max_iter 600, lr 0.025, min_leaf 15, nodes 45
   - Modele 4 (graine 456) : max_iter 450, lr 0.035, min_leaf 8,  nodes 25
   - Modele 5 (graine 789) : max_iter 500, lr 0.030, min_leaf 12, nodes 35
3. Evaluation rigoureuse sur GroupKFold (5 plis par patient).
4. Entrainement final et generation de submissions/10_super_blend.csv avec bornage strict [0.0, 132.0].
5. Publication du rapport sur Skore Hub sous la cle '10_super_blend'.
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
HUB_KEY = "10_super_blend"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_09_longitudinal import prepare_longitudinal_features


def build_super_ensemble() -> VotingRegressor:
    """Construit un VotingRegressor combinant des architectures HGBR diversifiees."""
    configs = [
        ("hgbr_base_0", 0, 500, 0.030, 10, 31, 1.0),
        ("hgbr_base_42", 42, 500, 0.030, 10, 31, 1.0),
        ("hgbr_deep_123", 123, 600, 0.025, 15, 45, 1.5),
        ("hgbr_light_456", 456, 450, 0.035, 8, 25, 0.5),
        ("hgbr_reg_789", 789, 500, 0.030, 12, 35, 1.0),
    ]

    estimators = [
        (
            name,
            tabular_pipeline(
                HistGradientBoostingRegressor(
                    random_state=seed,
                    max_iter=max_iter,
                    learning_rate=lr,
                    min_samples_leaf=min_leaf,
                    max_leaf_nodes=nodes,
                    l2_regularization=l2,
                )
            ),
        )
        for name, seed, max_iter, lr, min_leaf, nodes, l2 in configs
    ]
    return VotingRegressor(estimators=estimators, n_jobs=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble diversifie longitudinal (10_super_blend)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("ENSEMBLE DIVERSIFIE & MULTI-GRAINES LONGITUDINAL (10_super_blend)", flush=True)
    print("Sous-tache 8 - VotingRegressor multi-configurations sur variables 09", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement des donnees et calcul des caracteristiques
    print("\n[1/5] Chargement et enrichissement des donnees...", flush=True)
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

    # 2. Preparation de la validation croisee groupee par patient
    print("\n[2/5] Configuration de GroupKFold (5 plis par patient)...", flush=True)
    cv_grouped = GroupKFold(n_splits=5)
    cv_splits = list(cv_grouped.split(X, y, groups=patient_ids))

    # 3. Evaluation skore
    print("\n[3/5] Evaluation du super blend avec skore.evaluate...", flush=True)
    ensemble = build_super_ensemble()
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
    sub_path = SUBMISSIONS_DIR / "10_super_blend.csv"
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
