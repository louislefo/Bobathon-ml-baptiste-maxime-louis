"""HistGradientBoosting model with LOVO Trajectories and Early Stopping Regularization (13_regularized_lovo).

Sub-task: Modele LOVO regularise anti-surapprentissage avec Early Stopping (Sous-tache 11)
Project: Bobathon ESILV (Probabl x IBM)

Innovations :
1. Variables LOVO (Leave-One-Visit-Out) intra-patient sans auto-inclusion.
2. Mecanisme d'arret precoce (early_stopping=True, n_iter_no_change=25) pour empecher
   le gradient boosting de memoriser le bruit des derniers arbres.
3. Renforcement de la regularisation L2 (l2=4.0) et des feuilles minimales (min_samples_leaf=25)
   visant a reduire l'ecart train/test sous le seuil critique de 10% (verification SKD001).
4. Validation croisee groupee par patient (GroupKFold 5 plis).
5. Generation de submissions/13_regularized_lovo.csv et publication du rapport sur Skore Hub ('13_regularized_lovo').
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
HUB_KEY = "13_regularized_lovo"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_12_lovo import prepare_lovo_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Modele HGBR LOVO avec Early Stopping (13_regularized_lovo)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MODELE HGBR LOVO AVEC EARLY STOPPING REGULARISE (13_regularized_lovo)", flush=True)
    print("Sous-tache 11 - Elimination stricte du surapprentissage SKD001", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement et extraction des features LOVO
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
    print("\n[3/5] Evaluation skore.evaluate avec early_stopping...", flush=True)
    model = tabular_pipeline(
        HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.035,
            min_samples_leaf=25,
            max_leaf_nodes=25,
            l2_regularization=4.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
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
    sub_path = SUBMISSIONS_DIR / "13_regularized_lovo.csv"
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
