"""Modele 23 : Super Blend Orthogonal (Trajectoire Bayesienne 21 + Surrogate Calibre 22).

Project: Bobathon ESILV (Probabl x IBM)
Key Hub: '23_super_blend'

Architecture :
1. Pilier 1 (Model 21 Smooth) :
   - Trajectoire quadratique bayesienne intra-patient avec perte de Huber (Etage B).
   - Modele residuel d'arbres sur variables longitudinales (Etage C).
   - Lissage parabolique analytique et projection isotonique monotone.
2. Pilier 2 (Model 22 Adaptive) :
   - Debiaisage affine des 50 evaluateurs (rater_id) sur le signal OFF.
   - Inversion pharmacodynamique Levodopa (terme d'interaction ON * C(t)) et correction rater.
   - Surrogate de severite debiaise et regression HGBR strictement regularisee (early stopping, L2=5.0).
   - Lissage et forçage de monotonie.
3. Méta-Ensemble Orthogonal :
   - Combinaison convexe pondérée (50 % Model 21 + 50 % Model 22).
   - Annulation du bruit résiduel indépendant et projection isotonique monotone finale.
   - Bornage clinique strict [0.0, 132.0].
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "23_super_blend"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_15_trajectory import PatientTrajectoryRegressor, add_longitudinal_columns
from src.model_21_smooth import smooth_predictions
from src.model_22_adaptive_trajectory import AdaptiveTrajectoryRegressor


class SuperBlendTrajectoryRegressor(RegressorMixin, BaseEstimator):
    """Méta-estimateur unifié combinant Model 21 Smooth et Model 22 Adaptive."""

    def __init__(
        self,
        weight_21: float = 0.50,
        random_state: int = 42,
    ):
        self.weight_21 = weight_21
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        X_df = X.reset_index(drop=True).copy()
        y_arr = np.asarray(y, dtype=float)

        self.model_21_ = PatientTrajectoryRegressor(random_state=self.random_state).fit(X_df, y_arr)
        self.model_22_ = AdaptiveTrajectoryRegressor(random_state=self.random_state).fit(X_df, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_df = X.reset_index(drop=True).copy()

        pred_21_raw = self.model_21_.predict(X_df)
        pred_21_smooth = smooth_predictions(
            X_df["patient_id"].values,
            X_df["age"].values,
            pred_21_raw,
            alpha=1.0,
            degree=2,
            monotone=True,
        )

        pred_22 = self.model_22_.predict(X_df)

        # Melange orthogonal des deux familles
        w = self.weight_21
        blend = w * pred_21_smooth + (1.0 - w) * pred_22

        # Projection monotone isotonique finale
        final_pred = smooth_predictions(
            X_df["patient_id"].values,
            X_df["age"].values,
            blend,
            alpha=0.0,
            monotone=True,
        )
        return np.clip(final_pred, 0.0, 132.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Modele 23 : Super Blend Orthogonal")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    parser.add_argument("--weight-21", type=float, default=0.50, help="Poids attribue au modele 21 (reste au modele 22)")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MODELE 23 : SUPER BLEND ORTHOGONAL (23_super_blend)", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/5] Chargement et preparation des donnees...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")
    y = visits["target"]
    X = add_longitudinal_columns(visits.drop(columns=["target"]))
    X_test = add_longitudinal_columns(X_test_raw)

    print(f"  Train: {len(X)} visites | Test: {len(X_test)} visites", flush=True)

    print("\n[2/5] Configuration GroupKFold (5 plis sur patient_id)...", flush=True)
    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

    print(f"\n[3/5] Evaluation officielle via skore.evaluate (Poids 21 = {args.weight_21:.2f})...", flush=True)
    model = SuperBlendTrajectoryRegressor(
        weight_21=args.weight_21,
        random_state=42,
    )

    report = evaluate(model, X, y, splitter=cv_splits)

    try:
        rmse_val = report.metrics.rmse()
        print(f"\n  Score RMSE GroupKFold :\n{rmse_val}", flush=True)
    except Exception as exc:
        print(f"  Acces rmse : {exc}", flush=True)

    try:
        mae_val = report.metrics.mae()
        print(f"\n  Score MAE GroupKFold :\n{mae_val}", flush=True)
    except Exception as exc:
        print(f"  Acces mae : {exc}", flush=True)

    try:
        print(f"\n  Resume des verifications Skore :\n{report.checks.summarize(fast_mode=True)}", flush=True)
    except Exception as exc:
        print(f"  Controles Skore : {exc}", flush=True)

    print("\n[4/5] Entrainement final sur tout le Train et prediction sur X_test...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"{HUB_KEY}.csv"
    pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions}).to_csv(sub_path, index=False)
    print(f"  Soumission enregistree : {sub_path} ({len(predictions)} lignes)", flush=True)
    print(f"  Statistiques : min={predictions.min():.2f}, mean={predictions.mean():.2f}, max={predictions.max():.2f}", flush=True)

    if args.no_hub:
        print("\n[5/5] Publication Skore Hub ignoree (--no-hub).", flush=True)
        return

    print(f"\n[5/5] Publication sur Skore Hub sous la cle '{HUB_KEY}'...", flush=True)
    try:
        config = load_skore_credentials(REPO_ROOT)
        login(mode="hub")
        project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
        project.put(HUB_KEY, report)
        print("  Rapport publie sur Skore Hub avec succes !", flush=True)
    except Exception as exc:
        print(f"  Avertissement Skore Hub: {exc}", flush=True)

    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
