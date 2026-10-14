"""Ultimate ensemble: LOVO + Longitudinal + Isotonic calibration (19_ultimate).

Sub-task: Meta-Ensemble Ultimate (Sous-tache 19)
Project: Bobathon ESILV (Probabl x IBM)

Strategie :
Combiner les deux meilleures familles de features (longitudinale et LOVO+Bateman OFF)
dans un VotingRegressor diversifie, puis appliquer une calibration isotonique sur
les predictions OOF.

Composition du VotingRegressor (7 estimateurs) :
  - 3 HGBR longitudinaux (features model_09, graines 0/42/123, poids 0.18 chacun)
  - 2 HGBR LOVO+BatemanOFF (features model_15, graines 0/42, poids 0.18 chacun)
  - 1 Ridge longitudinal (features model_09, poids 0.07)
  - 1 Ridge LOVO (features model_15, poids 0.09)
Total poids = 3×0.18 + 2×0.18 + 0.07 + 0.09 = 0.54 + 0.36 + 0.16 = 1.10 -> normalise.

Calibration isotonique sur les predictions OOF du VotingRegressor.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.base import BaseEstimator, RegressorMixin
from skrub import tabular_pipeline
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "19_ultimate"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_09_longitudinal import prepare_longitudinal_features
from src.model_15_bateman_off import prepare_lovo_features_bateman_off


class UltimateEnsemble(BaseEstimator, RegressorMixin):
    """VotingRegressor sur deux sets de features distincts + calibration isotonique.

    Le VotingRegressor sklearn ne supporte pas nativement des features differentes
    par estimateur. Cette classe wrapper gere la dualite des features.
    """

    def __init__(self):
        self.models_long_: list = []
        self.models_lovo_: list = []
        self.weights_long_ = [0.18, 0.18, 0.18, 0.07]   # 3 HGBR + 1 Ridge
        self.weights_lovo_ = [0.18, 0.18, 0.09]           # 2 HGBR + 1 Ridge
        self.iso_: IsotonicRegression | None = None
        self._fitted = False

    def _build_long_models(self) -> list:
        return [
            tabular_pipeline(HistGradientBoostingRegressor(
                random_state=0, max_iter=500, learning_rate=0.030,
                min_samples_leaf=10, max_leaf_nodes=31, l2_regularization=1.0,
            )),
            tabular_pipeline(HistGradientBoostingRegressor(
                random_state=42, max_iter=500, learning_rate=0.030,
                min_samples_leaf=10, max_leaf_nodes=31, l2_regularization=1.0,
            )),
            tabular_pipeline(HistGradientBoostingRegressor(
                random_state=123, max_iter=600, learning_rate=0.025,
                min_samples_leaf=15, max_leaf_nodes=45, l2_regularization=1.5,
            )),
            tabular_pipeline(Ridge(alpha=50.0)),
        ]

    def _build_lovo_models(self) -> list:
        return [
            tabular_pipeline(HistGradientBoostingRegressor(
                random_state=0, max_iter=400, learning_rate=0.03,
                min_samples_leaf=15, max_leaf_nodes=31, l2_regularization=1.5,
            )),
            tabular_pipeline(HistGradientBoostingRegressor(
                random_state=42, max_iter=450, learning_rate=0.025,
                min_samples_leaf=20, max_leaf_nodes=31, l2_regularization=2.0,
            )),
            tabular_pipeline(Ridge(alpha=30.0)),
        ]

    def fit(self, X, y, X_lovo=None):
        """Entraine tous les modeles. X = longitudinal, X_lovo = LOVO+BatemanOFF."""
        self.models_long_ = self._build_long_models()
        self.models_lovo_ = self._build_lovo_models()
        for m in self.models_long_:
            m.fit(X, y)
        if X_lovo is not None:
            for m in self.models_lovo_:
                m.fit(X_lovo, y)
        self._fitted = True
        return self

    def predict_raw(self, X, X_lovo=None) -> np.ndarray:
        """Prediction brute (moyenne ponderee sans calibration isotonique)."""
        preds = []
        for m, w in zip(self.models_long_, self.weights_long_):
            preds.append(m.predict(X) * w)
        if X_lovo is not None:
            for m, w in zip(self.models_lovo_, self.weights_lovo_):
                preds.append(m.predict(X_lovo) * w)
        total_w = sum(self.weights_long_) + (sum(self.weights_lovo_) if X_lovo is not None else 0)
        return np.sum(preds, axis=0) / total_w

    def predict(self, X):
        """Utilise uniquement X longitudinal (pour compatibilite skore.evaluate)."""
        return self.predict_raw(X)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ultimate ensemble LOVO+Long+Isotonic (19_ultimate)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("META-ENSEMBLE ULTIMATE LOVO + LONGITUDINAL + ISOTONIC (19_ultimate)", flush=True)
    print("Sous-tache 19 - Meilleur ensemble toutes familles combinees", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement
    print("\n[1/6] Chargement et double extraction de features...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")
    patient_ids = visits["patient_id"]
    y = visits["target"]

    # Features longitudinales (model_09)
    X_long = prepare_longitudinal_features(visits)
    X_test_long = prepare_longitudinal_features(X_test_raw)

    # Features LOVO + Bateman OFF (model_15)
    X_lovo = prepare_lovo_features_bateman_off(visits)
    X_test_lovo = prepare_lovo_features_bateman_off(X_test_raw)

    print(f"  Features longitudinales : {X_long.shape[1]} colonnes", flush=True)
    print(f"  Features LOVO+BatemanOFF : {X_lovo.shape[1]} colonnes", flush=True)

    # 2. GroupKFold
    print("\n[2/6] Configuration GroupKFold (5 plis)...", flush=True)
    cv_grouped = GroupKFold(n_splits=5)
    cv_splits = list(cv_grouped.split(X_long, y, groups=patient_ids))

    # 3. Predictions OOF pour la calibration isotonique
    print("\n[3/6] Generation des predictions OOF (toutes familles)...", flush=True)
    oof_preds = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_long_tr = X_long.iloc[train_idx]
        X_long_va = X_long.iloc[val_idx]
        X_lovo_tr = X_lovo.iloc[train_idx]
        X_lovo_va = X_lovo.iloc[val_idx]
        y_tr = y.iloc[train_idx]

        ens = UltimateEnsemble()
        ens.fit(X_long_tr, y_tr, X_lovo=X_lovo_tr)
        oof_preds[val_idx] = ens.predict_raw(X_long_va, X_lovo=X_lovo_va)

        rmse_fold = np.sqrt(np.mean((oof_preds[val_idx] - y.iloc[val_idx].values) ** 2))
        print(f"  Fold {fold_idx + 1}/5 RMSE brut = {rmse_fold:.4f}", flush=True)

    rmse_oof_brut = np.sqrt(np.mean((oof_preds - y.values) ** 2))
    print(f"\n  RMSE OOF brut (avant calibration) : {rmse_oof_brut:.4f}", flush=True)

    # 4. Calibration isotonique sur les OOF
    print("\n[4/6] Calibration isotonique sur les OOF...", flush=True)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_preds, y.values)
    oof_calibrated = np.clip(iso.transform(oof_preds), 0.0, 132.0)
    rmse_oof_cal = np.sqrt(np.mean((oof_calibrated - y.values) ** 2))
    print(f"  RMSE OOF apres calibration isotonique : {rmse_oof_cal:.4f}", flush=True)
    print(f"  Gain calibration : {rmse_oof_brut - rmse_oof_cal:+.4f} RMSE", flush=True)

    # 5. Rapport skore sur les modeles longitudinaux seuls (pour compatibilite API)
    print("\n[5/6] Rapport skore.evaluate (famille longitudinale seule)...", flush=True)
    from sklearn.ensemble import VotingRegressor
    ensemble_long_only = VotingRegressor(
        estimators=[
            ("h0", tabular_pipeline(HistGradientBoostingRegressor(
                random_state=0, max_iter=500, learning_rate=0.030,
                min_samples_leaf=10, max_leaf_nodes=31, l2_regularization=1.0,
            ))),
            ("h42", tabular_pipeline(HistGradientBoostingRegressor(
                random_state=42, max_iter=500, learning_rate=0.030,
                min_samples_leaf=10, max_leaf_nodes=31, l2_regularization=1.0,
            ))),
            ("h123", tabular_pipeline(HistGradientBoostingRegressor(
                random_state=123, max_iter=600, learning_rate=0.025,
                min_samples_leaf=15, max_leaf_nodes=45, l2_regularization=1.5,
            ))),
            ("ridge", tabular_pipeline(Ridge(alpha=50.0))),
        ],
        weights=[0.30, 0.30, 0.30, 0.10],
        n_jobs=-1,
    )
    report = evaluate(ensemble_long_only, X_long, y, splitter=cv_splits)
    try:
        print(f"  RMSE skore.evaluate (longitudinal):\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  (erreur: {exc})", flush=True)

    # Entrainement final sur tout le train
    print("  Entrainement final ultimate ensemble...", flush=True)
    ens_final = UltimateEnsemble()
    ens_final.fit(X_long, y, X_lovo=X_lovo)

    raw_preds_test = ens_final.predict_raw(X_test_long, X_lovo=X_test_lovo)
    calibrated_test = iso.transform(raw_preds_test)
    predictions = np.clip(calibrated_test, 0.0, 132.0)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "19_ultimate.csv"
    submission_df = pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions})
    submission_df.to_csv(sub_path, index=False)
    print(f"  Fichier : {sub_path} ({len(submission_df)} lignes)", flush=True)
    print(f"  Predictions : min={predictions.min():.2f}, mean={predictions.mean():.2f}, max={predictions.max():.2f}", flush=True)

    # Hub
    if args.no_hub:
        print("\n[6/6] Hub ignore (--no-hub).", flush=True)
        print("=" * 70, flush=True)
        return

    print(f"\n[6/6] Publication Hub sous '{HUB_KEY}'...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
    project.put(HUB_KEY, report)
    print("  Publication terminee !", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
