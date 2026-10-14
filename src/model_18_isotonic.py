"""Isotonic calibration post-hoc on 11_meta_ensemble OOF predictions (18_isotonic).

Sub-task: Calibration isotonique post-hoc (Sous-tache 18)
Project: Bobathon ESILV (Probabl x IBM)

Idee :
Les HGBR peuvent systematiquement sur/sous-estimer aux extremes de la distribution
(scores < 10 ou > 80). La regression isotonique ajustee sur les predictions OOF
corrige ce biais de calibration sans changer l'architecture.

Protocole anti-leakage strict :
1. Generer les predictions OOF de 11_meta_ensemble via GroupKFold (cross_val_predict).
2. Ajuster IsotonicRegression(out_of_bounds='clip') sur (oof_preds, y_train).
3. Predictions finales = iso.transform(ensemble_trained_on_all.predict(X_test)).
4. Bornage final [0, 132].

Note : la calibration isotonique est monotone -> ne peut pas inverser les predictions.
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
from skrub import tabular_pipeline
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "18_isotonic"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_09_longitudinal import prepare_longitudinal_features


def build_meta_ensemble() -> VotingRegressor:
    """Meme architecture que 11_meta_ensemble pour une comparaison equitable."""
    estimators = [
        ("hgbr_seed_0", tabular_pipeline(HistGradientBoostingRegressor(
            random_state=0, max_iter=500, learning_rate=0.030,
            min_samples_leaf=10, max_leaf_nodes=31, l2_regularization=1.0,
        ))),
        ("hgbr_seed_42", tabular_pipeline(HistGradientBoostingRegressor(
            random_state=42, max_iter=500, learning_rate=0.030,
            min_samples_leaf=10, max_leaf_nodes=31, l2_regularization=1.0,
        ))),
        ("hgbr_seed_123", tabular_pipeline(HistGradientBoostingRegressor(
            random_state=123, max_iter=600, learning_rate=0.025,
            min_samples_leaf=15, max_leaf_nodes=45, l2_regularization=1.5,
        ))),
        ("hgbr_seed_456", tabular_pipeline(HistGradientBoostingRegressor(
            random_state=456, max_iter=450, learning_rate=0.035,
            min_samples_leaf=8, max_leaf_nodes=25, l2_regularization=0.5,
        ))),
        ("ridge_linear", tabular_pipeline(Ridge(alpha=50.0))),
    ]
    return VotingRegressor(estimators=estimators, weights=[0.22, 0.22, 0.22, 0.22, 0.12], n_jobs=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration isotonique sur 11_meta_ensemble (18_isotonic)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("META-ENSEMBLE + CALIBRATION ISOTONIQUE (18_isotonic)", flush=True)
    print("Sous-tache 18 - Correction du biais systematique aux extremes", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement
    print("\n[1/6] Chargement et extraction des variables longitudinales (model_09)...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")
    patient_ids = visits["patient_id"]
    y = visits["target"]

    X = prepare_longitudinal_features(visits)
    X_test = prepare_longitudinal_features(X_test_raw)

    print(f"  Shape X: {X.shape}, Shape X_test: {X_test.shape}", flush=True)

    # 2. GroupKFold
    print("\n[2/6] Configuration GroupKFold (5 plis)...", flush=True)
    cv_grouped = GroupKFold(n_splits=5)
    cv_splits = list(cv_grouped.split(X, y, groups=patient_ids))

    # 3. Generer les predictions OOF du meta-ensemble
    print("\n[3/6] Generation des predictions OOF via cross_val_predict...", flush=True)
    ensemble_for_oof = build_meta_ensemble()
    oof_preds = cross_val_predict(
        ensemble_for_oof, X, y,
        cv=cv_splits,
        n_jobs=1,
    )
    rmse_before = np.sqrt(np.mean((oof_preds - y.values) ** 2))
    print(f"  RMSE OOF avant calibration : {rmse_before:.4f}", flush=True)

    # Diagnostic de calibration : deciles de y vs deciles de oof_preds
    y_arr = y.values
    percentiles = [10, 25, 50, 75, 90]
    print("  Calibration diagnostique (target vs oof_pred par percentile) :", flush=True)
    for p in percentiles:
        t_p = np.percentile(y_arr, p)
        o_p = np.percentile(oof_preds, p)
        print(f"    P{p:3d}: target={t_p:.1f}, oof_pred={o_p:.1f}, biais={o_p - t_p:+.1f}", flush=True)

    # 4. Ajustement de la calibration isotonique sur les OOF
    print("\n[4/6] Ajustement de IsotonicRegression sur les predictions OOF...", flush=True)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_preds, y_arr)

    # Verification : RMSE apres calibration isotonique sur les OOF
    oof_calibrated = iso.transform(oof_preds)
    oof_calibrated_clipped = np.clip(oof_calibrated, 0.0, 132.0)
    rmse_after = np.sqrt(np.mean((oof_calibrated_clipped - y_arr) ** 2))
    print(f"  RMSE OOF apres calibration isotonique : {rmse_after:.4f}", flush=True)
    print(f"  Gain calibration : {rmse_before - rmse_after:+.4f} RMSE", flush=True)

    # 5. Rapport skore : evaluate le pipeline ensemble sur les splits (sans calibration)
    print("\n[5/6] Rapport skore.evaluate (ensemble brut pour rapport Hub)...", flush=True)
    ensemble_for_report = build_meta_ensemble()
    report = evaluate(ensemble_for_report, X, y, splitter=cv_splits)
    try:
        print(f"  RMSE skore.evaluate (sans calibration):\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  (erreur acces metrique: {exc})", flush=True)

    # Entrainement final + calibration isotonique pour soumission
    print("  Entrainement final de l'ensemble sur tout le train...", flush=True)
    ensemble_final = build_meta_ensemble()
    ensemble_final.fit(X, y)
    raw_preds_test = ensemble_final.predict(X_test)
    calibrated_preds_test = iso.transform(raw_preds_test)
    predictions = np.clip(calibrated_preds_test, 0.0, 132.0)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "18_isotonic.csv"
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
