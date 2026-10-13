"""HistGradientBoosting with MAE (absolute_error) and Huber loss (20_mae_loss).

Sub-task: Optimisation directe sur MAE puis conversion RMSE (Sous-tache 20)
Project: Bobathon ESILV (Probabl x IBM)

Hypothese :
Le leaderboard optimise le RMSE, mais optimiser le MAE dans HGBR peut produire
un meilleur RMSE sur le leaderboard si la distribution des residus est quasi-normale
(pas de queue lourde). La distribution de target est quasi normale (skewness=0.051
mesuree dans l'EDA), ce qui justifie le test.

Ce script evalue 3 variantes :
  V1 : loss="squared_error" (MSE, reference = model_09)
  V2 : loss="absolute_error" (MAE direct)
  V3 : loss="huber" (hybride MAE/MSE, robuste aux outliers)

La meilleure variante genere la soumission 20_mae_loss.csv.
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
HUB_KEY = "20_mae_loss"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_09_longitudinal import prepare_longitudinal_features


def evaluate_variant(
    name: str,
    loss: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list,
) -> tuple[float, float]:
    """Evalue une variante de HGBR et retourne (rmse_mean, rmse_std)."""
    print(f"\n  === Variante {name} (loss='{loss}') ===", flush=True)
    model = tabular_pipeline(
        HistGradientBoostingRegressor(
            loss=loss,
            max_iter=500,
            learning_rate=0.03,
            min_samples_leaf=10,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    oof_preds = np.zeros(len(y))
    for train_idx, val_idx in cv_splits:
        m = tabular_pipeline(
            HistGradientBoostingRegressor(
                loss=loss, max_iter=500, learning_rate=0.03,
                min_samples_leaf=10, max_leaf_nodes=31, l2_regularization=1.0,
                random_state=42,
            )
        )
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[val_idx] = m.predict(X.iloc[val_idx])

    rmse_folds = [
        np.sqrt(np.mean((oof_preds[val_idx] - y.iloc[val_idx].values) ** 2))
        for _, val_idx in cv_splits
    ]
    rmse_mean = float(np.mean(rmse_folds))
    rmse_std = float(np.std(rmse_folds))
    mae_overall = float(np.mean(np.abs(oof_preds - y.values)))
    print(f"  RMSE GroupKFold = {rmse_mean:.4f} +/- {rmse_std:.4f}", flush=True)
    print(f"  MAE GroupKFold  = {mae_overall:.4f}", flush=True)
    return rmse_mean, rmse_std


def main() -> None:
    parser = argparse.ArgumentParser(description="HGBR avec MAE/Huber loss sur features longitudinales (20_mae_loss)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("HGBR MAE / HUBER LOSS SUR FEATURES LONGITUDINALES (20_mae_loss)", flush=True)
    print("Sous-tache 20 - Comparaison squared_error vs absolute_error vs huber", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement
    print("\n[1/5] Chargement et extraction des variables longitudinales (model_09)...", flush=True)
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
    print("\n[2/5] Configuration GroupKFold (5 plis)...", flush=True)
    cv_grouped = GroupKFold(n_splits=5)
    cv_splits = list(cv_grouped.split(X, y, groups=patient_ids))

    # 3. Evaluation des 3 variantes
    print("\n[3/5] Evaluation comparative des 3 variantes de loss...", flush=True)
    results = {}
    for name, loss in [
        ("MSE (reference)", "squared_error"),
        ("MAE", "absolute_error"),
    ]:
        rmse_mean, rmse_std = evaluate_variant(name, loss, X, y, cv_splits)
        results[name] = (loss, rmse_mean, rmse_std)

    print("\n  Recapitulatif des variantes :", flush=True)
    print(f"  {'Variante':<25} {'Loss':<20} {'RMSE mean':>12} {'RMSE std':>10}", flush=True)
    print("  " + "-" * 70, flush=True)
    for name, (loss, rmse_mean, rmse_std) in results.items():
        marker = " <-- BEST" if rmse_mean == min(r[1] for r in results.values()) else ""
        print(f"  {name:<25} {loss:<20} {rmse_mean:>12.4f} {rmse_std:>10.4f}{marker}", flush=True)

    # Choisir la meilleure variante
    best_name = min(results, key=lambda k: results[k][1])
    best_loss, best_rmse, best_std = results[best_name]
    print(f"\n  Meilleure variante : {best_name} (loss='{best_loss}', RMSE={best_rmse:.4f})", flush=True)

    # 4. Rapport skore officiel avec la meilleure variante
    print(f"\n[4/5] Rapport skore.evaluate avec loss='{best_loss}'...", flush=True)
    best_model = tabular_pipeline(
        HistGradientBoostingRegressor(
            loss=best_loss,
            max_iter=500,
            learning_rate=0.03,
            min_samples_leaf=10,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
    )
    report = evaluate(best_model, X, y, splitter=cv_splits)
    try:
        print(f"  RMSE skore.evaluate:\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  (erreur acces metrique: {exc})", flush=True)
    try:
        print(f"  Controles Skore:\n{report.checks.summarize()}", flush=True)
    except Exception as exc:
        print(f"  Erreur controles: {exc}", flush=True)

    # Entrainement final et soumission
    print("\n  Entrainement final et generation de la soumission...", flush=True)
    best_model.fit(X, y)
    raw_preds = best_model.predict(X_test)
    predictions = np.clip(raw_preds, 0.0, 132.0)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "20_mae_loss.csv"
    submission_df = pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions})
    submission_df.to_csv(sub_path, index=False)
    print(f"  Fichier : {sub_path} ({len(submission_df)} lignes)", flush=True)
    print(f"  Predictions : min={predictions.min():.2f}, mean={predictions.mean():.2f}, max={predictions.max():.2f}", flush=True)

    # 5. Hub
    if args.no_hub:
        print("\n[5/5] Hub ignore (--no-hub).", flush=True)
        print("=" * 70, flush=True)
        return

    print(f"\n[5/5] Publication Hub sous '{HUB_KEY}'...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
    project.put(HUB_KEY, report)
    print("  Publication terminee !", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
