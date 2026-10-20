"""Blend du modele 22 (mesures debiaisees) et du modele 11 (meta-ensemble) + lissage (23_blend).

Sub-task: Blend de modeles complementaires (Sous-tache 16)
Project: Bobathon ESILV (Probabl x IBM)

Le 22 (trajectoires patient) et le 11 (HGBR + Ridge sur features longitudinales) font des
erreurs differentes : leur moyenne ponderee reduit la variance. Le lissage quadratique +
monotone par patient (modele 21) est applique APRES le blend.

Modeles 11, 21 et 22 importes, non modifies.

`--quick` : calcule les predictions hors-fold des deux modeles (cache dans `oof/`), puis
compare les poids de blend sans reentrainer.
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
OOF_DIR = REPO_ROOT / "oof"
HUB_KEY = "23_blend"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_11_meta_ensemble import build_meta_ensemble
from src.model_15_trajectory import LF_PREFIX, add_longitudinal_columns
from src.model_21_smooth import smooth_predictions
from src.model_22_debiased import DebiasedTrajectoryRegressor

WEIGHT_22 = 0.85


def longitudinal_view(X: pd.DataFrame) -> pd.DataFrame:
    """Features du modele 09/11 (colonnes `lf_` sans prefixe)."""
    cols = [c for c in X.columns if c.startswith(LF_PREFIX)]
    return X[cols].rename(columns=lambda c: c[len(LF_PREFIX):])


class BlendRegressor(RegressorMixin, BaseEstimator):
    """weight_22 * modele 22 (brut) + (1 - weight_22) * modele 11, puis lissage patient."""

    def __init__(self, weight_22=WEIGHT_22, smooth=True, random_state=0):
        self.weight_22 = weight_22
        self.smooth = smooth
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        self.model_22_ = DebiasedTrajectoryRegressor(smooth=False,
                                                     random_state=self.random_state).fit(X, y)
        self.model_11_ = build_meta_ensemble().fit(longitudinal_view(X), y)
        return self

    def predict_components(self, X: pd.DataFrame) -> tuple:
        X = X.reset_index(drop=True)
        p22 = self.model_22_.predict(X)
        p11 = np.clip(self.model_11_.predict(longitudinal_view(X)), 0.0, 132.0)
        return p22, p11

    def predict(self, X: pd.DataFrame):
        p22, p11 = self.predict_components(X)
        pred = self.weight_22 * p22 + (1 - self.weight_22) * p11
        if self.smooth:
            pred = smooth_predictions(X["patient_id"].values, X["age"].values, pred)
        return np.clip(pred, 0.0, 132.0)


def quick_grid(X: pd.DataFrame, y: pd.Series, cv_splits) -> None:
    """OOF des deux modeles (mises en cache), puis grille de poids."""
    path = OOF_DIR / "23_components_oof.csv"
    folds = np.zeros(len(X), dtype=int)
    for k, (_, va) in enumerate(cv_splits):
        folds[va] = k
    if path.is_file():
        print(f"  Cache OOF trouve : {path}", flush=True)
        oof = pd.read_csv(path)
        p22, p11 = oof["p22"].values, oof["p11"].values
    else:
        p22, p11 = np.zeros(len(X)), np.zeros(len(X))
        for k, (tr, va) in enumerate(cv_splits):
            model = BlendRegressor().fit(X.iloc[tr], y.iloc[tr])
            p22[va], p11[va] = model.predict_components(X.iloc[va])
            print(f"  pli {k} calcule", flush=True)
        OOF_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Index": X["Index"].values, "fold": folds, "p22": p22,
                      "p11": p11}).to_csv(path, index=False)
        print(f"  OOF sauvegardees : {path}", flush=True)
    yv = y.values
    pid, age = X["patient_id"].values, X["age"].values
    for w in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5):
        pred = smooth_predictions(pid, age, w * p22 + (1 - w) * p11)
        per_fold = [np.sqrt(np.mean((pred[folds == k] - yv[folds == k]) ** 2))
                    for k in range(len(cv_splits))]
        print(f"  poids 22 = {w:.2f} : RMSE = {np.mean(per_fold):.4f} +- {np.std(per_fold):.4f}",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend 22 + 11 + lissage (23_blend)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    parser.add_argument("--quick", action="store_true",
                        help="Grille de poids sur predictions hors-fold (pas de skore)")
    parser.add_argument("--weight-22", type=float, default=WEIGHT_22)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("BLEND 22 + 11 + LISSAGE PATIENT (23_blend)", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/5] Chargement des donnees...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
    visits = X_train_raw.merge(y_train_raw, on="Index")
    y = visits["target"]
    X = add_longitudinal_columns(visits.drop(columns=["target"]))
    X_test = add_longitudinal_columns(X_test_raw)

    print("\n[2/5] Configuration de GroupKFold (5 plis par patient)...", flush=True)
    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

    if args.quick:
        print("\n[3/5] Grille de poids...", flush=True)
        quick_grid(X, y, cv_splits)
        return

    print(f"\n[3/5] Evaluation skore.evaluate (poids 22 = {args.weight_22})...", flush=True)
    model = BlendRegressor(weight_22=args.weight_22)
    report = evaluate(model, X, y, splitter=cv_splits)
    try:
        print(f"  Score RMSE moyen en GroupKFold :\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  Rapport genere (erreur acces metrique: {exc})", flush=True)
    try:
        # fast_mode : SKD012 (permutation importance) serait beaucoup trop lent ici
        print(f"  Resume des controles Skore :\n{report.checks.summarize(fast_mode=True)}",
              flush=True)
    except Exception as exc:
        print(f"  Erreur controles: {exc}", flush=True)

    print("\n[4/5] Entrainement final sur tout le train et prediction sur X_test...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"{HUB_KEY}.csv"
    pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions}).to_csv(sub_path, index=False)
    print(f"  Fichier de soumission genere : {sub_path} ({len(predictions)} lignes)", flush=True)
    print(f"  Statistiques des predictions : min={predictions.min():.2f}, "
          f"mean={predictions.mean():.2f}, max={predictions.max():.2f}", flush=True)

    if args.no_hub:
        print("\n[5/5] Publication Skore Hub ignoree (--no-hub).", flush=True)
        return

    print(f"\n[5/5] Publication du rapport sur Skore Hub sous la cle '{HUB_KEY}'...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
    project.put(HUB_KEY, report)
    print("  Publication sur Skore Hub terminee avec succes !", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
