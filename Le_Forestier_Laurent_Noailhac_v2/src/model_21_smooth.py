"""Lissage quadratique + monotonie par patient des predictions du modele 15 (21_smooth).

Sub-task: Post-traitement de trajectoire (Sous-tache 14)
Project: Bobathon ESILV (Probabl x IBM)

Constat : la vraie cible est, par patient, une courbe lisse (parabole de l'age, RMSE
residuel 0.31) et jamais decroissante (0 baisse sur 39 014 transitions). Les predictions
du modele 15 ne respectent ni l'un ni l'autre.

Post-traitement (sans toucher au modele 15, simplement importe) :
1. Parabole ajustee par patient sur les predictions du 15 en fonction de l'age.
2. Melange : alpha * parabole + (1 - alpha) * prediction brute.
3. Optionnel : regression isotonique par patient (prediction croissante avec l'age).

`--quick` : calcule une seule fois les predictions hors-fold du 15 (sauvees dans
`oof/15_trajectory_oof.csv`, reutilisables pour des blends), puis compare toutes les
variantes de post-traitement sans reentrainer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
OOF_DIR = REPO_ROOT / "oof"
HUB_KEY = "21_smooth"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_15_trajectory import PatientTrajectoryRegressor, add_longitudinal_columns


def smooth_predictions(patient_id, age, pred, alpha=1.0, degree=2, monotone=True):
    """Lisse les predictions par patient (parabole + melange), puis impose la monotonie."""
    out = np.asarray(pred, dtype=float).copy()
    age = np.asarray(age, dtype=float)
    groups = pd.Series(np.arange(len(out))).groupby(np.asarray(patient_id)).indices
    for idx in groups.values():
        a, p = age[idx], out[idx]
        if alpha > 0 and len(idx) > degree:
            t = a - a.mean()
            fit = np.polyval(np.polyfit(t, p, degree), t)
            p = alpha * fit + (1 - alpha) * p
        if monotone and len(idx) > 1:
            p = IsotonicRegression(increasing=True).fit_transform(a, p)
        out[idx] = p
    return np.clip(out, 0.0, 132.0)


class SmoothedTrajectoryRegressor(RegressorMixin, BaseEstimator):
    """Modele 15 suivi du lissage par patient."""

    def __init__(self, alpha=1.0, degree=2, monotone=True, random_state=0):
        self.alpha = alpha
        self.degree = degree
        self.monotone = monotone
        self.random_state = random_state

    def fit(self, X, y):
        self.base_ = PatientTrajectoryRegressor(random_state=self.random_state).fit(X, y)
        return self

    def predict(self, X):
        base = self.base_.predict(X)
        return smooth_predictions(X["patient_id"].values, X["age"].values, base,
                                  self.alpha, self.degree, self.monotone)


def compute_oof(X, y, cv_splits, path: Path) -> np.ndarray:
    """Predictions hors-fold du modele 15 (mises en cache sur disque)."""
    if path.is_file():
        print(f"  Cache OOF trouve : {path}", flush=True)
        return pd.read_csv(path)["oof"].values
    oof = np.zeros(len(X))
    for k, (tr, va) in enumerate(cv_splits):
        model = PatientTrajectoryRegressor().fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict(X.iloc[va])
        rmse = np.sqrt(np.mean((oof[va] - y.iloc[va].values) ** 2))
        print(f"  pli {k}: RMSE 15 = {rmse:.4f}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Index": X["Index"].values, "fold": _fold_ids(cv_splits, len(X)),
                  "oof": oof}).to_csv(path, index=False)
    print(f"  OOF sauvegardees : {path}", flush=True)
    return oof


def _fold_ids(cv_splits, n):
    folds = np.zeros(n, dtype=int)
    for k, (_, va) in enumerate(cv_splits):
        folds[va] = k
    return folds


def quick_grid(X, y, cv_splits) -> None:
    oof = compute_oof(X, y, cv_splits, OOF_DIR / "15_trajectory_oof.csv")
    folds = _fold_ids(cv_splits, len(X))
    yv = y.values

    def score(pred):
        per_fold = [np.sqrt(np.mean((pred[folds == k] - yv[folds == k]) ** 2))
                    for k in range(len(cv_splits))]
        return np.mean(per_fold), np.std(per_fold)

    m, s = score(oof)
    print(f"\n  Reference modele 15 : {m:.4f} +- {s:.4f}", flush=True)
    rows = []
    for degree in (1, 2):
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            for monotone in (False, True):
                if alpha == 0.0 and not monotone:
                    continue
                pred = smooth_predictions(X["patient_id"].values, X["age"].values, oof,
                                          alpha, degree, monotone)
                m, s = score(pred)
                rows.append((degree, alpha, monotone, m, s))
    res = pd.DataFrame(rows, columns=["degree", "alpha", "monotone", "rmse", "std"])
    print(res.sort_values("rmse").to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lissage par patient du modele 15 (21_smooth)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    parser.add_argument("--quick", action="store_true",
                        help="Grille de post-traitements sur les OOF du 15 (pas de skore)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--no-monotone", action="store_true")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("LISSAGE PAR PATIENT DU MODELE 15 (21_smooth)", flush=True)
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
        print("\n[3/5] Grille de post-traitements sur les OOF du modele 15...", flush=True)
        quick_grid(X, y, cv_splits)
        return

    print(f"\n[3/5] Evaluation skore.evaluate (alpha={args.alpha}, degree={args.degree}, "
          f"monotone={not args.no_monotone})...", flush=True)
    model = SmoothedTrajectoryRegressor(alpha=args.alpha, degree=args.degree,
                                        monotone=not args.no_monotone)
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
