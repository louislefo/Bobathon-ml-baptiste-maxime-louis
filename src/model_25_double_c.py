"""Double etage final (base trajectoire / base coefficients) + blend a quatre (25_double_c).

Sub-task: Diversification de l'etage final (Sous-tache 18)
Project: Bobathon ESILV (Probabl x IBM)

Le 24 melange le 22, la regression des coefficients patient et le 11. Ici on ajoute un
second etage final : un HGBR qui corrige la prediction du modele de coefficients (au lieu
de corriger la trajectoire du 15 comme dans le 22), avec les coefficients predits en
variables. Les deux etages finaux partent de bases differentes, leurs erreurs sont moins
correlees, et le blend gagne nettement.

Prediction finale :
    0.4 * 22 (brut) + 0.4 * etage final sur base coefficients
  + 0.1 * coefficients + 0.1 * 11, puis lissage quadratique + monotone par patient (21).

Pas de fuite : sur le train, la base "coefficients" du second etage final est predite
hors-fold (GroupKFold patient interne) ; les variables d'etage A sont deja hors-fold (22).

Modeles 11, 21, 22, 24 importes, non modifies.
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
HUB_KEY = "25_double_c"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_11_meta_ensemble import build_meta_ensemble
from src.model_15_trajectory import _hgbr, add_longitudinal_columns
from src.model_21_smooth import smooth_predictions
from src.model_23_blend import longitudinal_view
from src.model_24_coef_blend import (COEF_PARAMS, COEFS, _Debiased22WithFeatures,
                                     patient_table, true_coefficients)

WEIGHTS = {"p22": 0.4, "c_coef": 0.4, "coef": 0.1, "p11": 0.1}
SECOND_C_PARAMS = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                       min_samples_leaf=100, l2_regularization=2.0)


def fit_coef_models(P: pd.DataFrame, C: pd.DataFrame, seed: int) -> dict:
    """Un HGBR par coefficient, en residu de la trajectoire debiaisee (t2) du 22."""
    return {b: _hgbr(random_state=seed, **COEF_PARAMS).fit(P, C[b].values - P[f"t2_{b}"].values)
            for b in COEFS}


def predict_coefs(models: dict, P: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Coefficients predits ramenes a chaque visite + prediction de la parabole."""
    pid = X["patient_id"].values
    coefs = {b: pd.Series(models[b].predict(P) + P[f"t2_{b}"].values, index=P.index).reindex(pid).values
             for b in COEFS}
    t = X["age"].values - P["age_mean"].reindex(pid).values
    out = pd.DataFrame({f"cf_{b}": coefs[b] for b in COEFS})
    out["cf_pred"] = np.clip(coefs["b0"] + coefs["b1"] * t + coefs["b2"] * t ** 2, 0.0, 132.0)
    return out


class DoubleFinalStageRegressor(RegressorMixin, BaseEstimator):
    """22 + second etage final sur base coefficients + coefficients + 11, puis lissage."""

    def __init__(self, w22=WEIGHTS["p22"], wccoef=WEIGHTS["c_coef"], wcoef=WEIGHTS["coef"],
                 w11=WEIGHTS["p11"], n_inner_splits=5, smooth=True, random_state=0):
        self.w22 = w22
        self.wccoef = wccoef
        self.wcoef = wcoef
        self.w11 = w11
        self.n_inner_splits = n_inner_splits
        self.smooth = smooth
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        seed = self.random_state
        self.model_22_ = _Debiased22WithFeatures(smooth=False, random_state=seed).fit(X, y)
        F = self.model_22_.train_features_.reset_index(drop=True)
        cats = self.model_22_.categories_

        # Coefficients : hors-fold sur le train (base du second etage), puis modele complet
        cf_oof = pd.DataFrame(index=range(len(X)), columns=["cf_b0", "cf_b1", "cf_b2", "cf_pred"],
                              dtype=float)
        splits = GroupKFold(n_splits=self.n_inner_splits).split(X, y, X["patient_id"])
        for tr, va in splits:
            Xtr, Xva = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True)
            Ptr = patient_table(F.iloc[tr], Xtr, cats)
            models = fit_coef_models(Ptr, true_coefficients(Xtr, y[tr], Ptr.index), seed)
            Pva = patient_table(F.iloc[va], Xva, cats)[Ptr.columns]
            cf_oof.iloc[va] = predict_coefs(models, Pva, Xva).values
        P = patient_table(F, X, cats)
        self.coef_columns_ = list(P.columns)
        self.coef_models_ = fit_coef_models(P, true_coefficients(X, y, P.index), seed)

        # Second etage final : corrige la prediction des coefficients
        F2 = pd.concat([F, cf_oof], axis=1)
        self.second_columns_ = list(F2.columns)
        self.second_c_ = _hgbr(random_state=seed, **SECOND_C_PARAMS).fit(
            F2, y - cf_oof["cf_pred"].values)

        self.model_11_ = (build_meta_ensemble().fit(longitudinal_view(X), y)
                          if self.w11 > 0 else None)
        return self

    def predict_components(self, X: pd.DataFrame) -> dict:
        X = X.reset_index(drop=True)
        out = {"p22": self.model_22_.predict(X)}
        F = self.model_22_.last_features_.reset_index(drop=True)
        P = patient_table(F, X, self.model_22_.categories_)[self.coef_columns_]
        cf = predict_coefs(self.coef_models_, P, X)
        out["coef"] = cf["cf_pred"].values
        F2 = pd.concat([F, cf], axis=1)[self.second_columns_]
        out["c_coef"] = np.clip(self.second_c_.predict(F2) + cf["cf_pred"].values, 0.0, 132.0)
        out["p11"] = (np.clip(self.model_11_.predict(longitudinal_view(X)), 0.0, 132.0)
                      if self.model_11_ is not None else np.zeros(len(X)))
        return out

    def predict(self, X: pd.DataFrame):
        c = self.predict_components(X)
        pred = (self.w22 * c["p22"] + self.wccoef * c["c_coef"] + self.wcoef * c["coef"]
                + self.w11 * c["p11"])
        if self.smooth:
            pred = smooth_predictions(X["patient_id"].values, X["age"].values, pred)
        return np.clip(pred, 0.0, 132.0)


def quick_grid(X: pd.DataFrame, y: pd.Series, cv_splits) -> None:
    """OOF des quatre composantes (mises en cache), puis grille de poids."""
    path = OOF_DIR / "25_components_oof.csv"
    names = ("p22", "c_coef", "coef", "p11")
    folds = np.zeros(len(X), dtype=int)
    for k, (_, va) in enumerate(cv_splits):
        folds[va] = k
    if path.is_file():
        print(f"  Cache OOF trouve : {path}", flush=True)
        comp = pd.read_csv(path)
    else:
        comp = pd.DataFrame({"Index": X["Index"].values, "fold": folds})
        for name in names:
            comp[name] = 0.0
        for k, (tr, va) in enumerate(cv_splits):
            c = DoubleFinalStageRegressor().fit(X.iloc[tr], y.iloc[tr]).predict_components(X.iloc[va])
            for name in names:
                comp.loc[va, name] = c[name]
            print(f"  pli {k} calcule", flush=True)
        OOF_DIR.mkdir(parents=True, exist_ok=True)
        comp.to_csv(path, index=False)
        print(f"  OOF sauvegardees : {path}", flush=True)
    yv = y.values
    pid, age = X["patient_id"].values, X["age"].values
    grid = [(1, 0, 0, 0), (0, 1, 0, 0), (0.6, 0, 0.3, 0.1), (0.4, 0.4, 0.1, 0.1),
            (0.3, 0.4, 0.1, 0.2), (0.5, 0.3, 0.1, 0.1), (0.4, 0.4, 0.15, 0.05), (0.35, 0.45, 0.1, 0.1)]
    for w in grid:
        pred = smooth_predictions(pid, age, sum(wi * comp[n].values for wi, n in zip(w, names)))
        per_fold = [np.sqrt(np.mean((pred[folds == k] - yv[folds == k]) ** 2))
                    for k in range(len(cv_splits))]
        print("  " + " ".join(f"{n}={wi:.2f}" for wi, n in zip(w, names))
              + f" : RMSE = {np.mean(per_fold):.4f} +- {np.std(per_fold):.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Double etage final + blend (25_double_c)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    parser.add_argument("--quick", action="store_true",
                        help="Grille de poids sur predictions hors-fold (pas de skore)")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("DOUBLE ETAGE FINAL + BLEND A QUATRE (25_double_c)", flush=True)
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

    print(f"\n[3/5] Evaluation skore.evaluate (poids {WEIGHTS})...", flush=True)
    model = DoubleFinalStageRegressor()
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
