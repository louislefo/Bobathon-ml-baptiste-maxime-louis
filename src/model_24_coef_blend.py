"""Regression des coefficients de trajectoire + blend 22 / coefficients / 11 (24_coef_blend).

Sub-task: Modele au niveau patient + blend a trois (Sous-tache 17)
Project: Bobathon ESILV (Probabl x IBM)

La cible est, par patient, une parabole de l'age. Au lieu de predire visite par visite,
un modele au niveau patient predit directement les 3 coefficients (niveau, pente, courbure
autour de l'age moyen du patient) a partir d'agregats patient des variables du modele 22.
Seul il est moins bon que le 22, mais ses erreurs sont differentes : en blend il apporte
nettement plus que le 11.

Prediction finale : 0.6 * 22 (brut) + 0.3 * coefficients + 0.1 * 11, puis lissage
quadratique + monotone par patient (modele 21).

Modeles 11, 21 et 22 importes, non modifies (le 22 est etendu par sous-classe).
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
HUB_KEY = "24_coef_blend"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_11_meta_ensemble import build_meta_ensemble
from src.model_15_trajectory import _hgbr, add_longitudinal_columns
from src.model_21_smooth import smooth_predictions
from src.model_22_debiased import DebiasedTrajectoryRegressor
from src.model_23_blend import longitudinal_view

WEIGHTS = {"p22": 0.6, "coef": 0.3, "p11": 0.1}
COEFS = ("b0", "b1", "b2")
COEF_PARAMS = dict(max_iter=800, learning_rate=0.02, max_leaf_nodes=31, min_samples_leaf=20,
                   l2_regularization=2.0)

# Variables visite agregees par patient / variables deja constantes par patient
AGG_COLS = ["A_est", "A_est_off", "A_est_on", "A_u_off", "A_u_on", "A_pop", "A_sig_est",
            "A_sl_off", "A_sl_on", "lf_off", "lf_on", "lf_ledd", "lf_time_since_intake_on",
            "lf_time_since_intake_off"]
CONST_COLS = [f"{t}_{c}" for t in ("t1", "t2", "t3") for c in ("b0", "b1", "b2", "se")] + [
    "lf_age_at_diagnosis", "lf_sexM", "lf_mean_off_patient", "lf_mean_on_patient",
    "lf_std_off_patient", "lf_n_visits_patient", "lf_patient_slope"]


class _Debiased22WithFeatures(DebiasedTrajectoryRegressor):
    """Modele 22 qui garde les variables d'etage C calculees (train hors-fold / test)."""

    def _stage_c_features(self, X, A):
        F, base = super()._stage_c_features(X, A)
        self.last_features_ = F
        return F, base

    def fit(self, X, y):
        super().fit(X, y)
        self.train_features_ = self.last_features_
        return self


def patient_table(F: pd.DataFrame, X: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Une ligne par patient : agregats des variables visite + variables constantes."""
    F = F.reset_index(drop=True).copy()
    F["pid"] = X["patient_id"].values
    F["age"] = X["age"].values
    F["has_off"] = X["off"].notna().values.astype(float)
    g = F.groupby("pid", sort=False)
    parts = [g[CONST_COLS].first(),
             g["age"].agg(["mean", "min", "max"]).add_prefix("age_"),
             g[["has_off"]].mean()]
    for col in AGG_COLS:
        parts.append(g[col].agg(["mean", "median", "std", "min", "max"]).add_prefix(col + "_"))
    T = pd.concat(parts, axis=1)
    T["span"] = T["age_max"] - T["age_min"]
    first = X.groupby(X["patient_id"].values, sort=False)[["cohort", "gene"]].first().reindex(T.index)
    for col in ("cohort", "gene"):
        T[col] = pd.Categorical(first[col].values, categories=categories[col])
    return T


def true_coefficients(X: pd.DataFrame, y: np.ndarray, index) -> pd.DataFrame:
    """Coefficients de la parabole (age centre sur la moyenne patient) ajustee sur la cible."""
    df = pd.DataFrame({"p": X["patient_id"].values, "a": X["age"].values, "y": y})
    out = {}
    for p, g in df.groupby("p", sort=False):
        t = g["a"].values - g["a"].values.mean()
        out[p] = np.polyfit(t, g["y"].values, 2)[::-1] if len(g) >= 3 else [g["y"].mean(), 0, 0]
    return pd.DataFrame(out, index=list(COEFS)).T.reindex(index)


class CoefBlendRegressor(RegressorMixin, BaseEstimator):
    """Blend 22 + regression des coefficients patient + 11, puis lissage patient."""

    def __init__(self, w22=WEIGHTS["p22"], wcoef=WEIGHTS["coef"], w11=WEIGHTS["p11"],
                 smooth=True, random_state=0):
        self.w22 = w22
        self.wcoef = wcoef
        self.w11 = w11
        self.smooth = smooth
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        self.model_22_ = _Debiased22WithFeatures(smooth=False,
                                                 random_state=self.random_state).fit(X, y)
        P = patient_table(self.model_22_.train_features_, X, self.model_22_.categories_)
        C = true_coefficients(X, y, P.index)
        self.coef_models_ = {
            b: _hgbr(random_state=self.random_state, **COEF_PARAMS).fit(
                P, C[b].values - P[f"t2_{b}"].values)
            for b in COEFS
        }
        self.coef_columns_ = list(P.columns)
        self.model_11_ = (build_meta_ensemble().fit(longitudinal_view(X), y)
                          if self.w11 > 0 else None)
        return self

    def predict_components(self, X: pd.DataFrame) -> dict:
        X = X.reset_index(drop=True)
        out = {"p22": self.model_22_.predict(X)}
        P = patient_table(self.model_22_.last_features_, X,
                          self.model_22_.categories_)[self.coef_columns_]
        coefs = {b: pd.Series(self.coef_models_[b].predict(P) + P[f"t2_{b}"].values, index=P.index)
                 for b in COEFS}
        pid = X["patient_id"].values
        t = X["age"].values - P["age_mean"].reindex(pid).values
        out["coef"] = np.clip(coefs["b0"].reindex(pid).values + coefs["b1"].reindex(pid).values * t
                              + coefs["b2"].reindex(pid).values * t ** 2, 0.0, 132.0)
        out["p11"] = (np.clip(self.model_11_.predict(longitudinal_view(X)), 0.0, 132.0)
                      if self.model_11_ is not None else np.zeros(len(X)))
        return out

    def predict(self, X: pd.DataFrame):
        c = self.predict_components(X)
        pred = self.w22 * c["p22"] + self.wcoef * c["coef"] + self.w11 * c["p11"]
        if self.smooth:
            pred = smooth_predictions(X["patient_id"].values, X["age"].values, pred)
        return np.clip(pred, 0.0, 132.0)


def quick_grid(X: pd.DataFrame, y: pd.Series, cv_splits) -> None:
    """OOF des trois composantes (mises en cache), puis grille de poids."""
    path = OOF_DIR / "24_components_oof.csv"
    folds = np.zeros(len(X), dtype=int)
    for k, (_, va) in enumerate(cv_splits):
        folds[va] = k
    if path.is_file():
        print(f"  Cache OOF trouve : {path}", flush=True)
        comp = pd.read_csv(path)
    else:
        comp = pd.DataFrame({"Index": X["Index"].values, "fold": folds})
        for name in ("p22", "coef", "p11"):
            comp[name] = 0.0
        for k, (tr, va) in enumerate(cv_splits):
            c = CoefBlendRegressor().fit(X.iloc[tr], y.iloc[tr]).predict_components(X.iloc[va])
            for name, values in c.items():
                comp.loc[va, name] = values
            print(f"  pli {k} calcule", flush=True)
        OOF_DIR.mkdir(parents=True, exist_ok=True)
        comp.to_csv(path, index=False)
        print(f"  OOF sauvegardees : {path}", flush=True)
    yv = y.values
    pid, age = X["patient_id"].values, X["age"].values
    for w22, wc, w11 in [(1, 0, 0), (0, 1, 0), (0.75, 0.25, 0), (0.7, 0.3, 0), (0.65, 0.25, 0.1),
                         (0.6, 0.3, 0.1), (0.55, 0.35, 0.1), (0.6, 0.35, 0.05)]:
        pred = smooth_predictions(pid, age, w22 * comp["p22"].values + wc * comp["coef"].values
                                  + w11 * comp["p11"].values)
        per_fold = [np.sqrt(np.mean((pred[folds == k] - yv[folds == k]) ** 2))
                    for k in range(len(cv_splits))]
        print(f"  22={w22:.2f} coef={wc:.2f} 11={w11:.2f} : RMSE = {np.mean(per_fold):.4f} "
              f"+- {np.std(per_fold):.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Coefficients patient + blend (24_coef_blend)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    parser.add_argument("--quick", action="store_true",
                        help="Grille de poids sur predictions hors-fold (pas de skore)")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("COEFFICIENTS PATIENT + BLEND 22 / COEF / 11 (24_coef_blend)", flush=True)
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
    model = CoefBlendRegressor()
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
