"""Mesures debiaisees + trajectoires multiples + etage final renforce + lissage (22_debiased).

Sub-task: Trajectoires sans retrecissement vers la moyenne (Sous-tache 15)
Project: Bobathon ESILV (Probabl x IBM)

Diagnostic sur le modele 15 : l'erreur est surtout au niveau patient, avec une forte
regression vers la moyenne (patients severes sous-estimes d'environ 5 points). Cause :
l'etage A predit E[cible | mesure], deja retreci vers la moyenne ; moyenner ces estimations
par patient conserve ce biais.

Ajouts par rapport au 15 (importe, non modifie) :
1. Etage A enrichi (tout en hors-fold, GroupKFold patient) :
   - estimation visite (comme le 15) + canaux off / on separes ;
   - modeles de MESURE : log(mesure+1) - log(cible+1) = g(clinicien, delais, ledd, ...)
     SANS la mesure en entree -> inversion non biaisee u = mesure / g (pas de retrecissement) ;
   - modele de population E[cible | demographie, temps] -> a priori de niveau patient.
2. Trois trajectoires par patient : celle du 15, une a 3 canaux, une sur mesures debiaisees
   avec bruit relatif et a priori de population.
3. Etage final HGBR plus profond sur toutes ces variables (residu de la trajectoire du 15).
4. Lissage quadratique + monotonie par patient (meilleure variante du modele 21).
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
HUB_KEY = "22_debiased"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_15_trajectory import (CATEGORICAL_COLS, LF_PREFIX, _hgbr, _sigma_hgbr,
                                     _visit_hgbr, add_longitudinal_columns, visit_features)
from src.model_21_smooth import smooth_predictions

MEASURES = ["on", "off", "off_minus_on", "off_over_on"]
LOG_EPS = 1.0
MAE_TO_SD = np.sqrt(np.pi / 2)

FINAL_PARAMS = dict(max_iter=1500, learning_rate=0.02, max_leaf_nodes=63,
                    min_samples_leaf=100, l2_regularization=2.0)


# ---------------------------------------------------------------------------
# Etage A : estimations visite par visite
# ---------------------------------------------------------------------------
def measurement_free(V: pd.DataFrame) -> pd.DataFrame:
    """Variables de la visite sans les mesures elles-memes (pour les modeles de mesure)."""
    VM = V.drop(columns=MEASURES)
    VM["has_on"] = V["on"].notna().astype(int)
    VM["has_off"] = V["off"].notna().astype(int)
    return VM


def population_features(VM: pd.DataFrame) -> pd.DataFrame:
    """Demographie et temps uniquement (a priori de niveau)."""
    return VM.drop(columns=["time_since_intake_on", "time_since_intake_off", "rater_id"])


def _log_ratio(a, b):
    return np.log(np.clip(a, 0, None) + LOG_EPS) - np.log(np.clip(b, 0, None) + LOG_EPS)


def fit_stage_a(V: pd.DataFrame, y: np.ndarray, seed: int) -> dict:
    """Entraine les modeles d'estimation de l'etage A."""
    VM = measurement_free(V)
    models = {"est": _visit_hgbr(seed).fit(V, y)}
    for chan in ("off", "on"):
        m = V[chan].notna().values
        # canal "conditionnel" (comme le 15) : la mesure est une entree -> E[cible | mesure]
        models["cond_" + chan] = _hgbr(max_iter=300, learning_rate=0.06, min_samples_leaf=40,
                                       loss="absolute_error", random_state=seed).fit(
            V[m], _log_ratio(y[m], V[chan].values[m]))
        # canal "mesure" : la mesure n'est PAS une entree -> inversion non biaisee
        models["meas_" + chan] = _hgbr(max_iter=300, learning_rate=0.05, min_samples_leaf=50,
                                       random_state=seed).fit(
            VM[m], _log_ratio(V[chan].values[m], y[m]))
    models["pop"] = _hgbr(max_iter=300, learning_rate=0.05, min_samples_leaf=50,
                          random_state=seed).fit(population_features(VM), y)
    return models


def predict_stage_a(models: dict, V: pd.DataFrame) -> pd.DataFrame:
    VM = measurement_free(V)
    out = {"est": models["est"].predict(V)}
    for chan in ("off", "on"):
        obs = V[chan].values
        has = ~np.isnan(obs)
        log_obs = np.log(np.clip(np.nan_to_num(obs), 0, None) + LOG_EPS)
        cond = np.exp(models["cond_" + chan].predict(V) + log_obs) - LOG_EPS
        g = models["meas_" + chan].predict(VM)
        meas = np.exp(log_obs - g) - LOG_EPS
        out["est_" + chan] = np.where(has, cond, np.nan)
        out["u_" + chan] = np.where(has, meas, np.nan)
        out["g_" + chan] = g
    out["pop"] = models["pop"].predict(population_features(VM))
    return pd.DataFrame(out)


SIGMA_TARGETS = {
    # nom de la colonne sigma : (colonne estimee, espace log ?)
    "sig_est": ("est", False),
    "sig_est_off": ("est_off", True),
    "sig_est_on": ("est_on", True),
    "sl_off": ("u_off", True),
    "sl_on": ("u_on", True),
}


def _abs_residual(A: pd.DataFrame, y: np.ndarray, col: str, log: bool) -> np.ndarray:
    est = A[col].values
    return np.abs(_log_ratio(y, est)) if log else np.abs(y - est)


# ---------------------------------------------------------------------------
# Etage B : trajectoires par patient
# ---------------------------------------------------------------------------
def fit_prior(X: pd.DataFrame, y: np.ndarray, min_span: float) -> tuple:
    """A priori de population sur (pente, courbure), estime sur les cibles du train."""
    df = pd.DataFrame({"p": X["patient_id"].values, "a": X["age"].values, "y": y})
    coefs = []
    for _, g in df.groupby("p"):
        a = g["a"].values
        if len(g) < 4 or a.max() - a.min() < min_span:
            continue
        coefs.append(np.polyfit(a - a.mean(), g["y"].values, 2)[::-1])
    coefs = np.array(coefs)
    return coefs[:, 1:].mean(axis=0), coefs[:, 1:].var(axis=0) + 1e-6


def level_sd(X: pd.DataFrame, y: np.ndarray, pop: np.ndarray) -> float:
    """Ecart-type du niveau patient autour du modele de population."""
    return float(pd.Series(y - pop).groupby(X["patient_id"].values).mean().std())


def _patient_rows(pid):
    return pd.Series(np.arange(len(pid))).groupby(pid).indices


def trajectory_conditional(X, A, channels, prior, strength, huber_c=1.5, n_iter=3):
    """Trajectoire quadratique (comme le 15) sur une ou plusieurs estimations conditionnelles."""
    pm, pv = prior
    pid = X["patient_id"].values
    age = X["age"].values.astype(float)
    rows_l, vals_l, sig_l = [], [], []
    for ch in channels:
        e = A[ch].values
        if ch == "est":
            s = np.clip(A["sig_est"].values, 0.5, None) * MAE_TO_SD
        else:
            s = np.clip(A["sig_" + ch].values * MAE_TO_SD * np.clip(e, 1, None), 0.5, None)
        m = ~np.isnan(e)
        rows_l.append(np.where(m)[0]); vals_l.append(e[m]); sig_l.append(s[m])
    r, v, s = map(np.concatenate, (rows_l, vals_l, sig_l))
    lam = np.diag([0.0, *(strength / pv)])
    mu = np.array([0.0, *pm])
    obs_by_p = pd.Series(np.arange(len(r))).groupby(pid[r]).indices
    res = np.full((len(X), 5), np.nan)
    for p, rows in _patient_rows(pid).items():
        oi = obs_by_p.get(p)
        if oi is None:
            continue
        c = age[rows].mean()
        Ao = np.vander(age[r[oi]] - c, 3, increasing=True)
        e, w0 = v[oi], 1.0 / s[oi] ** 2
        w = w0.copy()
        for _ in range(n_iter + 1):
            b = np.linalg.solve(Ao.T @ (Ao * w[:, None]) + lam, Ao.T @ (w * e) + lam @ mu)
            z = np.abs(e - Ao @ b) / s[oi]
            w = w0 * np.minimum(1.0, huber_c / np.maximum(z, 1e-9))
        cov = np.linalg.inv(Ao.T @ (Ao * w0[:, None]) + lam)
        Ar = np.vander(age[rows] - c, 3, increasing=True)
        res[rows, 0] = Ar @ b
        res[rows, 1:4] = b
        res[rows, 4] = np.sqrt(np.einsum("ij,jk,ik->i", Ar, cov, Ar))
    return pd.DataFrame(res, columns=["pred", "b0", "b1", "b2", "se"])


def trajectory_debiased(X, A, prior, lvl_sd, strength=10.0, huber_c=3.0, n_iter=6,
                        sl_scale=1.4):
    """Trajectoire sur mesures debiaisees : bruit relatif (poids reajustes sur la courbe),
    a priori de population sur le niveau (b0), la pente et la courbure."""
    pm, pv = prior
    pid = X["patient_id"].values
    age = X["age"].values.astype(float)
    rows_l, vals_l, sl_l = [], [], []
    for chan in ("off", "on"):
        u = A["u_" + chan].values
        sl = np.clip(A["sl_" + chan].values * MAE_TO_SD * sl_scale, 0.05, None)
        m = ~np.isnan(u)
        rows_l.append(np.where(m)[0]); vals_l.append(u[m]); sl_l.append(sl[m])
    r, v, sl = map(np.concatenate, (rows_l, vals_l, sl_l))
    obs_by_p = pd.Series(np.arange(len(r))).groupby(pid[r]).indices
    pop = A["pop"].values
    res = np.full((len(X), 5), np.nan)
    for p, rows in _patient_rows(pid).items():
        c = age[rows].mean()
        L = np.diag([1.0 / lvl_sd ** 2, *(strength / pv)])
        M0 = np.array([pop[rows].mean(), *pm])
        Ar = np.vander(age[rows] - c, 3, increasing=True)
        oi = obs_by_p.get(p)
        if oi is None:
            b, cov = M0, np.linalg.inv(L)
        else:
            Ao = np.vander(age[r[oi]] - c, 3, increasing=True)
            e = v[oi]
            fit = np.full(len(oi), np.median(e))
            hub = np.ones(len(oi))
            for _ in range(n_iter):
                s = sl[oi] * (np.clip(fit, 0, None) + 1.0)
                w = hub / s ** 2
                b = np.linalg.solve(Ao.T @ (Ao * w[:, None]) + L, Ao.T @ (w * e) + L @ M0)
                fit = Ao @ b
                hub = np.minimum(1.0, huber_c / np.maximum(np.abs(e - fit) / s, 1e-9))
            cov = np.linalg.inv(Ao.T @ (Ao * (1 / s ** 2)[:, None]) + L)
        res[rows, 0] = Ar @ b
        res[rows, 1:4] = b
        res[rows, 4] = np.sqrt(np.einsum("ij,jk,ik->i", Ar, cov, Ar))
    return pd.DataFrame(res, columns=["pred", "b0", "b1", "b2", "se"])


# ---------------------------------------------------------------------------
# Estimateur complet
# ---------------------------------------------------------------------------
class DebiasedTrajectoryRegressor(RegressorMixin, BaseEstimator):
    """Etage A enrichi -> 3 trajectoires patient -> HGBR final -> lissage par patient."""

    def __init__(self, n_inner_splits=5, smooth=True, random_state=0):
        self.n_inner_splits = n_inner_splits
        self.smooth = smooth
        self.random_state = random_state

    def _stage_c_features(self, X: pd.DataFrame, A: pd.DataFrame) -> tuple:
        T = {
            "t1": trajectory_conditional(X, A, ("est",), self.prior_, strength=1.0),
            "t3": trajectory_conditional(X, A, ("est", "est_off", "est_on"), self.prior_span_,
                                         strength=10.0),
            "t2": trajectory_debiased(X, A, self.prior_span_, self.level_sd_),
        }
        lf_cats = [LF_PREFIX + c for c in CATEGORICAL_COLS]
        F = X[[c for c in X.columns if c.startswith(LF_PREFIX) and c not in lf_cats]].copy()
        for col in CATEGORICAL_COLS:
            F[col] = pd.Categorical(X[col], categories=self.categories_[col])
        for col in A.columns:
            F["A_" + col] = A[col].values
        for name, traj in T.items():
            for col in traj.columns:
                F[f"{name}_{col}"] = traj[col].values
            F[f"{name}_resid"] = A["est"].values - traj["pred"].values
        return F, T["t1"]["pred"].values

    def _predict_sigmas(self, V: pd.DataFrame, A: pd.DataFrame, models: dict) -> pd.DataFrame:
        A = A.copy()
        for name, (col, _) in SIGMA_TARGETS.items():
            A[name] = models[name].predict(V)
            A.loc[A[col].isna(), name] = np.nan
        return A

    def fit(self, X: pd.DataFrame, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        groups = X["patient_id"].values
        self.categories_ = {c: sorted(X[c].dropna().unique()) for c in CATEGORICAL_COLS}
        V = visit_features(X, self.categories_)
        splits = list(GroupKFold(n_splits=self.n_inner_splits).split(V, y, groups))

        # Etage A hors-fold
        parts = []
        for tr, va in splits:
            part = predict_stage_a(fit_stage_a(V.iloc[tr], y[tr], self.random_state), V.iloc[va])
            part.index = va
            parts.append(part)
        A = pd.concat(parts).sort_index()

        # Bruit de chaque estimation : |residu hors-fold| predit hors-fold
        for name, (col, log) in SIGMA_TARGETS.items():
            target = _abs_residual(A, y, col, log)
            sig = np.full(len(X), np.nan)
            for tr, va in splits:
                m = ~np.isnan(target[tr])
                sig[va] = _sigma_hgbr(self.random_state).fit(V.iloc[tr][m], target[tr][m]).predict(V.iloc[va])
            A[name] = np.where(A[col].isna(), np.nan, sig)

        # Modeles finaux d'etage A (appliques au test)
        self.stage_a_ = fit_stage_a(V, y, self.random_state)
        self.sigma_models_ = {}
        for name, (col, log) in SIGMA_TARGETS.items():
            target = _abs_residual(A, y, col, log)
            m = ~np.isnan(target)
            self.sigma_models_[name] = _sigma_hgbr(self.random_state).fit(V[m], target[m])

        # Etage B (a prioris) puis C
        self.prior_ = fit_prior(X, y, min_span=0.0)
        self.prior_span_ = fit_prior(X, y, min_span=4.0)
        self.level_sd_ = level_sd(X, y, A["pop"].values)
        self.a_columns_ = list(A.columns)
        F, base = self._stage_c_features(X, A)
        self.feature_names_ = list(F.columns)
        self.final_model_ = _hgbr(random_state=self.random_state, **FINAL_PARAMS).fit(F, y - base)
        return self

    def predict(self, X: pd.DataFrame):
        X = X.reset_index(drop=True)
        V = visit_features(X, self.categories_)
        A = self._predict_sigmas(V, predict_stage_a(self.stage_a_, V), self.sigma_models_)
        F, base = self._stage_c_features(X, A[self.a_columns_])
        pred = np.clip(self.final_model_.predict(F[self.feature_names_]) + base, 0.0, 132.0)
        if self.smooth:
            pred = smooth_predictions(X["patient_id"].values, X["age"].values, pred,
                                      alpha=1.0, degree=2, monotone=True)
        return pred


def quick_cv(X: pd.DataFrame, y: pd.Series, cv_splits) -> None:
    """CV rapide hors skore : RMSE avant / apres lissage."""
    raw, smo = [], []
    for k, (tr, va) in enumerate(cv_splits):
        model = DebiasedTrajectoryRegressor(smooth=False).fit(X.iloc[tr], y.iloc[tr])
        p = model.predict(X.iloc[va])
        Xv = X.iloc[va]
        ps = smooth_predictions(Xv["patient_id"].values, Xv["age"].values, p)
        yv = y.iloc[va].values
        raw.append(np.sqrt(np.mean((p - yv) ** 2)))
        smo.append(np.sqrt(np.mean((ps - yv) ** 2)))
        print(f"  pli {k}: brut={raw[-1]:.4f}  lisse={smo[-1]:.4f}", flush=True)
    print(f"  MOYENNE: brut={np.mean(raw):.4f}+-{np.std(raw):.4f}  "
          f"lisse={np.mean(smo):.4f}+-{np.std(smo):.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mesures debiaisees + trajectoires (22_debiased)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    parser.add_argument("--quick", action="store_true",
                        help="CV rapide uniquement (pas de skore, pas de soumission)")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MESURES DEBIAISEES + TRAJECTOIRES MULTIPLES (22_debiased)", flush=True)
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
        print("\n[3/5] CV rapide...", flush=True)
        quick_cv(X, y, cv_splits)
        return

    print("\n[3/5] Evaluation skore.evaluate...", flush=True)
    model = DebiasedTrajectoryRegressor()
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
