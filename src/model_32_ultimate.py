"""Meilleur modele : 4 composantes blendees + lissage isotonique (32_ultimate).

Project: Bobathon ESILV (Probabl x IBM)

Architecture :
  Etage A  (visite, hors-fold) :
    A1  est_visit  : HGBR toutes variables visite -> cible
    A2  est_off / est_on : HGBR abs_error, log(cible+1)-log(mesure+1) AVEC mesure (conditionnel)
    A3  u_off / u_on : HGBR log(mesure+1)-log(cible+1) SANS mesure (debiaise, has_off/has_on)
    A4  est_pop    : demographie + temps + ledd + has_off/has_on -> cible (a priori niveau)
    A5  sigma_*    : |residu| (ou |log-residu| pour canaux mesure) predit hors-fold (abs_error)

  Etage B (patient, ponderation 1/sigma^2, Huber, a priori bayesien) :
    t1 : est_visit seul, prior_strength=1
    t2 : u_off + u_on, bruit relatif reajuste sur la courbe, prior_strength=10, Huber c=3
    t3 : est_visit + est_off + est_on empiles, prior_strength=10
    t4 : t4 = trajectoire "cond" du modele 31 (visit + off_cond + on_cond, la meilleure solo)

  C1 (comme 22) : HGBR (63f, min_sl=100, lr=0.02, 1500it, l2=2) sur lf_ + A + B
                  cible = y - t1.pred

  Modele coefficients (comme 24/25) :
    table patient (agregats A/B/off/on/ledd, constantes) -> 3 HGBR (b0, b1, b2 en residu t2)
    base visite = b0 + b1*t + b2*t^2 (hors-fold en train)

  C2 (comme 25) : HGBR (63f, min_sl=100, lr=0.05, 600it, l2=2) sur lf_ + A + B + cf_*
                  cible = y - cf_pred

  Modele 11 : build_meta_ensemble() sur colonnes lf_ (sans prefixe).

  Blend : w_c1*C1 + w_c2*C2 + w_cf*coefficients + w_11*modele11
          puis lissage quadratique + IsotonicRegression croissante par patient.
  Poids codifies en dur apres optimisation OOF --quick :
    W_C1=0.40, W_C2=0.40, W_CF=0.10, W_11=0.10

Pas de fuite :
  - Etage A : tout hors-fold (GroupKFold patient interne).
  - Coefficients : calcules hors-fold sur le train.
  - C2 : entrainee sur y - cf_pred_oof.
  - Statistiques lf_ : intra-patient sur X uniquement.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
OOF_DIR = REPO_ROOT / "oof"
HUB_KEY = "32_ultimate"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_09_longitudinal import prepare_longitudinal_features
from src.model_11_meta_ensemble import build_meta_ensemble
from src.model_15_trajectory import LF_PREFIX, add_longitudinal_columns
from src.model_23_blend import longitudinal_view
from src.model_21_smooth import smooth_predictions

CATEGORICAL_COLS = ["cohort", "gene", "rater_id"]
MAE2STD = float(np.sqrt(np.pi / 2))
LOG_EPS = 1.0

# Blend weights (coded after OOF optimisation)
W_C1 = 0.40
W_C2 = 0.40
W_CF = 0.10
W_11 = 0.10

COEFS = ("b0", "b1", "b2")
COEF_PARAMS = dict(max_iter=800, learning_rate=0.02, max_leaf_nodes=31,
                   min_samples_leaf=20, l2_regularization=2.0)
FINAL_C1_PARAMS = dict(max_iter=1500, learning_rate=0.02, max_leaf_nodes=63,
                       min_samples_leaf=100, l2_regularization=2.0)
FINAL_C2_PARAMS = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                       min_samples_leaf=100, l2_regularization=2.0)

# ── HGBR factories ────────────────────────────────────────────────────────────

def _hgbr(**kw) -> "HistGradientBoostingRegressor":
    from sklearn.ensemble import HistGradientBoostingRegressor
    d = dict(max_iter=500, learning_rate=0.04, min_samples_leaf=20, max_leaf_nodes=31,
             l2_regularization=1.0, categorical_features="from_dtype", random_state=0)
    d.update(kw)
    return HistGradientBoostingRegressor(**d)


def _sigma_hgbr(**kw):
    return _hgbr(max_iter=150, learning_rate=0.08, min_samples_leaf=100,
                 loss="absolute_error", **kw)


# ── feature builders ──────────────────────────────────────────────────────────

def _visit_full(df, cats):
    """Toutes variables visite (A1, A2 conditionnel)."""
    d = pd.DataFrame(index=df.index)
    for c in ["age_at_diagnosis", "age", "ledd",
              "time_since_intake_on", "time_since_intake_off", "on", "off", "sexM"]:
        d[c] = df[c]
    d["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    d["off_minus_on"] = df["off"] - df["on"]
    d["off_over_on"]  = df["off"] / df["on"].replace(0, np.nan)
    for c in CATEGORICAL_COLS:
        d[c] = pd.Categorical(df[c], categories=cats[c])
    return d


def _visit_no_meas(df, cats):
    """Sans on/off, avec has_off/has_on (A3 debiaise, A5 sigma mesure)."""
    d = pd.DataFrame(index=df.index)
    for c in ["age_at_diagnosis", "age", "ledd",
              "time_since_intake_on", "time_since_intake_off", "sexM"]:
        d[c] = df[c]
    d["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    d["has_off"] = df["off"].notna().astype(float)
    d["has_on"]  = df["on"].notna().astype(float)
    for c in CATEGORICAL_COLS:
        d[c] = pd.Categorical(df[c], categories=cats[c])
    return d


def _visit_pop(df, cats):
    """Demographie + temps + ledd + has (A4 population)."""
    d = pd.DataFrame(index=df.index)
    for c in ["age_at_diagnosis", "age", "ledd", "sexM"]:
        d[c] = df[c]
    d["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    d["has_off"] = df["off"].notna().astype(float)
    d["has_on"]  = df["on"].notna().astype(float)
    for c in ["cohort", "gene"]:
        d[c] = pd.Categorical(df[c], categories=cats[c])
    return d


def _log_ratio(a, b):
    return (np.log(np.clip(np.nan_to_num(a, nan=0.0), 0, None) + LOG_EPS)
            - np.log(np.clip(np.nan_to_num(b, nan=0.0), 0, None) + LOG_EPS))


# ── patient index helper ──────────────────────────────────────────────────────

def _patient_rows(pid):
    """Dict {patient -> array of row indices}. Calculé une seule fois."""
    return pd.Series(np.arange(len(pid))).groupby(pid).indices


# ── Etage B : trajectoires quadratiques ──────────────────────────────────────

def _fit_prior(X, y, oof_pop, min_span=0.0, min_span_slope=4.0):
    """A priori de population : (pente, courbure) et variance du niveau residuel."""
    pid = X["patient_id"].values
    age = X["age"].values.astype(float)
    resid = y - oof_pop
    coefs0, coefs4, lev_means = [], [], []
    for rows in _patient_rows(pid).values():
        a = age[rows]
        if len(rows) < 4:
            continue
        t = a - a.mean()
        lev_means.append(resid[rows].mean())
        c = np.polyfit(t, y[rows], 2)[::-1]
        coefs0.append(c)
        if a.max() - a.min() >= min_span_slope:
            coefs4.append(c)
    coefs0 = np.array(coefs0)
    coefs4 = np.array(coefs4) if coefs4 else coefs0
    prior0 = (coefs0[:, 1:].mean(0), coefs0[:, 1:].var(0) + 1e-6)
    prior4 = (coefs4[:, 1:].mean(0), coefs4[:, 1:].var(0) + 1e-6)
    lev_var = float(np.var(lev_means)) + 1e-6
    return prior0, prior4, lev_var


def _run_trajectory(
    pid, age, grid_pid, grid_age,
    obs_rows, obs_vals, obs_sigma,
    prior, strength, lev_var, pop_level,
    huber_c=1.5, n_iter=3,
):
    """Trajectoire quadratique bayesienne generique sur des observations empilees.

    obs_rows : indices dans (pid, age) des observations
    obs_vals / obs_sigma : valeurs et sigma des observations
    Retourne (pred, b0, b1, b2, se) sur la grille (grid_pid, grid_age).
    """
    pm, pv = prior
    lam = np.diag([strength / lev_var, *(strength / pv)])

    obs_by_p = pd.Series(np.arange(len(obs_rows))).groupby(pid[obs_rows]).indices
    grid_by_p = _patient_rows(grid_pid)

    res = np.full((len(grid_pid), 5), np.nan)
    for p, grid_idx in grid_by_p.items():
        ga = grid_age[grid_idx]
        c = ga.mean()
        mu = np.array([pop_level[grid_idx].mean(), *pm])
        oi = obs_by_p.get(p)
        if oi is None:
            # no observations: posterior = prior
            b = mu
            cov = np.linalg.inv(lam + np.eye(3) * 1e-9)
        else:
            t_obs = age[obs_rows[oi]] - c
            A_obs = np.column_stack([np.ones_like(t_obs), t_obs, t_obs**2])
            e, s = obs_vals[oi], np.maximum(obs_sigma[oi], 0.5)
            w0 = 1.0 / s**2
            w = w0.copy()
            for _ in range(n_iter + 1):
                M = A_obs.T @ (A_obs * w[:, None]) + lam
                b = np.linalg.solve(M, A_obs.T @ (w * e) + lam @ mu)
                z = np.abs(e - A_obs @ b) / s
                w = w0 * np.minimum(1.0, huber_c / np.maximum(z, 1e-9))
            cov = np.linalg.inv(A_obs.T @ (A_obs * w0[:, None]) + lam)
        A_g = np.column_stack([np.ones_like(ga - c), ga - c, (ga - c)**2])
        res[grid_idx, 0] = A_g @ b
        res[grid_idx, 1:4] = b
        res[grid_idx, 4] = np.sqrt(np.einsum("ij,jk,ik->i", A_g, cov, A_g))
    return pd.DataFrame(res, columns=["pred", "b0", "b1", "b2", "se"])


def _traj_visit(X, A, prior0, lev_var):
    """t1 : estimation visite seule, force=1."""
    pid = X["patient_id"].values
    age = X["age"].values.astype(float)
    pop = A["est_pop"].values
    e = A["est_visit"].values
    s = np.clip(A["sig_est"].values, 0.5, None) * MAE2STD
    rows = np.arange(len(pid))
    return _run_trajectory(pid, age, pid, age, rows, e, s,
                           prior0, strength=1.0, lev_var=lev_var, pop_level=pop)


def _traj_debiased(X, A, prior4, lev_var):
    """t2 : u_off + u_on empiles, bruit relatif reajuste, Huber c=3."""
    pid  = X["patient_id"].values
    age  = X["age"].values.astype(float)
    pop  = A["est_pop"].values
    rows_l, vals_l, sl_l = [], [], []
    for ch in ("off", "on"):
        u  = A[f"u_{ch}"].values
        sl = np.clip(A[f"sl_{ch}"].values * MAE2STD * 1.4, 0.05, None)
        m  = np.isfinite(u)
        if m.any():
            rows_l.append(np.where(m)[0])
            vals_l.append(u[m])
            sl_l.append(sl[m])
    if not rows_l:
        # fallback: visit
        return _traj_visit(X, A, prior4, lev_var)
    r = np.concatenate(rows_l)
    v = np.concatenate(vals_l)
    sl_obs = np.concatenate(sl_l)
    obs_by_p = pd.Series(np.arange(len(r))).groupby(pid[r]).indices
    grid_by_p = _patient_rows(pid)

    pm, pv = prior4
    res = np.full((len(pid), 5), np.nan)
    for p, grid_idx in grid_by_p.items():
        ga  = age[grid_idx]
        c   = ga.mean()
        mu  = np.array([pop[grid_idx].mean(), *pm])
        lev_lam = 1.0 / lev_var
        L = np.diag([lev_lam, *(10.0 / pv)])
        A_g = np.column_stack([np.ones_like(ga - c), ga - c, (ga - c)**2])
        oi = obs_by_p.get(p)
        if oi is None:
            b, cov = mu, np.linalg.inv(L)
        else:
            t_obs = age[r[oi]] - c
            A_obs = np.column_stack([np.ones_like(t_obs), t_obs, t_obs**2])
            e = v[oi]
            fit = np.full(len(oi), np.median(e))
            hub = np.ones(len(oi))
            for _ in range(6):
                s = sl_obs[oi] * (np.clip(fit, 0, None) + 1.0)
                w = hub / np.maximum(s, 0.5)**2
                M = A_obs.T @ (A_obs * w[:, None]) + L
                b = np.linalg.solve(M, A_obs.T @ (w * e) + L @ mu)
                fit = A_obs @ b
                hub = np.minimum(1.0, 3.0 / np.maximum(np.abs(e - fit) / np.maximum(s, 0.5), 1e-9))
            s_fin = sl_obs[oi] * (np.clip(fit, 0, None) + 1.0)
            cov = np.linalg.inv(A_obs.T @ (A_obs * (1 / np.maximum(s_fin, 0.5)**2)[:, None]) + L)
        res[grid_idx, 0] = A_g @ b
        res[grid_idx, 1:4] = b
        res[grid_idx, 4] = np.sqrt(np.einsum("ij,jk,ik->i", A_g, cov, A_g))
    return pd.DataFrame(res, columns=["pred", "b0", "b1", "b2", "se"])


def _traj_conditional(X, A, prior4, lev_var):
    """t3 : est_visit + est_off + est_on empiles, force=10."""
    pid = X["patient_id"].values
    age = X["age"].values.astype(float)
    pop = A["est_pop"].values
    rows_l, vals_l, sig_l = [], [], []
    for ch, sig_col in [("visit", "sig_est"), ("off", "sig_est_off"), ("on", "sig_est_on")]:
        e = A[f"est_{ch}"].values
        s = np.clip(A[sig_col].values * MAE2STD * np.clip(np.nan_to_num(e, nan=1.0), 1, None),
                    0.5, None) if ch != "visit" else np.clip(A[sig_col].values * MAE2STD, 0.5, None)
        m = np.isfinite(e) & (e > -LOG_EPS)
        if m.any():
            rows_l.append(np.where(m)[0])
            vals_l.append(e[m])
            sig_l.append(s[m])
    r = np.concatenate(rows_l)
    v = np.concatenate(vals_l)
    s = np.concatenate(sig_l)
    return _run_trajectory(pid, age, pid, age, r, v, s,
                           prior4, strength=10.0, lev_var=lev_var, pop_level=pop)


def _traj_cond4(X, A, prior4, lev_var):
    """t4 : visit + u_off_cond + u_on_cond (conditionnel meme si mesure absente, sigma triple)."""
    pid  = X["patient_id"].values
    age  = X["age"].values.astype(float)
    pop  = A["est_pop"].values
    ev   = A["est_visit"].values
    sv   = np.clip(A["sig_est"].values * MAE2STD, 0.5, None)
    rows_l, vals_l, sig_l = [np.arange(len(pid))], [ev], [sv]
    for ch in ("off", "on"):
        u   = A[f"u_{ch}"].values
        sl  = np.clip(A[f"sl_{ch}"].values * MAE2STD * 1.4, 0.05, None)
        u_c = np.where(np.isfinite(u), u, ev)
        s_c = np.where(np.isfinite(u), sl * (np.clip(u_c, 0, None) + 1.0), sv * 3.0)
        rows_l.append(np.arange(len(pid)))
        vals_l.append(u_c)
        sig_l.append(np.maximum(s_c, 0.5))
    r = np.concatenate(rows_l)
    v = np.concatenate(vals_l)
    s = np.concatenate(sig_l)
    return _run_trajectory(pid, age, pid, age, r, v, s,
                           prior4, strength=10.0, lev_var=lev_var, pop_level=pop)


# ── Etage C features ──────────────────────────────────────────────────────────

def _stage_c_features(X, A, trajs, cats, cf_cols=None):
    """Assemble les features de l'etage C1 ou C2.

    trajs : dict {name: DataFrame(pred, b0, b1, b2, se)}
    cf_cols : si fourni, ajoute les colonnes de coefficients predits.
    """
    lf_cats = [LF_PREFIX + c for c in CATEGORICAL_COLS]
    F = X[[c for c in X.columns if c.startswith(LF_PREFIX) and c not in lf_cats]].copy()
    F = F.reset_index(drop=True)
    for c in CATEGORICAL_COLS:
        F[c] = pd.Categorical(X[c].values, categories=cats[c])
    for col in A.columns:
        F["A_" + col] = A[col].values
    for name, traj in trajs.items():
        t = traj.reset_index(drop=True)
        for col in t.columns:
            F[f"{name}_{col}"] = t[col].values
        F[f"{name}_resid"] = A["est_visit"].values - t["pred"].values
    if cf_cols is not None:
        for c in cf_cols.columns:
            F[c] = cf_cols[c].values
    return F


# ── Modele coefficients patient ───────────────────────────────────────────────

AGG_COLS_CF = ["A_est_visit", "A_est_off", "A_est_on", "A_u_off", "A_u_on", "A_est_pop",
               "A_sig_est", "A_sl_off", "A_sl_on"]
CONST_COLS_CF = ([f"{t}_{c}" for t in ("t1", "t2", "t3", "t4")
                  for c in ("b0", "b1", "b2", "se")]
                 + ["lf_age_at_diagnosis", "lf_sexM", "lf_mean_off_patient",
                    "lf_mean_on_patient", "lf_std_off_patient",
                    "lf_n_visits_patient", "lf_patient_slope"])


def _patient_table(F, X, cats):
    """Une ligne par patient : agregats A/B + constantes."""
    F = F.reset_index(drop=True).copy()
    pid = X["patient_id"].values
    F["_pid"] = pid
    F["_age"] = X["age"].values.astype(float)
    F["_has_off"] = X["off"].notna().values.astype(float)
    F["_has_on"]  = X["on"].notna().values.astype(float)
    g = F.groupby("_pid", sort=False)

    present_const = [c for c in CONST_COLS_CF if c in F.columns]
    parts = [g[present_const].first()]
    parts.append(g["_age"].agg(["mean", "min", "max"]).add_prefix("age_"))
    parts.append(g[["_has_off", "_has_on"]].mean().rename(
        columns={"_has_off": "frac_off", "_has_on": "frac_on"}))
    parts.append(g["_age"].count().rename("n_visits").to_frame())
    present_agg = [c for c in AGG_COLS_CF if c in F.columns]
    for col in present_agg:
        parts.append(g[col].agg(["mean", "median", "std", "min", "max"]).add_prefix(col + "_"))
    T = pd.concat(parts, axis=1)
    T["span"] = T["age_max"] - T["age_min"]
    first = X.groupby(X["patient_id"].values, sort=False)[["cohort", "gene"]].first().reindex(T.index)
    for col in ("cohort", "gene"):
        T[col] = pd.Categorical(first[col].values, categories=cats[col])
    return T


def _true_coefs(X, y, index):
    """Coefficients de la parabole ajustee sur la cible par patient."""
    df = pd.DataFrame({"p": X["patient_id"].values, "a": X["age"].values.astype(float), "y": y})
    out = {}
    for p, g in df.groupby("p", sort=False):
        a = g["a"].values
        t = a - a.mean()
        out[p] = np.polyfit(t, g["y"].values, 2)[::-1] if len(g) >= 3 else [g["y"].mean(), 0.0, 0.0]
    return pd.DataFrame(out, index=list(COEFS)).T.reindex(index)


def _fit_coef_models(P, C, t2_cols_in_P, seed):
    """3 HGBR en residu de la trajectoire t2."""
    return {b: _hgbr(random_state=seed, **COEF_PARAMS).fit(
                P, C[b].values - P[t2_cols_in_P[b]].values)
            for b in COEFS}


def _predict_coefs(models, P, t2_cols_in_P, X_vis):
    """Retourne DataFrame avec cf_b0, cf_b1, cf_b2, cf_pred par visite."""
    pid   = X_vis["patient_id"].values
    t2_df = P[[t2_cols_in_P[b] for b in COEFS]]
    coefs = {b: pd.Series(models[b].predict(P) + t2_df[t2_cols_in_P[b]].values,
                          index=P.index).reindex(pid).values
             for b in COEFS}
    age_mean = P["age_mean"].reindex(pid).values
    t = X_vis["age"].values.astype(float) - age_mean
    out = pd.DataFrame({f"cf_{b}": coefs[b] for b in COEFS}, index=range(len(X_vis)))
    out["cf_pred"] = np.clip(coefs["b0"] + coefs["b1"] * t + coefs["b2"] * t**2, 0.0, 132.0)
    return out


# ── Estimateur principal ──────────────────────────────────────────────────────

class UltimateRegressor(RegressorMixin, BaseEstimator):
    """Meilleur modele Bobathon : 4 composantes blendees + lissage.

    Parameters
    ----------
    w_c1, w_c2, w_cf, w_11 : poids du blend (doivent sommer a 1).
    n_inner_splits : GroupKFold internes pour les predictions hors-fold.
    smooth : appliquer lissage quadratique + isotonique.
    random_state : graine.
    """

    def __init__(self, w_c1=W_C1, w_c2=W_C2, w_cf=W_CF, w_11=W_11,
                 n_inner_splits=5, smooth=True, random_state=0):
        self.w_c1 = w_c1
        self.w_c2 = w_c2
        self.w_cf = w_cf
        self.w_11 = w_11
        self.n_inner_splits = n_inner_splits
        self.smooth = smooth
        self.random_state = random_state

    # ── stage A hors-fold ────────────────────────────────────────────────────

    def _fit_one_a(self, V_full, V_nm, V_pop, y, seed):
        """Entraine tous les modeles d'etage A sur un sous-ensemble de train."""
        models = {}
        # A1
        models["est"] = _hgbr(max_iter=250, learning_rate=0.08, random_state=seed).fit(V_full, y)
        # A2 conditionnel (avec mesure)
        for ch in ("off", "on"):
            obs = V_full[ch].values
            m = np.isfinite(obs.astype(float))
            if m.sum() >= 10:
                tgt = _log_ratio(y[m], obs[m].astype(float))
                models[f"cond_{ch}"] = _hgbr(max_iter=300, learning_rate=0.06,
                                              min_samples_leaf=40, loss="absolute_error",
                                              random_state=seed).fit(V_full[m], tgt)
        # A3 debiaise (sans mesure)
        for ch in ("off", "on"):
            obs = V_full[ch].values.astype(float)
            m = np.isfinite(obs)
            if m.sum() >= 10:
                tgt = _log_ratio(obs[m], y[m])
                models[f"meas_{ch}"] = _hgbr(max_iter=300, learning_rate=0.05,
                                              min_samples_leaf=50, random_state=seed).fit(
                                                  V_nm[m], tgt)
        # A4 population
        models["pop"] = _hgbr(max_iter=300, learning_rate=0.05, min_samples_leaf=50,
                               random_state=seed).fit(V_pop, y)
        return models

    def _predict_one_a(self, models, V_full, V_nm, V_pop, X_slice):
        """Prediction d'etage A pour une tranche."""
        out = {"est_visit": models["est"].predict(V_full)}
        for ch in ("off", "on"):
            obs = V_full[ch].values.astype(float)
            has = np.isfinite(obs)
            log_obs = np.log(np.clip(np.nan_to_num(obs), 0, None) + LOG_EPS)
            if f"cond_{ch}" in models:
                cond_lr = models[f"cond_{ch}"].predict(V_full)
                out[f"est_{ch}"] = np.where(has, np.exp(cond_lr + log_obs) - LOG_EPS, np.nan)
            else:
                out[f"est_{ch}"] = np.where(has, obs, np.nan)
            if f"meas_{ch}" in models:
                g = models[f"meas_{ch}"].predict(V_nm)
                out[f"u_{ch}"]  = np.where(has, np.exp(log_obs - g) - LOG_EPS, np.nan)
                out[f"g_{ch}"]  = g
            else:
                out[f"u_{ch}"]  = np.where(has, obs, np.nan)
                out[f"g_{ch}"]  = np.zeros(len(V_nm))
        out["est_pop"] = models["pop"].predict(V_pop)
        return pd.DataFrame(out)

    def _compute_oof_a(self, V_full, V_nm, V_pop, X, y, splits):
        """Predictions hors-fold de l'etage A."""
        parts = []
        for tr, va in splits:
            mods = self._fit_one_a(V_full.iloc[tr], V_nm.iloc[tr], V_pop.iloc[tr],
                                   y[tr], self.random_state)
            p = self._predict_one_a(mods, V_full.iloc[va], V_nm.iloc[va],
                                     V_pop.iloc[va], X.iloc[va])
            p.index = va
            parts.append(p)
        return pd.concat(parts).sort_index()

    def _compute_oof_sigma(self, A_oof, y, V_full, V_nm, splits):
        """Bruits hors-fold : |residu| ou |log-residu| predit hors-fold."""
        sigma = pd.DataFrame(index=range(len(y)))
        targets = {
            "sig_est":     (A_oof["est_visit"].values, False),
            "sig_est_off": (A_oof["est_off"].values,   True),
            "sig_est_on":  (A_oof["est_on"].values,    True),
            "sl_off":      (A_oof["u_off"].values,     True),
            "sl_on":       (A_oof["u_on"].values,      True),
        }
        for name, (est, log) in targets.items():
            tgt = np.where(np.isfinite(est),
                           np.abs(_log_ratio(y, est)) if log else np.abs(y - est),
                           np.abs(y - A_oof["est_visit"].values))
            col = np.full(len(y), np.nan)
            V_src = V_nm if "sl_" in name else V_full
            for tr, va in splits:
                m = np.isfinite(tgt[tr])
                col[va] = _sigma_hgbr(random_state=self.random_state).fit(
                    V_src.iloc[tr][m], tgt[tr][m]).predict(V_src.iloc[va])
            sigma[name] = np.where(np.isfinite(est), col, np.nan)
        return sigma

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y):  # noqa: C901
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        groups = X["patient_id"].values
        seed = self.random_state

        self.cats_ = {c: sorted(X[c].dropna().unique()) for c in CATEGORICAL_COLS}
        splits = list(GroupKFold(n_splits=self.n_inner_splits).split(X, y, groups))

        V_full = _visit_full(X, self.cats_)
        V_nm   = _visit_no_meas(X, self.cats_)
        V_pop  = _visit_pop(X, self.cats_)

        # ── Etage A hors-fold ────────────────────────────────────────────
        A_oof = self._compute_oof_a(V_full, V_nm, V_pop, X, y, splits)
        sig   = self._compute_oof_sigma(A_oof, y, V_full, V_nm, splits)
        A_oof = pd.concat([A_oof, sig], axis=1)

        # Modeles finaux A (pour le test)
        self.stage_a_mods_  = self._fit_one_a(V_full, V_nm, V_pop, y, seed)
        # Sigma finaux
        self.sigma_mods_ = {}
        for name, (est, log) in {
            "sig_est":     (A_oof["est_visit"].values, False),
            "sig_est_off": (A_oof["est_off"].values,   True),
            "sig_est_on":  (A_oof["est_on"].values,    True),
            "sl_off":      (A_oof["u_off"].values,     True),
            "sl_on":       (A_oof["u_on"].values,      True),
        }.items():
            tgt = np.where(np.isfinite(est),
                           np.abs(_log_ratio(y, est)) if log else np.abs(y - est),
                           np.abs(y - A_oof["est_visit"].values))
            m = np.isfinite(tgt)
            V_src = V_nm if "sl_" in name else V_full
            self.sigma_mods_[name] = _sigma_hgbr(random_state=seed).fit(V_src[m], tgt[m])

        # ── A priori ─────────────────────────────────────────────────────
        self.prior0_, self.prior4_, self.lev_var_ = _fit_prior(X, y, A_oof["est_pop"].values)

        # ── Etage B : 4 trajectoires hors-fold ───────────────────────────
        t1 = _traj_visit(X, A_oof, self.prior0_, self.lev_var_)
        t2 = _traj_debiased(X, A_oof, self.prior4_, self.lev_var_)
        t3 = _traj_conditional(X, A_oof, self.prior4_, self.lev_var_)
        t4 = _traj_cond4(X, A_oof, self.prior4_, self.lev_var_)
        trajs_oof = {"t1": t1, "t2": t2, "t3": t3, "t4": t4}

        # ── C1 ───────────────────────────────────────────────────────────
        F1 = _stage_c_features(X, A_oof, trajs_oof, self.cats_)
        self.c1_cols_ = list(F1.columns)
        self.c1_model_ = _hgbr(random_state=seed, **FINAL_C1_PARAMS).fit(F1, y - t1["pred"].values)

        # ── Modele coefficients hors-fold ─────────────────────────────────
        t2_cols = {b: f"t2_{b}" for b in COEFS}
        cf_oof  = pd.DataFrame(np.zeros((len(X), 4)),
                               columns=["cf_b0", "cf_b1", "cf_b2", "cf_pred"])
        for tr, va in splits:
            Xtr = X.iloc[tr].reset_index(drop=True)
            Ptr = _patient_table(F1.iloc[tr], Xtr, self.cats_)
            if not all(tc in Ptr.columns for tc in t2_cols.values()):
                continue
            mods_cf = _fit_coef_models(Ptr, _true_coefs(Xtr, y[tr], Ptr.index), t2_cols, seed)
            Xva = X.iloc[va].reset_index(drop=True)
            Pva = _patient_table(F1.iloc[va], Xva, self.cats_)[Ptr.columns]
            cf_oof.iloc[va] = _predict_coefs(mods_cf, Pva, t2_cols, Xva).values
        P_full = _patient_table(F1, X, self.cats_)
        self.coef_cols_  = list(P_full.columns)
        self.t2_cols_    = t2_cols
        self.coef_mods_  = _fit_coef_models(P_full, _true_coefs(X, y, P_full.index), t2_cols, seed)

        # ── C2 ───────────────────────────────────────────────────────────
        F2 = _stage_c_features(X, A_oof, trajs_oof, self.cats_, cf_cols=cf_oof)
        self.c2_cols_ = list(F2.columns)
        self.c2_model_ = _hgbr(random_state=seed, **FINAL_C2_PARAMS).fit(
            F2, y - cf_oof["cf_pred"].values)

        # ── Modele 11 ────────────────────────────────────────────────────
        if self.w_11 > 0:
            self.model_11_ = build_meta_ensemble().fit(longitudinal_view(X), y)
        else:
            self.model_11_ = None
        return self

    # ── predict helpers ──────────────────────────────────────────────────────

    def _predict_a(self, X):
        V_full = _visit_full(X, self.cats_)
        V_nm   = _visit_no_meas(X, self.cats_)
        V_pop  = _visit_pop(X, self.cats_)
        A = self._predict_one_a(self.stage_a_mods_, V_full, V_nm, V_pop, X)
        for name, _ in self.sigma_mods_.items():
            V_src = V_nm if "sl_" in name else V_full
            pred_sig = self.sigma_mods_[name].predict(V_src)
            col_base = name.replace("sig_", "est_").replace("sl_", "u_") if "sl_" in name \
                else name.replace("sig_", "est_")
            # Attach to A only where the base exists
            A[name] = pred_sig
        return A

    def _predict_trajs(self, X, A):
        return {
            "t1": _traj_visit(X, A, self.prior0_, self.lev_var_),
            "t2": _traj_debiased(X, A, self.prior4_, self.lev_var_),
            "t3": _traj_conditional(X, A, self.prior4_, self.lev_var_),
            "t4": _traj_cond4(X, A, self.prior4_, self.lev_var_),
        }

    def predict_components(self, X: pd.DataFrame) -> dict:
        X = X.reset_index(drop=True)
        A = self._predict_a(X)
        trajs = self._predict_trajs(X, A)

        F1 = _stage_c_features(X, A, trajs, self.cats_)[self.c1_cols_]
        c1 = np.clip(self.c1_model_.predict(F1) + trajs["t1"]["pred"].values, 0.0, 132.0)

        P = _patient_table(F1, X, self.cats_)[self.coef_cols_]
        cf = _predict_coefs(self.coef_mods_, P, self.t2_cols_, X)

        F2 = _stage_c_features(X, A, trajs, self.cats_, cf_cols=cf)[self.c2_cols_]
        c2 = np.clip(self.c2_model_.predict(F2) + cf["cf_pred"].values, 0.0, 132.0)

        p11 = (np.clip(self.model_11_.predict(longitudinal_view(X)), 0.0, 132.0)
               if self.model_11_ is not None else np.zeros(len(X)))

        return {"c1": c1, "c2": c2, "cf": cf["cf_pred"].values, "p11": p11,
                "t1": trajs["t1"]["pred"].values, "t2": trajs["t2"]["pred"].values,
                "t3": trajs["t3"]["pred"].values, "t4": trajs["t4"]["pred"].values}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        c = self.predict_components(X)
        pred = self.w_c1 * c["c1"] + self.w_c2 * c["c2"] + self.w_cf * c["cf"] + self.w_11 * c["p11"]
        if self.smooth:
            pred = smooth_predictions(X["patient_id"].values, X["age"].values, pred)
        return np.clip(pred, 0.0, 132.0)


# ── quick mode (OOF cache + grille poids) ────────────────────────────────────

def _fold_ids(cv_splits, n):
    folds = np.zeros(n, dtype=int)
    for k, (_, va) in enumerate(cv_splits):
        folds[va] = k
    return folds


def quick_mode(X: pd.DataFrame, y: pd.Series, cv_splits) -> None:
    """Calcule les 4 composantes OOF (ou recharge le cache), optimise les poids."""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    path = OOF_DIR / "32_components_oof.csv"
    names = ("c1", "c2", "cf", "p11", "t1", "t2", "t3", "t4")
    folds = _fold_ids(cv_splits, len(X))

    if path.is_file():
        print(f"  Cache OOF trouve : {path}", flush=True)
        comp = pd.read_csv(path)
    else:
        comp = pd.DataFrame({"Index": X["Index"].values if "Index" in X.columns
                             else np.arange(len(X)), "fold": folds})
        for n in names:
            comp[n] = 0.0
        for k, (tr, va) in enumerate(cv_splits):
            model = UltimateRegressor(smooth=False).fit(X.iloc[tr], y.iloc[tr])
            c = model.predict_components(X.iloc[va])
            for n in names:
                comp.loc[va, n] = c[n]
            rmse_c1 = np.sqrt(np.mean((c["c1"] - y.iloc[va].values)**2))
            print(f"  pli {k} calcule  C1={rmse_c1:.4f}", flush=True)
        comp.to_csv(path, index=False)
        print(f"  OOF sauvegardees : {path}", flush=True)

    yv = y.values
    pid, age = X["patient_id"].values, X["age"].values

    # --- RMSE par composante seule et par trajectoire ---
    print("\n  RMSE par composante seule :", flush=True)
    for n in names:
        v = comp[n].values
        v_sm = smooth_predictions(pid, age, v)
        rmse_fold = [np.sqrt(np.mean((v[folds == k] - yv[folds == k])**2))
                     for k in range(len(cv_splits))]
        rmse_sm   = [np.sqrt(np.mean((v_sm[folds == k] - yv[folds == k])**2))
                     for k in range(len(cv_splits))]
        print(f"    {n:<6} brut={np.mean(rmse_fold):.4f}+-{np.std(rmse_fold):.4f}"
              f"  lisse={np.mean(rmse_sm):.4f}+-{np.std(rmse_sm):.4f}", flush=True)

    # --- C1 sans t4 (pour mesurer l'apport de t4) ---
    # Recharge depuis oof uniquement si disponible
    path_no_t4 = OOF_DIR / "32_c1_no_t4_oof.csv"
    if path_no_t4.is_file():
        print("\n  C1 sans t4 (cache) :", flush=True)
        c1_no_t4 = pd.read_csv(path_no_t4)["c1_no_t4"].values
    else:
        print("\n  Calcul C1 sans t4 pour mesurer son apport...", flush=True)
        c1_no_t4 = np.zeros(len(X))
        for k, (tr, va) in enumerate(cv_splits):
            model_no_t4 = UltimateRegressor(smooth=False).fit(X.iloc[tr], y.iloc[tr])
            # Desactiver t4 en le remplacant par t1
            model_no_t4.prior4_ = model_no_t4.prior0_
            c_tmp = model_no_t4.predict_components(X.iloc[va])
            c1_no_t4[va] = c_tmp["c1"]
        pd.DataFrame({"c1_no_t4": c1_no_t4}).to_csv(path_no_t4, index=False)
    rmse_no_t4 = [np.sqrt(np.mean((c1_no_t4[folds == k] - yv[folds == k])**2))
                  for k in range(len(cv_splits))]
    rmse_with_t4 = [np.sqrt(np.mean((comp["c1"].values[folds == k] - yv[folds == k])**2))
                    for k in range(len(cv_splits))]
    print(f"    C1 sans t4  : {np.mean(rmse_no_t4):.4f}+-{np.std(rmse_no_t4):.4f}", flush=True)
    print(f"    C1 avec t4  : {np.mean(rmse_with_t4):.4f}+-{np.std(rmse_with_t4):.4f}", flush=True)
    print(f"    Apport t4   : {np.mean(rmse_no_t4) - np.mean(rmse_with_t4):.4f}", flush=True)

    # --- Grille de poids ---
    print("\n  Optimisation des poids du blend (pas=0.05) :", flush=True)
    c1v = comp["c1"].values
    c2v = comp["c2"].values
    cfv = comp["cf"].values
    p11v = comp["p11"].values

    best_rmse, best_w = 999.0, None
    results = []
    step = 0.05
    grid_pts = np.arange(0, 1.0 + step / 2, step)
    for w1 in grid_pts:
        for w2 in grid_pts:
            for w3 in grid_pts:
                w4 = round(1.0 - w1 - w2 - w3, 10)
                if w4 < -1e-9:
                    continue
                w4 = max(w4, 0.0)
                pred = smooth_predictions(pid, age, w1*c1v + w2*c2v + w3*cfv + w4*p11v)
                rmse_f = [np.sqrt(np.mean((pred[folds == k] - yv[folds == k])**2))
                          for k in range(len(cv_splits))]
                m = float(np.mean(rmse_f))
                results.append((m, w1, w2, w3, w4))
                if m < best_rmse:
                    best_rmse, best_w = m, (w1, w2, w3, w4)

    results.sort()
    print(f"  Top 5 combinaisons (c1, c2, cf, p11) :", flush=True)
    for rm, w1, w2, w3, w4 in results[:5]:
        print(f"    c1={w1:.2f} c2={w2:.2f} cf={w3:.2f} p11={w4:.2f}  RMSE={rm:.4f}", flush=True)
    print(f"  Meilleure : c1={best_w[0]:.2f} c2={best_w[1]:.2f} "
          f"cf={best_w[2]:.2f} p11={best_w[3]:.2f}  RMSE={best_rmse:.4f}", flush=True)

    # --- Verification Ridge positive ---
    C_mat = np.column_stack([c1v, c2v, cfv, p11v])
    try:
        ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False)
        ridge.fit(C_mat, yv)
        w_ridge = ridge.coef_ / ridge.coef_.sum()
        pred_r = smooth_predictions(pid, age, C_mat @ ridge.coef_)
        rmse_r = [np.sqrt(np.mean((pred_r[folds == k] - yv[folds == k])**2))
                  for k in range(len(cv_splits))]
        print(f"  Ridge positive : c1={w_ridge[0]:.3f} c2={w_ridge[1]:.3f} "
              f"cf={w_ridge[2]:.3f} p11={w_ridge[3]:.3f}  RMSE={np.mean(rmse_r):.4f}", flush=True)
    except Exception as exc:
        print(f"  Ridge positive echouee : {exc}", flush=True)

    # --- Comparaison finale ---
    print("\n  Comparaison avec les references :", flush=True)
    refs = [("21_smooth", 3.689), ("31_bob_traj", 3.664), ("22_debiased", 3.558),
            ("25_double_c", 3.440), ("32_best_oof", best_rmse)]
    for name, val in refs:
        print(f"    {name:<16} RMSE = {val:.4f}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meilleur modele : blend 4 composantes + trajectoire + lissage (32_ultimate)"
    )
    parser.add_argument("--no-hub",  action="store_true", help="Ne pas publier sur Skore Hub")
    parser.add_argument("--quick",   action="store_true",
                        help="OOF + grille poids uniquement (pas de skore)")
    parser.add_argument("--w-c1",  type=float, default=W_C1)
    parser.add_argument("--w-c2",  type=float, default=W_C2)
    parser.add_argument("--w-cf",  type=float, default=W_CF)
    parser.add_argument("--w-11",  type=float, default=W_11)
    args = parser.parse_args()
    params = dict(w_c1=args.w_c1, w_c2=args.w_c2, w_cf=args.w_cf, w_11=args.w_11)

    print("=" * 70, flush=True)
    print("MODELE ULTIMATE (32_ultimate)", flush=True)
    print("4 composantes : C1 / C2 / coefficients / meta-ensemble", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/5] Chargement des donnees...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv")
    visits = X_train_raw.merge(y_train_raw, on="Index")
    y = visits["target"]
    X = add_longitudinal_columns(visits.drop(columns=["target"]))
    X_test = add_longitudinal_columns(X_test_raw)
    print(f"  Shape X train: {X.shape}  Shape X test: {X_test.shape}", flush=True)

    print("\n[2/5] Configuration de GroupKFold(5) par patient...", flush=True)
    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

    if args.quick:
        print("\n[3/5] Mode --quick : OOF + grille de poids...", flush=True)
        quick_mode(X, y, cv_splits)
        return

    print("\n[3/5] Evaluation skore.evaluate (GroupKFold 5, cle '32_ultimate')...", flush=True)
    model = UltimateRegressor(**params)
    report = evaluate(model, X, y, splitter=cv_splits)
    try:
        print(f"  RMSE moyen GroupKFold :\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  Rapport genere (erreur acces metrique: {exc})", flush=True)
    try:
        print(f"  Resume controles Skore :\n{report.checks.summarize(fast_mode=True)}", flush=True)
    except Exception as exc:
        print(f"  Erreur controles : {exc}", flush=True)

    print("\n[4/5] Entrainement final et prediction sur X_test...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"{HUB_KEY}.csv"
    pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions}).to_csv(sub_path, index=False)
    print(f"  Fichier de soumission : {sub_path} ({len(predictions)} lignes)", flush=True)
    print(f"  Statistiques : min={predictions.min():.2f}  mean={predictions.mean():.2f}"
          f"  max={predictions.max():.2f}", flush=True)

    if args.no_hub:
        print("\n[5/5] Publication Skore Hub ignoree (--no-hub).", flush=True)
        print("=" * 70, flush=True)
        return

    print(f"\n[5/5] Publication sur Skore Hub sous la cle '{HUB_KEY}'...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
    project.put(HUB_KEY, report)
    print("  Publication terminee avec succes !", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
