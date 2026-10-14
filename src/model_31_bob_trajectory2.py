"""Trajectoire patient v2 : 4 corrections vs 30_bob_trajectory (31_bob_trajectory2).

Project: Bobathon ESILV (Probabl x IBM)

Corrections par rapport au modele 30 :

1. A priori de niveau reel dans l'etage B
   Modele 30 : lam[0,0] = 0 (pas d'a priori sur b0, juste un mu[0] = moyenne pop).
   Modele 31 : lam[0,0] = prior_strength / level_var_ ou level_var_ est l'ecart-type
   des moyennes patient de (cible - pop_prediction) estime sur le train.  Cela ancre
   vraiment b0 vers le residu moyen attendu au-dessus du modele de population.

2. u_off et u_on comme observations separees dans la trajectoire
   Modele 30 : priorite off, fallback on (une seule observation par visite).
   Modele 31 : si les deux sont disponibles, ils sont empiles comme deux observations
   independantes avec leurs propres sigma ; chaque canal contribue au fit bayesien.

3. Bruit des modeles de mesure en log-ratio, avec has_off/has_on comme features
   Modele 30 : bruit = |u - y| sur les visites avec mesure.
   Modele 31 : bruit = |log(u+1) - log(y+1)| (erreur relative, plus stable), converti
   en sigma absolu via sigma_abs = sigma_rel * (traj_ajustee + 1).  On ajoute aussi
   has_off et has_on en indicateurs dans V_no_meas pour que les modeles log-ratio
   sachent si l'autre canal est disponible (co-information sur la qualite de la visite).

4. Trois trajectoires en entree de l'etage C
   Modele 30 : une seule trajectoire bayesienne (sur combined = off or on or visit).
   Modele 31 : trois trajectoires independantes passees a l'etage C comme features :
     traj_main  : off+on empiles (correction 2)
     traj_visit : estimation visite seule (Aa, sans mesures)
     traj_cond  : visite + u_off conditionnel + u_on conditionnel (off/on inverses
                  meme quand la mesure est manquante, avec sigma eleve)
   L'etage C peut choisir de les combiner ou d'ignorer les moins fiables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "31_bob_trajectory2"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_09_longitudinal import prepare_longitudinal_features

CATEGORICAL_COLS = ["cohort", "gene", "rater_id"]
LF_PREFIX = "lf_"
MAE2STD = float(np.sqrt(np.pi / 2))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def add_longitudinal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features du modele 09 prefixees lf_ (calcul intra-patient, pas de fuite)."""
    lf = prepare_longitudinal_features(df).add_prefix(LF_PREFIX)
    return pd.concat([df.reset_index(drop=True), lf.reset_index(drop=True)], axis=1)


def _hgbr(**kwargs) -> HistGradientBoostingRegressor:
    defaults = dict(
        max_iter=500,
        learning_rate=0.04,
        min_samples_leaf=20,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        random_state=0,
    )
    defaults.update(kwargs)
    return HistGradientBoostingRegressor(**defaults)


def _hgbr_c(**kwargs) -> HistGradientBoostingRegressor:
    """HGBR pour l'etage C : 63 feuilles, min_samples=100, lr=0.02, 1500 iter."""
    return _hgbr(
        max_iter=1500,
        learning_rate=0.02,
        min_samples_leaf=100,
        max_leaf_nodes=63,
        l2_regularization=2.0,
        **kwargs,
    )


def _sigma_hgbr(**kwargs) -> HistGradientBoostingRegressor:
    """HGBR pour predire le bruit (MAE loss)."""
    return _hgbr(
        max_iter=150,
        learning_rate=0.08,
        min_samples_leaf=100,
        loss="absolute_error",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# feature builders
# ---------------------------------------------------------------------------

def _visit_features_full(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Toutes les variables de la visite incluant on/off (etage Aa)."""
    data = pd.DataFrame(index=df.index)
    for col in ["age_at_diagnosis", "age", "ledd",
                "time_since_intake_on", "time_since_intake_off", "on", "off", "sexM"]:
        data[col] = df[col]
    data["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    data["off_minus_on"] = df["off"] - df["on"]
    data["off_over_on"] = df["off"] / df["on"].replace(0, np.nan)
    for col in CATEGORICAL_COLS:
        data[col] = pd.Categorical(df[col], categories=categories[col])
    return data


def _visit_features_no_meas(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Variables contextuelles SANS on/off, AVEC has_off/has_on (correction 3).

    has_off et has_on informent le modele de ratio log sur la qualite de la visite :
    la presence simultanee de off et on est un signal de visite complete et fiable.
    En excluant les mesures brutes elles-memes on preserve le caractere non-retreci
    de l'estimation par ratio log.
    """
    data = pd.DataFrame(index=df.index)
    for col in ["age_at_diagnosis", "age", "ledd",
                "time_since_intake_on", "time_since_intake_off", "sexM"]:
        data[col] = df[col]
    data["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    # Correction 3 : indicateurs de disponibilite
    data["has_off"] = df["off"].notna().astype(float)
    data["has_on"]  = df["on"].notna().astype(float)
    for col in CATEGORICAL_COLS:
        data[col] = pd.Categorical(df[col], categories=categories[col])
    return data


def _visit_features_demo(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Demographie + temps uniquement (modele de population Ac)."""
    data = pd.DataFrame(index=df.index)
    for col in ["age_at_diagnosis", "age", "sexM"]:
        data[col] = df[col]
    data["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    for col in ["cohort", "gene"]:
        data[col] = pd.Categorical(df[col], categories=categories[col])
    return data


# ---------------------------------------------------------------------------
# estimateur principal
# ---------------------------------------------------------------------------

class BobTrajectoryRegressor2(RegressorMixin, BaseEstimator):
    """Estimateur trajectoire patient v2 : 4 corrections vs modele 30.

    Parameters
    ----------
    n_inner_splits : int
        Nombre de plis GroupKFold internes pour les predictions hors-fold en train.
    prior_strength : float
        Force commune de regularisation bayesienne (pente, courbure, niveau).
    huber_c : float
        Seuil Huber (en unites de sigma) pour l'IRLS de l'etage B.
    huber_iter : int
        Nombre d'iterations IRLS.
    apply_isotonic : bool
        Post-traitement isotonique croissant par patient.
    random_state : int
        Graine aleatoire.
    """

    def __init__(
        self,
        n_inner_splits: int = 5,
        prior_strength: float = 2.0,
        huber_c: float = 1.5,
        huber_iter: int = 3,
        apply_isotonic: bool = True,
        random_state: int = 0,
    ):
        self.n_inner_splits = n_inner_splits
        self.prior_strength = prior_strength
        self.huber_c = huber_c
        self.huber_iter = huber_iter
        self.apply_isotonic = apply_isotonic
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Etage B : a priori de population
    # ------------------------------------------------------------------

    def _fit_prior(
        self,
        pid: np.ndarray,
        age: np.ndarray,
        y: np.ndarray,
        oof_pop: np.ndarray,
    ) -> None:
        """Estime les a priori de population sur (niveau, pente, courbure).

        Correction 1 : l'a priori de niveau est estime sur les residus (y - pop)
        par patient, pas sur y brut.  level_var_ = variance des moyennes par patient
        de (y - oof_pop).  Cela calibre lam[0,0] pour que l'ancrage de b0 soit
        proportionnel a l'incertitude reelle du niveau residuel.
        """
        df = pd.DataFrame({"pid": pid, "age": age, "y": y, "pop": oof_pop})
        coefs, level_means = [], []
        for _, g in df.groupby("pid"):
            if len(g) < 4:
                continue
            t = g["age"].values - g["age"].values.mean()
            # residus par rapport a la population pour le niveau
            resid = g["y"].values - g["pop"].values
            level_means.append(resid.mean())
            # fit sur les cibles originales pour pente/courbure
            c = np.polyfit(t, g["y"].values, 2)[::-1]  # [b0, b1, b2]
            coefs.append(c)
        coefs = np.array(coefs)
        # Prior pente (b1) et courbure (b2)
        self.prior_mean_ = coefs[:, 1:].mean(axis=0)
        self.prior_var_  = coefs[:, 1:].var(axis=0) + 1e-6
        # Correction 1 : prior niveau
        level_means = np.array(level_means)
        self.level_var_ = float(level_means.var()) + 1e-6

    def _trajectories(
        self,
        pid: np.ndarray,
        age: np.ndarray,
        est: np.ndarray,        # estimations empilees (N_obs,)
        sigma: np.ndarray,      # sigma par observation
        pop_level: np.ndarray,  # a priori de niveau par obs (prediction pop)
        obs_pid: np.ndarray,    # patient_id de chaque obs (peut differer de pid si empile)
        obs_age: np.ndarray,    # age de chaque obs
    ) -> pd.DataFrame:
        """Ajuste b0 + b1*t + b2*t^2 par patient, retourne les predictions sur la grille pid/age.

        Accepte une liste d'observations (obs_pid, obs_age, est, sigma) qui peut contenir
        plusieurs observations par visite (correction 2 : u_off et u_on separes).
        La prediction est evaluee sur la grille originale (pid, age).
        """
        out = pd.DataFrame(
            index=np.arange(len(pid)),
            columns=["traj_pred", "traj_b0", "traj_b1", "traj_b2", "traj_se"],
            dtype=float,
        )
        # Correction 1 : lam[0,0] = prior_strength / level_var_
        lam = np.diag([
            self.prior_strength / self.level_var_,
            *(self.prior_strength / self.prior_var_),
        ])
        mu_base = np.array([0.0, *self.prior_mean_])

        # Groupes sur les observations
        obs_df = pd.DataFrame({"pid": obs_pid, "age": obs_age, "est": est, "sigma": sigma,
                               "pop": pop_level})
        # Grille de prediction
        grid_df = pd.DataFrame({"pid": pid, "age": age})
        grid_order = grid_df.groupby("pid").indices

        for p, idx_grid in grid_order.items():
            obs_p = obs_df[obs_df["pid"] == p]
            if len(obs_p) == 0:
                # fallback : mu du prior
                out.iloc[idx_grid, 0] = mu_base[0]
                out.iloc[idx_grid, 1:] = 0.0
                continue

            t_obs = obs_p["age"].values - obs_p["age"].values.mean()
            A_obs = np.column_stack([np.ones_like(t_obs), t_obs, t_obs**2])
            e_obs = obs_p["est"].values
            s_obs = np.maximum(obs_p["sigma"].values, 0.5)

            # A priori de niveau : mu[0] = moyenne du residu attendu (y-pop) par patient
            mu = mu_base.copy()
            mu[0] = obs_p["pop"].values.mean()

            w0 = 1.0 / s_obs**2
            w  = w0.copy()
            for _ in range(self.huber_iter + 1):
                M = A_obs.T @ (A_obs * w[:, None]) + lam
                rhs = A_obs.T @ (w * e_obs) + lam @ mu
                b = np.linalg.solve(M, rhs)
                z = np.abs(e_obs - A_obs @ b) / s_obs
                w = w0 * np.minimum(1.0, self.huber_c / np.maximum(z, 1e-9))

            cov = np.linalg.inv(A_obs.T @ (A_obs * w0[:, None]) + lam)

            # Prediction sur la grille
            t_grid = grid_df.loc[idx_grid, "age"].values - obs_p["age"].values.mean()
            A_grid = np.column_stack([np.ones_like(t_grid), t_grid, t_grid**2])
            traj = A_grid @ b
            se   = np.sqrt(np.einsum("ij,jk,ik->i", A_grid, cov, A_grid))

            out.iloc[idx_grid] = np.column_stack([
                traj,
                np.full(len(idx_grid), b[0]),
                np.full(len(idx_grid), b[1]),
                np.full(len(idx_grid), b[2]),
                se,
            ])
        return out

    # ------------------------------------------------------------------
    # Assemblage etage C
    # ------------------------------------------------------------------

    def _build_obs_stacked(
        self,
        pid: np.ndarray,
        age: np.ndarray,
        est_visit: np.ndarray,
        sigma_visit: np.ndarray,
        u_off: np.ndarray,      # nan si mesure absente
        sigma_off: np.ndarray,
        u_on: np.ndarray,
        sigma_on: np.ndarray,
        pop_level: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Correction 2 : empile u_off et u_on comme observations separees.

        Pour une visite i :
        - si off disponible  : ajoute (pid[i], age[i], u_off[i], sigma_off[i], pop[i])
        - si on disponible   : ajoute (pid[i], age[i], u_on[i],  sigma_on[i],  pop[i])
        - si ni l'un ni l'autre : utilise est_visit[i] comme observation unique
        Retourne (obs_pid, obs_age, obs_est, obs_sigma, obs_pop).
        """
        rows_pid, rows_age, rows_est, rows_sigma, rows_pop = [], [], [], [], []
        for i in range(len(pid)):
            added = False
            if np.isfinite(u_off[i]):
                rows_pid.append(pid[i])
                rows_age.append(age[i])
                rows_est.append(u_off[i])
                rows_sigma.append(sigma_off[i])
                rows_pop.append(pop_level[i])
                added = True
            if np.isfinite(u_on[i]):
                rows_pid.append(pid[i])
                rows_age.append(age[i])
                rows_est.append(u_on[i])
                rows_sigma.append(sigma_on[i])
                rows_pop.append(pop_level[i])
                added = True
            if not added:
                rows_pid.append(pid[i])
                rows_age.append(age[i])
                rows_est.append(est_visit[i])
                rows_sigma.append(sigma_visit[i])
                rows_pop.append(pop_level[i])
        return (np.array(rows_pid), np.array(rows_age, dtype=float),
                np.array(rows_est, dtype=float), np.array(rows_sigma, dtype=float),
                np.array(rows_pop, dtype=float))

    def _stage_c_features(
        self,
        X: pd.DataFrame,
        est_visit: np.ndarray,
        sigma_visit: np.ndarray,
        u_off: np.ndarray,
        sigma_off: np.ndarray,
        u_on: np.ndarray,
        sigma_on: np.ndarray,
        est_pop: np.ndarray,
        sigma_pop: np.ndarray,
    ) -> pd.DataFrame:
        pid = X["patient_id"].values
        age = X["age"].values.astype(float)

        # Correction 4 : trois trajectoires independantes
        # traj_main : off + on empiles (correction 2)
        obs = self._build_obs_stacked(
            pid, age, est_visit, sigma_visit, u_off, sigma_off, u_on, sigma_on, est_pop,
        )
        traj_main = self._trajectories(pid, age, obs[2], obs[3], obs[4], obs[0], obs[1])

        # traj_visit : estimation visite seule (Aa)
        traj_visit = self._trajectories(
            pid, age, est_visit, sigma_visit, est_pop, pid, age,
        )

        # traj_cond : visite + u_off conditionnel + u_on conditionnel
        # Quand la mesure est absente on utilise est_visit avec sigma triple
        u_off_cond   = np.where(np.isfinite(u_off), u_off, est_visit)
        sig_off_cond = np.where(np.isfinite(u_off), sigma_off, sigma_visit * 3.0)
        u_on_cond    = np.where(np.isfinite(u_on), u_on, est_visit)
        sig_on_cond  = np.where(np.isfinite(u_on), sigma_on, sigma_visit * 3.0)
        # Empile visit + off_cond + on_cond (3 obs par visite)
        obs_c_pid   = np.tile(pid, 3)
        obs_c_age   = np.tile(age, 3)
        obs_c_est   = np.concatenate([est_visit,  u_off_cond,   u_on_cond])
        obs_c_sigma = np.concatenate([sigma_visit, sig_off_cond, sig_on_cond])
        obs_c_pop   = np.tile(est_pop, 3)
        traj_cond = self._trajectories(
            pid, age, obs_c_est, obs_c_sigma, obs_c_pop, obs_c_pid, obs_c_age,
        )

        # Features longitudinales du modele 09
        lf_cats = [LF_PREFIX + c for c in CATEGORICAL_COLS]
        feats = X[[c for c in X.columns if c.startswith(LF_PREFIX) and c not in lf_cats]].copy()
        feats = feats.reset_index(drop=True)
        for col in CATEGORICAL_COLS:
            feats[col] = pd.Categorical(X[col].values, categories=self.categories_[col])

        # Sorties A
        feats["A_visit_est"]    = est_visit
        feats["A_visit_sigma"]  = sigma_visit
        feats["A_u_off"]        = np.where(np.isfinite(u_off), u_off, np.nan)
        feats["A_sigma_off"]    = sigma_off
        feats["A_u_on"]         = np.where(np.isfinite(u_on),  u_on,  np.nan)
        feats["A_sigma_on"]     = sigma_on
        feats["A_pop_est"]      = est_pop
        feats["A_pop_sigma"]    = sigma_pop
        feats["has_off"]        = np.isfinite(u_off).astype(float)
        feats["has_on"]         = np.isfinite(u_on).astype(float)

        # Correction 4 : 3 trajectoires
        traj_main_r  = traj_main.reset_index(drop=True).add_suffix("_main")
        traj_visit_r = traj_visit.reset_index(drop=True).add_suffix("_visit")
        traj_cond_r  = traj_cond.reset_index(drop=True).add_suffix("_cond")
        feats = pd.concat([feats, traj_main_r, traj_visit_r, traj_cond_r], axis=1)

        # Ecarts visite - trajectoire pour chaque canal
        feats["visit_minus_traj_main"]  = est_visit - traj_main["traj_pred"].values
        feats["visit_minus_traj_visit"] = est_visit - traj_visit["traj_pred"].values
        feats["visit_minus_traj_cond"]  = est_visit - traj_cond["traj_pred"].values
        return feats

    # ------------------------------------------------------------------
    # Correction 3 : bruit en log-ratio
    # ------------------------------------------------------------------

    def _log_sigma_to_abs(
        self,
        log_sigma_rel: np.ndarray,
        traj_level: np.ndarray,
    ) -> np.ndarray:
        """Convertit sigma_rel en sigma_abs : sigma_abs = sigma_rel * (traj + 1).

        sigma_rel = ecart-type de |log(u+1) - log(y+1)|, qui est approximativement
        l'ecart-type relatif du ratio u/y.  La trajectoire courante sert d'ancrage
        pour convertir en unite absolue.
        """
        return log_sigma_rel * np.maximum(traj_level + 1.0, 1.0)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y):  # noqa: C901
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        groups = X["patient_id"].values

        self.categories_ = {c: sorted(X[c].dropna().unique()) for c in CATEGORICAL_COLS}
        splits = list(GroupKFold(n_splits=self.n_inner_splits).split(X, y, groups))

        V_full    = _visit_features_full(X, self.categories_)
        V_no_meas = _visit_features_no_meas(X, self.categories_)   # + has_off/has_on
        V_demo    = _visit_features_demo(X, self.categories_)

        # -- Aa : HGBR visite -> y_visit (hors-fold) ----------------------
        oof_visit = np.zeros(len(X))
        for tr, va in splits:
            m = _hgbr(max_iter=250, learning_rate=0.08, random_state=self.random_state)
            m.fit(V_full.iloc[tr], y[tr])
            oof_visit[va] = m.predict(V_full.iloc[va])
        self.visit_model_ = _hgbr(
            max_iter=250, learning_rate=0.08, random_state=self.random_state,
        ).fit(V_full, y)

        # -- Ab : modeles de mesure log-ratio (hors-fold) -----------------
        off_v = X["off"].values.astype(float)
        on_v  = X["on"].values.astype(float)
        log_ratio_off = np.where(
            np.isfinite(off_v), np.log1p(off_v) - np.log1p(y), np.nan,
        )
        log_ratio_on = np.where(
            np.isfinite(on_v), np.log1p(on_v) - np.log1p(y), np.nan,
        )

        oof_lr_off = np.full(len(X), np.nan)
        oof_lr_on  = np.full(len(X), np.nan)
        for tr, va in splits:
            mask_off_tr = np.isfinite(log_ratio_off[tr])
            if mask_off_tr.sum() >= 10:
                m_off = _hgbr(max_iter=200, learning_rate=0.06, random_state=self.random_state)
                m_off.fit(V_no_meas.iloc[tr][mask_off_tr], log_ratio_off[tr][mask_off_tr])
                oof_lr_off[va] = m_off.predict(V_no_meas.iloc[va])
            mask_on_tr = np.isfinite(log_ratio_on[tr])
            if mask_on_tr.sum() >= 10:
                m_on = _hgbr(max_iter=200, learning_rate=0.06, random_state=self.random_state)
                m_on.fit(V_no_meas.iloc[tr][mask_on_tr], log_ratio_on[tr][mask_on_tr])
                oof_lr_on[va] = m_on.predict(V_no_meas.iloc[va])

        mask_off_all = np.isfinite(log_ratio_off)
        self.meas_off_model_ = _hgbr(
            max_iter=200, learning_rate=0.06, random_state=self.random_state,
        ).fit(V_no_meas[mask_off_all], log_ratio_off[mask_off_all])
        mask_on_all = np.isfinite(log_ratio_on)
        self.meas_on_model_ = _hgbr(
            max_iter=200, learning_rate=0.06, random_state=self.random_state,
        ).fit(V_no_meas[mask_on_all], log_ratio_on[mask_on_all])

        # Inverser log-ratio -> u (correction 2 : garder les deux separes)
        oof_u_off = np.where(
            np.isfinite(off_v) & np.isfinite(oof_lr_off),
            (off_v + 1.0) / np.exp(oof_lr_off) - 1.0, np.nan,
        )
        oof_u_on = np.where(
            np.isfinite(on_v) & np.isfinite(oof_lr_on),
            (on_v + 1.0) / np.exp(oof_lr_on) - 1.0, np.nan,
        )

        # -- Ac : modele de population ------------------------------------
        oof_pop = np.zeros(len(X))
        for tr, va in splits:
            m_pop = _hgbr(max_iter=200, learning_rate=0.06, random_state=self.random_state)
            m_pop.fit(V_demo.iloc[tr], y[tr])
            oof_pop[va] = m_pop.predict(V_demo.iloc[va])
        self.pop_model_ = _hgbr(
            max_iter=200, learning_rate=0.06, random_state=self.random_state,
        ).fit(V_demo, y)

        # -- Ad : modeles de bruit ----------------------------------------
        # Visite : bruit absolu classique
        abs_res_visit = np.abs(y - oof_visit)
        abs_res_pop   = np.abs(y - oof_pop)

        # Correction 3 : bruit en log-ratio pour off/on
        # log_err_off[i] = |log(u_off+1) - log(y+1)| quand u_off disponible
        log_err_off = np.where(
            np.isfinite(oof_u_off),
            np.abs(np.log1p(np.maximum(oof_u_off, 0.0)) - np.log1p(y)),
            np.abs(np.log1p(np.maximum(oof_visit, 0.0)) - np.log1p(y)),
        )
        log_err_on = np.where(
            np.isfinite(oof_u_on),
            np.abs(np.log1p(np.maximum(oof_u_on, 0.0)) - np.log1p(y)),
            np.abs(np.log1p(np.maximum(oof_visit, 0.0)) - np.log1p(y)),
        )

        oof_sv = np.zeros(len(X))
        oof_sp = np.zeros(len(X))
        oof_slog_off = np.zeros(len(X))
        oof_slog_on  = np.zeros(len(X))
        for tr, va in splits:
            ms_v = _sigma_hgbr(random_state=self.random_state).fit(V_full.iloc[tr], abs_res_visit[tr])
            oof_sv[va] = ms_v.predict(V_full.iloc[va])
            ms_p = _sigma_hgbr(random_state=self.random_state).fit(V_demo.iloc[tr], abs_res_pop[tr])
            oof_sp[va] = ms_p.predict(V_demo.iloc[va])
            ms_lo = _sigma_hgbr(random_state=self.random_state).fit(V_no_meas.iloc[tr], log_err_off[tr])
            oof_slog_off[va] = ms_lo.predict(V_no_meas.iloc[va])
            ms_ln = _sigma_hgbr(random_state=self.random_state).fit(V_no_meas.iloc[tr], log_err_on[tr])
            oof_slog_on[va] = ms_ln.predict(V_no_meas.iloc[va])

        oof_sigma_visit = np.clip(oof_sv, 0.01, None) * MAE2STD
        oof_sigma_pop   = np.clip(oof_sp, 0.01, None) * MAE2STD
        # sigma_rel (log) -> conserve pour la conversion plus bas
        oof_slog_off = np.clip(oof_slog_off, 1e-3, None) * MAE2STD
        oof_slog_on  = np.clip(oof_slog_on,  1e-3, None) * MAE2STD

        self.sigma_visit_model_  = _sigma_hgbr(random_state=self.random_state).fit(V_full, abs_res_visit)
        self.sigma_pop_model_    = _sigma_hgbr(random_state=self.random_state).fit(V_demo, abs_res_pop)
        self.slog_off_model_     = _sigma_hgbr(random_state=self.random_state).fit(V_no_meas, log_err_off)
        self.slog_on_model_      = _sigma_hgbr(random_state=self.random_state).fit(V_no_meas, log_err_on)

        # Correction 1 : a priori avec level_var_ estime sur residus (y - pop)
        self._fit_prior(groups, X["age"].values.astype(float), y, oof_pop)

        # Conversion sigma_rel -> sigma_abs (necessaire pour l'etage B en unites absolues)
        # On utilise oof_visit comme approximation de la trajectoire courante
        oof_sigma_off = self._log_sigma_to_abs(oof_slog_off, oof_visit)
        oof_sigma_on  = self._log_sigma_to_abs(oof_slog_on,  oof_visit)

        # -- C : HGBR sur features enrichies ------------------------------
        F = self._stage_c_features(
            X,
            oof_visit, oof_sigma_visit,
            oof_u_off, oof_sigma_off,
            oof_u_on,  oof_sigma_on,
            oof_pop,   oof_sigma_pop,
        )
        self.feature_names_c_ = list(F.columns)
        target_c = y - F["traj_pred_main"].values
        self.final_model_ = _hgbr_c(random_state=self.random_state).fit(F, target_c)
        return self

    # ------------------------------------------------------------------
    # predict helpers
    # ------------------------------------------------------------------

    def _predict_stage_a(self, X: pd.DataFrame):
        """Retourne (est_visit, sigma_visit, u_off, sigma_off, u_on, sigma_on, est_pop, sigma_pop)."""
        V_full    = _visit_features_full(X, self.categories_)
        V_no_meas = _visit_features_no_meas(X, self.categories_)
        V_demo    = _visit_features_demo(X, self.categories_)

        est_visit = self.visit_model_.predict(V_full)
        est_pop   = self.pop_model_.predict(V_demo)

        sigma_visit = np.clip(self.sigma_visit_model_.predict(V_full), 0.01, None) * MAE2STD
        sigma_pop   = np.clip(self.sigma_pop_model_.predict(V_demo),   0.01, None) * MAE2STD
        slog_off    = np.clip(self.slog_off_model_.predict(V_no_meas), 1e-3, None) * MAE2STD
        slog_on     = np.clip(self.slog_on_model_.predict(V_no_meas),  1e-3, None) * MAE2STD

        r_off = self.meas_off_model_.predict(V_no_meas)
        r_on  = self.meas_on_model_.predict(V_no_meas)

        off_v = X["off"].values.astype(float)
        on_v  = X["on"].values.astype(float)

        u_off = np.where(np.isfinite(off_v), (off_v + 1.0) / np.exp(r_off) - 1.0, np.nan)
        u_on  = np.where(np.isfinite(on_v),  (on_v  + 1.0) / np.exp(r_on)  - 1.0, np.nan)

        sigma_off = self._log_sigma_to_abs(slog_off, est_visit)
        sigma_on  = self._log_sigma_to_abs(slog_on,  est_visit)

        return est_visit, sigma_visit, u_off, sigma_off, u_on, sigma_on, est_pop, sigma_pop

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        (est_visit, sigma_visit,
         u_off, sigma_off,
         u_on,  sigma_on,
         est_pop, sigma_pop) = self._predict_stage_a(X)

        F = self._stage_c_features(
            X,
            est_visit, sigma_visit,
            u_off, sigma_off,
            u_on,  sigma_on,
            est_pop, sigma_pop,
        )[self.feature_names_c_]

        traj = F["traj_pred_main"].values
        raw  = traj + self.final_model_.predict(F)

        if self.apply_isotonic:
            raw = self._apply_isotonic(X["patient_id"].values, X["age"].values.astype(float), raw)

        return np.clip(raw, 0.0, 132.0)

    def _apply_isotonic(
        self, pid: np.ndarray, age: np.ndarray, pred: np.ndarray,
    ) -> np.ndarray:
        """Parabole sur les predictions puis isotonique croissant par patient."""
        out = pred.copy()
        order = pd.Series(np.arange(len(pid))).groupby(np.asarray(pid)).indices
        for idx in order.values():
            t = age[idx]
            p = pred[idx]
            sort_idx = np.argsort(t)
            if len(idx) >= 3:
                tc = t - t.mean()
                smoothed = np.polyval(np.polyfit(tc, p, 2), tc)
            else:
                smoothed = p.copy()
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(t[sort_idx], smoothed[sort_idx])
            out[idx] = iso.predict(t)
        return out

    def predict_stages(self, X: pd.DataFrame) -> pd.DataFrame:
        """Diagnostic par etage pour --quick."""
        X = X.reset_index(drop=True)
        (est_visit, sigma_visit,
         u_off, sigma_off,
         u_on,  sigma_on,
         est_pop, sigma_pop) = self._predict_stage_a(X)

        # Aa : visite seule
        # Ab : combinaison off>on>visit (pour reference avec modele 30)
        ab_combined = np.where(np.isfinite(u_off), u_off,
                      np.where(np.isfinite(u_on),  u_on, est_visit))
        # Ac : population
        # B_main : trajectoire principale (off+on empiles)
        F = self._stage_c_features(
            X,
            est_visit, sigma_visit,
            u_off, sigma_off,
            u_on,  sigma_on,
            est_pop, sigma_pop,
        )[self.feature_names_c_]

        traj_main  = F["traj_pred_main"].values
        traj_visit = F["traj_pred_visit"].values
        traj_cond  = F["traj_pred_cond"].values
        stage_c    = traj_main + self.final_model_.predict(F)
        final      = self.predict(X)

        return pd.DataFrame({
            "Aa_visit":    est_visit,
            "Ab_combined": ab_combined,
            "Ac_pop":      est_pop,
            "B_main":      traj_main,
            "B_visit":     traj_visit,
            "B_cond":      traj_cond,
            "C_final":     stage_c,
            "D_isoton":    final,
        })


# ---------------------------------------------------------------------------
# CV rapide
# ---------------------------------------------------------------------------

def quick_cv(X: pd.DataFrame, y: pd.Series, cv_splits, **params) -> None:
    """CV rapide hors skore : RMSE de chaque etage sur les 5 plis."""
    scores = []
    for k, (tr, va) in enumerate(cv_splits):
        model = BobTrajectoryRegressor2(**params)
        model.fit(X.iloc[tr], y.iloc[tr])
        stages = model.predict_stages(X.iloc[va])
        y_va = y.iloc[va].values
        rmse = np.sqrt(((stages.values - y_va[:, None]) ** 2).mean(axis=0))
        scores.append(rmse)
        print(
            f"  pli {k}: " + "  ".join(f"{c}={r:.4f}" for c, r in zip(stages.columns, rmse)),
            flush=True,
        )
    s = np.array(scores)
    print(
        "  MOYENNE: "
        + "  ".join(
            f"{c}={m:.4f}+-{d:.4f}" for c, m, d in zip(stages.columns, s.mean(0), s.std(0))
        ),
        flush=True,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trajectoire patient v2 : 4 corrections vs 30 (31_bob_trajectory2)"
    )
    parser.add_argument("--no-hub",      action="store_true", help="Ne pas publier sur Skore Hub")
    parser.add_argument("--quick",       action="store_true", help="CV rapide par etage (pas de skore)")
    parser.add_argument("--prior-strength", type=float, default=2.0)
    parser.add_argument("--no-isotonic", action="store_true", help="Desactiver post-traitement isotonique")
    args = parser.parse_args()
    params = dict(
        prior_strength=args.prior_strength,
        apply_isotonic=not args.no_isotonic,
    )

    print("=" * 70, flush=True)
    print("MODELE TRAJECTOIRE PATIENT v2 (31_bob_trajectory2)", flush=True)
    print("4 corrections vs modele 30", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/5] Chargement des donnees...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv")
    visits = X_train_raw.merge(y_train_raw, on="Index")
    y = visits["target"]
    X = add_longitudinal_columns(visits.drop(columns=["target"]))
    X_test = add_longitudinal_columns(X_test_raw)
    print(f"  Shape X train: {X.shape}, Shape X test: {X_test.shape}", flush=True)

    print("\n[2/5] Configuration de GroupKFold(5) par patient...", flush=True)
    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

    if args.quick:
        print("\n[3/5] CV rapide par etage (--quick)...", flush=True)
        quick_cv(X, y, cv_splits, **params)
        return

    print("\n[3/5] Evaluation skore.evaluate (GroupKFold 5, cle '31_bob_trajectory2')...", flush=True)
    model = BobTrajectoryRegressor2(**params)
    report = evaluate(model, X, y, splitter=cv_splits)
    try:
        print(f"  RMSE moyen GroupKFold :\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  Rapport genere (erreur acces metrique: {exc})", flush=True)
    try:
        print(
            f"  Resume des controles Skore :\n{report.checks.summarize(fast_mode=True)}",
            flush=True,
        )
    except Exception as exc:
        print(f"  Erreur controles: {exc}", flush=True)

    print("\n[4/5] Entrainement final et prediction sur X_test...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"{HUB_KEY}.csv"
    pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions}).to_csv(sub_path, index=False)
    print(f"  Fichier de soumission : {sub_path} ({len(predictions)} lignes)", flush=True)
    print(
        f"  Statistiques : min={predictions.min():.2f}, "
        f"mean={predictions.mean():.2f}, max={predictions.max():.2f}",
        flush=True,
    )

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
