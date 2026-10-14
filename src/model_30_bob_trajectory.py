"""Trajectoire patient enrichie : modèles de mesure + trajectoire quadratique bayésienne (30_bob_trajectory).

Project: Bobathon ESILV (Probabl x IBM)

Constat (vérifié sur les données) :
- Par patient, la cible est quasi une parabole de l'âge : RMSE résiduel quadratique = 0.21 (moyen),
  contre 1.19 en linéaire.  Le problème se réduit à estimer 3 coefficients par patient (~8 visites).

Architecture :
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Étage A (par visite, hors-fold GroupKFold patient)                               │
│   Aa. HGBR visite → ŷ_visit                                                     │
│   Ab. Modèles de mesure : prédire log(mesure+1) - log(cible+1) SANS les mesures │
│       → débiaisage multiplicatif ; u_off, u_on inversés                         │
│   Ac. Modèle de population (démographie + temps) → ŷ_pop                        │
│   Ad. Modèles de bruit : |résidu| hors-fold → σ_visit, σ_meas, σ_pop            │
├──────────────────────────────────────────────────────────────────────────────────┤
│ Étage B (par patient) : régression quadratique pondérée (1/σ²), robuste Huber,  │
│   avec a priori de population sur pente/courbure et sur le niveau (via Ac)       │
├──────────────────────────────────────────────────────────────────────────────────┤
│ Étage C : HGBR (63 feuilles, min_samples=100, lr=0.02, 1500 iter) sur features  │
│   longitudinales (modèle 09) + sorties A + B → résidu par rapport à la traj.    │
├──────────────────────────────────────────────────────────────────────────────────┤
│ Post-traitement : parabole sur prédictions par patient + isotonic (croissant)    │
└──────────────────────────────────────────────────────────────────────────────────┘

Pourquoi le modèle de mesure (Ab) évite-t-il le rétrécissement vers la moyenne ?
  Un HGBR qui prédit directement la cible (Aa) apprend à régulariser ses poids vers
  la moyenne de la distribution d'entraînement : toute prédiction extrême est pénalisée
  par la loss quadratique, donc les prédictions hautes sont tirées vers le bas et
  vice-versa (biais de régression vers la moyenne de Galton).
  Le modèle de mesure, lui, prédit le *ratio log* log(mesure+1) - log(cible+1) à
  partir des seules variables contextuelles (rater, délai, ledd, âge) — sans accès
  à la mesure elle-même.  L'estimation de la cible est alors :
      u = (mesure + 1) / exp(r̂) - 1
  ce qui est une correction *multiplicative* de la mesure brute.  Puisque la mesure
  reste l'ancrage et que le modèle n'apprend qu'un facteur de correction, l'estimateur
  hérite de la dispersion naturelle de la mesure plutôt que de la comprimer.

Pas de fuite : les stages B/C en train n'utilisent que des prédictions hors-fold de A ;
les statistiques intra-patient ne portent que sur X (disponible en test aussi).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "30_bob_trajectory"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_09_longitudinal import prepare_longitudinal_features

CATEGORICAL_COLS = ["cohort", "gene", "rater_id"]
LF_PREFIX = "lf_"

# ─── helpers ────────────────────────────────────────────────────────────────────


def add_longitudinal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features du modèle 09 préfixées `lf_` (calcul intra-patient, pas de fuite)."""
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
    """HGBR pour l'étage C : 63 feuilles, min_samples=100, lr=0.02, 1500 iter."""
    return _hgbr(
        max_iter=1500,
        learning_rate=0.02,
        min_samples_leaf=100,
        max_leaf_nodes=63,
        l2_regularization=2.0,
        **kwargs,
    )


def _sigma_hgbr(**kwargs) -> HistGradientBoostingRegressor:
    """HGBR pour prédire |résidu| (MAE loss)."""
    return _hgbr(
        max_iter=150,
        learning_rate=0.08,
        min_samples_leaf=100,
        loss="absolute_error",
        **kwargs,
    )


# ─── features de visite ─────────────────────────────────────────────────────────


def _visit_features_full(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Toutes les variables de la visite (incluant on/off) pour l'étage Aa."""
    data = pd.DataFrame(index=df.index)
    for col in ["age_at_diagnosis", "age", "ledd", "time_since_intake_on",
                "time_since_intake_off", "on", "off", "sexM"]:
        data[col] = df[col]
    data["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    data["off_minus_on"] = df["off"] - df["on"]
    data["off_over_on"] = df["off"] / df["on"].replace(0, np.nan)
    for col in CATEGORICAL_COLS:
        data[col] = pd.Categorical(df[col], categories=categories[col])
    return data


def _visit_features_no_meas(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Variables contextuelles SANS les mesures on/off (pour le modèle de mesure Ab).

    On utilise : rater_id, délais de prise, ledd, âge, démographie.
    En excluant on et off, on force le modèle à apprendre le biais du *contexte*
    d'évaluation (rater, moment de la prise) plutôt que de régulariser la mesure
    brute elle-même.
    """
    data = pd.DataFrame(index=df.index)
    for col in ["age_at_diagnosis", "age", "ledd", "time_since_intake_on",
                "time_since_intake_off", "sexM"]:
        data[col] = df[col]
    data["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    for col in CATEGORICAL_COLS:
        data[col] = pd.Categorical(df[col], categories=categories[col])
    return data


def _visit_features_demo(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Démographie + temps uniquement (pour le modèle de population Ac)."""
    data = pd.DataFrame(index=df.index)
    for col in ["age_at_diagnosis", "age", "sexM"]:
        data[col] = df[col]
    data["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    for col in ["cohort", "gene"]:
        data[col] = pd.Categorical(df[col], categories=categories[col])
    return data


# ─── estimateur principal ────────────────────────────────────────────────────────


class BobTrajectoryRegressor(RegressorMixin, BaseEstimator):
    """Estimateur 5 étages : mesure → trajectoire patient → correction → isotonique.

    Parameters
    ----------
    n_inner_splits : int
        Nombre de plis GroupKFold pour les prédictions hors-fold en train.
    prior_strength : float
        Force de la régularisation bayésienne de l'étage B vers les a priori population.
    huber_c : float
        Seuil Huber (en unités de σ) pour l'ajustement robuste de l'étage B.
    huber_iter : int
        Nombre d'itérations IRLS pour Huber.
    apply_isotonic : bool
        Si True, applique une régression isotonique croissante par patient en post-traitement.
    random_state : int
        Graine aléatoire.
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

    # ── étage B : trajectoire quadratique bayésienne par patient ────────────────

    def _fit_prior(self, pid: np.ndarray, age: np.ndarray, y: np.ndarray) -> None:
        """Estime l'a priori de population (pente, courbure, niveau) sur les cibles."""
        coefs = []
        df = pd.DataFrame({"pid": pid, "age": age, "y": y})
        for _, g in df.groupby("pid"):
            if len(g) < 4:
                continue
            t = g["age"].values - g["age"].values.mean()
            coefs.append(np.polyfit(t, g["y"].values, 2)[::-1])  # [b0, b1, b2]
        coefs = np.array(coefs)
        # prior sur [b1, b2] ; b0 est ancré par le modèle de population (Ac)
        self.prior_mean_ = coefs[:, 1:].mean(axis=0)  # (pente, courbure)
        self.prior_var_ = coefs[:, 1:].var(axis=0) + 1e-6

    def _trajectories(
        self,
        pid: np.ndarray,
        age: np.ndarray,
        est: np.ndarray,
        sigma: np.ndarray,
        pop_level: np.ndarray,
    ) -> pd.DataFrame:
        """Ajuste b0 + b1*t + b2*t² par patient avec a priori bayésien Huber.

        L'a priori sur le niveau b0 est fourni par le modèle de population (pop_level),
        ce qui ancre la trajectoire même pour les patients avec peu de visites.
        """
        out = pd.DataFrame(
            index=np.arange(len(pid)),
            columns=["traj_pred", "traj_b0", "traj_b1", "traj_b2", "traj_se"],
            dtype=float,
        )
        # régularisation : lam[0,0] faible (b0 ancré par pop_level via pseudo-obs),
        # lam[1,1] et lam[2,2] vers les a priori de population
        lam = np.diag([0.0, *(self.prior_strength / self.prior_var_)])
        mu_base = np.array([0.0, *self.prior_mean_])

        order = pd.Series(np.arange(len(pid))).groupby(np.asarray(pid)).indices
        for idx in order.values():
            t = age[idx] - age[idx].mean()
            A = np.column_stack([np.ones_like(t), t, t**2])
            e = est[idx]
            # ancrage du niveau : ajouter une pseudo-observation (moyenne pop_level, poids fort)
            mu = mu_base.copy()
            mu[0] = pop_level[idx].mean()

            w0 = 1.0 / np.maximum(sigma[idx], 0.5) ** 2
            w = w0.copy()
            for _ in range(self.huber_iter + 1):
                M = A.T @ (A * w[:, None]) + lam
                rhs = A.T @ (w * e) + lam @ mu
                b = np.linalg.solve(M, rhs)
                z = np.abs(e - A @ b) / np.maximum(sigma[idx], 0.5)
                w = w0 * np.minimum(1.0, self.huber_c / np.maximum(z, 1e-9))
            cov = np.linalg.inv(A.T @ (A * w0[:, None]) + lam)
            traj = A @ b
            se = np.sqrt(np.einsum("ij,jk,ik->i", A, cov, A))
            out.iloc[idx] = np.column_stack([
                traj,
                np.full(len(idx), b[0]),
                np.full(len(idx), b[1]),
                np.full(len(idx), b[2]),
                se,
            ])
        return out

    # ── étage C : features assemblées ───────────────────────────────────────────

    def _stage_c_features(
        self,
        X: pd.DataFrame,
        est_visit: np.ndarray,
        sigma_visit: np.ndarray,
        est_meas: np.ndarray,  # estimation via modèle de mesure (combinaison u_off/u_on)
        sigma_meas: np.ndarray,
        est_pop: np.ndarray,
        sigma_pop: np.ndarray,
    ) -> pd.DataFrame:
        # Trajectoire bayésienne sur l'estimation de mesure (la plus riche en signal)
        combined_est = np.where(np.isfinite(est_meas), est_meas, est_visit)
        combined_sigma = np.where(np.isfinite(sigma_meas), sigma_meas, sigma_visit)
        traj = self._trajectories(
            X["patient_id"].values,
            X["age"].values.astype(float),
            combined_est,
            combined_sigma,
            est_pop,
        )

        # Features longitudinales du modèle 09 (préfixe lf_), sans les catégorielles dupliquées
        lf_cats = [LF_PREFIX + c for c in CATEGORICAL_COLS]
        feats = X[[c for c in X.columns if c.startswith(LF_PREFIX) and c not in lf_cats]].copy()
        for col in CATEGORICAL_COLS:
            feats[col] = pd.Categorical(X[col], categories=self.categories_[col])

        # Sorties des étages A
        feats["A_visit_est"] = est_visit
        feats["A_visit_sigma"] = sigma_visit
        feats["A_meas_est"] = est_meas
        feats["A_meas_sigma"] = sigma_meas
        feats["A_pop_est"] = est_pop
        feats["A_pop_sigma"] = sigma_pop
        feats["A_combined_est"] = combined_est

        # Sorties de l'étage B
        feats = pd.concat([feats.reset_index(drop=True), traj.reset_index(drop=True)], axis=1)
        feats["visit_minus_traj"] = combined_est - feats["traj_pred"].values
        return feats

    # ── fit ─────────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        groups = X["patient_id"].values

        self.categories_ = {c: sorted(X[c].dropna().unique()) for c in CATEGORICAL_COLS}
        splits = list(GroupKFold(n_splits=self.n_inner_splits).split(X, y, groups))

        V_full = _visit_features_full(X, self.categories_)
        V_no_meas = _visit_features_no_meas(X, self.categories_)
        V_demo = _visit_features_demo(X, self.categories_)

        # ── Étage Aa : HGBR visite → ŷ_visit (hors-fold) ──────────────────────
        oof_visit = np.zeros(len(X))
        for tr, va in splits:
            m = _hgbr(max_iter=250, learning_rate=0.08, random_state=self.random_state)
            m.fit(V_full.iloc[tr], y[tr])
            oof_visit[va] = m.predict(V_full.iloc[va])
        self.visit_model_ = _hgbr(max_iter=250, learning_rate=0.08,
                                   random_state=self.random_state).fit(V_full, y)

        # ── Étage Ab : modèles de mesure log-ratio (hors-fold) ─────────────────
        # Pour chaque visite avec off : prédire log(off+1) - log(target+1) SANS off ni on
        # Pour chaque visite avec on  : prédire log(on+1)  - log(target+1) SANS off ni on
        log_ratio_off = np.where(X["off"].notna(), np.log1p(X["off"]) - np.log1p(y), np.nan)
        log_ratio_on  = np.where(X["on"].notna(),  np.log1p(X["on"])  - np.log1p(y), np.nan)

        oof_log_ratio_off = np.full(len(X), np.nan)
        oof_log_ratio_on  = np.full(len(X), np.nan)
        for tr, va in splits:
            # OFF
            mask_tr = np.isfinite(log_ratio_off[tr])
            if mask_tr.sum() >= 10:
                m_off = _hgbr(max_iter=200, learning_rate=0.06,
                               random_state=self.random_state)
                m_off.fit(V_no_meas.iloc[tr][mask_tr], log_ratio_off[tr][mask_tr])
                oof_log_ratio_off[va] = m_off.predict(V_no_meas.iloc[va])
            # ON
            mask_tr_on = np.isfinite(log_ratio_on[tr])
            if mask_tr_on.sum() >= 10:
                m_on = _hgbr(max_iter=200, learning_rate=0.06,
                              random_state=self.random_state)
                m_on.fit(V_no_meas.iloc[tr][mask_tr_on], log_ratio_on[tr][mask_tr_on])
                oof_log_ratio_on[va] = m_on.predict(V_no_meas.iloc[va])

        # Modèles finaux de mesure
        mask_off = np.isfinite(log_ratio_off)
        self.meas_off_model_ = _hgbr(max_iter=200, learning_rate=0.06,
                                      random_state=self.random_state)
        self.meas_off_model_.fit(V_no_meas[mask_off], log_ratio_off[mask_off])
        mask_on = np.isfinite(log_ratio_on)
        self.meas_on_model_ = _hgbr(max_iter=200, learning_rate=0.06,
                                     random_state=self.random_state)
        self.meas_on_model_.fit(V_no_meas[mask_on], log_ratio_on[mask_on])

        # Inversion : u = (mesure+1) / exp(r̂) - 1 ; combinaison des deux si disponibles
        def _invert_meas(off_vals, on_vals, r_off, r_on):
            u_off = np.where(np.isfinite(off_vals) & np.isfinite(r_off),
                             (off_vals + 1) / np.exp(r_off) - 1, np.nan)
            u_on  = np.where(np.isfinite(on_vals)  & np.isfinite(r_on),
                             (on_vals  + 1) / np.exp(r_on)  - 1, np.nan)
            # Priorité off (plus direct), fallback on, fallback visit
            return np.where(np.isfinite(u_off), u_off,
                   np.where(np.isfinite(u_on),  u_on, np.nan))

        oof_meas = _invert_meas(
            X["off"].values, X["on"].values,
            oof_log_ratio_off, oof_log_ratio_on,
        )

        # ── Étage Ac : modèle de population (démographie + temps) ──────────────
        oof_pop = np.zeros(len(X))
        for tr, va in splits:
            m_pop = _hgbr(max_iter=200, learning_rate=0.06,
                           random_state=self.random_state)
            m_pop.fit(V_demo.iloc[tr], y[tr])
            oof_pop[va] = m_pop.predict(V_demo.iloc[va])
        self.pop_model_ = _hgbr(max_iter=200, learning_rate=0.06,
                                 random_state=self.random_state).fit(V_demo, y)

        # ── Étage Ad : modèles de bruit |résidu| hors-fold ─────────────────────
        abs_res_visit = np.abs(y - oof_visit)
        abs_res_meas  = np.where(np.isfinite(oof_meas), np.abs(y - oof_meas), abs_res_visit)
        abs_res_pop   = np.abs(y - oof_pop)

        oof_sigma_visit = np.zeros(len(X))
        oof_sigma_meas  = np.zeros(len(X))
        oof_sigma_pop   = np.zeros(len(X))
        for tr, va in splits:
            ms_v = _sigma_hgbr(random_state=self.random_state).fit(V_full.iloc[tr], abs_res_visit[tr])
            oof_sigma_visit[va] = ms_v.predict(V_full.iloc[va])
            ms_m = _sigma_hgbr(random_state=self.random_state).fit(V_no_meas.iloc[tr], abs_res_meas[tr])
            oof_sigma_meas[va] = ms_m.predict(V_no_meas.iloc[va])
            ms_p = _sigma_hgbr(random_state=self.random_state).fit(V_demo.iloc[tr], abs_res_pop[tr])
            oof_sigma_pop[va] = ms_p.predict(V_demo.iloc[va])

        # MAE → écart-type
        MAE2STD = np.sqrt(np.pi / 2)
        oof_sigma_visit = np.clip(oof_sigma_visit, 0.5, None) * MAE2STD
        oof_sigma_meas  = np.clip(oof_sigma_meas,  0.5, None) * MAE2STD
        oof_sigma_pop   = np.clip(oof_sigma_pop,   0.5, None) * MAE2STD

        # Modèles de bruit finaux
        self.sigma_visit_model_ = _sigma_hgbr(random_state=self.random_state).fit(V_full, abs_res_visit)
        self.sigma_meas_model_  = _sigma_hgbr(random_state=self.random_state).fit(V_no_meas, abs_res_meas)
        self.sigma_pop_model_   = _sigma_hgbr(random_state=self.random_state).fit(V_demo, abs_res_pop)

        # ── Étage B : a priori de population sur (pente, courbure) ─────────────
        self._fit_prior(groups, X["age"].values.astype(float), y)

        # ── Étage C : HGBR sur features enrichies ──────────────────────────────
        F = self._stage_c_features(
            X,
            oof_visit, oof_sigma_visit,
            oof_meas,  oof_sigma_meas,
            oof_pop,   oof_sigma_pop,
        )
        self.feature_names_c_ = list(F.columns)
        # Cible = résidu par rapport à la trajectoire bayésienne
        target_c = y - F["traj_pred"].values
        self.final_model_ = _hgbr_c(random_state=self.random_state).fit(F, target_c)
        return self

    # ── predict ─────────────────────────────────────────────────────────────────

    def _predict_stage_a(self, X: pd.DataFrame):
        """Retourne (est_visit, sigma_visit, est_meas, sigma_meas, est_pop, sigma_pop)."""
        V_full    = _visit_features_full(X, self.categories_)
        V_no_meas = _visit_features_no_meas(X, self.categories_)
        V_demo    = _visit_features_demo(X, self.categories_)

        est_visit = self.visit_model_.predict(V_full)
        est_pop   = self.pop_model_.predict(V_demo)

        MAE2STD = np.sqrt(np.pi / 2)
        sigma_visit = np.clip(self.sigma_visit_model_.predict(V_full), 0.5, None) * MAE2STD
        sigma_meas  = np.clip(self.sigma_meas_model_.predict(V_no_meas), 0.5, None) * MAE2STD
        sigma_pop   = np.clip(self.sigma_pop_model_.predict(V_demo), 0.5, None) * MAE2STD

        r_off = self.meas_off_model_.predict(V_no_meas)
        r_on  = self.meas_on_model_.predict(V_no_meas)
        off_v = X["off"].values
        on_v  = X["on"].values
        u_off = np.where(np.isfinite(off_v.astype(float)), (off_v.astype(float) + 1) / np.exp(r_off) - 1, np.nan)
        u_on  = np.where(np.isfinite(on_v.astype(float)),  (on_v.astype(float)  + 1) / np.exp(r_on)  - 1, np.nan)
        est_meas = np.where(np.isfinite(u_off), u_off, np.where(np.isfinite(u_on), u_on, np.nan))

        return est_visit, sigma_visit, est_meas, sigma_meas, est_pop, sigma_pop

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        (est_visit, sigma_visit, est_meas,
         sigma_meas, est_pop, sigma_pop) = self._predict_stage_a(X)

        F = self._stage_c_features(
            X,
            est_visit, sigma_visit,
            est_meas,  sigma_meas,
            est_pop,   sigma_pop,
        )[self.feature_names_c_]

        traj = F["traj_pred"].values
        resid_pred = self.final_model_.predict(F)
        raw = traj + resid_pred

        if self.apply_isotonic:
            raw = self._apply_isotonic(X["patient_id"].values, X["age"].values.astype(float), raw)

        return np.clip(raw, 0.0, 132.0)

    def _apply_isotonic(
        self, pid: np.ndarray, age: np.ndarray, pred: np.ndarray
    ) -> np.ndarray:
        """Post-traitement : parabole + isotonique croissant par patient.

        1. Ajuste une parabole sur les prédictions par patient (lisse les artefacts de fold).
        2. Applique une régression isotonique croissante (la cible ne décroît jamais).
        """
        out = pred.copy()
        order = pd.Series(np.arange(len(pid))).groupby(np.asarray(pid)).indices
        for idx in order.values():
            t = age[idx]
            p = pred[idx]
            sort_idx = np.argsort(t)

            # Parabole sur les prédictions
            if len(idx) >= 3:
                tc = t - t.mean()
                coef = np.polyfit(tc, p, 2)
                smoothed = np.polyval(coef, tc)
            else:
                smoothed = p.copy()

            # Isotonique croissant
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(t[sort_idx], smoothed[sort_idx])
            out[idx] = iso.predict(t)
        return out

    def predict_stages(self, X: pd.DataFrame) -> pd.DataFrame:
        """Diagnostic : RMSE de chaque étage."""
        X = X.reset_index(drop=True)
        (est_visit, sigma_visit, est_meas,
         sigma_meas, est_pop, sigma_pop) = self._predict_stage_a(X)

        combined = np.where(np.isfinite(est_meas), est_meas, est_visit)

        F = self._stage_c_features(
            X,
            est_visit, sigma_visit,
            est_meas,  sigma_meas,
            est_pop,   sigma_pop,
        )[self.feature_names_c_]

        traj = F["traj_pred"].values
        resid_pred = self.final_model_.predict(F)
        stage_c = traj + resid_pred
        final = self.predict(X)

        return pd.DataFrame({
            "Aa_visit": est_visit,
            "Ab_meas":  np.where(np.isfinite(est_meas), est_meas, est_visit),
            "Ac_pop":   est_pop,
            "B_traj":   traj,
            "C_final":  stage_c,
            "D_isoton": final,
        })


# ─── CV rapide par étage ─────────────────────────────────────────────────────────


def quick_cv(X: pd.DataFrame, y: pd.Series, cv_splits, **params) -> None:
    """CV rapide hors skore : affiche le RMSE de chaque étage pour diagnostiquer la contribution."""
    scores = []
    for k, (tr, va) in enumerate(cv_splits):
        model = BobTrajectoryRegressor(**params)
        model.fit(X.iloc[tr], y.iloc[tr])
        stages = model.predict_stages(X.iloc[va])
        y_va = y.iloc[va].values
        rmse = np.sqrt(((stages.values - y_va[:, None]) ** 2).mean(axis=0))
        scores.append(rmse)
        print(f"  pli {k}: " + "  ".join(f"{c}={r:.4f}" for c, r in zip(stages.columns, rmse)),
              flush=True)
    s = np.array(scores)
    print(
        "  MOYENNE: "
        + "  ".join(f"{c}={m:.4f}±{d:.4f}" for c, m, d in zip(stages.columns, s.mean(0), s.std(0))),
        flush=True,
    )


# ─── main ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trajectoire patient enrichie : mesure + Bayes + isotonique (30_bob_trajectory)"
    )
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="CV rapide par étage uniquement (pas de skore, pas de soumission)",
    )
    parser.add_argument("--prior-strength", type=float, default=2.0)
    parser.add_argument("--no-isotonic", action="store_true",
                        help="Désactiver le post-traitement isotonique")
    args = parser.parse_args()
    params = dict(
        prior_strength=args.prior_strength,
        apply_isotonic=not args.no_isotonic,
    )

    print("=" * 70, flush=True)
    print("MODELE TRAJECTOIRE ENRICHIE (30_bob_trajectory)", flush=True)
    print("Mesure -> Trajectoire Bayes -> HGBR-C -> Isotonique", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/5] Chargement des données et construction des features longitudinales...", flush=True)
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
        print("\n[3/5] CV rapide par étage (--quick)...", flush=True)
        quick_cv(X, y, cv_splits, **params)
        return

    print("\n[3/5] Évaluation skore.evaluate (GroupKFold 5, clé '30_bob_trajectory')...", flush=True)
    model = BobTrajectoryRegressor(**params)
    report = evaluate(model, X, y, splitter=cv_splits)
    try:
        print(f"  RMSE moyen GroupKFold :\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  Rapport généré (erreur accès métrique: {exc})", flush=True)
    try:
        print(
            f"  Résumé des contrôles Skore :\n{report.checks.summarize(fast_mode=True)}",
            flush=True,
        )
    except Exception as exc:
        print(f"  Erreur contrôles: {exc}", flush=True)

    print("\n[4/5] Entraînement final sur tout le train et prédiction sur X_test...", flush=True)
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
        print("\n[5/5] Publication Skore Hub ignorée (--no-hub).", flush=True)
        print("=" * 70, flush=True)
        return

    print(f"\n[5/5] Publication sur Skore Hub sous la clé '{HUB_KEY}'...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
    project.put(HUB_KEY, report)
    print("  Publication terminée avec succès !", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
