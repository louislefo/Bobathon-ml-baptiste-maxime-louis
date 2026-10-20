"""Blend bayesien 4 composantes + estimateur courbe patient (33_bayes_blend).

Project: Bobathon ESILV (Probabl x IBM)

Architecture :
  Composantes importees du modele 32 :
    C1   : HGBR qui corrige la trajectoire t1 (etage final 1 du 32)
    C2   : HGBR qui corrige la prediction du modele de coefficients (etage final 2 du 32)
    CF   : prediction du modele de coefficients patient du 32

  Estimateur bayesien (BayesCurveEstimator) :
    Modele de bruit    : residus r = log(mesure+1) - beta*log(cible+1) - decalage_clinicien
                         - decalage_delai. Ajustement par medianes alternees (3 iterations).
    Distribution bruit : GaussianMixture 3 composantes par tranche de niveau de cible,
                         separement pour (r_off, r_on), r_off seul, r_on seul.
    A priori parabole  : 1500 voisins (NearestNeighbors sur cles patient), paraboles ajustees
                         sur la cible, evaluees aux ages du nouveau patient.
    Effet patient on   : grille de 7 decalages log pi dans [-2.5, 2.5] * 0.17,
                         marginalisation par logsumexp.
    Prediction         : moyenne ponderee des courbes candidates (poids = softmax
                         log-vraisemblance).

  Blend final :
    0.30 * C1 + 0.30 * C2 + 0.05 * CF + 0.35 * bayesien
    puis lissage par patient (parabole + IsotonicRegression croissante)
    puis clip [0, 132].

Pas de fuite :
  - Composantes 32 recalculees hors-fold (GroupKFold 5 patient).
  - BayesCurveEstimator entraine sur le pli train, predit sur le pli val.
  - En mode --quick, les composantes sont mises en cache dans oof/33_components_oof.csv.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm as _norm_dist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
OOF_DIR = REPO_ROOT / "oof"
HUB_KEY = "33_bayes_blend"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_15_trajectory import LF_PREFIX, add_longitudinal_columns
from src.model_21_smooth import smooth_predictions
from src.model_32_ultimate import (
    UltimateRegressor,
    _patient_rows,
    _visit_full,
    _visit_no_meas,
    _visit_pop,
    _stage_c_features,
    _patient_table,
    _predict_coefs,
    _traj_visit,
    _traj_debiased,
    _traj_conditional,
    _traj_cond4,
    _fit_prior,
    _hgbr,
    _sigma_hgbr,
    _log_ratio,
    CATEGORICAL_COLS,
    COEFS,
    FINAL_C1_PARAMS,
    FINAL_C2_PARAMS,
    COEF_PARAMS,
    MAE2STD,
    LOG_EPS,
)

# ── Poids du blend (codes apres optimisation OOF) ───────────────────────────
W_C1 = 0.30
W_C2 = 0.30
W_CF = 0.05
W_BAY = 0.35

# ── Parametres du modele de bruit ───────────────────────────────────────────
BETA_OFF = 1.18
BETA_ON = 1.15

# Bornes de tranches de delai off (time_since_intake_off, en heures)
DELAY_BINS_OFF = [0, 8, 9, 10, 11, 12, 13, 14, 16, 18, 48]
# Bornes de tranches de delai on (time_since_intake_on, en heures)
DELAY_BINS_ON = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 24]

# Bornes de tranches de niveau de cible
LEVEL_BINS = [-1, 12, 20, 28, 36, 45, 56, 1000]

# Seuil minimum de visites par tranche pour utiliser un GMM specifique
MIN_VISITS_PER_BAND = 80

# Parametres de l'effet patient on
PI_SIGMA = 0.17
PI_GRID = np.linspace(-2.5, 2.5, 7) * PI_SIGMA  # (7,)
LOG_PI_PRIOR = _norm_dist.logpdf(PI_GRID, 0, PI_SIGMA)
LOG_PI_PRIOR -= logsumexp(LOG_PI_PRIOR)

N_COMPONENTS_GMM = 3
RANDOM_STATE_GMM = 0

# Nombre de voisins pour l'a priori parabole
N_NEIGHBORS = 1500
# Nombre minimal de visites pour ajuster une parabole patient dans le train
MIN_VISITS_PRIOR = 4

LOG_EPS_BAYES = 1.0  # meme que LOG_EPS dans le 32


# ── Helpers de binning ────────────────────────────────────────────────────────

def _delay_bin(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Affecte chaque valeur a une tranche ; NaN -> derniere tranche (= manquant)."""
    out = np.full(len(values), len(edges), dtype=int)  # tranche manquante
    valid = np.isfinite(values)
    out[valid] = np.digitize(values[valid], edges) - 1
    out[valid] = np.clip(out[valid], 0, len(edges) - 1)
    return out


def _level_bin(values: np.ndarray) -> np.ndarray:
    """Affecte chaque valeur a une tranche de niveau de cible."""
    return np.clip(np.digitize(values, LEVEL_BINS) - 1, 0, len(LEVEL_BINS) - 2)


# ── Residus de bruit et decalages ─────────────────────────────────────────────

def _log_meas(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.nan_to_num(x, nan=0.0), 0, None) + LOG_EPS_BAYES)


def _raw_residual(meas: np.ndarray, target: np.ndarray, beta: float) -> np.ndarray:
    """r = log(mesure+1) - beta*log(cible+1) ; NaN la ou la mesure est absente."""
    valid = np.isfinite(meas)
    r = np.full(len(meas), np.nan)
    r[valid] = _log_meas(meas[valid]) - beta * _log_meas(target[valid])
    return r


def _alternating_medians(
    r: np.ndarray,
    rater_idx: np.ndarray,
    delay_idx: np.ndarray,
    n_raters: int,
    n_delay_bins: int,
    n_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estime decalages_clinicien et decalages_delai par medianes alternees.

    Seules les lignes ou r est fini sont utilisees.
    Retourne (delta_rater[n_raters+1], delta_delay[n_delay_bins+1]) ; la derniere
    entree de chaque tableau correspond a la tranche manquante (valeur 0.0).
    """
    valid = np.isfinite(r)
    rv = r[valid]
    riv = rater_idx[valid]
    div = delay_idx[valid]

    delta_r = np.zeros(n_raters + 1)
    delta_d = np.zeros(n_delay_bins + 1)

    for _ in range(n_iter):
        # Mise a jour des decalages par clinicien
        for i in range(n_raters + 1):
            m = riv == i
            if m.sum() >= 2:
                delta_r[i] = float(np.median(rv[m] - delta_d[div[m]]))
        # Mise a jour des decalages par tranche de delai
        for j in range(n_delay_bins + 1):
            m = div == j
            if m.sum() >= 2:
                delta_d[j] = float(np.median(rv[m] - delta_r[riv[m]]))

    return delta_r, delta_d


def _compute_residuals(
    df: pd.DataFrame,
    target: np.ndarray,
    rater_cats: list,
    delta_r_off: np.ndarray,
    delta_d_off: np.ndarray,
    delta_r_on: np.ndarray,
    delta_d_on: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule r_off et r_on corriges des decalages sur un ensemble de donnees."""
    rater_idx = np.array([rater_cats.index(r) if r in rater_cats else len(rater_cats)
                          for r in df["rater_id"].values])

    delay_idx_off = _delay_bin(df["time_since_intake_off"].values, DELAY_BINS_OFF)
    delay_idx_on = _delay_bin(df["time_since_intake_on"].values, DELAY_BINS_ON)

    raw_off = _raw_residual(df["off"].values, target, BETA_OFF)
    raw_on = _raw_residual(df["on"].values, target, BETA_ON)

    n_r = len(rater_cats)
    r_off = raw_off - delta_r_off[rater_idx] - delta_d_off[delay_idx_off]
    r_on = raw_on - delta_r_on[rater_idx] - delta_d_on[delay_idx_on]
    # Preserve NaN pour les visites sans mesure
    r_off = np.where(np.isfinite(df["off"].values.astype(float)), r_off, np.nan)
    r_on = np.where(np.isfinite(df["on"].values.astype(float)), r_on, np.nan)
    return r_off, r_on


# ── Modeles de melange gaussien par tranche de niveau ─────────────────────────

def _fit_gmm_band(
    r_off: np.ndarray,
    r_on: np.ndarray,
) -> dict:
    """Ajuste trois GMMs sur une tranche de niveau : 2D (off+on), 1D off, 1D on.

    Retourne un dict avec cles 'both', 'off_only', 'on_only'.
    """
    gmms = {}
    has_both = np.isfinite(r_off) & np.isfinite(r_on)
    has_off = np.isfinite(r_off) & ~np.isfinite(r_on)
    has_on = ~np.isfinite(r_off) & np.isfinite(r_on)

    if has_both.sum() >= N_COMPONENTS_GMM:
        X2 = np.column_stack([r_off[has_both], r_on[has_both]])
        gmms["both"] = GaussianMixture(
            n_components=N_COMPONENTS_GMM, covariance_type="full",
            random_state=RANDOM_STATE_GMM
        ).fit(X2)
    if has_off.sum() >= N_COMPONENTS_GMM:
        gmms["off_only"] = GaussianMixture(
            n_components=N_COMPONENTS_GMM, covariance_type="full",
            random_state=RANDOM_STATE_GMM
        ).fit(r_off[has_off].reshape(-1, 1))
    if has_on.sum() >= N_COMPONENTS_GMM:
        gmms["on_only"] = GaussianMixture(
            n_components=N_COMPONENTS_GMM, covariance_type="full",
            random_state=RANDOM_STATE_GMM
        ).fit(r_on[has_on].reshape(-1, 1))
    return gmms


def _fit_noise_gmms(
    r_off: np.ndarray,
    r_on: np.ndarray,
    level_idx: np.ndarray,
    n_bands: int,
) -> list[dict]:
    """Ajuste les GMMs pour chaque tranche de niveau.

    Si une tranche a < MIN_VISITS_PER_BAND visites d'un type, utilise le GMM
    ajuste sur toutes les tranches.
    """
    # GMM global (fallback)
    fallback = _fit_gmm_band(r_off, r_on)

    gmms = []
    for b in range(n_bands):
        m = level_idx == b
        if m.sum() >= MIN_VISITS_PER_BAND:
            g = _fit_gmm_band(r_off[m], r_on[m])
            # Pour chaque cle, utilise le GMM global si pas assez de visites
            for key in ("both", "off_only", "on_only"):
                if key not in g and key in fallback:
                    g[key] = fallback[key]
            gmms.append(g)
        else:
            gmms.append(fallback)
    return gmms


# ── Evaluation vectorisee de log-densite GMM ─────────────────────────────────
#
# Pour eviter score_samples point par point (overhead Python par appel),
# on extrait les parametres du GMM une seule fois apres le fit et on evalue
# la formule analytique sur des tableaux (K, V) entiers.

_LOG2PI = math.log(2.0 * math.pi)


def _gmm_params(gmm: GaussianMixture) -> tuple:
    """Extrait (log_weights, means, precisions_chol) d'un GaussianMixture fitté."""
    log_w = np.log(gmm.weights_ + 1e-300)          # (C,)
    means = gmm.means_                              # (C, d)
    # precisions_cholesky_ : triangulaire inférieur, shape (C, d, d) pour "full"
    prec_chol = gmm.precisions_cholesky_            # (C, d, d)
    return log_w, means, prec_chol


def _gmm_logpdf_vec(log_w, means, prec_chol, x):
    """Log-densite d'un GMM full-covariance sur x de shape (N, d).

    Formule : log sum_c [ w_c * N(x ; mu_c, Sigma_c) ]
    avec Sigma_c^{-1} = L_c @ L_c^T (L_c = precisions_cholesky_[c]).
    """
    C, d = means.shape
    N = x.shape[0]
    # log-densite de chaque composante : (N, C)
    log_p = np.empty((N, C), dtype=float)
    for c in range(C):
        diff = x - means[c]                       # (N, d)
        # y = L_c^T @ diff^T  => shape (d, N)
        # mahal2 = sum(y**2, axis=0)
        y = prec_chol[c].T @ diff.T               # (d, N)
        mahal2 = np.einsum("ij,ij->j", y, y)      # (N,)
        log_det = np.sum(np.log(np.abs(np.diag(prec_chol[c]))))
        log_p[:, c] = log_w[c] + log_det - 0.5 * (d * _LOG2PI + mahal2)
    return logsumexp(log_p, axis=1)               # (N,)


def _gmm_logpdf_vec_1d(log_w, means, prec_chol, x):
    """Specialisation 1-D (plus rapide, evite la boucle sur C)."""
    # means : (C, 1), prec_chol : (C, 1, 1), x : (N, 1)
    C = len(log_w)
    N = x.shape[0]
    log_p = np.empty((N, C), dtype=float)
    for c in range(C):
        prec = prec_chol[c, 0, 0]                 # scalaire
        diff = x[:, 0] - means[c, 0]              # (N,)
        log_p[:, c] = (log_w[c]
                       + math.log(abs(prec))
                       - 0.5 * (_LOG2PI + (prec * diff) ** 2))
    return logsumexp(log_p, axis=1)               # (N,)


def _precompute_gmm_params(noise_gmms: list[dict]) -> list[dict]:
    """Extrait et met en cache les parametres de tous les GMMs."""
    out = []
    for band in noise_gmms:
        d = {}
        for key in ("both", "off_only", "on_only"):
            if key in band:
                lw, mu, pc = _gmm_params(band[key])
                d[key] = (lw, mu, pc)
        out.append(d)
    return out


# ── Estimateur bayesien de courbe patient ────────────────────────────────────

class BayesCurveEstimator(RegressorMixin, BaseEstimator):
    """Estimateur bayesien de trajectoire patient par candidats + GMM de bruit.

    Parameters
    ----------
    n_neighbors : nombre de voisins pour l'a priori.
    random_state : graine.
    with_pi_effect : si False, desactive l'effet patient sur on (controle : ~4.14).
    """

    def __init__(self, n_neighbors: int = N_NEIGHBORS, random_state: int = 0,
                 with_pi_effect: bool = True):
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.with_pi_effect = with_pi_effect

    # ── Construction du modele de bruit ──────────────────────────────────────

    def _fit_noise_model(self, df: pd.DataFrame, y: np.ndarray) -> None:
        """Ajuste les decalages (medianes alternees) et les GMMs de bruit."""
        rater_cats = sorted(df["rater_id"].dropna().unique().tolist())
        self.rater_cats_ = rater_cats
        n_r = len(rater_cats)
        rater_idx = np.array([rater_cats.index(r) if r in rater_cats else n_r
                               for r in df["rater_id"].values])
        delay_idx_off = _delay_bin(df["time_since_intake_off"].values, DELAY_BINS_OFF)
        delay_idx_on = _delay_bin(df["time_since_intake_on"].values, DELAY_BINS_ON)

        raw_off = _raw_residual(df["off"].values, y, BETA_OFF)
        raw_on = _raw_residual(df["on"].values, y, BETA_ON)

        n_delay_off = len(DELAY_BINS_OFF)
        n_delay_on = len(DELAY_BINS_ON)

        self.delta_r_off_, self.delta_d_off_ = _alternating_medians(
            raw_off, rater_idx, delay_idx_off, n_r, n_delay_off)
        self.delta_r_on_, self.delta_d_on_ = _alternating_medians(
            raw_on, rater_idx, delay_idx_on, n_r, n_delay_on)

        r_off, r_on = _compute_residuals(
            df, y, rater_cats,
            self.delta_r_off_, self.delta_d_off_,
            self.delta_r_on_, self.delta_d_on_,
        )
        level_idx = _level_bin(y)
        n_bands = len(LEVEL_BINS) - 1
        self.n_bands_ = n_bands
        self.noise_gmms_ = _fit_noise_gmms(r_off, r_on, level_idx, n_bands)

    # ── Construction de l'a priori parabole ──────────────────────────────────

    def _fit_prior_model(self, df: pd.DataFrame, y: np.ndarray) -> None:
        """Ajuste les paraboles candidats et le modele NearestNeighbors."""
        pid = df["patient_id"].values
        age = df["age"].values.astype(float)
        pat_rows = _patient_rows(pid)

        # -- Cles patient pour le NearestNeighbors --
        # (temps moyen depuis diagnostic, age au diagnostic, ecart premiere-derniere visite)
        keys_list = []
        coefs_list = []
        pids_with_coefs = []

        for p, rows in pat_rows.items():
            a = age[rows]
            span = a.max() - a.min() if len(a) > 1 else 0.0
            # Temps depuis diagnostic : age - age_at_diagnosis
            if "age_at_diagnosis" in df.columns:
                diag = df["age_at_diagnosis"].values[rows]
                tsd = float(np.nanmedian(a - diag))
                age_diag = float(np.nanmedian(diag))
            else:
                tsd = 0.0
                age_diag = float(np.nanmedian(a))
            keys_list.append([tsd, age_diag, span])

            # Ajustement parabole si assez de visites
            if len(rows) >= MIN_VISITS_PRIOR:
                t = a - a.mean()
                c = np.polyfit(t, y[rows], 2)[::-1]
                coefs_list.append((p, a, c))
                pids_with_coefs.append(p)

        # Standardisation des cles
        all_pids_ordered = list(pat_rows.keys())
        keys_arr = np.array(keys_list, dtype=float)

        # Remplacement des NaN par les medianes
        col_medians = np.nanmedian(keys_arr, axis=0)
        for j in range(keys_arr.shape[1]):
            nans = ~np.isfinite(keys_arr[:, j])
            keys_arr[nans, j] = col_medians[j]

        self.key_medians_ = col_medians
        self.key_stds_ = np.maximum(np.nanstd(keys_arr, axis=0), 1e-6)
        keys_norm = (keys_arr - self.key_medians_) / self.key_stds_

        # NearestNeighbors sur les patients qui ont des paraboles
        pid_to_idx = {p: i for i, p in enumerate(all_pids_ordered)}
        coef_patient_keys = np.array(
            [keys_norm[pid_to_idx[p]] for p, _, _ in coefs_list], dtype=float
        )

        self.nn_ = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(coefs_list)),
                                    algorithm="auto")
        self.nn_.fit(coef_patient_keys)

        # Stockage des paraboles : pour chaque patient avec coefficients,
        # on garde ses ages et ses coefficients
        self.coef_ages_ = [(a, c) for _, a, c in coefs_list]  # list of (ages, (b0,b1,b2))
        self.coef_patient_keys_ = coef_patient_keys
        self.all_pids_ordered_ = all_pids_ordered
        self.pid_to_key_idx_ = pid_to_idx

        # Stockage de tous les ages par patient (pour evaluer les candidats)
        self.pat_ages_ = {p: age[rows] for p, rows in pat_rows.items()}

    def _patient_key(self, df_patient: pd.DataFrame) -> np.ndarray:
        """Calcule la cle standardisee d'un nouveau patient."""
        a = df_patient["age"].values.astype(float)
        span = a.max() - a.min() if len(a) > 1 else 0.0
        if "age_at_diagnosis" in df_patient.columns:
            diag = df_patient["age_at_diagnosis"].values
            tsd = float(np.nanmedian(a - diag))
            age_diag = float(np.nanmedian(diag))
        else:
            tsd = 0.0
            age_diag = float(np.nanmedian(a))
        key = np.array([tsd, age_diag, span], dtype=float)
        for j in range(3):
            if not np.isfinite(key[j]):
                key[j] = self.key_medians_[j]
        return (key - self.key_medians_) / self.key_stds_

    def _candidate_curves(self, key: np.ndarray, ages_patient: np.ndarray) -> np.ndarray:
        """Retourne les courbes candidates (N_neighbors, len(ages_patient)) tronquees a 0."""
        _, inds = self.nn_.kneighbors(key.reshape(1, -1))
        inds = inds[0]
        curves = np.zeros((len(inds), len(ages_patient)), dtype=float)
        t_pat = ages_patient - ages_patient.mean()
        for k, idx in enumerate(inds):
            _, (b0, b1, b2) = self.coef_ages_[idx]
            curves[k] = np.maximum(b0 + b1 * t_pat + b2 * t_pat**2, 0.0)
        return curves

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y) -> "BayesCurveEstimator":
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        self._fit_noise_model(X, y)
        self._fit_prior_model(X, y)
        return self

    # ── predict (vectorise par patient) ─────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: C901
        X = X.reset_index(drop=True)
        pid = X["patient_id"].values
        age = X["age"].values.astype(float)

        rater_cats = self.rater_cats_
        n_r = len(rater_cats)

        rater_idx_all = np.array([rater_cats.index(r) if r in rater_cats else n_r
                                   for r in X["rater_id"].values])
        delay_idx_off_all = _delay_bin(X["time_since_intake_off"].values, DELAY_BINS_OFF)
        delay_idx_on_all = _delay_bin(X["time_since_intake_on"].values, DELAY_BINS_ON)

        # log(mesure+1) par visite — NaN conserve pour les mesures absentes
        off_raw = X["off"].values.astype(float)
        on_raw = X["on"].values.astype(float)
        has_off_all = np.isfinite(off_raw)
        has_on_all = np.isfinite(on_raw)
        log_off_all = np.where(has_off_all, _log_meas(off_raw), np.nan)
        log_on_all = np.where(has_on_all, _log_meas(on_raw), np.nan)

        # Decalages fixes par visite (vecteurs de longueur N)
        dr_off = self.delta_r_off_[rater_idx_all]
        dd_off = self.delta_d_off_[delay_idx_off_all]
        dr_on = self.delta_r_on_[rater_idx_all]
        dd_on = self.delta_d_on_[delay_idx_on_all]

        # Parametres GMM pre-extraits une seule fois
        gmm_params = _precompute_gmm_params(self.noise_gmms_)

        pat_rows_dict = _patient_rows(pid)
        pred_out = np.full(len(X), np.nan)
        self.posterior_std_ = np.full(len(X), np.nan)

        pi_grid = PI_GRID          # (n_pi,)
        log_pi_prior = LOG_PI_PRIOR  # (n_pi,)
        n_pi = len(pi_grid)
        with_pi = self.with_pi_effect

        for p, rows in pat_rows_dict.items():
            rows = np.asarray(rows)
            ages_p = age[rows]
            sort_order = np.argsort(ages_p)
            rows = rows[sort_order]
            ages_p = ages_p[sort_order]

            # Courbes candidates : (K, V)
            key_p = self._patient_key(X.iloc[rows])
            candidates = self._candidate_curves(key_p, ages_p)
            K, V = candidates.shape

            # log(cand+1) : (K, V)
            log_cand = _log_meas(candidates)

            # Residus par candidat et par visite : (K, V), NaN si mesure absente
            loff_p = log_off_all[rows]          # (V,)
            lon_p = log_on_all[rows]            # (V,)
            hoff_p = has_off_all[rows]          # (V,) bool
            hon_p = has_on_all[rows]            # (V,) bool

            # r_off[k, v] = log_off[v] - beta_off * log_cand[k,v] - dr_off[v] - dd_off[v]
            r_off_kv = np.full((K, V), np.nan)
            r_on_kv = np.full((K, V), np.nan)
            if hoff_p.any():
                vo = hoff_p  # bool mask over V
                r_off_kv[:, vo] = (loff_p[vo]
                                   - BETA_OFF * log_cand[:, vo]
                                   - dr_off[rows][vo]
                                   - dd_off[rows][vo])
            if hon_p.any():
                vn = hon_p
                r_on_kv[:, vn] = (lon_p[vn]
                                  - BETA_ON * log_cand[:, vn]
                                  - dr_on[rows][vn]
                                  - dd_on[rows][vn])

            # Tranche de niveau par (k, v) : (K, V)
            band_kv = _level_bin(candidates)   # (K, V)

            # log-vraisemblance accumulee : (K, n_pi)
            log_lik = np.zeros((K, n_pi), dtype=float)

            # Visites avec au moins une mesure
            vis_mask = hoff_p | hon_p
            if not vis_mask.any():
                # Aucune mesure -> poids uniformes
                pred_out[rows] = candidates.mean(axis=0)
                self.posterior_std_[rows] = candidates.std(axis=0)
                continue

            # Groupes de visites par tranche de niveau et type de mesure,
            # pour appeler _gmm_logpdf_vec par bloc homogene plutot que point par point.
            #
            # Pour l'effet pi : on boucle sur les n_pi valeurs (7 iterations seulement)
            # et pour chaque pi on calcule d'un coup la log-densite sur tous les (k,v)
            # qui ont la mesure on (ou les deux mesures).
            #
            # Strategy :
            #   - visites off-only   : log-densite 1D sur r_off_kv -> contribution independante de pi
            #   - visites on-only    : log-densite 1D sur r_on_kv - pi_j pour chaque j
            #   - visites both       : log-densite 2D sur [r_off_kv, r_on_kv - pi_j]

            both_mask = hoff_p & hon_p         # (V,) bool
            off_only_mask = hoff_p & ~hon_p
            on_only_mask = hon_p & ~hoff_p

            n_bands_total = len(self.noise_gmms_)

            # ── Contribution off-only (independante de pi) ────────────────
            if off_only_mask.any():
                v_idx = np.where(off_only_mask)[0]  # indices dans V
                for b in range(n_bands_total):
                    gp = gmm_params[b]
                    if "off_only" not in gp:
                        continue
                    lw, mu, pc = gp["off_only"]
                    # Candidats de la tranche b dans les visites off-only : mask (K, len(v_idx))
                    band_match = (band_kv[:, v_idx] == b)  # (K, n_v)
                    if not band_match.any():
                        continue
                    k_idx, vi_idx = np.where(band_match)
                    x_eval = r_off_kv[k_idx, v_idx[vi_idx]].reshape(-1, 1)
                    ll = _gmm_logpdf_vec_1d(lw, mu, pc, x_eval)  # (n_pts,)
                    # Accumule sur toutes les colonnes pi (contribution independante de pi)
                    np.add.at(log_lik, (k_idx, slice(None)), ll[:, None])

            # ── Contributions dependantes de pi ───────────────────────────
            for j, pi_j in enumerate(pi_grid):
                # on-only
                if on_only_mask.any():
                    v_idx = np.where(on_only_mask)[0]
                    for b in range(n_bands_total):
                        gp = gmm_params[b]
                        if "on_only" not in gp:
                            continue
                        lw, mu, pc = gp["on_only"]
                        band_match = (band_kv[:, v_idx] == b)
                        if not band_match.any():
                            continue
                        k_idx, vi_idx = np.where(band_match)
                        x_eval = (r_on_kv[k_idx, v_idx[vi_idx]] - pi_j).reshape(-1, 1)
                        ll = _gmm_logpdf_vec_1d(lw, mu, pc, x_eval)
                        np.add.at(log_lik[:, j], k_idx, ll)

                # both
                if both_mask.any():
                    v_idx = np.where(both_mask)[0]
                    for b in range(n_bands_total):
                        gp = gmm_params[b]
                        key2d = "both" if "both" in gp else None
                        if key2d is None:
                            # fallback separable
                            for ch, r_kv, pi_shift in (
                                ("off_only", r_off_kv, 0.0),
                                ("on_only",  r_on_kv,  pi_j),
                            ):
                                if ch not in gp:
                                    continue
                                lw, mu, pc = gp[ch]
                                band_match = (band_kv[:, v_idx] == b)
                                if not band_match.any():
                                    continue
                                k_idx, vi_idx = np.where(band_match)
                                x_eval = (r_kv[k_idx, v_idx[vi_idx]] - pi_shift).reshape(-1, 1)
                                ll = _gmm_logpdf_vec_1d(lw, mu, pc, x_eval)
                                np.add.at(log_lik[:, j], k_idx, ll)
                        else:
                            lw, mu, pc = gp[key2d]
                            band_match = (band_kv[:, v_idx] == b)
                            if not band_match.any():
                                continue
                            k_idx, vi_idx = np.where(band_match)
                            r_off_pts = r_off_kv[k_idx, v_idx[vi_idx]]
                            r_on_pts = r_on_kv[k_idx, v_idx[vi_idx]] - pi_j
                            x_eval = np.column_stack([r_off_pts, r_on_pts])
                            ll = _gmm_logpdf_vec(lw, mu, pc, x_eval)
                            np.add.at(log_lik[:, j], k_idx, ll)

                if not with_pi:
                    # Sans effet pi, toutes les colonnes sont identiques apres la premiere
                    if j == 0:
                        log_lik[:, 1:] = log_lik[:, 0:1]
                    break

            # Marginalisation sur pi
            if with_pi:
                log_lik_marg = logsumexp(log_lik + log_pi_prior[None, :], axis=1)
            else:
                log_lik_marg = log_lik[:, 0]

            # Poids a posteriori
            log_weights = log_lik_marg - logsumexp(log_lik_marg)
            weights = np.exp(log_weights)  # (K,)

            pred_p = weights @ candidates       # (V,)
            var_p = weights @ (candidates ** 2) - pred_p ** 2
            self.posterior_std_[rows] = np.sqrt(np.maximum(var_p, 0.0))
            pred_out[rows] = pred_p

        pred_out = np.nan_to_num(pred_out, nan=0.0)
        return pred_out


# ── Estimateur principal 33 ───────────────────────────────────────────────────

class BayesBlendRegressor(RegressorMixin, BaseEstimator):
    """Blend 33 : C1 + C2 + CF + BayesCurveEstimator + lissage patient.

    Parameters
    ----------
    w_c1, w_c2, w_cf, w_bay : poids du blend.
    n_inner_splits : GroupKFold internes pour les predictions hors-fold.
    smooth : appliquer lissage quadratique + isotonique.
    random_state : graine.
    with_pi_effect : transmet a BayesCurveEstimator.
    """

    def __init__(
        self,
        w_c1: float = W_C1,
        w_c2: float = W_C2,
        w_cf: float = W_CF,
        w_bay: float = W_BAY,
        n_inner_splits: int = 5,
        smooth: bool = True,
        random_state: int = 0,
        with_pi_effect: bool = True,
    ):
        self.w_c1 = w_c1
        self.w_c2 = w_c2
        self.w_cf = w_cf
        self.w_bay = w_bay
        self.n_inner_splits = n_inner_splits
        self.smooth = smooth
        self.random_state = random_state
        self.with_pi_effect = with_pi_effect

    # ── Prediction des 3 composantes issues du 32 (hors-fold ou finale) ─────

    def _fit_32_fold(self, X_tr, y_tr, seed):
        """Entraine un UltimateRegressor (sans modele 11) sur un pli."""
        m = UltimateRegressor(w_c1=1.0, w_c2=0.0, w_cf=0.0, w_11=0.0,
                              n_inner_splits=self.n_inner_splits,
                              smooth=False, random_state=seed)
        m.fit(X_tr, y_tr)
        return m

    def _predict_32_components(self, model_32, X_va):
        """Retourne c1, c2, cf pour un ensemble de validation."""
        c = model_32.predict_components(X_va)
        return c["c1"], c["c2"], c["cf"]

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y):  # noqa: C901
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        groups = X["patient_id"].values
        seed = self.random_state

        splits = list(GroupKFold(n_splits=self.n_inner_splits).split(X, y, groups))

        # ── Composantes 32 et bayesien hors-fold (pour C2 et CF corrects) ──
        # Ces composantes ne sont pas utilisees dans le fit, mais gardees pour
        # permettre quick_mode sans reentrainement.
        # Le modele 32 final et le bayesien final sont entraines sur tout le train.

        # ── Modele 32 final ──────────────────────────────────────────────────
        self.model_32_ = UltimateRegressor(
            w_c1=1.0, w_c2=0.0, w_cf=0.0, w_11=0.0,
            n_inner_splits=self.n_inner_splits,
            smooth=False, random_state=seed,
        )
        self.model_32_.fit(X, y)

        # ── BayesCurveEstimator final ────────────────────────────────────────
        self.bayes_ = BayesCurveEstimator(
            n_neighbors=N_NEIGHBORS, random_state=seed,
            with_pi_effect=self.with_pi_effect,
        )
        self.bayes_.fit(X, y)
        return self

    # ── predict ──────────────────────────────────────────────────────────────

    def predict_components(self, X: pd.DataFrame) -> dict:
        X = X.reset_index(drop=True)
        c = self.model_32_.predict_components(X)
        c1 = np.clip(c["c1"], 0.0, 132.0)
        c2 = np.clip(c["c2"], 0.0, 132.0)
        cf = np.clip(c["cf"], 0.0, 132.0)
        bay = np.clip(self.bayes_.predict(X), 0.0, 132.0)
        return {"c1": c1, "c2": c2, "cf": cf, "bay": bay}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        c = self.predict_components(X)
        pred = (self.w_c1 * c["c1"] + self.w_c2 * c["c2"]
                + self.w_cf * c["cf"] + self.w_bay * c["bay"])
        if self.smooth:
            pred = smooth_predictions(X["patient_id"].values, X["age"].values, pred)
        return np.clip(pred, 0.0, 132.0)


# ── Calcul OOF des composantes par pli ────────────────────────────────────────

def _fold_ids(cv_splits, n):
    folds = np.zeros(n, dtype=int)
    for k, (_, va) in enumerate(cv_splits):
        folds[va] = k
    return folds


def _compute_oof_bay_only(X, y, cv_splits, seed=0):
    """Calcule uniquement les composantes bay hors-fold (les 32 viennent du cache)."""
    n = len(X)
    bay = np.full(n, np.nan)
    for k, (tr, va) in enumerate(cv_splits):
        X_tr = X.iloc[tr].reset_index(drop=True)
        y_tr = y.iloc[tr].values
        X_va = X.iloc[va].reset_index(drop=True)
        bayes = BayesCurveEstimator(n_neighbors=N_NEIGHBORS, random_state=seed)
        bayes.fit(X_tr, y_tr)
        bay[va] = np.clip(bayes.predict(X_va), 0.0, 132.0)
        rmse_bay = np.sqrt(np.mean((bay[va] - y.iloc[va].values) ** 2))
        print(f"  pli {k}  bay={rmse_bay:.4f}", flush=True)
    return bay


def _compute_oof_components(X, y, cv_splits, seed=0):
    """Calcule c1, c2, cf, bay hors-fold sans aucun cache disponible."""
    n = len(X)
    comp = {k: np.full(n, np.nan) for k in ("c1", "c2", "cf", "bay")}

    for k, (tr, va) in enumerate(cv_splits):
        X_tr = X.iloc[tr].reset_index(drop=True)
        y_tr = y.iloc[tr].values
        X_va = X.iloc[va].reset_index(drop=True)

        m32 = UltimateRegressor(w_c1=1.0, w_c2=0.0, w_cf=0.0, w_11=0.0,
                                n_inner_splits=5, smooth=False, random_state=seed)
        m32.fit(X_tr, y_tr)
        c_32 = m32.predict_components(X_va)
        comp["c1"][va] = np.clip(c_32["c1"], 0.0, 132.0)
        comp["c2"][va] = np.clip(c_32["c2"], 0.0, 132.0)
        comp["cf"][va] = np.clip(c_32["cf"], 0.0, 132.0)

        bayes = BayesCurveEstimator(n_neighbors=N_NEIGHBORS, random_state=seed)
        bayes.fit(X_tr, y_tr)
        comp["bay"][va] = np.clip(bayes.predict(X_va), 0.0, 132.0)

        rmse_c1 = np.sqrt(np.mean((comp["c1"][va] - y.iloc[va].values) ** 2))
        rmse_bay = np.sqrt(np.mean((comp["bay"][va] - y.iloc[va].values) ** 2))
        print(f"  pli {k}  C1={rmse_c1:.4f}  bay={rmse_bay:.4f}", flush=True)

    return comp


# ── Mode --quick ──────────────────────────────────────────────────────────────

def quick_mode(X: pd.DataFrame, y: pd.Series, cv_splits) -> None:
    """Calcule (ou recharge) les 4 composantes OOF, optimise les poids."""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    path33 = OOF_DIR / "33_components_oof.csv"
    path32 = OOF_DIR / "32_components_oof.csv"
    names = ("c1", "c2", "cf", "bay")
    folds = _fold_ids(cv_splits, len(X))
    X_index = X["Index"].values if "Index" in X.columns else np.arange(len(X))

    if path33.is_file():
        print(f"  Cache OOF 33 trouve : {path33}", flush=True)
        df_oof = pd.read_csv(path33)
        comp = {n: df_oof[n].values for n in names}
    else:
        # Tente de recuperer c1/c2/cf depuis le cache du modele 32 ──────────
        if path32.is_file():
            print(f"  Cache OOF 32 trouve : {path32} (c1/c2/cf reutilises)", flush=True)
            df32 = pd.read_csv(path32)
            # Alignement par Index et verification des plis
            # On s'assure que l'ordre des lignes correspond via merge sur Index+fold
            df_ref = pd.DataFrame({"Index": X_index, "fold": folds})
            merged = df_ref.merge(df32[["Index", "fold", "c1", "c2", "cf"]],
                                  on=["Index", "fold"], how="left")
            missing = merged["c1"].isna().sum()
            if missing > 0:
                print(f"  Attention : {missing} lignes sans correspondance dans 32 OOF,"
                      " recalcul complet.", flush=True)
                comp = _compute_oof_components(X, y, cv_splits)
            else:
                comp = {n: merged[n].values for n in ("c1", "c2", "cf")}
                print("  Calcul des composantes bay hors-fold (bayesien uniquement)...",
                      flush=True)
                comp["bay"] = _compute_oof_bay_only(X, y, cv_splits)
        else:
            print("  Calcul de toutes les composantes OOF (long — aucun cache 32)...",
                  flush=True)
            comp = _compute_oof_components(X, y, cv_splits)

        df_out = pd.DataFrame({"Index": X_index, "fold": folds})
        for n in names:
            df_out[n] = comp[n]
        df_out.to_csv(path33, index=False)
        print(f"  OOF sauvegardees : {path33}", flush=True)

    yv = y.values
    pid = X["patient_id"].values
    age = X["age"].values

    # ── RMSE par composante seule ──────────────────────────────────────────
    print("\n  RMSE par composante seule (lisse) :", flush=True)
    for n in names:
        v = comp[n]
        v_sm = smooth_predictions(pid, age, v)
        rmse_f = [np.sqrt(np.mean((v[folds == k] - yv[folds == k]) ** 2))
                  for k in range(len(cv_splits))]
        rmse_sm = [np.sqrt(np.mean((v_sm[folds == k] - yv[folds == k]) ** 2))
                   for k in range(len(cv_splits))]
        print(f"    {n:<6} brut={np.mean(rmse_f):.4f}+-{np.std(rmse_f):.4f}"
              f"  lisse={np.mean(rmse_sm):.4f}+-{np.std(rmse_sm):.4f}", flush=True)

    # ── Grille de poids : pre-filtrage sans lissage, lissage sur top-20 ────
    print("\n  Optimisation des poids du blend (pas=0.05) :", flush=True)
    c1v = comp["c1"]
    c2v = comp["c2"]
    cfv = comp["cf"]
    bayv = comp["bay"]

    step = 0.05
    grid_pts = np.arange(0, 1.0 + step / 2, step)

    # Etape 1 : RMSE brut (sans lissage) sur toutes les combinaisons
    raw_results = []
    for w1 in grid_pts:
        for w2 in grid_pts:
            for w3 in grid_pts:
                w4 = round(1.0 - w1 - w2 - w3, 10)
                if w4 < -1e-9:
                    continue
                w4 = max(w4, 0.0)
                pred_raw = w1 * c1v + w2 * c2v + w3 * cfv + w4 * bayv
                rmse_f = [np.sqrt(np.mean((pred_raw[folds == k] - yv[folds == k]) ** 2))
                          for k in range(len(cv_splits))]
                raw_results.append((float(np.mean(rmse_f)), w1, w2, w3, w4))

    raw_results.sort()
    top20 = raw_results[:20]

    # Etape 2 : RMSE lisse uniquement sur les top-20
    smooth_results = []
    best_rmse, best_w = 999.0, None
    for _, w1, w2, w3, w4 in top20:
        pred = smooth_predictions(pid, age, w1 * c1v + w2 * c2v + w3 * cfv + w4 * bayv)
        rmse_f = [np.sqrt(np.mean((pred[folds == k] - yv[folds == k]) ** 2))
                  for k in range(len(cv_splits))]
        m = float(np.mean(rmse_f))
        smooth_results.append((m, w1, w2, w3, w4))
        if m < best_rmse:
            best_rmse, best_w = m, (w1, w2, w3, w4)

    smooth_results.sort()
    print("  Top 5 (c1, c2, cf, bay) [lisse] :", flush=True)
    for rm, w1, w2, w3, w4 in smooth_results[:5]:
        print(f"    c1={w1:.2f} c2={w2:.2f} cf={w3:.2f} bay={w4:.2f}  RMSE={rm:.4f}",
              flush=True)
    print(f"  Meilleure : c1={best_w[0]:.2f} c2={best_w[1]:.2f} "
          f"cf={best_w[2]:.2f} bay={best_w[3]:.2f}  RMSE={best_rmse:.4f}", flush=True)

    # ── Tableau des RMSE et explication ───────────────────────────────────
    print("\n  Tableau des RMSE de reference :", flush=True)
    refs = [
        ("modele 25 / 32",  "~3.44", "meilleur modele precedent"),
        ("bayesien seul",   "~3.84", "estimateur bayesien (BayesCurveEstimator)"),
        ("blend 33 final",  f"{best_rmse:.4f}", "blend 4 composantes + lissage"),
    ]
    print(f"    {'Modele':<18} {'RMSE':>8}  Commentaire", flush=True)
    for name, val, comment in refs:
        print(f"    {name:<18} {val:>8}  {comment}", flush=True)

    print("""
  Pourquoi le bayesien apporte autant au blend alors qu'il est moins bon seul ?
  Le BayesCurveEstimator fait des erreurs tres differentes de C1/C2 :
    - Il exploite la FORME globale de la courbe par similarite de patient,
      la ou C1/C2 corrigent des biais locaux visite par visite.
    - Son bruit residuel depend du niveau de la cible (GMM par tranche),
      ce qui le rend peu correle avec les residus du HGBR.
    - La diversite des sources d'erreur (paraboles vs gradients) reduit
      la variance du blend bien au-dela de ce que chaque composante offre seule.
    => Un modele mediocre mais decorrelé est souvent plus utile en blend
       qu'un bon modele dont les erreurs sont deja corrigees par les autres.
""", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blend bayesien 4 composantes + lissage patient (33_bayes_blend)"
    )
    parser.add_argument("--no-hub", action="store_true",
                        help="Ne pas publier sur Skore Hub")
    parser.add_argument("--quick", action="store_true",
                        help="OOF + grille poids uniquement (pas de skore)")
    parser.add_argument("--w-c1",  type=float, default=W_C1)
    parser.add_argument("--w-c2",  type=float, default=W_C2)
    parser.add_argument("--w-cf",  type=float, default=W_CF)
    parser.add_argument("--w-bay", type=float, default=W_BAY)
    args = parser.parse_args()
    params = dict(w_c1=args.w_c1, w_c2=args.w_c2, w_cf=args.w_cf, w_bay=args.w_bay)

    print("=" * 70, flush=True)
    print("MODELE 33 BAYES BLEND", flush=True)
    print("4 composantes : C1 / C2 / CF / Bayesien + lissage patient", flush=True)
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

    print("\n[3/5] Evaluation skore.evaluate (GroupKFold 5, cle '33_bayes_blend')...",
          flush=True)
    model = BayesBlendRegressor(**params)
    report = evaluate(model, X, y, splitter=cv_splits)
    try:
        print(f"  RMSE moyen GroupKFold :\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  Rapport genere (erreur acces metrique: {exc})", flush=True)
    try:
        print(f"  Resume controles Skore :\n{report.checks.summarize(fast_mode=True)}",
              flush=True)
    except Exception as exc:
        print(f"  Erreur controles : {exc}", flush=True)

    print("\n[4/5] Entrainement final et prediction sur X_test...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"{HUB_KEY}.csv"
    pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions}).to_csv(
        sub_path, index=False)
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
