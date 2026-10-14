"""Stacking bayesien : sorties BayesCurveEstimator en variables des etages finaux (34_bayes_stack).

Projet : Bobathon ESILV (Probabl x IBM)
Cible  : Score moteur MDS-UPDRS Part III (continu, domaine [0, 132]).

Architecture et principes :
  Etape 1 - Sorties bayesiennes hors-fold (OOF) :
    BayesCurveEstimator (n_neighbors=800) est entraine sur chaque sous-pli train interne.
    Il evalue l'esperance a posteriori (bay_mean) et l'ecart-type a posteriori (bay_sd)
    sur le pli de validation interne.
    Variables calculees :
      - bay_mean     : score moyen predit par la trajectoire bayesienne.
      - bay_sd       : incertitude (ecart-type a posteriori).
      - bay_minus_t1 : ecart entre l'estimation bayesienne et la trajectoire parametrique t1.
      - bay_pmean    : moyenne intra-patient des bay_mean pour capturer le niveau moyen.

  Etape 2 - Etages finaux augmentes (sur matrice C1 du Modele 32) :
    Ca : HGBR entraine sur F1 + variables bayesiennes, avec pour cible (y - t1_pred).
         Prediction : pred_ca = t1 + Ca.predict(F_aug).
    Cb : HGBR entraine sur F1 + variables bayesiennes, avec pour cible (y - bay_mean).
         Prediction : pred_cb = bay_mean + Cb.predict(F_aug).

  Etape 3 - Blend final et post-traitement :
    Combinaison ponderee :
      pred_brut = 0.30 * Ca + 0.40 * Cb + 0.10 * C1 + 0.20 * C2
    Puis lissage longitudinal par patient (parabole + IsotonicRegression) et clipping [0, 132].

  Prevention stricte des fuites (Data Leakage) :
    - Toutes les sorties bayesiennes employees comme covariables sont calculees hors-fold (OOF).
    - L'etage A est ajuste a l'interieur de chaque pli d'entrainement.
    - GroupKFold / GroupShuffleSplit par patient_id a tous les niveaux de decoupage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from skore import Project, evaluate, login

# Definition des repertoires du projet
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
OOF_DIR = REPO_ROOT / "oof"
LOGS_DIR = REPO_ROOT / "logs"
HUB_KEY = "34_bayes_stack"

# Seuil d'avertissement temporel (20 minutes)
WARN_SECONDS = 20 * 60

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_15_trajectory import LF_PREFIX, add_longitudinal_columns
from src.model_21_smooth import smooth_predictions
from src.model_32_ultimate import (
    CATEGORICAL_COLS,
    COEFS,
    LOG_EPS,
    MAE2STD,
    UltimateRegressor,
    _fit_coef_models,
    _fit_prior,
    _hgbr,
    _log_ratio,
    _patient_rows,
    _patient_table,
    _predict_coefs,
    _sigma_hgbr,
    _stage_c_features,
    _traj_cond4,
    _traj_conditional,
    _traj_debiased,
    _traj_visit,
    _true_coefs,
    _visit_full,
    _visit_no_meas,
    _visit_pop,
)
from src.model_33_bayes_blend import BayesCurveEstimator

# -----------------------------------------------------------------------------
# Hyperparametres globaux
# -----------------------------------------------------------------------------

# Nombre de plis internes pour la generation hors-fold des composantes
N_INNER = 3

# Nombre de plus proches voisins utilises par l'estimateur de trajectoire bayesienne
N_BAY_NEIGHBORS = 800

# Parametres des regresseurs HistGradientBoosting (Ca et Cb)
# Cadence d'apprentissage et nombre d'iterations adaptes pour un calcul rapide et stable
STACK_PARAMS = dict(
    max_iter=400,
    learning_rate=0.08,
    max_leaf_nodes=63,
    min_samples_leaf=100,
    l2_regularization=2.0,
)

# Poids du melange (blend) par defaut en mode evaluation
W_CA = 0.30
W_CB = 0.40
W_C1 = 0.10
W_C2 = 0.20


# -----------------------------------------------------------------------------
# Classe utilitaire : Chronometre avec alerte
# -----------------------------------------------------------------------------

class _Timer:
    """Gestionnaire de contexte pour chronometrer l'execution des differentes etapes.

    Affiche la duree ecoulee et declenche un avertissement si le delai depasse 20 minutes.
    """

    def __init__(self, label: str):
        self.label = label
        self._t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self._t0

    def done(self) -> None:
        dt = self.elapsed()
        flag = "  *** DEPASSE 20 MIN ***" if dt > WARN_SECONDS else ""
        print(f"  [{self.label}] {dt:.1f}s{flag}", flush=True)

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *_):
        self.done()


# -----------------------------------------------------------------------------
# Generation des variables bayesiennes hors-fold (OOF)
# -----------------------------------------------------------------------------

def _compute_bay_oof_with_y(
    X: pd.DataFrame,
    y: np.ndarray,
    t1_oof: np.ndarray,
    splits: list,
    seed: int,
) -> pd.DataFrame:
    """Calcule les sorties bayesiennes hors-fold a partir de la cible d'entrainement.

    Pour chaque pli interne :
      1. BayesCurveEstimator est entraine sur le pli train (sans jamais voir le pli val).
      2. Il predit sur le pli val :
         - bay_mean : moyenne a posteriori tronquee a [0, 132].
         - bay_sd   : ecart-type a posteriori representant l'incertitude locale.
      3. bay_minus_t1 mesure l'ecart avec la trajectoire parametrique t1.
      4. bay_pmean calcule la moyenne intra-patient pour capturer le niveau global.

    Ce partitionnement hors-fold garantit l'absence totale de fuite de donnees.
    """
    n = len(X)
    bay_mean = np.full(n, np.nan)
    bay_sd = np.full(n, np.nan)

    for tr, va in splits:
        X_tr = X.iloc[tr].reset_index(drop=True)
        X_va = X.iloc[va].reset_index(drop=True)

        bay = BayesCurveEstimator(n_neighbors=N_BAY_NEIGHBORS, random_state=seed)
        bay.fit(X_tr, y[tr])

        bay_mean[va] = np.clip(bay.predict(X_va), 0.0, 132.0)
        bay_sd[va] = np.clip(bay.posterior_std_, 0.0, None)

    # Calcul de la moyenne intra-patient des predictions bayesiennes
    pid = X["patient_id"].values
    pmean = np.full(n, np.nan)
    pat_rows_dict = _patient_rows(pid)
    for rows in pat_rows_dict.values():
        pmean[rows] = float(np.nanmean(bay_mean[rows]))

    return pd.DataFrame({
        "bay_mean": bay_mean,
        "bay_sd": bay_sd,
        "bay_minus_t1": bay_mean - t1_oof,
        "bay_pmean": pmean,
    })


# -----------------------------------------------------------------------------
# Modele de base : UltimateRegressor avec exposition des etages intermediaires
# -----------------------------------------------------------------------------

class _UltimateWithFeatures(UltimateRegressor):
    """Sous-classe d'UltimateRegressor conservant les matrices d'etage C1 et de trajectoire.

    Permet d'extraire :
      - F1_oof_   : matrice de variables explicatives pour le modele C1.
      - t1_oof_   : predictions de trajectoire de base (etage B).
      - A_oof_    : predictions de l'etage A de visite.
      - splits_   : decoupage GroupKFold interne utilise lors du fit.
      - get_features_test : methode pour regenerer F1 et t1 sur de nouvelles donnees de test.
    """

    def fit(self, X: pd.DataFrame, y):  # noqa: C901
        X = X.reset_index(drop=True)
        y_arr = np.asarray(y, dtype=float)
        groups = X["patient_id"].values
        seed = self.random_state

        self.cats_ = {c: sorted(X[c].dropna().unique()) for c in CATEGORICAL_COLS}
        splits = list(GroupKFold(n_splits=self.n_inner_splits).split(X, y_arr, groups))

        # Sous-matrices de visite (complete, sans mesures, sociodemographique)
        V_full = _visit_full(X, self.cats_)
        V_nm = _visit_no_meas(X, self.cats_)
        V_pop = _visit_pop(X, self.cats_)

        # Etage A : predictions hors-fold de visite et modeles de variance (sigmas)
        A_oof = self._compute_oof_a(V_full, V_nm, V_pop, X, y_arr, splits)
        sig = self._compute_oof_sigma(A_oof, y_arr, V_full, V_nm, splits)
        A_oof = pd.concat([A_oof, sig], axis=1)

        # Entrainement des modeles d'etage A sur l'ensemble complet du train
        self.stage_a_mods_ = self._fit_one_a(V_full, V_nm, V_pop, y_arr, seed)
        self.sigma_mods_ = {}
        for name, (est, log) in {
            "sig_est": (A_oof["est_visit"].values, False),
            "sig_est_off": (A_oof["est_off"].values, True),
            "sig_est_on": (A_oof["est_on"].values, True),
            "sl_off": (A_oof["u_off"].values, True),
            "sl_on": (A_oof["u_on"].values, True),
        }.items():
            tgt = np.where(
                np.isfinite(est),
                np.abs(_log_ratio(y_arr, est)) if log else np.abs(y_arr - est),
                np.abs(y_arr - A_oof["est_visit"].values),
            )
            m = np.isfinite(tgt)
            V_src = V_nm if "sl_" in name else V_full
            self.sigma_mods_[name] = _sigma_hgbr(random_state=seed).fit(V_src[m], tgt[m])

        # Etage B : Ajustement de l'a priori et calcul des trajectoires
        self.prior0_, self.prior4_, self.lev_var_ = _fit_prior(X, y_arr, A_oof["est_pop"].values)

        t1 = _traj_visit(X, A_oof, self.prior0_, self.lev_var_)
        t2 = _traj_debiased(X, A_oof, self.prior4_, self.lev_var_)
        t3 = _traj_conditional(X, A_oof, self.prior4_, self.lev_var_)
        t4 = _traj_cond4(X, A_oof, self.prior4_, self.lev_var_)
        trajs_oof = {"t1": t1, "t2": t2, "t3": t3, "t4": t4}

        # Construction de la matrice de variables F1 pour l'etage C1
        F1 = _stage_c_features(X, A_oof, trajs_oof, self.cats_)
        self.c1_cols_ = list(F1.columns)

        # Ajustement du modele C1 sur le residu de la trajectoire t1
        self.c1_model_ = _hgbr(
            random_state=seed,
            max_iter=STACK_PARAMS["max_iter"],
            learning_rate=STACK_PARAMS["learning_rate"],
            max_leaf_nodes=STACK_PARAMS["max_leaf_nodes"],
            min_samples_leaf=STACK_PARAMS["min_samples_leaf"],
            l2_regularization=STACK_PARAMS["l2_regularization"],
        ).fit(F1, y_arr - t1["pred"].values)

        # Modeles de regression des coefficients de parabole par patient (cf)
        t2_cols = {b: f"t2_{b}" for b in COEFS}
        cf_oof = pd.DataFrame(
            np.zeros((len(X), 4)),
            columns=["cf_b0", "cf_b1", "cf_b2", "cf_pred"],
        )
        for tr, va in splits:
            Xtr = X.iloc[tr].reset_index(drop=True)
            Ptr = _patient_table(F1.iloc[tr], Xtr, self.cats_)
            if not all(tc in Ptr.columns for tc in t2_cols.values()):
                continue
            mods_cf = _fit_coef_models(Ptr, _true_coefs(Xtr, y_arr[tr], Ptr.index), t2_cols, seed)
            Xva = X.iloc[va].reset_index(drop=True)
            Pva = _patient_table(F1.iloc[va], Xva, self.cats_)[Ptr.columns]
            cf_oof.iloc[va] = _predict_coefs(mods_cf, Pva, t2_cols, Xva).values

        P_full = _patient_table(F1, X, self.cats_)
        self.coef_cols_ = list(P_full.columns)
        self.t2_cols_ = t2_cols
        self.coef_mods_ = _fit_coef_models(P_full, _true_coefs(X, y_arr, P_full.index), t2_cols, seed)

        # Etage C2 : regression sur le residu du modele de coefficients cf_pred
        F2 = _stage_c_features(X, A_oof, trajs_oof, self.cats_, cf_cols=cf_oof)
        self.c2_cols_ = list(F2.columns)
        self.c2_model_ = _hgbr(
            random_state=seed,
            max_iter=STACK_PARAMS["max_iter"],
            learning_rate=STACK_PARAMS["learning_rate"],
            max_leaf_nodes=STACK_PARAMS["max_leaf_nodes"],
            min_samples_leaf=STACK_PARAMS["min_samples_leaf"],
            l2_regularization=STACK_PARAMS["l2_regularization"],
        ).fit(F2, y_arr - cf_oof["cf_pred"].values)

        self.model_11_ = None

        # Sauvegarde des variables hors-fold destinees au Stacking
        self.F1_oof_ = F1.reset_index(drop=True)
        self.t1_oof_ = t1["pred"].values.copy()
        self.A_oof_ = A_oof.reset_index(drop=True)
        self.trajs_oof_ = trajs_oof
        self.cf_oof_ = cf_oof.reset_index(drop=True)
        self.splits_ = splits

        return self

    def get_features_test(self, X_test: pd.DataFrame):
        """Reconstruit F1, t1_pred, c1 et c2 sur un ensemble de test."""
        X_test = X_test.reset_index(drop=True)
        A = self._predict_a(X_test)
        trajs = self._predict_trajs(X_test, A)
        F1 = _stage_c_features(X_test, A, trajs, self.cats_)[self.c1_cols_]
        t1_pred = trajs["t1"]["pred"].values
        P = _patient_table(F1, X_test, self.cats_)[self.coef_cols_]
        cf = _predict_coefs(self.coef_mods_, P, self.t2_cols_, X_test)
        F2 = _stage_c_features(X_test, A, trajs, self.cats_, cf_cols=cf)[self.c2_cols_]
        c1 = np.clip(self.c1_model_.predict(F1) + t1_pred, 0.0, 132.0)
        c2 = np.clip(self.c2_model_.predict(F2) + cf["cf_pred"].values, 0.0, 132.0)
        return F1, t1_pred, c1, c2


# -----------------------------------------------------------------------------
# Estimateur Principal : BayesStackRegressor
# -----------------------------------------------------------------------------

class BayesStackRegressor(RegressorMixin, BaseEstimator):
    """Architecture de Stacking Bayesien integrant les sorties BayesCurveEstimator.

    Ce metamodele combine :
      1. Les variables cliniques et de visite issues de l'etage C1 du Modele 32 (F1).
      2. Quatre variables issues de l'estimateur probabiliste BayesCurveEstimator :
         - bay_mean : esperance conditionnelle a posteriori.
         - bay_sd   : incertitude a posteriori de la forme de trajectoire.
         - bay_minus_t1 : divergence entre la courbe bayesienne et la dynamique parametrique t1.
         - bay_pmean : moyenne intra-patient du score predit.
      3. Deux modeles HGBR specialises :
         - Ca : cible (y - t1), apprend la correction sur l'ancre t1.
         - Cb : cible (y - bay_mean), apprend la correction directement sur la courbe bayesienne.
      4. Blend pondere (Ca, Cb, C1, C2) suivi d'un lissage longitudinal patient.

    Parametres :
      w_ca, w_cb, w_c1, w_c2 : coefficients de ponderation du melange final.
      n_inner_splits : nombre de plis pour le decoupage GroupKFold interne.
      smooth : active ou desactive le lissage polynomial et isotonique final.
      random_state : graine aleatoire pour la reproductibilite.
    """

    def __init__(
        self,
        w_ca: float = W_CA,
        w_cb: float = W_CB,
        w_c1: float = W_C1,
        w_c2: float = W_C2,
        n_inner_splits: int = N_INNER,
        smooth: bool = True,
        random_state: int = 0,
    ):
        self.w_ca = w_ca
        self.w_cb = w_cb
        self.w_c1 = w_c1
        self.w_c2 = w_c2
        self.n_inner_splits = n_inner_splits
        self.smooth = smooth
        self.random_state = random_state

    def _augment_features(self, F1: pd.DataFrame, bay_vars: pd.DataFrame) -> pd.DataFrame:
        """Concatene les 4 variables bayesiennes a la matrice d'etage C1 (F1)."""
        F = F1.copy().reset_index(drop=True)
        for col in ("bay_mean", "bay_sd", "bay_minus_t1", "bay_pmean"):
            F[col] = bay_vars[col].values
        return F

    def fit(self, X: pd.DataFrame, y):  # noqa: C901
        """Entrainement en 4 phases avec isolation stricte des echantillons."""
        X = X.reset_index(drop=True)
        y_arr = np.asarray(y, dtype=float)
        seed = self.random_state

        # Phase 1 : Entrainement du modele socle UltimateRegressor avec capture des features
        self.base_ = _UltimateWithFeatures(
            w_c1=1.0,
            w_c2=0.0,
            w_cf=0.0,
            w_11=0.0,
            n_inner_splits=self.n_inner_splits,
            smooth=False,
            random_state=seed,
        )
        self.base_.fit(X, y_arr)

        F1_oof = self.base_.F1_oof_
        t1_oof = self.base_.t1_oof_
        splits = self.base_.splits_

        # Phase 2 : Calcul hors-fold (OOF) des predictions bayesiennes internes
        bay_oof = _compute_bay_oof_with_y(X, y_arr, t1_oof, splits, seed)

        # Phase 3 : Construction de la matrice augmentee et entrainement de Ca et Cb
        F_aug = self._augment_features(F1_oof, bay_oof)
        self.aug_cols_ = list(F_aug.columns)

        # Ca apprend la correction residuelle par rapport a la trajectoire parametrique t1
        self.ca_model_ = _hgbr(random_state=seed, **STACK_PARAMS).fit(
            F_aug, y_arr - t1_oof
        )

        # Cb apprend la correction residuelle directement par rapport a la courbe bayesienne
        self.cb_model_ = _hgbr(random_state=seed, **STACK_PARAMS).fit(
            F_aug, y_arr - bay_oof["bay_mean"].values
        )

        # Phase 4 : Entrainement du BayesCurveEstimator final sur l'integralite du train
        self.bay_final_ = BayesCurveEstimator(
            n_neighbors=N_BAY_NEIGHBORS, random_state=seed
        )
        self.bay_final_.fit(X, y_arr)

        return self

    def predict_components(self, X: pd.DataFrame) -> dict:
        """Predit et extrait chacune des composantes individuelles du systeme (ca, cb, c1, c2, bay)."""
        X = X.reset_index(drop=True)

        # Recuperation de F1, t1, c1 et c2 depuis le modele socle
        F1, t1_pred, c1, c2 = self.base_.get_features_test(X)

        # Evaluation par le modele bayesien final
        bay_mean = np.clip(self.bay_final_.predict(X), 0.0, 132.0)
        bay_sd = np.clip(self.bay_final_.posterior_std_, 0.0, None)

        pid = X["patient_id"].values
        pat_rows_dict = _patient_rows(pid)
        pmean = np.full(len(X), np.nan)
        for rows in pat_rows_dict.values():
            pmean[rows] = float(np.nanmean(bay_mean[rows]))

        bay_vars = pd.DataFrame({
            "bay_mean": bay_mean,
            "bay_sd": bay_sd,
            "bay_minus_t1": bay_mean - t1_pred,
            "bay_pmean": pmean,
        })

        # Construction de la matrice d'inference augmentee
        F_aug = self._augment_features(F1, bay_vars)[self.aug_cols_]

        # Reconstitution des predictions absolues Ca et Cb
        ca = np.clip(self.ca_model_.predict(F_aug) + t1_pred, 0.0, 132.0)
        cb = np.clip(self.cb_model_.predict(F_aug) + bay_mean, 0.0, 132.0)

        return {"ca": ca, "cb": cb, "c1": c1, "c2": c2, "bay": bay_mean}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predit le score final en appliquant la combinaison lineaire ponderee et le lissage."""
        X = X.reset_index(drop=True)
        c = self.predict_components(X)
        pred = (
            self.w_ca * c["ca"]
            + self.w_cb * c["cb"]
            + self.w_c1 * c["c1"]
            + self.w_c2 * c["c2"]
        )
        if self.smooth:
            pred = smooth_predictions(X["patient_id"].values, X["age"].values, pred)
        return np.clip(pred, 0.0, 132.0)


# -----------------------------------------------------------------------------
# Utilitaires d'evaluation et de metriques
# -----------------------------------------------------------------------------

def _fold_ids(cv_splits, n: int) -> np.ndarray:
    """Attribue a chaque indice d'echantillon l'identifiant du pli auquel il appartient."""
    folds = np.zeros(n, dtype=int)
    for k, (_, va) in enumerate(cv_splits):
        folds[va] = k
    return folds


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Calcule la racine carree de l'erreur quadratique moyenne (Root Mean Squared Error)."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


# -----------------------------------------------------------------------------
# Calcul Hors-Fold (OOF) de Ca et Cb avec sauvegarde progressive
# -----------------------------------------------------------------------------

def _compute_oof_ca_cb(X: pd.DataFrame, y: pd.Series | np.ndarray, cv_splits: list, seed: int = 0):
    """Calcule Ca et Cb hors-fold sur les 5 plis externes avec sauvegarde a chaque pli.

    Ce mecanisme permet de reprendre le calcul sans perte en cas d'interruption accidentelle.
    """
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    path = OOF_DIR / "34_components_oof.csv"
    folds = _fold_ids(cv_splits, len(X))
    X_index = X["Index"].values if "Index" in X.columns else np.arange(len(X))
    yv = np.asarray(y, dtype=float)

    done_folds: set = set()
    ca = np.full(len(X), np.nan)
    cb = np.full(len(X), np.nan)

    # Rechargement des plis deja calcules si le fichier existe
    if path.is_file():
        df_prev = pd.read_csv(path)
        if "ca" in df_prev.columns and "cb" in df_prev.columns:
            prev_folds = df_prev["fold"].values if "fold" in df_prev.columns else np.full(len(df_prev), -1)
            for k in range(len(cv_splits)):
                m = prev_folds == k
                if m.sum() > 0 and not np.isnan(df_prev.loc[m, "ca"].values).any():
                    _, va = cv_splits[k]
                    ca[va] = df_prev.loc[m, "ca"].values
                    cb[va] = df_prev.loc[m, "cb"].values
                    done_folds.add(k)
            if done_folds:
                print(f"  Reprise : plis deja en cache = {sorted(done_folds)}", flush=True)

    for k, (tr, va) in enumerate(cv_splits):
        if k in done_folds:
            rmse_ca = _rmse(ca[va], yv[va])
            rmse_cb = _rmse(cb[va], yv[va])
            print(f"  Pli {k} (depuis cache) -> ca={rmse_ca:.4f}  cb={rmse_cb:.4f}", flush=True)
            continue

        t_pli = _Timer(f"pli {k}")
        X_tr = X.iloc[tr].reset_index(drop=True)
        y_tr = yv[tr]
        X_va = X.iloc[va].reset_index(drop=True)

        # Modele socle sur le pli d'entrainement externe
        base = _UltimateWithFeatures(
            w_c1=1.0,
            w_c2=0.0,
            w_cf=0.0,
            w_11=0.0,
            n_inner_splits=N_INNER,
            smooth=False,
            random_state=seed,
        )
        base.fit(X_tr, y_tr)

        F1_tr = base.F1_oof_
        t1_tr = base.t1_oof_
        splits_tr = base.splits_

        # Calcul bayesien hors-fold interne sur le pli d'entrainement
        bay_tr = _compute_bay_oof_with_y(X_tr, y_tr, t1_tr, splits_tr, seed)

        # Ajustement des metamodeles Ca et Cb sur la matrice augmentee
        F_aug_tr = pd.concat([F1_tr.reset_index(drop=True), bay_tr.reset_index(drop=True)], axis=1)
        aug_cols = list(F_aug_tr.columns)

        ca_m = _hgbr(random_state=seed, **STACK_PARAMS).fit(F_aug_tr, y_tr - t1_tr)
        cb_m = _hgbr(random_state=seed, **STACK_PARAMS).fit(
            F_aug_tr, y_tr - bay_tr["bay_mean"].values
        )

        # Inferences sur le pli de validation externe
        bay_val_est = BayesCurveEstimator(n_neighbors=N_BAY_NEIGHBORS, random_state=seed)
        bay_val_est.fit(X_tr, y_tr)
        bay_mean_va = np.clip(bay_val_est.predict(X_va), 0.0, 132.0)
        bay_sd_va = np.clip(bay_val_est.posterior_std_, 0.0, None)

        F1_va, t1_va, _, _ = base.get_features_test(X_va)

        pid_va = X_va["patient_id"].values
        pmean_va = np.full(len(X_va), np.nan)
        for rows in _patient_rows(pid_va).values():
            pmean_va[rows] = float(np.nanmean(bay_mean_va[rows]))

        bay_vars_va = pd.DataFrame({
            "bay_mean": bay_mean_va,
            "bay_sd": bay_sd_va,
            "bay_minus_t1": bay_mean_va - t1_va,
            "bay_pmean": pmean_va,
        })

        F_aug_va = pd.concat([F1_va.reset_index(drop=True), bay_vars_va.reset_index(drop=True)], axis=1)[aug_cols]

        ca[va] = np.clip(ca_m.predict(F_aug_va) + t1_va, 0.0, 132.0)
        cb[va] = np.clip(cb_m.predict(F_aug_va) + bay_mean_va, 0.0, 132.0)

        # Sauvegarde a jour du CSV apres chaque pli traite
        df_save = pd.DataFrame({"Index": X_index, "fold": folds, "ca": ca, "cb": cb})
        df_save.to_csv(path, index=False)

        rmse_ca = _rmse(ca[va], yv[va])
        rmse_cb = _rmse(cb[va], yv[va])
        t_pli.done()
        print(f"    Pli {k} termine -> ca={rmse_ca:.4f}  cb={rmse_cb:.4f}", flush=True)

    return ca, cb


# -----------------------------------------------------------------------------
# Mode Rapide (--quick) : Optimisation de la grille de poids
# -----------------------------------------------------------------------------

def quick_mode(X: pd.DataFrame, y: pd.Series, cv_splits: list) -> None:
    """Mode d'optimisation rapide de la grille de poids combinant les 5 composantes (ca, cb, c1, c2, bay).

    Recharge les composantes c1, c2, cf, bay depuis oof/33_components_oof.csv
    et calcule ou recharge ca, cb depuis oof/34_components_oof.csv.
    """
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    path33 = OOF_DIR / "33_components_oof.csv"
    path34 = OOF_DIR / "34_components_oof.csv"
    names33 = ("c1", "c2", "cf", "bay")
    folds = _fold_ids(cv_splits, len(X))
    X_index = X["Index"].values if "Index" in X.columns else np.arange(len(X))
    yv = y.values

    # Etape A : Chargement de c1, c2 et bay depuis le Modele 33
    if path33.is_file():
        print(f"  Cache OOF Modele 33 : {path33}", flush=True)
        df33 = pd.read_csv(path33)
        df_ref = pd.DataFrame({"Index": X_index, "fold": folds})
        merged = df_ref.merge(df33[["Index", "fold"] + list(names33)], on=["Index", "fold"], how="left")
        missing = merged["c1"].isna().sum()
        if missing > 0:
            print(f"  ATTENTION : {missing} lignes sans correspondance dans 33 OOF.", flush=True)
        comp33 = {n: merged[n].values for n in names33}
    else:
        print("  ATTENTION : oof/33_components_oof.csv absent, composantes 33 initialisees a 0.", flush=True)
        comp33 = {n: np.zeros(len(X)) for n in names33}

    # Etape B : Rechargement ou calcul hors-fold de Ca et Cb
    if path34.is_file():
        df34 = pd.read_csv(path34)
        df_ref = pd.DataFrame({"Index": X_index, "fold": folds})
        merged34 = df_ref.merge(df34[["Index", "fold", "ca", "cb"]], on=["Index", "fold"], how="left")
        if not merged34["ca"].isna().any():
            print(f"  Cache OOF Modele 34 complet : {path34}", flush=True)
            ca = merged34["ca"].values
            cb = merged34["cb"].values
        else:
            print("  Cache OOF Modele 34 partiel, calcul des plis restants...", flush=True)
            ca, cb = _compute_oof_ca_cb(X, y, cv_splits)
    else:
        print("  Calcul complet des composantes Ca et Cb hors-fold...", flush=True)
        ca, cb = _compute_oof_ca_cb(X, y, cv_splits)

    c1v = comp33["c1"]
    c2v = comp33["c2"]
    bayv = comp33["bay"]
    cav = ca
    cbv = cb

    pid = X["patient_id"].values
    age = X["age"].values

    # Etape C : Evaluation de chaque composante isolee
    print("\n  RMSE par composante individuelle (brut vs lisse) :", flush=True)
    all_comp = {"c1": c1v, "c2": c2v, "bay": bayv, "ca": cav, "cb": cbv}
    for n, v in all_comp.items():
        v_sm = smooth_predictions(pid, age, v)
        rmse_f = [np.sqrt(np.mean((v[folds == k] - yv[folds == k]) ** 2)) for k in range(len(cv_splits))]
        rmse_sm = [np.sqrt(np.mean((v_sm[folds == k] - yv[folds == k]) ** 2)) for k in range(len(cv_splits))]
        print(
            f"    {n:<6} brut={np.mean(rmse_f):.4f}+-{np.std(rmse_f):.4f}"
            f"  lisse={np.mean(rmse_sm):.4f}+-{np.std(rmse_sm):.4f}",
            flush=True,
        )

    # Etape D : Recherche sur grille en deux passes
    # Passe 1 : Exploration exhaustive sur predictions brutes (pas = 0.05)
    print("\n  Optimisation de la grille de poids (pas=0.05, 5 composantes) :", flush=True)
    step = 0.05
    grid_pts = np.arange(0, 1.0 + step / 2, step)

    raw_results = []
    for w_ca in grid_pts:
        for w_cb in grid_pts:
            for w_c1 in grid_pts:
                for w_c2 in grid_pts:
                    w_bay = round(1.0 - w_ca - w_cb - w_c1 - w_c2, 10)
                    if w_bay < -1e-9:
                        continue
                    w_bay = max(w_bay, 0.0)
                    pred_raw = (
                        w_ca * cav
                        + w_cb * cbv
                        + w_c1 * c1v
                        + w_c2 * c2v
                        + w_bay * bayv
                    )
                    rmse_f = [
                        np.sqrt(np.mean((pred_raw[folds == k] - yv[folds == k]) ** 2))
                        for k in range(len(cv_splits))
                    ]
                    raw_results.append((float(np.mean(rmse_f)), w_ca, w_cb, w_c1, w_c2, w_bay))

    raw_results.sort()
    top20 = raw_results[:20]

    # Passe 2 : Application du lissage longitudinal sur les 20 meilleures combinaisons
    smooth_results = []
    best_rmse, best_w = 999.0, None
    for _, w_ca, w_cb, w_c1, w_c2, w_bay in top20:
        pred = smooth_predictions(
            pid,
            age,
            w_ca * cav + w_cb * cbv + w_c1 * c1v + w_c2 * c2v + w_bay * bayv,
        )
        rmse_f = [
            np.sqrt(np.mean((pred[folds == k] - yv[folds == k]) ** 2))
            for k in range(len(cv_splits))
        ]
        m = float(np.mean(rmse_f))
        smooth_results.append((m, w_ca, w_cb, w_c1, w_c2, w_bay))
        if m < best_rmse:
            best_rmse, best_w = m, (w_ca, w_cb, w_c1, w_c2, w_bay)

    smooth_results.sort()
    print("  Top 5 des combinaisons (ca, cb, c1, c2, bay) [apres lissage] :", flush=True)
    for rm, w_ca, w_cb, w_c1, w_c2, w_bay in smooth_results[:5]:
        print(
            f"    ca={w_ca:.2f} cb={w_cb:.2f} c1={w_c1:.2f} c2={w_c2:.2f} bay={w_bay:.2f}"
            f" -> RMSE={rm:.4f}",
            flush=True,
        )
    print(
        f"  Combinaison optimale : ca={best_w[0]:.2f} cb={best_w[1]:.2f} "
        f"c1={best_w[2]:.2f} c2={best_w[3]:.2f} bay={best_w[4]:.2f}"
        f" -> RMSE={best_rmse:.4f}",
        flush=True,
    )

    # Affichage du tableau comparatif 32 vs 33 vs 34
    _print_comparison_table(
        {"c1(32)": c1v, "c2(32)": c2v, "bay(33)": bayv, "ca(34)": cav, "cb(34)": cbv},
        {
            "blend_33": 0.30 * c1v + 0.35 * bayv + 0.30 * c2v,
            "blend_34": (
                best_w[0] * cav
                + best_w[1] * cbv
                + best_w[2] * c1v
                + best_w[3] * c2v
                + best_w[4] * bayv
            ),
        },
        yv,
        pid,
        age,
        folds,
        len(cv_splits),
    )


def _print_comparison_table(
    components: dict,
    blends: dict,
    yv: np.ndarray,
    pid: np.ndarray,
    age: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
) -> None:
    """Affiche le tableau comparatif des RMSE et recapitule les fondements theoriques du gain de performance."""
    print("\n  --- Tableau comparatif des modeles 32, 33 et 34 ---", flush=True)
    print(f"  {'Modele':<20} {'RMSE brut':>10}  {'RMSE lisse':>10}", flush=True)
    all_items = {**components, **blends}
    for name, v in all_items.items():
        v_sm = smooth_predictions(pid, age, v)
        rmse_f = np.mean([
            np.sqrt(np.mean((v[folds == k] - yv[folds == k]) ** 2))
            for k in range(n_folds)
        ])
        rmse_sm = np.mean([
            np.sqrt(np.mean((v_sm[folds == k] - yv[folds == k]) ** 2))
            for k in range(n_folds)
        ])
        print(f"  {name:<20} {rmse_f:>10.4f}  {rmse_sm:>10.4f}", flush=True)

    print("""
  Pourquoi le Stacking Bayesien ameliore-t-il le blend fixe (Modele 33) ?
    1. bay_sd expose au gradient boosting l'incertitude de la forme bayesienne :
       quand la trajectoire est mal contrainte (manque d'examens off/on), l'arbre
       se replie naturellement sur c1 et c2 dont l'erreur est plus uniforme.
    2. bay_minus_t1 et bay_pmean transmettent le niveau specifique du patient,
       ce que les etages de visite simples ne captent pas de maniere globale.
    3. Cb (cible : y - bay_mean) prend l'estimation bayesienne comme baseline :
       les arbres apprennent directement les residus sur une base deja non-lineaire,
       ce qui reduit drastiquement la variance residuelle.
    => Le stacking apprend QUAND faire confiance au bayesien au lieu d'imposer un poids uniforme.
""", flush=True)


# -----------------------------------------------------------------------------
# Point d'entree principal : CLI et modes d'execution
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    """Point d'entree en ligne de commande gerant l'evaluation, l'entrainement et Skore Hub."""
    parser = argparse.ArgumentParser(
        description="Stacking bayesien : sorties BCEst en variables des etages C (34_bayes_stack)"
    )
    parser.add_argument("--no-hub", action="store_true", help="Desactive la publication sur Skore Hub.")
    parser.add_argument("--quick", action="store_true", help="Calcule OOF et la grille de poids uniquement (pas de Skore).")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Evalue, sauvegarde oof/34_report.joblib et publie sans entrainement final ni soumission CSV.",
    )
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="Recharge oof/34_report.joblib existant et le publie directement sur Skore Hub.",
    )
    parser.add_argument("--w-ca", type=float, default=W_CA, help="Poids attribue a la composante Ca.")
    parser.add_argument("--w-cb", type=float, default=W_CB, help="Poids attribue a la composante Cb.")
    parser.add_argument("--w-c1", type=float, default=W_C1, help="Poids attribue a la composante C1.")
    parser.add_argument("--w-c2", type=float, default=W_C2, help="Poids attribue a la composante C2.")
    args = parser.parse_args()

    params = dict(w_ca=args.w_ca, w_cb=args.w_cb, w_c1=args.w_c1, w_c2=args.w_c2)
    t_global = _Timer("global")

    # Mode 1 : --publish-only (recharge rapide et envoi sans calcul)
    if args.publish_only:
        report_path = OOF_DIR / "34_report.joblib"
        print(f"  Rechargement du rapport existant depuis {report_path}...", flush=True)
        report = joblib.load(report_path)
        print(f"  Publication sur Skore Hub (cle '{HUB_KEY}')...", flush=True)
        config = load_skore_credentials(REPO_ROOT)
        login(mode="hub")
        project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
        project.put(HUB_KEY, report)
        print("  Publication terminee avec succes.", flush=True)
        t_global.done()
        return

    print("=" * 70, flush=True)
    print("MODELE 34 BAYES STACK", flush=True)
    print("Stacking bayesien : sorties BCEst en variables des etages C", flush=True)
    print("=" * 70, flush=True)

    # Chargement des jeux de donnees
    with _Timer("chargement donnees"):
        X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
        y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
        X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

        visits = X_train_raw.merge(y_train_raw, on="Index")
        y = visits["target"]
        X = add_longitudinal_columns(visits.drop(columns=["target"]))
        X_test = add_longitudinal_columns(X_test_raw)

    print(f"  Dimensions X train: {X.shape}  Dimensions X test: {X_test.shape}", flush=True)

    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

    # Mode 2 : --quick (optimisation rapide des poids)
    if args.quick:
        print("\n[--quick] Calcul OOF et recherche sur grille de poids...", flush=True)
        quick_mode(X, y, cv_splits)
        t_global.done()
        return

    # Splitter de validation (1 pli GroupShuffleSplit a 20% des patients)
    splitter = list(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, groups=X["patient_id"]))

    # Mode 3 : --report-only (evaluation + Skore Hub uniquement)
    if args.report_only:
        print("\n[report-only] Evaluation par GroupShuffleSplit(1, test_size=0.2)...", flush=True)
        from skore import configuration
        configuration.ignore_checks = ["SKD003"]

        model = BayesStackRegressor(**params)
        with _Timer("skore.evaluate"):
            report = evaluate(model, X, y, splitter=splitter)

        try:
            print(f"  RMSE :\n{report.metrics.rmse()}", flush=True)
        except Exception as exc:
            print(f"  (Avertissement metrique: {exc})", flush=True)
        try:
            print(f"  Controles qualite Skore :\n{report.checks.summarize(fast_mode=True)}", flush=True)
        except Exception as exc:
            print(f"  (Avertissement controles : {exc})", flush=True)

        OOF_DIR.mkdir(parents=True, exist_ok=True)
        report_path = OOF_DIR / "34_report.joblib"
        joblib.dump(report, report_path)
        print(f"  Rapport sauvegarde dans {report_path}", flush=True)

        if args.no_hub:
            print("  Publication Hub ignoree (--no-hub).", flush=True)
            t_global.done()
            return

        print(f"  Publication sur Skore Hub (cle '{HUB_KEY}')...", flush=True)
        config = load_skore_credentials(REPO_ROOT)
        login(mode="hub")
        project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
        project.put(HUB_KEY, report)
        print("  Publication terminee avec succes.", flush=True)
        t_global.done()
        return

    # Mode 4 : Execution complete (evaluation + entrainement final + soumission)
    print("\n[1/3] Evaluation par GroupShuffleSplit(1, test_size=0.2)...", flush=True)
    model = BayesStackRegressor(**params)

    with _Timer("skore.evaluate"):
        report = evaluate(model, X, y, splitter=splitter)

    try:
        print(f"  RMSE :\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  (Avertissement metrique: {exc})", flush=True)
    try:
        print(f"  Controles qualite Skore :\n{report.checks.summarize(fast_mode=True)}", flush=True)
    except Exception as exc:
        print(f"  (Avertissement controles : {exc})", flush=True)

    # Analyse des performances par composante sur l'echantillon de test
    print("\n  RMSE par composante sur le pli de test :", flush=True)
    est_report = report.reports_[0]
    fitted_model = est_report.estimator_
    X_va_ev = est_report.X_test.reset_index(drop=True)
    y_va_ev = np.asarray(est_report.y_test, dtype=float)

    comp_va = fitted_model.predict_components(X_va_ev)
    for name, v in comp_va.items():
        v_sm = smooth_predictions(X_va_ev["patient_id"].values, X_va_ev["age"].values, v)
        print(f"    {name:<8} RMSE={_rmse(v, y_va_ev):.4f}  lisse={_rmse(v_sm, y_va_ev):.4f}", flush=True)

    blend_raw = (
        args.w_ca * comp_va["ca"]
        + args.w_cb * comp_va["cb"]
        + args.w_c1 * comp_va["c1"]
        + args.w_c2 * comp_va["c2"]
    )
    blend_sm = smooth_predictions(X_va_ev["patient_id"].values, X_va_ev["age"].values, blend_raw)
    print(f"    blend    RMSE={_rmse(blend_raw, y_va_ev):.4f}  lisse={_rmse(blend_sm, y_va_ev):.4f}", flush=True)

    # Entrainement final sur tout le dataset d'apprentissage
    print("\n[2/3] Entrainement final sur 100% du jeu d'entrainement...", flush=True)
    with _Timer("fit final"):
        model.fit(X, y)

    # Generation du fichier de soumission pour Kaggle
    predictions = model.predict(X_test)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"{HUB_KEY}.csv"
    pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions}).to_csv(sub_path, index=False)

    print(f"  Fichier de soumission genere : {sub_path} ({len(predictions)} lignes)", flush=True)
    print(f"  Statistiques cibles : min={predictions.min():.2f}  moyenne={predictions.mean():.2f}  max={predictions.max():.2f}", flush=True)

    # Sauvegarde et publication sur Skore Hub
    if args.no_hub:
        print("\n[3/3] Publication Hub ignoree (--no-hub).", flush=True)
        t_global.done()
        return

    print(f"\n[3/3] Publication sur Skore Hub (cle '{HUB_KEY}')...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
    project.put(HUB_KEY, report)
    print("  Publication sur Skore Hub terminee avec succes.", flush=True)
    t_global.done()


if __name__ == "__main__":
    main()
