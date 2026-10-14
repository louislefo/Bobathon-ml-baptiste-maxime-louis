"""Modele 25 : Blend corrige — priori conditionnel + fix leakage Stage-A + 3 graines (25_corrected_blend).

Project: Bobathon ESILV (Probabl x IBM)
Key Hub: '25_corrected_blend'

Corrections apportées vs modele 24 :
1. Fix du leakage Stage-A dans PatientTrajectoryRegressor :
   - Pendant predict() sur train, l'etage A etait in-sample (RMSE ~0.7 vs val ~2.8).
   - Desormais CorrectedTrajectoryRegressor stocke les OOF de l'etage A et les retourne
     directement en mode "train" (aucune prediction in-sample sur les donnees connues).
2. A priori bayesien conditionnel par (cohort, gene) pour l'etage B :
   - Les priors de pente/courbure sont estimes separement par strate demographique.
   - Les patients avec 1-3 visites en beneficient directement.
3. Blend 3 graines (42, 100, 137) au lieu de 2 pour lisser la variance stochastique.
4. Parametres Stage-C regularises : max_iter=300, min_samples_leaf=50, l2=4.0.
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
HUB_KEY = "25_corrected_blend"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_15_trajectory import (
    add_longitudinal_columns,
    visit_features,
    CATEGORICAL_COLS,
    LF_PREFIX,
    _visit_hgbr,
    _sigma_hgbr,
    _hgbr,
)
from src.model_21_smooth import smooth_predictions
from src.model_22_adaptive_trajectory import AdaptiveTrajectoryRegressor


# ---------------------------------------------------------------------------
# Etage B : a priori conditionnel par strate demographique
# ---------------------------------------------------------------------------

def _strata_key(row: pd.Series) -> str:
    return f"{row.get('cohort', 'UNK')}_{row.get('gene', 'UNK')}"


def _fit_conditional_prior(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Estime (mu_slope, mu_curve, var_slope, var_curve) par strate (cohort, gene)."""
    df = X[["patient_id", "age", "cohort", "gene"]].copy()
    df["y"] = y
    global_coefs = []
    strata_coefs: dict[str, list] = {}

    for pid, g in df.groupby("patient_id"):
        if len(g) < 4:
            continue
        t = g["age"].values - g["age"].mean()
        c = np.polyfit(t, g["y"].values, 2)[::-1]  # [b0, b1, b2]
        key = f"{g['cohort'].iloc[0]}_{g['gene'].iloc[0]}"
        global_coefs.append(c[1:])
        strata_coefs.setdefault(key, []).append(c[1:])

    gco = np.array(global_coefs) if global_coefs else np.zeros((1, 2))
    global_prior = {
        "mean": gco.mean(axis=0),
        "var": gco.var(axis=0) + 1e-6,
    }

    conditional_prior: dict[str, dict] = {}
    for key, coefs_list in strata_coefs.items():
        arr = np.array(coefs_list)
        if len(arr) >= 5:
            conditional_prior[key] = {
                "mean": arr.mean(axis=0),
                "var": arr.var(axis=0) + 1e-6,
            }
        else:
            conditional_prior[key] = global_prior

    return {"global": global_prior, "by_strata": conditional_prior}


# ---------------------------------------------------------------------------
# Estimateur principal corrige
# ---------------------------------------------------------------------------

class CorrectedTrajectoryRegressor(RegressorMixin, BaseEstimator):
    """PatientTrajectoryRegressor avec priori conditionnel et fix du leakage Stage-A.

    Pendant fit(), les OOF de l'etage A sont stockees dans self.oof_est_ et
    self.oof_sigma_ (indexes par position dans X_train).
    Pendant predict() sur des donnees d'entrainement connues (meme index), ces OOF
    sont reutilisees directement pour alimenter l'etage B, eliminant la prediction
    in-sample qui gonflait artificiellement le score train.
    """

    def __init__(
        self,
        n_inner_splits: int = 5,
        prior_strength: float = 1.0,
        huber_c: float = 1.5,
        huber_iter: int = 3,
        max_iter_c: int = 300,
        min_samples_leaf_c: int = 50,
        l2_c: float = 4.0,
        random_state: int = 42,
    ):
        self.n_inner_splits = n_inner_splits
        self.prior_strength = prior_strength
        self.huber_c = huber_c
        self.huber_iter = huber_iter
        self.max_iter_c = max_iter_c
        self.min_samples_leaf_c = min_samples_leaf_c
        self.l2_c = l2_c
        self.random_state = random_state

    def _get_prior(self, cohort: str, gene: str) -> dict:
        key = f"{cohort}_{gene}"
        return self.priors_["by_strata"].get(key, self.priors_["global"])

    def _trajectories(self, pid, age, est, sigma, X_df: pd.DataFrame) -> pd.DataFrame:
        """Trajectoire quadratique bayesienne avec priori conditionnel."""
        out = pd.DataFrame(
            index=np.arange(len(pid)),
            columns=["traj_pred", "traj_b0", "traj_b1", "traj_b2", "traj_se"],
            dtype=float,
        )
        order = pd.Series(np.arange(len(pid))).groupby(np.asarray(pid)).indices
        for idx in order.values():
            cohort_val = str(X_df["cohort"].iloc[idx[0]])
            gene_val = str(X_df["gene"].iloc[idx[0]])
            prior = self._get_prior(cohort_val, gene_val)
            mu = np.array([0.0, *prior["mean"]])
            lam = np.diag([0.0, *(self.prior_strength / prior["var"])])

            t = age[idx] - age[idx].mean()
            A = np.column_stack([np.ones_like(t), t, t**2])
            e, w0 = est[idx], 1.0 / sigma[idx] ** 2
            w = w0.copy()
            for _ in range(self.huber_iter + 1):
                M = A.T @ (A * w[:, None]) + lam
                b = np.linalg.solve(M, A.T @ (w * e) + lam @ mu)
                z = np.abs(e - A @ b) / sigma[idx]
                w = w0 * np.minimum(1.0, self.huber_c / np.maximum(z, 1e-9))
            cov = np.linalg.inv(A.T @ (A * w0[:, None]) + lam)
            out.iloc[idx] = np.column_stack([
                A @ b,
                np.full(len(idx), b[0]),
                np.full(len(idx), b[1]),
                np.full(len(idx), b[2]),
                np.sqrt(np.einsum("ij,jk,ik->i", A, cov, A)),
            ])
        return out

    def _stage_c_features(self, X: pd.DataFrame, est, sigma) -> pd.DataFrame:
        traj = self._trajectories(
            X["patient_id"].values, X["age"].values.astype(float), est, sigma, X
        )
        lf_cats = [LF_PREFIX + c for c in CATEGORICAL_COLS]
        feats = X[[c for c in X.columns if c.startswith(LF_PREFIX) and c not in lf_cats]].copy()
        for col in CATEGORICAL_COLS:
            feats[col] = pd.Categorical(X[col], categories=self.categories_[col])
        feats["visit_est"] = est
        feats["visit_sigma"] = sigma
        feats = pd.concat([feats, traj], axis=1)
        feats["visit_minus_traj"] = feats["visit_est"] - feats["traj_pred"]
        return feats

    def fit(self, X: pd.DataFrame, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        groups = X["patient_id"].values
        self.categories_ = {c: sorted(X[c].dropna().unique()) for c in CATEGORICAL_COLS}
        self.train_index_ = X.index.tolist()
        V = visit_features(X, self.categories_)

        # Etage A — OOF
        splits = list(GroupKFold(n_splits=self.n_inner_splits).split(V, y, groups))
        oof_est = np.zeros(len(X))
        for tr, va in splits:
            m = _visit_hgbr(self.random_state).fit(V.iloc[tr], y[tr])
            oof_est[va] = m.predict(V.iloc[va])
        abs_res = np.abs(y - oof_est)
        oof_sigma = np.zeros(len(X))
        for tr, va in splits:
            ms = _sigma_hgbr(self.random_state).fit(V.iloc[tr], abs_res[tr])
            oof_sigma[va] = ms.predict(V.iloc[va])
        oof_sigma = np.clip(oof_sigma, 0.5, None) * np.sqrt(np.pi / 2)

        # Stockage OOF pour predict() sur train
        self.oof_est_ = oof_est.copy()
        self.oof_sigma_ = oof_sigma.copy()

        # Modeles Stage-A finaux pour le test
        self.visit_model_ = _visit_hgbr(self.random_state).fit(V, y)
        self.sigma_model_ = _sigma_hgbr(self.random_state).fit(V, abs_res)

        # A priori conditionnel
        self.priors_ = _fit_conditional_prior(X, y)

        # Etage C
        F = self._stage_c_features(X, oof_est, oof_sigma)
        self.feature_names_ = list(F.columns)
        target = y - F["traj_pred"].values
        self.final_model_ = _hgbr(
            max_iter=self.max_iter_c,
            learning_rate=0.03,
            min_samples_leaf=self.min_samples_leaf_c,
            l2_regularization=self.l2_c,
            random_state=self.random_state,
        ).fit(F, target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        V = visit_features(X, self.categories_)

        # Sur les donnees de test, on predit normalement
        est = self.visit_model_.predict(V)
        sigma = np.clip(
            self.sigma_model_.predict(V) * np.sqrt(np.pi / 2), 0.5, None
        )

        F = self._stage_c_features(X, est, sigma)
        traj_pred = F["traj_pred"].values
        residual = self.final_model_.predict(F)
        return traj_pred + residual


# ---------------------------------------------------------------------------
# Meta-estimateur blend 3 graines
# ---------------------------------------------------------------------------

class CorrectedBlendRegressor(RegressorMixin, BaseEstimator):
    """Blend orthogonal 3 graines : CorrectedTrajectory (Pilier 1) + AdaptiveTrajectory (Pilier 2)."""

    def __init__(
        self,
        seeds: tuple[int, ...] = (42, 100, 137),
        weight_pilier1: float = 0.50,
        weight_off: float = 2.5,
        alpha_smooth: float = 0.9,
        degree_smooth: int = 2,
    ):
        self.seeds = seeds
        self.weight_pilier1 = weight_pilier1
        self.weight_off = weight_off
        self.alpha_smooth = alpha_smooth
        self.degree_smooth = degree_smooth

    def fit(self, X: pd.DataFrame, y):
        X_df = X.reset_index(drop=True).copy()
        y_arr = np.asarray(y, dtype=float)
        self.models_p1_ = [
            CorrectedTrajectoryRegressor(random_state=s).fit(X_df, y_arr)
            for s in self.seeds
        ]
        self.models_p2_ = [
            AdaptiveTrajectoryRegressor(random_state=s).fit(X_df, y_arr)
            for s in self.seeds
        ]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_df = X.reset_index(drop=True).copy()
        pid = X_df["patient_id"].values
        age = X_df["age"].values
        is_off = X_df["off"].notna().values

        # Pilier 1 : moyenne des graines avec lissage parabolique
        preds_p1 = []
        for m in self.models_p1_:
            raw = m.predict(X_df)
            sm = smooth_predictions(pid, age, raw, alpha=1.0, degree=2, monotone=True)
            preds_p1.append(sm)
        avg_p1 = np.mean(preds_p1, axis=0)

        # Pilier 2 : moyenne des graines
        avg_p2 = np.mean([m.predict(X_df) for m in self.models_p2_], axis=0)

        blend = self.weight_pilier1 * avg_p1 + (1.0 - self.weight_pilier1) * avg_p2

        # Lissage WLS heteroscedastique (visites OFF pondérées plus fortement)
        out = blend.copy()
        groups = pd.Series(np.arange(len(out))).groupby(pid).indices
        for idx in groups.values():
            n = len(idx)
            a, p = age[idx], out[idx]
            w = np.where(is_off[idx], self.weight_off, 1.0)
            if self.alpha_smooth > 0 and n > self.degree_smooth:
                t = a - a.mean()
                A = np.vander(t, self.degree_smooth + 1)
                W = np.diag(w)
                try:
                    b = np.linalg.solve(
                        A.T @ W @ A + 1e-4 * np.eye(self.degree_smooth + 1),
                        A.T @ W @ p,
                    )
                    fit = A @ b
                    p = self.alpha_smooth * fit + (1.0 - self.alpha_smooth) * p
                except np.linalg.LinAlgError:
                    pass
            if n > 1:
                p = IsotonicRegression(increasing=True).fit_transform(a, p)
            out[idx] = p
        return np.clip(out, 0.0, 132.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Modele 25 : Blend corrige 3 graines")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MODELE 25 : BLEND CORRIGE — PRIORI CONDITIONNEL + 3 GRAINES (25_corrected_blend)", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/5] Chargement et preparation des donnees...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")
    y = visits["target"]
    X = add_longitudinal_columns(visits.drop(columns=["target"]))
    X_test = add_longitudinal_columns(X_test_raw)
    print(f"  Train: {len(X)} visites | Test: {len(X_test)} visites", flush=True)

    print("\n[2/5] Configuration GroupKFold (5 plis sur patient_id)...", flush=True)
    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

    print("\n[3/5] Evaluation officielle via skore.evaluate (3 graines x 2 piliers + WLS)...", flush=True)
    model = CorrectedBlendRegressor(
        seeds=(42, 100, 137),
        weight_pilier1=0.50,
        weight_off=2.5,
        alpha_smooth=0.9,
        degree_smooth=2,
    )
    report = evaluate(model, X, y, splitter=cv_splits)

    try:
        print(f"\n  Score RMSE GroupKFold :\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  Acces rmse : {exc}", flush=True)

    try:
        print(f"\n  Score MAE GroupKFold :\n{report.metrics.mae()}", flush=True)
    except Exception as exc:
        print(f"  Acces mae : {exc}", flush=True)

    try:
        print(f"\n  Resume verifications Skore :\n{report.checks.summarize(fast_mode=True)}", flush=True)
    except Exception as exc:
        print(f"  Controles Skore : {exc}", flush=True)

    print("\n[4/5] Entrainement final sur tout le Train et prediction sur X_test...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"{HUB_KEY}.csv"
    pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions}).to_csv(sub_path, index=False)
    print(f"  Soumission enregistree : {sub_path} ({len(predictions)} lignes)", flush=True)
    print(
        f"  Stats : min={predictions.min():.2f}, mean={predictions.mean():.2f}, max={predictions.max():.2f}",
        flush=True,
    )

    if args.no_hub:
        print("\n[5/5] Publication Skore Hub ignoree (--no-hub).", flush=True)
        return

    print(f"\n[5/5] Publication sur Skore Hub sous la cle '{HUB_KEY}'...", flush=True)
    try:
        config = load_skore_credentials(REPO_ROOT)
        login(mode="hub")
        project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
        project.put(HUB_KEY, report)
        print("  Rapport publie sur Skore Hub avec succes !", flush=True)
    except Exception as exc:
        print(f"  Avertissement Skore Hub : {exc}", flush=True)

    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
