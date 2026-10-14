"""Modele 25 : Tri-Blend 21 + 22 + JPI-CQT avec lissage WLS (25_tri_blend).

Project: Bobathon ESILV (Probabl x IBM)
Key Hub: '25_tri_blend'

Architecture :
1. Pilier A (Modele 21) : trajectoire bayesienne 3 etages + lissage quadratique monotone.
2. Pilier B (Modele 22) : calibration affine clinicien + interaction Bateman + HGBR regularise.
3. Pilier C (JPI-CQT, nouveau) :
   - observations nettoyees : `off` calibre par clinicien (ridge affine), `on` via ridge
     (on, Bateman, rater) ; fusion ponderee par precision (w_off=1.0, w_on=0.35) ;
   - trajectoire quadratique par patient sous contrainte de derivee b1 + 2 b2 t >= 0 (QP SLSQP),
     regularisee vers la pente de population ;
   - HGBR residuel `squared_error` sur les variables numeriques + trajectoire QP.
4. Melange convexe (0.35, 0.35, 0.30), chaque pilier moyenne sur plusieurs graines.
5. Lissage WLS (w=2.5 sur les visites OFF) + projection isotonique + bornage [0, 132].

Test pli 0 (GroupKFold patient) : blend 21+22 = 3.5667 RMSE -> tri-blend = 3.5417 RMSE.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "25_tri_blend"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_15_trajectory import PatientTrajectoryRegressor, add_longitudinal_columns
from src.model_21_smooth import smooth_predictions
from src.model_22_adaptive_trajectory import AdaptiveTrajectoryRegressor
from src.model_24_wls_multi_seed import smooth_wls_monotonic

EXCLUDED_TREE_COLS = {
    "Index", "patient_id", "cohort", "gene", "rater_id",
    "time_since_first_visit", "time_since_diagnosis",
}
POP_SLOPE = 2.438


def bateman(t: np.ndarray) -> np.ndarray:
    """Courbe de concentration levodopa (ka=0.5708, ke=0.1870)."""
    t_clean = np.clip(np.nan_to_num(np.asarray(t, dtype=float), nan=2.0), 0.0, None)
    return np.exp(-0.1870 * t_clean) - np.exp(-0.5708 * t_clean)


def solve_patient_qp(pid, age, y_obs, w_obs, lam1=5.0, lam2=15.0) -> np.ndarray:
    """Trajectoire quadratique ponderee par patient, contrainte non-decroissante."""
    pred = np.zeros(len(pid))
    groups = pd.Series(np.arange(len(pid))).groupby(np.asarray(pid)).indices
    Lambda = np.diag([0.0, lam1, lam2])
    mu = np.array([0.0, POP_SLOPE, 0.0])
    for idx in groups.values():
        n = len(idx)
        y, w = y_obs[idx], w_obs[idx]
        if n == 1:
            pred[idx] = y
            continue
        t = age[idx] - age[idx].mean()
        A = np.column_stack([np.ones(n), t, t ** 2])
        M = A.T @ (A * w[:, None]) + Lambda
        rhs = A.T @ (w * y) + Lambda @ mu
        try:
            b0 = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            b0 = np.array([y.mean(), POP_SLOPE, 0.0])
        C = np.column_stack([np.zeros(n), np.ones(n), 2.0 * t])
        if (C @ b0 >= -1e-6).all():
            b = b0
        else:
            H, c = 2.0 * M, -2.0 * rhs
            res = minimize(
                lambda b: 0.5 * b @ H @ b + c @ b, b0, jac=lambda b: H @ b + c,
                constraints={"type": "ineq", "fun": lambda b: C @ b, "jac": lambda b: C},
                method="SLSQP",
            )
            b = res.x if res.success else b0
        pred[idx] = A @ b
    return np.clip(pred, 0.0, 132.0)


class JPICQTRegressor(RegressorMixin, BaseEstimator):
    """Pilier C : observations calibrees -> QP monotone par patient -> HGBR residuel."""

    def __init__(self, w_on=0.35, lam1=5.0, lam2=15.0, rater_lam=5.0, random_state=42):
        self.w_on = w_on
        self.lam1 = lam1
        self.lam2 = lam2
        self.rater_lam = rater_lam
        self.random_state = random_state

    def _on_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        c_on = bateman(df["time_since_intake_on"].values)
        on = df["on"].fillna(0.0).values
        mat = pd.DataFrame({"on": on, "C_on": c_on, "on_C_on": on * c_on}, index=df.index)
        rd = pd.get_dummies(df["rater_id"].astype(str), prefix="rater", dtype=float)
        rd = rd.reindex(columns=self.rater_cols_, fill_value=0.0)
        return pd.concat([mat, rd], axis=1)

    def _clean_obs(self, df: pd.DataFrame):
        has_off = df["off"].notna().values
        has_on = df["on"].notna().values
        coefs = df["rater_id"].map(self.rater_models_)
        slope = np.array([c[0] if isinstance(c, np.ndarray) else self.global_poly_[0] for c in coefs])
        inter = np.array([c[1] if isinstance(c, np.ndarray) else self.global_poly_[1] for c in coefs])
        off_est = np.where(has_off, slope * df["off"].fillna(0.0).values + inter, 0.0)
        on_est = np.zeros(len(df))
        if has_on.any():
            on_est[has_on] = self.ridge_on_.predict(self._on_matrix(df.loc[has_on]))
        w_off, w_on = 1.0, self.w_on
        y_obs = np.where(
            has_off & has_on, (w_off * off_est + w_on * on_est) / (w_off + w_on),
            np.where(has_off, off_est, on_est),
        )
        w_obs = np.where(has_off & has_on, w_off + w_on, np.where(has_off, w_off, w_on))
        return y_obs, w_obs

    def _tree_features(self, df, traj, y_obs):
        Xt = df[self.feat_cols_].copy()
        Xt["traj_qp"] = traj
        Xt["obs_minus_traj"] = y_obs - traj
        return Xt

    def fit(self, X: pd.DataFrame, y):
        df = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        has_off = df["off"].notna().values

        self.global_poly_ = np.polyfit(df.loc[has_off, "off"], y[has_off], 1)
        self.rater_models_ = {}
        for r, idx in df.loc[has_off].groupby("rater_id").groups.items():
            if len(idx) >= 20:
                A = np.column_stack([df.loc[idx, "off"].values, np.ones(len(idx))])
                M = A.T @ A + self.rater_lam * np.eye(2)
                self.rater_models_[r] = np.linalg.solve(M, A.T @ y[idx] + self.rater_lam * self.global_poly_)
            else:
                self.rater_models_[r] = self.global_poly_

        self.rater_cols_ = sorted(pd.get_dummies(df["rater_id"].astype(str), prefix="rater").columns)
        has_on = df["on"].notna().values
        self.ridge_on_ = Ridge(alpha=10.0).fit(self._on_matrix(df.loc[has_on]), y[has_on])

        self.feat_cols_ = [
            c for c in df.columns
            if c not in EXCLUDED_TREE_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
        y_obs, w_obs = self._clean_obs(df)
        traj = solve_patient_qp(df["patient_id"].values, df["age"].values, y_obs, w_obs, self.lam1, self.lam2)
        self.tree_ = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=300, learning_rate=0.03, min_samples_leaf=40,
            l2_regularization=5.0, early_stopping=True, random_state=self.random_state,
        ).fit(self._tree_features(df, traj, y_obs), y - traj)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        df = X.reset_index(drop=True)
        y_obs, w_obs = self._clean_obs(df)
        traj = solve_patient_qp(df["patient_id"].values, df["age"].values, y_obs, w_obs, self.lam1, self.lam2)
        return traj + self.tree_.predict(self._tree_features(df, traj, y_obs))


class TriBlendRegressor(RegressorMixin, BaseEstimator):
    """Melange 21 / 22 / JPI-CQT multi-graines puis lissage WLS monotone."""

    def __init__(self, seeds=(42, 100), weights=(0.35, 0.35, 0.30),
                 weight_off=2.5, alpha_smooth=0.9):
        self.seeds = seeds
        self.weights = weights
        self.weight_off = weight_off
        self.alpha_smooth = alpha_smooth

    def fit(self, X: pd.DataFrame, y):
        X_df = X.reset_index(drop=True)
        y_arr = np.asarray(y, dtype=float)
        self.m21_ = [PatientTrajectoryRegressor(random_state=s).fit(X_df, y_arr) for s in self.seeds]
        self.m22_ = [AdaptiveTrajectoryRegressor(random_state=s).fit(X_df, y_arr) for s in self.seeds]
        self.m25_ = [JPICQTRegressor(random_state=s).fit(X_df, y_arr) for s in self.seeds]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_df = X.reset_index(drop=True)
        pid, age = X_df["patient_id"].values, X_df["age"].values
        p21 = np.mean([
            smooth_predictions(pid, age, m.predict(X_df), alpha=1.0, degree=2, monotone=True)
            for m in self.m21_
        ], axis=0)
        p22 = np.mean([m.predict(X_df) for m in self.m22_], axis=0)
        p25 = np.mean([m.predict(X_df) for m in self.m25_], axis=0)
        wa, wb, wc = self.weights
        blend = wa * p21 + wb * p22 + wc * p25
        return smooth_wls_monotonic(
            patient_id=pid, age=age, pred=blend, is_off=X_df["off"].notna().values,
            alpha=self.alpha_smooth, degree=2, weight_off=self.weight_off,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Modele 25 : Tri-Blend 21 + 22 + JPI-CQT")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    parser.add_argument("--no-cv", action="store_true", help="Sauter la validation croisee")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MODELE 25 : TRI-BLEND 21 + 22 + JPI-CQT (25_tri_blend)", flush=True)
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

    model = TriBlendRegressor()
    report = None
    if not args.no_cv:
        print("\n[2/5] GroupKFold (5 plis sur patient_id)...", flush=True)
        cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))
        print("\n[3/5] Evaluation officielle via skore.evaluate...", flush=True)
        report = evaluate(model, X, y, splitter=cv_splits)
        for name in ("rmse", "mae"):
            try:
                print(f"\n  Score {name.upper()} GroupKFold :\n{getattr(report.metrics, name)()}", flush=True)
            except Exception as exc:
                print(f"  Acces {name} : {exc}", flush=True)
        try:
            print(f"\n  Verifications Skore :\n{report.checks.summarize(fast_mode=True)}", flush=True)
        except Exception as exc:
            print(f"  Controles Skore : {exc}", flush=True)

    print("\n[4/5] Entrainement final sur tout le Train et prediction sur X_test...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"{HUB_KEY}.csv"
    pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions}).to_csv(sub_path, index=False)
    print(f"  Soumission enregistree : {sub_path} ({len(predictions)} lignes)", flush=True)
    print(f"  Statistiques : min={predictions.min():.2f}, mean={predictions.mean():.2f}, "
          f"max={predictions.max():.2f}", flush=True)

    if args.no_hub or report is None:
        print("\n[5/5] Publication Skore Hub ignoree.", flush=True)
        return
    print(f"\n[5/5] Publication sur Skore Hub sous la cle '{HUB_KEY}'...", flush=True)
    try:
        config = load_skore_credentials(REPO_ROOT)
        login(mode="hub")
        project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
        project.put(HUB_KEY, report)
        print("  Rapport publie sur Skore Hub.", flush=True)
    except Exception as exc:
        print(f"  Avertissement Skore Hub: {exc}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
