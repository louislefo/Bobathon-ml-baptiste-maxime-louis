"""Modele 22 : Debiaisage Affine Clinicien + Interaction PKPD + Trajectoire Adaptee et Lissage.

Project: Bobathon ESILV (Probabl x IBM)
Key Hub: '22_adaptive_trajectory'

Architecture :
1. Etage A (Debiaisage de visite) :
   - Calibration affine par evaluateur pour le signal OFF (target ~ alpha_r * off + beta_r).
   - Inversion pharmacodynamique Levodopa regularisee pour le signal ON via interaction
     PKPD (on * C(t)) et indicatrices cliniciens.
   - Construction d'un substitut continu debiaise de severite (surrogate_calib).
2. Etage B (Dynamique temporelle et interpolations intra-patient) :
   - Pente de progression, projection lineaire et interpolations chronologiques (off_age_interp, on_age_interp).
   - Dynamique locale (surr_prev, surr_next).
3. Etage C (Regression regularisee anti-overfitting) :
   - HistGradientBoostingRegressor bride (l2_regularization=5.0, min_samples_leaf=40, early_stopping=True).
4. Etage D (Post-traitement analytique) :
   - Ajustement de trajectoire parabolique par patient (degre 2, alpha=0.9).
   - Forcage de la monotonie non-decroissante stricte via IsotonicRegression.
   - Bornage clinique strict [0.0, 132.0].
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "22_adaptive_trajectory"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials


def compute_bateman_c(t: np.ndarray) -> np.ndarray:
    """Concentration cinetique normalisee de Levodopa (Bateman bicompartimentale)."""
    t_clean = np.clip(np.nan_to_num(t, nan=2.0), 0.0, None)
    return np.exp(-0.1870 * t_clean) - np.exp(-0.5708 * t_clean)


def smooth_and_monotonic(patient_id, age, pred, alpha=0.9, degree=2):
    """Ajustement parabolique par patient suivi d'une projection isotonique stricte."""
    out = np.asarray(pred, dtype=float).copy()
    age = np.asarray(age, dtype=float)
    groups = pd.Series(np.arange(len(out))).groupby(np.asarray(patient_id)).indices
    for idx in groups.values():
        a, p = age[idx], out[idx]
        if alpha > 0 and len(idx) > degree:
            t = a - a.mean()
            fit = np.polyval(np.polyfit(t, p, degree), t)
            p = alpha * fit + (1 - alpha) * p
        if len(idx) > 1:
            p = IsotonicRegression(increasing=True).fit_transform(a, p)
        out[idx] = p
    return np.clip(out, 0.0, 132.0)


class AdaptiveTrajectoryRegressor(RegressorMixin, BaseEstimator):
    """Pipeline complet Model 22 respectant l'interface Scikit-learn sans fuite inter-patients."""

    def __init__(
        self,
        alpha_smooth: float = 0.9,
        degree_smooth: int = 2,
        max_iter: int = 400,
        learning_rate: float = 0.03,
        min_samples_leaf: int = 40,
        l2_regularization: float = 5.0,
        random_state: int = 42,
    ):
        self.alpha_smooth = alpha_smooth
        self.degree_smooth = degree_smooth
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def _fit_rater_models(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Apprend la calibration affine des 50 evaluateurs et le debiaisage ON."""
        # 1. Calibration affine OFF
        has_off = X["off"].notna().values
        off_vals = X.loc[has_off, "off"].values
        y_off = y[has_off]

        self.global_poly_off_ = np.polyfit(off_vals, y_off, 1)

        self.rater_models_off_ = {}
        for r, g in X[has_off].groupby("rater_id"):
            idx_g = g.index.values
            y_g = y[has_off][np.isin(np.where(has_off)[0], idx_g)]
            if len(g) >= 20:
                A = np.column_stack([g["off"].values, np.ones(len(g))])
                lam = 5.0
                M = A.T @ A + lam * np.eye(2)
                self.rater_models_off_[r] = np.linalg.solve(M, A.T @ y_g + lam * self.global_poly_off_)
            else:
                self.rater_models_off_[r] = self.global_poly_off_

        # 2. Calibration interaction ON
        self.rater_cols_ = sorted(list(pd.get_dummies(X["rater_id"], prefix="rater").columns))
        has_on = X["on"].notna().values

        X_on_sub = X[has_on].copy()
        y_on = y[has_on]

        C_on = compute_bateman_c(X_on_sub["time_since_intake_on"].values)
        on_vals = X_on_sub["on"].fillna(0.0).values
        mat_on = pd.DataFrame(
            {"on": on_vals, "C_on": C_on, "on_C_on": on_vals * C_on},
            index=X_on_sub.index,
        )
        rd_on = pd.get_dummies(X_on_sub["rater_id"], prefix="rater", dtype=float)
        for c in self.rater_cols_:
            if c not in rd_on.columns:
                rd_on[c] = 0.0
        X_on_mat = pd.concat([mat_on, rd_on[self.rater_cols_]], axis=1)

        self.ridge_on_ = Ridge(alpha=10.0).fit(X_on_mat, y_on)

    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Construit les variables longitudinales et temporelles a partir des debiaisages appris."""
        d = X.copy().sort_values(["patient_id", "age"])

        has_off = d["off"].notna().values
        has_on = d["on"].notna().values

        # Calcul OFF calibre
        off_est = np.zeros(len(d))
        for i, (_, row) in enumerate(d.iterrows()):
            if has_off[i]:
                coefs = self.rater_models_off_.get(row["rater_id"], self.global_poly_off_)
                off_est[i] = coefs[0] * row["off"] + coefs[1]

        # Calcul ON calibre
        on_est = np.zeros(len(d))
        if has_on.any():
            idx_on = np.where(has_on)[0]
            d_on = d.iloc[idx_on]
            C_on = compute_bateman_c(d_on["time_since_intake_on"].values)
            on_vals = d_on["on"].fillna(0.0).values
            mat_on = pd.DataFrame(
                {"on": on_vals, "C_on": C_on, "on_C_on": on_vals * C_on},
                index=d_on.index,
            )
            rd_on = pd.get_dummies(d_on["rater_id"], prefix="rater", dtype=float)
            for c in self.rater_cols_:
                if c not in rd_on.columns:
                    rd_on[c] = 0.0
            X_on_mat = pd.concat([mat_on, rd_on[self.rater_cols_]], axis=1)
            on_est[idx_on] = self.ridge_on_.predict(X_on_mat)

        # Fusion en surrogate de visite
        surrogate = np.full(len(d), np.nan)
        for i in range(len(d)):
            if has_off[i] and has_on[i]:
                surrogate[i] = 0.70 * off_est[i] + 0.30 * on_est[i]
            elif has_off[i]:
                surrogate[i] = off_est[i]
            elif has_on[i]:
                surrogate[i] = on_est[i]

        d["surrogate"] = surrogate
        d["off_calib"] = np.where(has_off, off_est, np.nan)
        d["on_calib"] = np.where(has_on, on_est, np.nan)

        # Variables temporelles
        d["time_since_diagnosis"] = d["age"] - d["age_at_diagnosis"]
        d["visit_number"] = d.groupby("patient_id")["age"].rank(method="first").astype(int)
        d["n_visits_patient"] = d.groupby("patient_id")["age"].transform("count")
        d["visit_ratio"] = d["visit_number"] / d["n_visits_patient"]
        min_age = d.groupby("patient_id")["age"].transform("min")
        d["time_since_first_visit"] = d["age"] - min_age

        d["mean_surrogate_patient"] = d.groupby("patient_id")["surrogate"].transform("mean")
        d["std_surrogate_patient"] = d.groupby("patient_id")["surrogate"].transform("std")

        # Dynamique intra-patient
        def _compute_patient_traj(g):
            x = g["age"].values
            y_surr = g["surrogate"].values
            n = len(x)
            if n >= 2:
                poly = np.polyfit(x, y_surr, 1)
                g["patient_slope"] = poly[0]
                g["patient_linear_surrogate"] = poly[0] * x + poly[1]
            else:
                g["patient_slope"] = 2.44
                g["patient_linear_surrogate"] = y_surr[0]

            val_off = g["off_calib"].notnull()
            n_off = val_off.sum()
            if n_off >= 2:
                g["off_age_interp"] = np.interp(g["age"], g.loc[val_off, "age"], g.loc[val_off, "off_calib"])
            elif n_off == 1:
                g["off_age_interp"] = g.loc[val_off, "off_calib"].iloc[0]
            else:
                g["off_age_interp"] = np.nan

            val_on = g["on_calib"].notnull()
            n_on = val_on.sum()
            if n_on >= 2:
                g["on_age_interp"] = np.interp(g["age"], g.loc[val_on, "age"], g.loc[val_on, "on_calib"])
            elif n_on == 1:
                g["on_age_interp"] = g.loc[val_on, "on_calib"].iloc[0]
            else:
                g["on_age_interp"] = np.nan

            g["surr_prev"] = g["surrogate"].ffill().shift(1)
            g["surr_next"] = g["surrogate"].bfill().shift(-1)
            return g

        processed = d.groupby("patient_id", group_keys=False).apply(_compute_patient_traj, include_groups=False)
        for col in ["patient_slope", "patient_linear_surrogate", "off_age_interp", "on_age_interp", "surr_prev", "surr_next"]:
            d[col] = processed[col]

        return d.sort_index()

    def fit(self, X: pd.DataFrame, y):
        X_df = X.reset_index(drop=True).copy()
        y_arr = np.asarray(y, dtype=float)

        self._fit_rater_models(X_df, y_arr)
        feats_df = self._transform_features(X_df)

        self.feature_cols_ = [
            "age", "age_at_diagnosis", "time_since_diagnosis", "sexM", "ledd",
            "surrogate", "off_calib", "on_calib", "patient_slope", "patient_linear_surrogate",
            "off_age_interp", "on_age_interp", "surr_prev", "surr_next",
            "mean_surrogate_patient", "visit_number", "n_visits_patient"
        ]

        self.model_ = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=self.random_state,
        ).fit(feats_df[self.feature_cols_], y_arr)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_df = X.reset_index(drop=True).copy()
        feats_df = self._transform_features(X_df)
        pred_raw = self.model_.predict(feats_df[self.feature_cols_])

        # Post-traitement parabolique + monotonie stricte
        pred_final = smooth_and_monotonic(
            feats_df["patient_id"].values,
            feats_df["age"].values,
            pred_raw,
            alpha=self.alpha_smooth,
            degree=self.degree_smooth,
        )
        return pred_final


def main() -> None:
    parser = argparse.ArgumentParser(description="Modele 22 : Trajectoire Adaptee et Calibree")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MODELE 22 : TRAJECTOIRE ADAPTEE, CALIBREE ET MONOTONE (22_adaptive_trajectory)", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/5] Chargement des donnees...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")
    y = visits["target"]
    X = visits.drop(columns=["target"])

    print(f"  Train: {len(X)} visites | Test: {len(X_test_raw)} visites", flush=True)

    print("\n[2/5] Configuration GroupKFold (5 plis sur patient_id)...", flush=True)
    groups = X["patient_id"]
    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

    print("\n[3/5] Evaluation officielle via skore.evaluate...", flush=True)
    model = AdaptiveTrajectoryRegressor(
        alpha_smooth=0.9,
        degree_smooth=2,
        max_iter=400,
        learning_rate=0.03,
        min_samples_leaf=40,
        l2_regularization=5.0,
        random_state=42,
    )

    report = evaluate(model, X, y, splitter=cv_splits)

    try:
        rmse_val = report.metrics.rmse()
        print(f"\n  Score RMSE GroupKFold :\n{rmse_val}", flush=True)
    except Exception as exc:
        print(f"  Acces rmse : {exc}", flush=True)

    try:
        mae_val = report.metrics.mae()
        print(f"\n  Score MAE GroupKFold :\n{mae_val}", flush=True)
    except Exception as exc:
        print(f"  Acces mae : {exc}", flush=True)

    try:
        print(f"\n  Resume des verifications Skore :\n{report.checks.summarize(fast_mode=True)}", flush=True)
    except Exception as exc:
        print(f"  Controles Skore : {exc}", flush=True)

    print("\n[4/5] Entrainement final sur l'integralite du Train et prediction sur X_test...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test_raw)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"{HUB_KEY}.csv"
    pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions}).to_csv(sub_path, index=False)
    print(f"  Soumission enregistree : {sub_path} ({len(predictions)} lignes)", flush=True)
    print(f"  Statistiques : min={predictions.min():.2f}, mean={predictions.mean():.2f}, max={predictions.max():.2f}", flush=True)

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
        print(f"  Avertissement Skore Hub: {exc}", flush=True)

    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
