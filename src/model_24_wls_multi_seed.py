"""Modele 24 : Super Blend Multi-Graines avec Lissage Pondere WLS (24_wls_multi_seed).

Project: Bobathon ESILV (Probabl x IBM)
Key Hub: '24_wls_multi_seed'

Innovations majeures :
1. Ensembling Multi-Graines (Bagging orthogonal 2x2) :
   - Pilier 1 (Trajectoire bayesienne 3 etages) entraine sur 2 graines stochastiques distinctes (42, 100).
   - Pilier 2 (Surrogate calibre clinicien + interaction PKPD) entraine sur 2 graines stochastiques distinctes (42, 100).
   - Moyennage des 4 estimateurs pour eliminer la variance stochastique des arbres de decision.
2. Lissage Intra-Patient par Moindres Carres Ponderes (WLS) :
   - Les visites avec mesure OFF observee sont nettement plus fiables (RMSE ~2.40 vs ~4.25 sur ON seul).
   - Ajustement parabolique ponderant les visites OFF avec un poids w = 2.5 vs w = 1.0 sur ON seul.
3. Forcage strict de monotonie temporelle non-decroissante (IsotonicRegression) et bornage clinique [0.0, 132.0].
4. Validation croisee officielle GroupKFold (5 plis par patient) et publication sur Skore Hub.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "24_wls_multi_seed"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_15_trajectory import PatientTrajectoryRegressor, add_longitudinal_columns
from src.model_21_smooth import smooth_predictions
from src.model_22_adaptive_trajectory import AdaptiveTrajectoryRegressor


def smooth_wls_monotonic(patient_id, age, pred, is_off, alpha=0.9, degree=2, weight_off=2.5):
    """Lissage par moindres carres ponderes (WLS) suivi d'une projection monotone isotonique."""
    out = np.asarray(pred, dtype=float).copy()
    age = np.asarray(age, dtype=float)
    is_off = np.asarray(is_off, dtype=bool)
    groups = pd.Series(np.arange(len(out))).groupby(np.asarray(patient_id)).indices
    for idx in groups.values():
        n = len(idx)
        a, p = age[idx], out[idx]
        w = np.where(is_off[idx], weight_off, 1.0)
        if alpha > 0 and n > degree:
            t = a - a.mean()
            A = np.vander(t, degree + 1)
            W = np.diag(w)
            try:
                b = np.linalg.solve(A.T @ W @ A + 1e-4 * np.eye(degree + 1), A.T @ W @ p)
                fit = A @ b
                p = alpha * fit + (1.0 - alpha) * p
            except np.linalg.LinAlgError:
                pass
        if n > 1:
            p = IsotonicRegression(increasing=True).fit_transform(a, p)
        out[idx] = p
    return np.clip(out, 0.0, 132.0)


class SuperBlendWLSRegressor(RegressorMixin, BaseEstimator):
    """Meta-estimateur multi-graines avec lissage heteroscedastique WLS."""

    def __init__(
        self,
        seeds: tuple[int, ...] = (42, 100),
        weight_21: float = 0.50,
        weight_off: float = 2.5,
        alpha_smooth: float = 0.9,
        degree_smooth: int = 2,
    ):
        self.seeds = seeds
        self.weight_21 = weight_21
        self.weight_off = weight_off
        self.alpha_smooth = alpha_smooth
        self.degree_smooth = degree_smooth

    def fit(self, X: pd.DataFrame, y):
        X_df = X.reset_index(drop=True).copy()
        y_arr = np.asarray(y, dtype=float)

        self.models_21_ = []
        self.models_22_ = []

        for seed in self.seeds:
            m21 = PatientTrajectoryRegressor(random_state=seed).fit(X_df, y_arr)
            self.models_21_.append(m21)
            m22 = AdaptiveTrajectoryRegressor(random_state=seed).fit(X_df, y_arr)
            self.models_22_.append(m22)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_df = X.reset_index(drop=True).copy()
        pid = X_df["patient_id"].values
        age = X_df["age"].values
        is_off = X_df["off"].notna().values

        # Predictions moyennees pour le Pilier 1 (avec lissage 21)
        preds_21 = []
        for m21 in self.models_21_:
            raw = m21.predict(X_df)
            sm = smooth_predictions(pid, age, raw, alpha=1.0, degree=2, monotone=True)
            preds_21.append(sm)
        pred_21_avg = np.mean(preds_21, axis=0)

        # Predictions moyennees pour le Pilier 2
        preds_22 = []
        for m22 in self.models_22_:
            preds_22.append(m22.predict(X_df))
        pred_22_avg = np.mean(preds_22, axis=0)

        # Melange convexe
        w = self.weight_21
        blend = w * pred_21_avg + (1.0 - w) * pred_22_avg

        # Lissage WLS privilegiant les visites avec OFF observe
        final_pred = smooth_wls_monotonic(
            patient_id=pid,
            age=age,
            pred=blend,
            is_off=is_off,
            alpha=self.alpha_smooth,
            degree=self.degree_smooth,
            weight_off=self.weight_off,
        )
        return final_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Modele 24 : Super Blend WLS Multi-Graines")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MODELE 24 : SUPER BLEND WLS MULTI-GRAINES (24_wls_multi_seed)", flush=True)
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

    print("\n[3/5] Evaluation officielle via skore.evaluate (4 modeles 2x2 + WLS)...", flush=True)
    model = SuperBlendWLSRegressor(
        seeds=(42, 100),
        weight_21=0.50,
        weight_off=2.5,
        alpha_smooth=0.9,
        degree_smooth=2,
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

    print("\n[4/5] Entrainement final sur tout le Train et prediction sur X_test...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test)

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
