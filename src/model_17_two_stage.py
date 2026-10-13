"""Two-stage model: biological progression + rater residual (17_two_stage).

Sub-task: Architecture deux etapes bio + clinicien (Sous-tache 17)
Project: Bobathon ESILV (Probabl x IBM)

Architecture :
Stage 1 — Modele biologique (sans rater_id, sans scores moteurs bruts) :
  Features : age, age_at_diagnosis, time_since_diagnosis, ledd, visit_number,
             lovo_trajectory_slope, lovo_trajectory_pred, mean_ledd_patient, etc.
  Predit : composante de progression biologique pure du score OFF.

Stage 2 — Modele de residu clinicien :
  Features : rater_id, off, on, off_corrected, bateman_off, diff_from_lovo_pred, ...
  Predit : residu (target - y_hat_bio), i.e., le biais du clinicien + signal motor.

Prediction finale = np.clip(y_hat_bio + y_hat_resid, 0, 132).

Justification :
- CONTEXT.md : correlation des biais clinicien r=0.975 entre OFF et ON -> isole bien
- La progression biologique est monotone (94.1% variance intra-patient expliquee par age)
- Le Stage 2 capture le signal contextuel de la visite (etat du patient CE JOUR)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from skrub import tabular_pipeline
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "17_two_stage"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_15_bateman_off import prepare_lovo_features_bateman_off

# Features strictement biologiques (pas de rater_id, pas de scores moteurs bruts)
BIO_FEATURES = [
    "age", "age_at_diagnosis", "time_since_diagnosis", "ledd",
    "visit_number", "n_visits_patient", "visit_ratio",
    "mean_ledd_patient",
    "lovo_trajectory_slope", "lovo_trajectory_pred", "lovo_mean_surrogate",
    "mean_surrogate_patient",
]

# Features cliniciennes et motrices (biais rater + etat ce jour)
CLINIC_FEATURES = [
    "rater_id",
    "off", "on", "on_adjusted", "bateman_on",
    "off_corrected", "bateman_off", "time_since_intake_off", "time_since_intake_on",
    "off_minus_on", "diff_from_lovo_pred",
    "mean_off_patient", "mean_on_patient",
    "diff_from_mean_off", "diff_from_mean_on",
    "off_prev", "off_next",
    "off_age_interp", "diff_from_interp_off",
    "std_off_patient", "std_on_patient",
]


def select_cols(X: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """Selectionne uniquement les colonnes disponibles dans feature_list."""
    available = [c for c in feature_list if c in X.columns]
    return X[available]


def build_bio_model() -> object:
    """HGBR pour la composante biologique (peu de features, besoin de generalisation)."""
    return tabular_pipeline(
        HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.025,
            min_samples_leaf=20,
            max_leaf_nodes=25,
            l2_regularization=2.0,
            random_state=42,
        )
    )


def build_clinic_model() -> object:
    """HGBR pour le residu clinicien (features motrices, biais rater)."""
    return tabular_pipeline(
        HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.03,
            min_samples_leaf=15,
            max_leaf_nodes=31,
            l2_regularization=1.5,
            random_state=0,
        )
    )


class TwoStageRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        Xb = select_cols(X, BIO_FEATURES)
        Xc = select_cols(X, CLINIC_FEATURES)
        self.bio_ = build_bio_model()
        self.bio_.fit(Xb, y)
        y_bio = self.bio_.predict(Xb)
        Xc_aug = Xc.copy()
        Xc_aug["bio_pred"] = y_bio
        self.clinic_ = build_clinic_model()
        self.clinic_.fit(Xc_aug, np.array(y) - y_bio)
        return self

    def predict(self, X):
        Xb = select_cols(X, BIO_FEATURES)
        Xc = select_cols(X, CLINIC_FEATURES)
        y_bio = self.bio_.predict(Xb)
        Xc_aug = Xc.copy()
        Xc_aug["bio_pred"] = y_bio
        return np.clip(y_bio + self.clinic_.predict(Xc_aug), 0.0, 132.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture deux etapes bio + clinicien (17_two_stage)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MODELE DEUX ETAPES BIO + CLINICIEN (17_two_stage)", flush=True)
    print("Sous-tache 17 - Separation progression biologique et biais clinicien", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement
    print("\n[1/5] Chargement et extraction des features LOVO + Bateman OFF...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")
    patient_ids = visits["patient_id"]
    y = visits["target"]

    X_all = prepare_lovo_features_bateman_off(visits)
    X_test_all = prepare_lovo_features_bateman_off(X_test_raw)

    X_bio = select_cols(X_all, BIO_FEATURES)
    X_clinic = select_cols(X_all, CLINIC_FEATURES)
    X_bio_test = select_cols(X_test_all, BIO_FEATURES)
    X_clinic_test = select_cols(X_test_all, CLINIC_FEATURES)

    print(f"  Features biologiques : {X_bio.shape[1]} colonnes", flush=True)
    print(f"  Features cliniciennes : {X_clinic.shape[1]} colonnes", flush=True)

    # 2. GroupKFold
    print("\n[2/5] CV deux etapes avec GroupKFold (5 plis)...", flush=True)
    cv_grouped = GroupKFold(n_splits=5)
    cv_splits = list(cv_grouped.split(X_all, y, groups=patient_ids))

    # 3. Evaluation deux etapes avec CV manuelle
    oof_bio = np.zeros(len(y))
    oof_final = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_bio_tr = X_bio.iloc[train_idx]
        X_bio_va = X_bio.iloc[val_idx]
        X_cl_tr = X_clinic.iloc[train_idx]
        X_cl_va = X_clinic.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_va = y.iloc[val_idx]

        # Stage 1 : fit bio sur train, predire val
        m_bio = build_bio_model()
        m_bio.fit(X_bio_tr, y_tr)
        y_hat_bio_tr = m_bio.predict(X_bio_tr)
        y_hat_bio_va = m_bio.predict(X_bio_va)

        # Stage 2 : fit residu clinicien sur train (residu OOF interne)
        resid_tr = y_tr.values - y_hat_bio_tr

        # Enrichir les features cliniciennes avec la prediction bio (signal pour le residu)
        X_cl_tr_aug = X_cl_tr.copy()
        X_cl_va_aug = X_cl_va.copy()
        X_cl_tr_aug["bio_pred"] = y_hat_bio_tr
        X_cl_va_aug["bio_pred"] = y_hat_bio_va

        m_clinic = build_clinic_model()
        m_clinic.fit(X_cl_tr_aug, resid_tr)
        resid_va = m_clinic.predict(X_cl_va_aug)

        oof_bio[val_idx] = y_hat_bio_va
        oof_final[val_idx] = np.clip(y_hat_bio_va + resid_va, 0.0, 132.0)

        rmse_bio = np.sqrt(np.mean((y_hat_bio_va - y_va.values) ** 2))
        rmse_final = np.sqrt(np.mean((oof_final[val_idx] - y_va.values) ** 2))
        print(f"  Fold {fold_idx + 1}/5 — RMSE bio: {rmse_bio:.4f}, RMSE final (bio+clinic): {rmse_final:.4f}", flush=True)

    rmse_global_bio = np.sqrt(np.mean((oof_bio - y.values) ** 2))
    rmse_global_final = np.sqrt(np.mean((oof_final - y.values) ** 2))
    print(f"\n  RMSE Stage 1 (bio seul) : {rmse_global_bio:.4f}", flush=True)
    print(f"  RMSE Stage 1+2 (bio + clinicien) : {rmse_global_final:.4f}", flush=True)

    # Evaluation skore officielle (Stage 2 seul sur X_all pour le rapport Hub,
    # avec bio_pred comme feature — pas ideal mais necessaire pour skore.evaluate API)
    print("\n[3/5] Rapport skore.evaluate sur le modele combine...", flush=True)
    two_stage = TwoStageRegressor()

    report = evaluate(two_stage, X_all, y, splitter=cv_splits)

    try:
        print(f"  RMSE skore.evaluate:\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  (erreur acces metrique: {exc})", flush=True)

    try:
        print(f"  Controles Skore:\n{report.checks.summarize()}", flush=True)
    except Exception as exc:
        print(f"  Erreur controles: {exc}", flush=True)

    # 4. Entrainement final sur tout le train
    print("\n[4/5] Entrainement final et prediction X_test...", flush=True)
    m_bio_final = build_bio_model()
    m_bio_final.fit(X_bio, y)
    y_bio_test = m_bio_final.predict(X_bio_test)

    X_clinic_test_aug = X_clinic_test.copy()
    X_clinic_test_aug["bio_pred"] = y_bio_test

    # Residus OOF pour le Stage 2 (train complet)
    y_bio_train = m_bio_final.predict(X_bio)
    resid_train = y.values - y_bio_train
    X_clinic_aug_train = X_clinic.copy()
    X_clinic_aug_train["bio_pred"] = y_bio_train

    m_clinic_final = build_clinic_model()
    m_clinic_final.fit(X_clinic_aug_train, resid_train)

    raw_preds = y_bio_test + m_clinic_final.predict(X_clinic_test_aug)
    predictions = np.clip(raw_preds, 0.0, 132.0)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "17_two_stage.csv"
    submission_df = pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions})
    submission_df.to_csv(sub_path, index=False)
    print(f"  Fichier : {sub_path} ({len(submission_df)} lignes)", flush=True)
    print(f"  Predictions : min={predictions.min():.2f}, mean={predictions.mean():.2f}, max={predictions.max():.2f}", flush=True)

    # 5. Hub
    if args.no_hub:
        print("\n[5/5] Hub ignore (--no-hub).", flush=True)
        print("=" * 70, flush=True)
        return

    print(f"\n[5/5] Publication Hub sous '{HUB_KEY}'...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
    project.put(HUB_KEY, report)
    print("  Publication terminee !", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
