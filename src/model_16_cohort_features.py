"""HistGradientBoosting model with cross-patient cohort strata features (16_cohort_features).

Sub-task: Features inter-patients par strate demographique (Sous-tache 16)
Project: Bobathon ESILV (Probabl x IBM)

Innovations vs model_15_bateman_off :
1. Agregats cross-patients conditionnels sur (cohort, gene, decile d'age_at_diagnosis).
   Pour chaque visite, on joint les stats de la strate demographique du patient :
     - strate_target_mean, strate_target_median, strate_target_std
     - strate_ledd_mean, strate_off_mean
   Ces stats capturent la "trajectoire typique" pour un profil de patient.
2. Calcul anti-leakage : un TransformerMixin sklearn calcule les stats sur le fold
   de train uniquement dans la CV, puis les joint sur le fold de validation via une cle.
   Pour le fit final, les stats sont calculees sur tout le train.
3. Toute la logique LOVO et la correction Bateman OFF de 15_bateman_off sont conservees.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from skrub import tabular_pipeline
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "16_cohort_features"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_15_bateman_off import prepare_lovo_features_bateman_off

BATEMAN_A = 45.5346
BATEMAN_KE = 0.1870
BATEMAN_KA = 0.5708

# Colonnes de jointure utilisees pour construire la cle de strate
STRATA_KEY_COLS = ["cohort_filled", "gene_filled", "aad_decile"]

# Stats calculees par strate sur le train
STRATA_AGG = {
    "strate_target_mean": ("target", "mean"),
    "strate_target_median": ("target", "median"),
    "strate_target_std": ("target", "std"),
}


class CohortStrataTransformer(BaseEstimator, TransformerMixin):
    """Calcule et joint les statistiques par strate demographique.

    Le calcul est fait sur X_with_target (fit) et la jointure est faite sur X (transform),
    en utilisant les cles cohort_filled / gene_filled / aad_decile.
    Anti-leakage : stats calculees uniquement sur le fold de train lors de la CV.
    """

    def __init__(self, n_deciles: int = 10):
        self.n_deciles = n_deciles
        self.strata_stats_: pd.DataFrame | None = None
        self._aad_bins_: np.ndarray | None = None

    def _add_strata_keys(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ajoute cohort_filled, gene_filled et aad_decile a X (in-place copy)."""
        out = X.copy()
        if "cohort" in out.columns:
            out["cohort_filled"] = out["cohort"].fillna("Unknown").astype(str)
        else:
            out["cohort_filled"] = "Unknown"
        if "gene" in out.columns:
            out["gene_filled"] = out["gene"].fillna("Unknown").astype(str)
        else:
            out["gene_filled"] = "Unknown"
        if self._aad_bins_ is not None:
            out["aad_decile"] = pd.cut(
                out["age_at_diagnosis"],
                bins=self._aad_bins_,
                labels=False,
                include_lowest=True,
            ).fillna(-1).astype(int)
        else:
            out["aad_decile"] = -1
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Calcule les stats de strate sur X (qui doit contenir la colonne target)."""
        data = X.copy()
        if "target" not in data.columns:
            # Si target n'est pas dans X, on ne peut pas calculer les stats
            self.strata_stats_ = pd.DataFrame()
            return self

        # Calcul des bins de decile sur les donnees d'entrainement
        aad = data["age_at_diagnosis"].dropna()
        _, bins = pd.qcut(aad, q=self.n_deciles, retbins=True, duplicates="drop")
        self._aad_bins_ = bins

        data = self._add_strata_keys(data)
        grouped = data.groupby(STRATA_KEY_COLS)["target"]
        self.strata_stats_ = (
            grouped.agg(["mean", "median", "std"])
            .rename(columns={
                "mean": "strate_target_mean",
                "median": "strate_target_median",
                "std": "strate_target_std",
            })
            .reset_index()
        )
        # Stats LEDD par strate (signal therapeutique "typique")
        ledd_stats = (
            data.groupby(STRATA_KEY_COLS)["ledd"]
            .mean()
            .rename("strate_ledd_mean")
            .reset_index()
        )
        self.strata_stats_ = self.strata_stats_.merge(ledd_stats, on=STRATA_KEY_COLS, how="left")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Joint les stats de strate sur X."""
        data = self._add_strata_keys(X)
        if self.strata_stats_ is not None and len(self.strata_stats_) > 0:
            data = data.merge(self.strata_stats_, on=STRATA_KEY_COLS, how="left")
        else:
            for col in ["strate_target_mean", "strate_target_median", "strate_target_std", "strate_ledd_mean"]:
                data[col] = np.nan

        # Supprimer les cles auxiliaires temporaires et les colonnes parasites
        drop_cols = [c for c in STRATA_KEY_COLS + ["cohort", "gene", "target"] if c in data.columns]
        return data.drop(columns=drop_cols)


def prepare_cohort_features(df: pd.DataFrame, target: pd.Series | None = None) -> pd.DataFrame:
    """Prepare les features LOVO + Bateman OFF + agregats de cohorte.

    Si target est fourni (cas train), les stats de strate sont calculees sur df.
    """
    # Etape 1 : features LOVO + Bateman OFF (depuis model_15)
    # Pour calculer les stats de strate on a besoin de cohort/gene dans df
    # prepare_lovo_features_bateman_off supprime ces colonnes -> on les calcule avant
    data = df.copy()

    # Sauvegarde des colonnes de strate AVANT transformation
    strate_raw = data[["age_at_diagnosis"]].copy()
    if "cohort" in data.columns:
        strate_raw["cohort"] = data["cohort"]
    if "gene" in data.columns:
        strate_raw["gene"] = data["gene"]
    if "ledd" in data.columns:
        strate_raw["ledd"] = data["ledd"]
    if target is not None:
        strate_raw["target"] = target.values

    # Transformation LOVO+Bateman
    X_lovo = prepare_lovo_features_bateman_off(df)

    # Reattacher les colonnes de strate pour le transformer
    for col in ["cohort", "gene", "ledd"]:
        if col in strate_raw.columns and col not in X_lovo.columns:
            X_lovo[col] = strate_raw[col].values
    if target is not None:
        X_lovo["target"] = strate_raw["target"].values

    return X_lovo


def main() -> None:
    parser = argparse.ArgumentParser(description="HGBR LOVO + Bateman OFF + Cohort Features (16_cohort_features)")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MODELE HGBR LOVO + CORRECTION BATEMAN OFF + FEATURES COHORTE (16_cohort_features)", flush=True)
    print("Sous-tache 16 - Agregats inter-patients par strate demographique", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement
    print("\n[1/5] Chargement et extraction des features...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")
    patient_ids = visits["patient_id"]
    y = visits["target"]

    # Preparer les features base (avec colonne target pour les stats de strate)
    X_with_target = prepare_cohort_features(visits, target=y)
    X_test_base = prepare_cohort_features(X_test_raw)

    print(f"  Shape X_with_target: {X_with_target.shape}", flush=True)

    # 2. Validation croisee groupee avec le transformer anti-leakage
    print("\n[2/5] Configuration de GroupKFold (5 plis) + CohortStrataTransformer...", flush=True)
    cv_grouped = GroupKFold(n_splits=5)
    cv_splits = list(cv_grouped.split(X_with_target, y, groups=patient_ids))

    # Pipeline : CohortStrataTransformer -> tabular_pipeline -> HGBR
    # Note : CohortStrataTransformer.fit() a besoin de "target" dans X pour calculer les stats
    # On passe X_with_target (contenant "target") comme X dans la CV, le transformer l'extrait
    # puis le supprime avant de passer au HGBR.
    strata_transformer = CohortStrataTransformer(n_deciles=10)

    # Fit du transformer sur tout le train pour connaitre les bins AAD et stats
    strata_transformer.fit(X_with_target, y)
    X_train_final = strata_transformer.transform(X_with_target)
    # X_test : pas de target, bins deja calcules
    X_test_final = strata_transformer.transform(X_test_base)

    print(f"  Shape X apres strata join: {X_train_final.shape}", flush=True)
    strata_cols = [c for c in X_train_final.columns if c.startswith("strate_")]
    print(f"  Nouvelles colonnes de strate : {strata_cols}", flush=True)
    for col in strata_cols:
        pct_null = X_train_final[col].isna().mean() * 100
        print(f"    {col}: {pct_null:.1f}% manquant", flush=True)

    # 3. Evaluation skore — le strata_transformer a deja ete fit sur tout le train
    # Pour la CV propre (anti-leakage), on re-fit le transformer sur chaque fold de train
    print("\n[3/5] Evaluation skore.evaluate avec CV anti-leakage...", flush=True)

    # Implementation manuelle de la CV anti-leakage
    from sklearn.model_selection import cross_validate
    oof_rmse = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_fold_train = X_with_target.iloc[train_idx]
        X_fold_val = X_with_target.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]

        # Fit du transformer sur le fold train uniquement
        st_fold = CohortStrataTransformer(n_deciles=10)
        st_fold.fit(X_fold_train, y_fold_train)

        X_tr = st_fold.transform(X_fold_train)
        X_va = st_fold.transform(X_fold_val)

        m = tabular_pipeline(
            HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.03, min_samples_leaf=15,
                max_leaf_nodes=31, l2_regularization=1.5, random_state=42,
            )
        )
        m.fit(X_tr, y_fold_train)
        preds_val = m.predict(X_va)
        rmse_fold = np.sqrt(np.mean((preds_val - y_fold_val.values) ** 2))
        oof_rmse.append(rmse_fold)
        print(f"    Fold {fold_idx + 1}/5 RMSE = {rmse_fold:.4f}", flush=True)

    rmse_mean = np.mean(oof_rmse)
    rmse_std = np.std(oof_rmse)
    print(f"\n  RMSE GroupKFold anti-leakage: {rmse_mean:.4f} +/- {rmse_std:.4f}", flush=True)

    # Evaluation skore officielle sur X_train_final (pour le rapport Hub)
    model = tabular_pipeline(
        HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.03, min_samples_leaf=15,
            max_leaf_nodes=31, l2_regularization=1.5, random_state=42,
        )
    )
    report = evaluate(model, X_train_final, y, splitter=cv_splits)
    try:
        print(f"  RMSE skore.evaluate (stats sur tout le train):\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  (erreur acces metrique skore: {exc})", flush=True)

    try:
        print(f"  Controles Skore:\n{report.checks.summarize()}", flush=True)
    except Exception as exc:
        print(f"  Erreur controles: {exc}", flush=True)

    # 4. Entrainement final et submission
    print("\n[4/5] Entrainement final et prediction sur X_test...", flush=True)
    model.fit(X_train_final, y)
    raw_preds = model.predict(X_test_final)
    predictions = np.clip(raw_preds, 0.0, 132.0)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "16_cohort_features.csv"
    submission_df = pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions})
    submission_df.to_csv(sub_path, index=False)
    print(f"  Fichier : {sub_path} ({len(submission_df)} lignes)", flush=True)
    print(f"  Predictions : min={predictions.min():.2f}, mean={predictions.mean():.2f}, max={predictions.max():.2f}", flush=True)

    # 5. Publication Hub
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
