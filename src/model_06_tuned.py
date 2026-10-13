"""HistGradientBoosting model with hyperparameter tuning via RandomizedSearchCV (06_hgbr_tuned).

Sub-task: Sous-tâche 4 - Optimisation des hyperparamètres HGBR
Project: Bobathon ESILV (Probabl x IBM)

Features:
- Caractéristiques temporelles et intra-patient (issue de 04_hgbr_features)
- Encodage catégoriel (cohort, gene, rater_id)
- Recherche d'hyperparamètres avec RandomizedSearchCV (20 itérations, GroupKFold 5 plis par patient)
- Évaluation skore du meilleur modèle sur GroupKFold
- Entraînement final sur l'ensemble du train
- Génération de submissions/06_hgbr_tuned.csv
- Publication du rapport 06_hgbr_tuned sur Skore Hub
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from skrub import tabular_pipeline
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les variables explicatives avec l'ingénierie temporelle et intra-patient."""
    data = df.copy()

    # 1. Progression de la maladie
    data["time_since_diagnosis"] = data["age"] - data["age_at_diagnosis"]

    # 2. Caractéristiques temporelles et de suivi par patient
    data["visit_number"] = data.groupby("patient_id")["age"].rank(method="first").astype(int)
    data["n_visits_patient"] = data.groupby("patient_id")["age"].transform("count")
    data["time_since_first_visit"] = data["age"] - data.groupby("patient_id")["age"].transform("min")

    # 3. Statistiques intra-patient (profil de base de sévérité motrice)
    mean_off = data.groupby("patient_id")["off"].transform("mean")
    mean_on = data.groupby("patient_id")["on"].transform("mean")
    data["mean_off_patient"] = mean_off
    data["mean_on_patient"] = mean_on
    data["std_off_patient"] = data.groupby("patient_id")["off"].transform("std")
    data["std_on_patient"] = data.groupby("patient_id")["on"].transform("std")
    data["mean_ledd_patient"] = data.groupby("patient_id")["ledd"].transform("mean")

    # 4. Dynamique thérapeutique et écarts à la moyenne du patient
    data["off_minus_on"] = data["off"] - data["on"]
    data["diff_from_mean_off"] = data["off"] - mean_off
    data["diff_from_mean_on"] = data["on"] - mean_on

    # 5. Typage des colonnes catégorielles
    for col in ["cohort", "gene", "rater_id"]:
        if col in data.columns:
            data[col] = data[col].astype("category")

    # 6. Exclusion des identifiants et de la cible
    drop_cols = [c for c in ["Index", "patient_id", "target"] if c in data.columns]
    return data.drop(columns=drop_cols)


def main() -> None:
    print("=" * 70, flush=True)
    print("OPTIMISATION DES HYPERPARAMÈTRES HGBR (06_hgbr_tuned)", flush=True)
    print("Sous-tâche 4 - Recherche par RandomizedSearchCV avec GroupKFold", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement et préparation des données
    print("\n[1/6] Chargement des données et calcul des features...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")

    patient_ids = visits["patient_id"]
    y = visits["target"]

    X = prepare_features(visits)
    X_test = prepare_features(X_test_raw)

    print(f"  Shape X: {X.shape}, Shape X_test: {X_test.shape}", flush=True)
    print(f"  Nombre de features : {X.shape[1]}", flush=True)

    # 2. Configuration du pipeline et de la grille d'hyperparamètres
    print("\n[2/6] Définition de la grille de recherche d'hyperparamètres...", flush=True)
    base_pipeline = tabular_pipeline(HistGradientBoostingRegressor(random_state=0))

    param_distributions = {
        "histgradientboostingregressor__max_iter": [200, 300, 500],
        "histgradientboostingregressor__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "histgradientboostingregressor__max_leaf_nodes": [31, 63, 127],
        "histgradientboostingregressor__min_samples_leaf": [10, 20, 40],
        "histgradientboostingregressor__l2_regularization": [0.0, 0.1, 1.0],
    }

    print("  Grille configurée :")
    for param, values in param_distributions.items():
        print(f"    - {param}: {values}")

    # 3. Exécution de RandomizedSearchCV avec GroupKFold
    print("\n[3/6] Lancement du RandomizedSearchCV (20 itérations, GroupKFold 5 plis)...", flush=True)
    cv_grouped = GroupKFold(n_splits=5)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=cv_grouped,
        scoring="neg_root_mean_squared_error",
        random_state=0,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X, y, groups=patient_ids)

    print(f"\n  Meilleur score CV (RMSE) : {-search.best_score_:.4f}", flush=True)
    print("  Meilleurs hyperparamètres trouvés :", flush=True)
    for k, v in search.best_params_.items():
        print(f"    * {k}: {v}", flush=True)

    best_pipeline = search.best_estimator_

    # 4. Évaluation propre du meilleur modèle avec skore.evaluate
    print("\n[4/6] Évaluation propre avec skore.evaluate sur GroupKFold...", flush=True)
    cv_splits = list(cv_grouped.split(X, y, groups=patient_ids))
    report = evaluate(best_pipeline, X, y, splitter=cv_splits)

    try:
        rmse_summary = report.metrics.rmse()
        print(f"  -> Score skore RMSE GroupKFold :\n{rmse_summary}", flush=True)
    except Exception as exc:
        print(f"  Rapport généré (erreur affichage direct métrique: {exc})", flush=True)

    # 5. Entraînement final sur tout le train et prédiction sur X_test
    print("\n[5/6] Entraînement final sur 100% du jeu d'entraînement...", flush=True)
    best_pipeline.fit(X, y)
    predictions = np.clip(best_pipeline.predict(X_test), 0.0, 132.0)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "06_hgbr_tuned.csv"
    submission_df = pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions})
    submission_df.to_csv(sub_path, index=False)
    print(f"  Fichier de soumission généré : {sub_path} ({len(submission_df)} lignes)", flush=True)

    # 6. Publication du rapport sur Skore Hub
    print("\n[6/6] Publication du rapport sur Skore Hub sous la clé '06_hgbr_tuned'...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    workspace = config.get("workspace")
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=workspace)
    project.put("06_hgbr_tuned", report)
    print("  Publication sur Skore Hub terminée avec succès !", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
