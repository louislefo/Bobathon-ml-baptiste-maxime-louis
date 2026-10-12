"""HistGradientBoosting avec hyperparamètres optimisés (06_hgbr_tuned) for Bobathon ESILV.

Sous-tâche 4 du plan `kaggle-best-score-plan.md` :
1. S'appuie directement sur les features de la sous-tâche 3 (model_05_rater).
2. Optimisation des hyperparamètres HGBR via RandomizedSearchCV avec GroupKFold (5 plis par patient).
3. Utilisation des meilleurs paramètres trouvés (max_iter=500, lr=0.05, max_leaf_nodes=31, min_samples_leaf=10, l2=0.1).
4. Évaluation propre avec skore.evaluate sur GroupKFold.
5. Génération de submissions/06_hgbr_tuned.csv et publication sur Skore Hub.

Usage :
    python src/model_06_tuned.py            # évaluation avec paramètres optimaux + soumission + push Hub
    python src/model_06_tuned.py --no-hub   # sans publication sur Skore Hub
    python src/model_06_tuned.py --tune     # ré-exécute le RandomizedSearchCV complet (100 fits)
"""

from __future__ import annotations

import argparse
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
HUB_KEY = "06_hgbr_tuned"

for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from parkinson.hub import load_skore_credentials
from model_05_rater import prepare_features

# Hyperparamètres optimaux identifiés via RandomizedSearchCV (20 itérations, 5 plis GroupKFold)
OPTIMAL_HGBR_PARAMS = {
    "max_iter": 500,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 10,
    "l2_regularization": 0.1,
    "random_state": 0,
}

HGBR_PARAM_GRID = {
    "histgradientboostingregressor__max_iter": [200, 300, 500],
    "histgradientboostingregressor__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "histgradientboostingregressor__max_leaf_nodes": [31, 63, 127],
    "histgradientboostingregressor__min_samples_leaf": [10, 20, 40],
    "histgradientboostingregressor__l2_regularization": [0.0, 0.1, 1.0],
}


def run_tuning_search(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_iter: int = 20,
) -> tuple[dict, any]:
    """Exécute une recherche d'hyperparamètres RandomizedSearchCV avec GroupKFold."""
    print(f"\n  Lancement de RandomizedSearchCV ({n_iter} itérations, GroupKFold 5 plis)...", flush=True)
    base_pipeline = tabular_pipeline(HistGradientBoostingRegressor(random_state=0))
    cv_grouped = GroupKFold(n_splits=5)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=HGBR_PARAM_GRID,
        n_iter=n_iter,
        cv=cv_grouped,
        scoring="neg_root_mean_squared_error",
        random_state=0,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y, groups=groups)

    print(f"\n  Meilleur score CV (RMSE) : {-search.best_score_:.4f}", flush=True)
    print("  Meilleurs hyperparamètres trouvés :", flush=True)
    for k, v in search.best_params_.items():
        print(f"    * {k}: {v}", flush=True)

    return search.best_params_, search.best_estimator_


def main() -> None:
    parser = argparse.ArgumentParser(description="Modèle HGBR optimisé (06_hgbr_tuned)")
    parser.add_argument("--no-hub", action="store_true", help="ne pas publier sur Skore Hub")
    parser.add_argument("--tune", action="store_true", help="ré-exécuter le RandomizedSearchCV complet")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("OPTIMISATION DES HYPERPARAMÈTRES HGBR (06_hgbr_tuned)", flush=True)
    print("Sous-tâche 4 - Basé sur la sous-tâche 3 (model_05_rater)", flush=True)
    print("=" * 70, flush=True)

    # 1. Chargement des données et feature engineering hérité de model_05_rater
    print("\n[1/5] Chargement et enrichissement des données (via model_05_rater)...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")

    groups = visits["patient_id"]
    y = visits["target"]

    X = prepare_features(visits)
    X_test = prepare_features(X_test_raw)

    # Alignement des catégories de rater_id
    X_test["rater_id"] = X_test["rater_id"].cat.set_categories(X["rater_id"].cat.categories)

    print(f"  Shape train X : {X.shape}, Shape test X : {X_test.shape}", flush=True)
    print(f"  Nombre de features : {X.shape[1]} {list(X.columns)}", flush=True)

    # 2. Préparation du modèle (RandomizedSearchCV ou paramètres optimaux gagnants)
    print("\n[2/5] Configuration du modèle HGBR optimisé...", flush=True)
    cv_grouped = GroupKFold(n_splits=5)
    cv_splits = list(cv_grouped.split(X, y, groups=groups))

    if args.tune:
        _, model = run_tuning_search(X, y, groups, n_iter=20)
    else:
        print("  Utilisation des hyperparamètres optimaux retenus :", flush=True)
        for k, v in OPTIMAL_HGBR_PARAMS.items():
            print(f"    * {k}: {v}", flush=True)
        model = tabular_pipeline(HistGradientBoostingRegressor(**OPTIMAL_HGBR_PARAMS))

    # 3. Évaluation propre du modèle avec skore.evaluate sur GroupKFold
    print("\n[3/5] Évaluation du modèle avec skore.evaluate sur GroupKFold...", flush=True)
    report = evaluate(model, X, y, splitter=cv_splits)

    try:
        rmse_summary = report.metrics.rmse()
        print(f"  -> Score skore RMSE GroupKFold :\n{rmse_summary}", flush=True)
    except Exception as exc:
        print(f"  Rapport généré (erreur affichage direct métrique: {exc})", flush=True)

    # 4. Entraînement final sur tout le train et prédiction sur X_test
    print("\n[4/5] Entraînement final sur 100% du jeu d'entraînement...", flush=True)
    model.fit(X, y)
    predictions = np.clip(model.predict(X_test), 0.0, 132.0)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"{HUB_KEY}.csv"
    submission_df = pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions})
    submission_df.to_csv(sub_path, index=False)
    print(f"  Fichier de soumission généré : {sub_path} ({len(submission_df)} lignes)", flush=True)

    # 5. Publication sur Skore Hub
    if args.no_hub:
        print("\n[5/5] Publication Skore Hub ignorée (--no-hub).", flush=True)
        print("=" * 70, flush=True)
        return

    print(f"\n[5/5] Publication du rapport sur Skore Hub sous la clé '{HUB_KEY}'...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    workspace = config.get("workspace")
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=workspace)
    project.put(HUB_KEY, report)
    print("  Publication sur Skore Hub terminée avec succès !", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
