"""Stacking Ridge + HGBR (07_ensemble) for Bobathon ESILV.

Sous-tache 5 du plan `kaggle-best-score-plan.md` :
1. Reprend les features de 05_hgbr_rater (features patient + rater_id categoriel).
2. Estimateurs de base : Ridge (signal lineaire off/on) et HGBR (interactions, NaN natifs).
3. Meta-modele : Ridge(alpha=1.0), entraine sur des predictions out-of-fold groupees par patient.
4. Validation croisee groupee par patient (GroupKFold sur patient_id).
5. Generation de submissions/07_ensemble.csv et reporting Skore Hub.

Usage :
    python src/model_07_ensemble.py            # evaluation + soumission + push Hub
    python src/model_07_ensemble.py --no-hub   # sans publication sur Skore Hub
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "07_ensemble"

for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from parkinson.hub import load_skore_credentials
from model_05_rater import prepare_features


class PatientGroupKFold:
    """GroupKFold qui retrouve le patient_id de chaque ligne via l'index de X.

    Le StackingRegressor n'accepte pas `groups=` et des splits precalcules sur tout
    le train seraient faux une fois imbriques dans la CV externe de skore (les
    indices ne correspondent plus au sous-ensemble d'entrainement du pli). Ce
    splitter recalcule donc des plis groupes sur n'importe quel sous-ensemble de X.
    """

    def __init__(self, groups: pd.Series, n_splits: int = 5):
        self.groups = groups
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        return GroupKFold(n_splits=self.n_splits).split(X, y, groups=self.groups.loc[X.index])


def build_model(groups: pd.Series) -> StackingRegressor:
    ridge = tabular_pipeline(Ridge(alpha=10.0))
    hgbr = tabular_pipeline(
        HistGradientBoostingRegressor(
            random_state=42,
            max_iter=150,
            learning_rate=0.08,
            min_samples_leaf=20,
        )
    )
    return StackingRegressor(
        estimators=[("ridge", ridge), ("hgbr", hgbr)],
        final_estimator=Ridge(alpha=1.0),
        cv=PatientGroupKFold(groups, n_splits=5),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hub", action="store_true", help="ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("STACKING RIDGE + HGBR (07_ensemble)", flush=True)
    print("=" * 60, flush=True)

    # 1. Chargement des donnees
    print("\n[1/5] Chargement et enrichissement des donnees...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train_raw.merge(y_train_raw, on="Index")

    groups = visits["patient_id"]
    y = visits["target"]
    X = prepare_features(visits)
    X_test = prepare_features(X_test_raw)
    X_test["rater_id"] = X_test["rater_id"].cat.set_categories(X["rater_id"].cat.categories)
    print(f"  Nombre de features utilisees : {X.shape[1]} {list(X.columns)}", flush=True)

    # 2. Preparation de la validation croisee groupee par patient (GroupKFold)
    print("\n[2/5] Preparation de GroupKFold (5 plis par patient)...", flush=True)
    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

    # 3. Stacking evalue avec skore
    print("\n[3/5] Evaluation du stacking avec skore...", flush=True)
    model = build_model(groups)

    report = evaluate(model, X, y, splitter=cv_splits)
    try:
        rmse = report.metrics.rmse()
        print(f"  Score RMSE moyen en GroupKFold : {rmse}", flush=True)
    except Exception:
        print("  Rapport d'evaluation genere.", flush=True)

    # 4. Entrainement final et generation de la soumission
    print("\n[4/5] Entrainement final sur tout le jeu d'entrainement...", flush=True)
    model.fit(X, y)
    weights = {name: round(float(w), 3) for name, w in zip(["ridge", "hgbr"], model.final_estimator_.coef_)}
    print(f"  Poids du meta-modele : {weights}", flush=True)
    predictions = model.predict(X_test)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "07_ensemble.csv"
    submission_df = pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions})
    submission_df.to_csv(sub_path, index=False)
    print(f"  Fichier de soumission genere : {sub_path} ({len(submission_df)} lignes)", flush=True)

    # 5. Publication sur Skore Hub
    if args.no_hub:
        print("\n[5/5] Publication Skore Hub ignoree (--no-hub).", flush=True)
        print("=" * 60, flush=True)
        return
    print(f"\n[5/5] Publication du rapport sur Skore Hub sous la cle '{HUB_KEY}'...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    workspace = config.get("workspace")
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=workspace)
    project.put(HUB_KEY, report)
    print("  Publication terminee avec succes !", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
