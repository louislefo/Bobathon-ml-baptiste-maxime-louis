"""HistGradientBoosting + correction du biais clinicien (05_hgbr_rater) for Bobathon ESILV.

Sous-tache 3 du plan `kaggle-best-score-plan.md` :
1. Reutilise `prepare_features` de 04_hgbr_features (features temporelles intra-patient).
2. `rater_id` y est conserve comme variable categorielle (biais fixe par clinicien) :
   gain mesure de ~0.17 RMSE en GroupKFold (4.97 sans -> 4.80 avec).
3. Aligne les categories de `rater_id` entre train et test.
4. Validation croisee groupee par patient (GroupKFold sur patient_id).
5. Generation de submissions/05_hgbr_rater.csv et reporting Skore Hub.

Usage :
    python src/model_05_rater.py            # evaluation + soumission + push Hub
    python src/model_05_rater.py --no-hub   # sans publication sur Skore Hub
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skrub import tabular_pipeline
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
HUB_KEY = "05_hgbr_rater"

for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from parkinson.hub import load_skore_credentials

def add_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features de contexte patient (sous-tache 2).

    Calculees uniquement a partir des variables explicatives (jamais de `target`),
    donc sans fuite : le meme calcul s'applique a X_test, dont les patients sont
    nouveaux mais dont toutes les visites sont disponibles.
    """
    data = df.sort_values(["patient_id", "age"]).copy()
    by_patient = data.groupby("patient_id")

    data["visit_number"] = by_patient.cumcount() + 1
    data["n_visits_patient"] = by_patient["age"].transform("size")
    data["time_since_first_visit"] = data["age"] - by_patient["age"].transform("min")

    mean_off = by_patient["off"].transform("mean")
    mean_on = by_patient["on"].transform("mean")
    data["mean_off_patient"] = mean_off
    data["mean_on_patient"] = mean_on
    data["std_off_patient"] = by_patient["off"].transform("std")
    data["std_on_patient"] = by_patient["on"].transform("std")
    data["mean_ledd_patient"] = by_patient["ledd"].transform("mean")

    data["off_minus_on"] = data["off"] - data["on"]
    data["diff_from_mean_off"] = data["off"] - mean_off
    data["diff_from_mean_on"] = data["on"] - mean_on

    # Restaure l'ordre d'origine pour rester aligne avec y et l'Index de soumission
    return data.loc[df.index]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare les variables explicatives, y compris le biais clinicien `rater_id`."""
    data = add_patient_features(df)

    # Duree depuis le diagnostic clinique
    data["time_since_diagnosis"] = data["age"] - data["age_at_diagnosis"]

    # Variables categorielles : rater_id est conserve (biais clinicien), pas supprime
    for col in ["cohort", "gene", "rater_id"]:
        if col in data.columns:
            data[col] = data[col].astype("category")

    # Colonnes a exclure de l'apprentissage (identifiants ou cible)
    drop_cols = [c for c in ["Index", "patient_id", "target"] if c in data.columns]
    return data.drop(columns=drop_cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hub", action="store_true", help="ne pas publier sur Skore Hub")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("ENTRAINEMENT HGBR + BIAIS CLINICIEN (05_hgbr_rater)", flush=True)
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

    # rater_id doit partager les memes categories entre train et test
    # (un clinicien absent du train devient NaN categoriel)
    X_test["rater_id"] = X_test["rater_id"].cat.set_categories(X["rater_id"].cat.categories)
    n_unknown = int(X_test["rater_id"].isna().sum() - X_test_raw["rater_id"].isna().sum())
    print(
        f"  rater_id : {X['rater_id'].nunique()} cliniciens en train, "
        f"{n_unknown} visites test avec clinicien inconnu",
        flush=True,
    )
    print(f"  Nombre de features utilisees : {X.shape[1]} {list(X.columns)}", flush=True)

    # 2. Preparation de la validation croisee groupee par patient (GroupKFold)
    print("\n[2/5] Preparation de GroupKFold (5 plis par patient)...", flush=True)
    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

    # 3. Modele HistGradientBoosting via pipeline skrub (memes parametres que 03_hgbr)
    print("\n[3/5] Evaluation du modele avec skore (gestion native des NaNs)...", flush=True)
    model = tabular_pipeline(
        HistGradientBoostingRegressor(
            random_state=42,
            max_iter=150,
            learning_rate=0.08,
            min_samples_leaf=20,
        )
    )

    report = evaluate(model, X, y, splitter=cv_splits)
    try:
        rmse = report.metrics.rmse()
        print(f"  Score RMSE moyen en GroupKFold : {rmse}", flush=True)
    except Exception:
        print("  Rapport d'evaluation genere.", flush=True)

    # 4. Entrainement final et generation de la soumission
    print("\n[4/5] Entrainement final sur tout le jeu d'entrainement...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "05_hgbr_rater.csv"
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
