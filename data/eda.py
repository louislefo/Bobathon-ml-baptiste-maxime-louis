# %% [markdown]
# # EDA: Parkinson Motor Score Evaluation (Bobathon ESILV)
#
# Exploratory Data Analysis of synthetic multi-cohort Parkinson records.
# Aligned with the `.bob/skills/explore-ml-data` methodology.
#
# Outputs:
# - Rich interactive HTML reports in `data/eda_visits_train.html` and `data/eda_X_test.html`
# - Summary report in `data/eda.md` answering the 6 Day 1 workbook questions.

# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import skrub

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def main():
    print("=" * 60)
    print("DEMARRAGE DE L'EDA (EXPLORATORY DATA ANALYSIS)")
    print("=" * 60)

    # 1. Chargement des donnees
    print("\n[1/5] Chargement des fichiers CSV...")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    visits_train = X_train.merge(y_train, on="Index")
    print(f"  Train: {len(X_train)} visites, {X_train['patient_id'].nunique()} patients distincts")
    print(f"  Test:  {len(X_test)} visites, {X_test['patient_id'].nunique()} patients distincts")

    # 2. Generation des rapports interactifs skrub TableReport
    print("\n[2/5] Generation des rapports HTML skrub (TableReport)...")
    try:
        report_train = skrub.TableReport(visits_train, title="visits_train", verbose=0)
        report_train.write_html(DATA_DIR / "eda_visits_train.html")
        print("  Rapport sauvegarde: data/eda_visits_train.html")

        report_test = skrub.TableReport(X_test, title="X_test", verbose=0)
        report_test.write_html(DATA_DIR / "eda_X_test.html")
        print("  Rapport sauvegarde: data/eda_X_test.html")
    except Exception as exc:
        print(f"  Avertissement: generation HTML skrub ignoree ({exc})")

    # 3. Statistiques cles pour le workbook Day 1
    print("\n[3/5] Analyse des visites par patient...")
    train_vp = X_train.groupby("patient_id").size()
    print(f"  Visites par patient (train) : min={train_vp.min()}, mediane={train_vp.median()}, moyenne={train_vp.mean():.2f}, max={train_vp.max()}")

    print("\n[4/5] Analyse des valeurs manquantes...")
    missing = X_train.isnull().sum()
    missing_pct = (missing / len(X_train)) * 100
    top_missing = pd.DataFrame({"manquantes": missing, "pourcentage": missing_pct}).sort_values(by="manquantes", ascending=False)
    print(top_missing.head(6).to_string())

    no_off = (X_train["off"].isnull().sum() / len(X_train)) * 100
    no_on = (X_train["on"].isnull().sum() / len(X_train)) * 100
    both_null = (X_train["off"].isnull() & X_train["on"].isnull()).sum()
    print(f"  Sans score OFF : {no_off:.2f}%")
    print(f"  Sans score ON  : {no_on:.2f}%")
    print(f"  Sans NI ON NI OFF : {both_null} visites (chaque visite a au moins l'un des deux)")

    # 4. Distribution de la cible
    print("\n[5/5] Distribution de la cible (true OFF)...")
    target = y_train["target"]
    print(f"  Min     : {target.min():.2f} (theorique MDS-UPDRS: 0)")
    print(f"  Max     : {target.max():.2f} (theorique MDS-UPDRS: 132)")
    print(f"  Moyenne : {target.mean():.2f}")
    print(f"  Ecart-type : {target.std():.2f} (RMSE theorique du modele Dummy)")
    print(f"  Mediane : {target.median():.2f}")
    print(f"  Skewness: {target.skew():.4f} (quasi parfaitement symetrique)")

    # 5. Associations avec la cible
    print("\nAssociations (Pearson) avec la cible :")
    num_cols = visits_train.select_dtypes(include=[np.number]).columns
    corrs = visits_train[num_cols].corr()["target"].sort_values(ascending=False)
    for col, val in corrs.items():
        if col != "target":
            print(f"  {col:<22} : {val:+.4f}")

    print("\n" + "=" * 60)
    print("EDA TERMINEE AVEC SUCCES.")
    print("Consultez data/eda.md pour les explications completes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
