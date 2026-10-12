---
id: guide-de-reproduction
title: Guide de Reproduction Pratique
sidebar_position: 7
---

# Guide de Reproduction Pratique

Ce guide pas-a-pas detaille l'ensemble des commandes a executer pour installer le projet, valider l'environnement, reproduire les entrainements et generer les fichiers de soumission.

---

## Etape 1 : Preparation de l'Environnement

Clonez le depot et activez l'environnement virtuel Python 3.12+ :

```bash
# Cloner le depot
git clone https://github.com/louislefo/Bobathon-ml-baptiste-maxime-louis.git
cd Bobathon-ml-baptiste-maxime-louis

# Creation et activation du venv (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Installation des dependances
pip install -r requirements.txt
```

Verifiez ensuite que tous les tests de validation de configuration passent :

```bash
python -m pytest
```

:::info Validation Requise
La suite de tests doit afficher `all passed`. Elle certifie la validite de la structure `.skore`, la version de Python et la presence des donnees dans `data/`.
:::

---

## Etape 2 : Entrainement et Generation des Soumissions

Chaque modele dispose d'un script dedie dans le dossier [`src/`](file:///c:/Users/louis/OneDrive/Desktop/Esilv/A4/Machine%20Learning/Hackathon/bobathon-esilv/src).

### 1. Baseline Moyenne Naive (`01_dummy`)
```bash
python src/baseline_dummy.py
```
- Genere : `submissions/01_dummy.csv` (11 013 lignes)
- Pousse le rapport sur Skore Hub sous la cle `01_dummy`.

### 2. Regression Ridge Linéaire (`02_ridge`)
```bash
python src/model_ridge.py
```
- Genere : `submissions/02_ridge.csv`
- Publication Hub : cle `02_ridge`.

### 3. HistGradientBoosting avec tabular_pipeline (`03_hgbr`)
```bash
python src/model_hgbr.py
```
- Genere : `submissions/03_hgbr.csv`
- Publication Hub : cle `03_hgbr`.

### 4. Modele avec Feature Engineering Temporel (`04_features`)
```bash
python src/model_04_features.py
```
- Genere : `submissions/04_features.csv`
- Publication Hub : cle `04_features`.

### 5. Modele avec Biais Clinique (`05_rater`)
```bash
python src/model_05_rater.py
```
- Genere : `submissions/05_rater.csv`
- Publication Hub : cle `05_rater`.

### 6. Ensemble Stacking Ridge + HGBR (`07_ensemble`)
```bash
python src/model_07_ensemble.py
```
- Genere : `submissions/07_ensemble.csv`
- Publication Hub : cle `07_ensemble`.

:::tip Mode Hors-Ligne
Pour executer les modeles sans envoyer de rapport sur Skore Hub (ex. en local sans connexion), ajoutez le drapeau `--no-hub` :
```bash
python src/model_07_ensemble.py --no-hub
```
:::

---

## Etape 3 : Deposer sur Kaggle

1. Rendez-vous sur la page officielle [Kaggle Bobathon ESILV](https://www.kaggle.com/competitions/bobathon-esilv/).
2. Cliquez sur **Submit Predictions**.
3. Selectionnez le fichier genere, par exemple `submissions/07_ensemble.csv`.
4. Dans le champ **Description**, collez obligatoirement l'URL affichee par skore lors de l'execution :
   `https://skore.probabl.ai/workspace/estimations/...`
5. Validez le depot et suivez l'evolution de votre score sur le leaderboard public.
