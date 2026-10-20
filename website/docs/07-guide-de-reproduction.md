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

Chaque modele dispose d'un script dedie dans le dossier `src/` ou directement dans le dossier d'evaluation `rendu/`.

### 1. Modele Champion : Stacking Bayesien Adaptatif (`34_bayes_stack`)

Pour executer le modele champion complet (validation sur holdout patient 20%, generation des features bayesiennes OOF, entrainement sur 100% des donnees et export de la soumission finale) :

```bash
python rendu/model_34_bayes_stack.py
```
- Genere : `rendu/submissions/34_bayes_stack.csv` (11 013 lignes au format `Index,target`).
- Metriques evaluees : RMSE globale ~3.38.
- Publication automatique sur Skore Hub sous la cle `34_bayes_stack`.

#### Options d'execution disponibles :
- **Mode hors-ligne** (sans communication avec Skore Hub) :
  ```bash
  python rendu/model_34_bayes_stack.py --no-hub
  ```
- **Mode calibration rapide** (recherche sur grille de poids sans re-entrainement final) :
  ```bash
  python rendu/model_34_bayes_stack.py --quick
  ```
- **Mode rapport seul** (evaluation et export du rapport `.joblib` sans soumission CSV) :
  ```bash
  python rendu/model_34_bayes_stack.py --report-only
  ```
- **Poids de melange personnalises** :
  ```bash
  python rendu/model_34_bayes_stack.py --w-ca 0.30 --w-cb 0.40 --w-c1 0.10 --w-c2 0.20
  ```

---

### 2. Modeles de Reference et Jalons Historiques

Pour reproduire les etapes incrementales precedentes :

#### Baseline Moyenne Naive (`01_dummy`)
```bash
python src/baseline_dummy.py
```
- Genere : `submissions/01_dummy.csv` (RMSE 16.48).

#### Regression Ridge Linéaire (`02_ridge`)
```bash
python src/model_ridge.py
```
- Genere : `submissions/02_ridge.csv` (RMSE 10.49).

#### HistGradientBoosting avec tabular_pipeline (`03_hgbr`)
```bash
python src/model_hgbr.py
```
- Genere : `submissions/03_hgbr.csv` (RMSE 7.45).

#### Ensemble Stacking Ridge + HGBR (`07_ensemble`)
```bash
python src/model_07_ensemble.py
```
- Genere : `submissions/07_ensemble.csv` (RMSE 4.80).

#### Trajectoires Longitudinales et Pharmacodynamie (`09_longitudinal` & `15_trajectory`)
```bash
python src/model_09_longitudinal.py
python src/model_15_trajectory.py
```

#### Socle Multi-Etages (`32_ultimate`)
```bash
python src/model_32_ultimate.py
```

---

## Etape 3 : Deposer sur Kaggle

1. Rendez-vous sur la page officielle [Kaggle Bobathon ESILV](https://www.kaggle.com/competitions/bobathon-esilv/).
2. Cliquez sur **Submit Predictions**.
3. Selectionnez le fichier de soumission du modele champion : `submissions/34_bayes_stack.csv`.
4. Dans le champ **Description**, collez obligatoirement l'URL du rapport generee sur Skore Hub :
   `https://skore.probabl.ai/workspace/estimations/...`
5. Validez le depot et verifiez la conformite de votre score sur le leaderboard public.
