---
id: architecture-et-outils
title: Architecture et Environnement
sidebar_position: 2
---

# Architecture et Environnement Technique

## Chaine d'Outils Officielle (Stack Imposée)

Dans le cadre du Bobathon ESILV, l'usage des outils est rigoureusement encadre pour garantir l'integrite experimentale, la reproductibilite scientifique et la tracabilite complete de chaque modele.

Toutes les experimentations doivent s'appuyer sur la stack suivante :

| Composant | Role dans le Projet | Justification & Specificite |
| :--- | :--- | :--- |
| **Bob IDE / Bob CLI** | Agent de codage et environnement d'execution | Seul agent et environnement autorise pour rediger et executer le code de soumission. |
| **skore** | Suivi d'experiences et evaluation | Generation d'objets `EstimatorReport` via `skore.evaluate` et persistance des runs. |
| **Skore Hub** | Plateforme collaborative | Hebergement en ligne des rapports de modele sur `skore.probabl.ai`. Chaque soumission Kaggle doit pointer vers son rapport Hub. |
| **scikit-learn** | Coeur algorithmique | Regresseurs lineaires (`Ridge`), arbres de gradient boosting (`HistGradientBoostingRegressor`), validation croisee (`GroupKFold`). |
| **skrub** | Traitement tabulaire avance | Pipelines automatisés avec `tabular_pipeline` pour le traitement natif des categoriels et des valeurs manquantes. |

:::caution Regle de Conformite Kaggle
Tout fichier CSV produit en dehors de l'environnement Bob + skore ou depourvu d'une URL de rapport Skore Hub valide dans sa description Kaggle est declare non conforme et disqualifie.
:::

---

## Structure de l'Arborescence du Projet

Le depot de code est organise de maniere modulaire :

```
bobathon-esilv/
|-- .github/workflows/
|   `-- pages.yml              # Pipeline GitHub Actions deployant le site et la doc
|-- data/                      # Donnees du hackathon et rapports exploratoires
|   |-- X_train.csv            # Features d'entrainement (44 590 visites)
|   |-- y_train.csv            # Cible d'entrainement (Index, target)
|   |-- X_test.csv             # Features de test (11 013 visites)
|   |-- sample_submission.csv  # Gabarit officiel de soumission
|   |-- eda_visits_train.html  # Rapport interactif HTML du jeu d'entrainement
|   `-- eda_X_test.html        # Rapport interactif HTML du jeu de test
|-- parkinson/                 # Modules Python utilitaires
|   |-- __init__.py
|   `-- hub.py                 # Gestion securisee des identifiants Skore Hub
|-- src/                       # Scripts des modeles de Machine Learning
|   |-- baseline_dummy.py      # 01_dummy : moyenne empirique naive
|   |-- model_ridge.py         # 02_ridge : regression lineaire regularisee
|   |-- model_hgbr.py          # 03_hgbr : HistGradientBoosting avec tabular_pipeline
|   |-- model_04_features.py   # 04_features : feature engineering temporel
|   |-- model_05_rater.py      # 05_rater : correction du biais clinicien
|   `-- model_07_ensemble.py   # 07_ensemble : stacking Ridge + HGBR
|-- submissions/               # Fichiers de soumission CSV generes
|   |-- 01_dummy.csv
|   |-- 02_ridge.csv
|   `-- ...
|-- website/                   # Site web moderne et documentation (Docusaurus)
|   |-- docs/                  # Fichiers sources Markdown de la documentation
|   |-- static/                # Rapports EDA et figures cliniques
|   `-- src/                   # Composants React de la page d'accueil
|-- requirements.txt           # Dependances Python
`-- pytest.ini                 # Configuration des tests d'integration
```

---

## Gestion des Identifiants Skore Hub

Le module utilitaire [`parkinson/hub.py`](file:///c:/Users/louis/OneDrive/Desktop/Esilv/A4/Machine%20Learning/Hackathon/bobathon-esilv/parkinson/hub.py) centralise la configuration des acces pour eviter toute exposition accidentelle de secrets.

Il lit les preferences dans le fichier local `.skore` ou via les variables d'environnement du projet :

```python
from parkinson.hub import load_skore_credentials
from skore import Project, login

# Chargement de la configuration
config = load_skore_credentials(repo_root)
workspace = config.get("workspace")

# Connexion au Hub
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=workspace)

# Publication du rapport
project.put("03_hgbr", report)
```

---

## Validation Automatisee de l'Environnement

Avant de lancer le moindre entrainement, la suite de tests unitaires permet de certifier que l'environnement virtuel Python 3.12+ et les acces Skore sont operationnels :

```bash
python -m pytest
```

Les tests verifies par la suite couvrent :
1. La version minimale de Python (>= 3.12).
2. L'existence et la syntaxe du fichier `.skore`.
3. La bonne installation de `skore`, `skrub` et `scikit-learn`.
4. La resolution correcte des chemins de donnees dans `data/`.
