---
id: analyse-donnees-eda
title: Analyse Exploratoire des Donnees (EDA)
sidebar_position: 3
---

# Analyse Exploratoire des Donnees (EDA)

L'exploration rigoureuse des donnees cliniques est une etape preliminaire incontournable pour apprehender la structure longitudinale des visites, la nature des donnees manquantes et les correlats majeurs du score cible `target`.

:::tip Rapports Interactifs EDA
Deux rapports interactifs complets generes lors du Day 1 sont directement consultables sur cette plateforme :
- [Consulter le Rapport EDA - Visites Train (44 590 visites)](pathname:///reports/eda_visits_train.html)
- [Consulter le Rapport EDA - Visites Test (11 013 visites)](pathname:///reports/eda_X_test.html)
:::

---

## Volumetrie et Repartition des Patients

Le tableau ci-dessous resume les dimensions fondamentales des donnees d'apprentissage et de test :

| Dimension Clinique | Jeu d'Entrainement (`X_train`) | Jeu de Test (`X_test`) |
| :--- | :--- | :--- |
| **Nombre total de visites** | 44 590 | 11 013 |
| **Nombre de patients uniques** | 5 576 | 1 395 |
| **Visites par patient (Min / Mediane / Max)** | 4 / 7 / 12 | 4 / 7 / 12 |
| **Chevauchement des patients** | Aucun (patients strictement disjoints) | - |
| **Disponibilite de la cible** | Cible fournie dans `y_train.csv` | Cachee (a predire pour Kaggle) |

### Consequence Directe sur la Validation
Les patients du jeu de test ne figurant jamais dans l'entrainement, un decoupage aleatoire standard par ligne introduirait une fuite d'information (*data leakage*) majeure. L'evaluation doit imperativement regrouper les echantillons par `patient_id`.

---

## Cartographie des Valeurs Manquantes

Contrairement a des donnees tabulaires synthetiques classiques, les valeurs manquantes de ce jeu de donnees portent un sens clinique direct.

| Variable | Visites Manquantes | Pourcentage Manquant | Explication Clinique |
| :--- | :--- | :--- | :--- |
| `time_since_intake_off` | 35 116 | **78.75 %** | Delai avant examen OFF, tres rarement trace avec precision. |
| `time_since_intake_on` | 20 678 | **46.37 %** | Heure exacte de la prise de medicament precedente. |
| `off` | 18 913 | **42.42 %** | Examen en privation therapeutique penible et frequemment omis. |
| `ledd` | 16 339 | **36.64 %** | Posologie journaliere en Levodopa (variations selon centres). |
| `gene` | 14 434 | **32.37 %** | Profilage genetique non realise systematiquement. |
| `on` | 13 222 | **29.65 %** | Mesure en condition medicamenteuse active. |

:::info Propriete Remarquable
Il n'existe **aucune visite** ou les deux mesures `on` et `off` sont simultanement absentes (0.0 % de visites sans aucune note motrice). Chaque consultation dispose d'au minimum une mesure directe de l'etat du patient.
:::

### Analyse de `ledd` selon la Cohorte
Le jeu de donnees rassemble deux cohortes de recueil distinctes :
- **Cohorte A** : 39 636 visites, 35.90 % de valeurs manquantes, moyenne de dose LEDD a 647.1 mg.
- **Cohorte B** : 4 954 visites, 42.59 % de valeurs manquantes, moyenne de dose LEDD a 555.3 mg.

Cette disparite temoigne du fait que les donnees manquantes ne sont pas manquantes completement au hasard (*not Missing Completely at Random*, non-MCAR). Les modeles a base d'arbres tels que `HistGradientBoostingRegressor` (qui bifurquent nativement sur la presence de NaN) exploitent directement ce signal.

---

## Distribution de la Cible `target` (true OFF)

La variable a predire represente le score moteur MDS-UPDRS reel debiaise :

- **Minimum observe** : 0.0
- **Maximum observe** : 109.5 (sur un maximum theorique de 132)
- **Moyenne empirique** : 37.47
- **Mediane** : 37.30
- **Ecart-type ($\sigma$)** : 16.50
- **Coefficient d'asymetrie (skewness)** : 0.051

La distribution est remarquablement symetrique et quasi normale autour de 37.3. Il n'est donc pas pertinent d'appliquer une transformation logarithmique forcee, l'optimisation directe du critere quadratique (**RMSE**) etant alignee avec l'esperance conditionnelle.

---

## Correlats et Pouvoir Predictif des Variables

Les correlations lineaires (Pearson) et de rang (Spearman) avec la cible revelent la hierarchie des predicteurs :

| Caracteristique | Correlation de Pearson | Correlation de Spearman | Interpretation Clinique |
| :--- | :--- | :--- | :--- |
| `off` | **+0.871** | **+0.872** | Tres fort predicteur de base, mais entache de bruit d'observation. |
| `on` | **+0.669** | **+0.672** | Signal moteur puissant mais attenue par l'effet medicamenteux (50-100%). |
| `age` | **+0.310** | **+0.295** | La degenerescence globale s'intensifie avec l'age chronologique. |
| `ledd` | **+0.298** | **+0.280** | Les doses prescrites sont relevees pour les stades de maladie avances. |
| `age_at_diagnosis` | **+0.133** | **+0.120** | Les formes tardives presentent souvent une progression plus marquee. |

---

## Colonnes a Exclure des Features Directes

1. **`Index`** : Identifiant technique de soumission servant uniquement de cle primaire pour le fichier de resultat.
2. **`patient_id`** : Identifiant individuel du patient. Comme aucun patient d'apprentissage n'est present dans le jeu de test, injecter `patient_id` dans un arbre ou un regresseur provoquerait un surapprentissage pur sans capacite de generalisation. Cette colonne est strictement reservee au partitionnement de la validation croisee.
