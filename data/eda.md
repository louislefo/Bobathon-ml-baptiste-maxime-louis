# Rapport d'analyse exploratoire des donnees (EDA) - Day 1

Ce rapport documente les caracteristiques statistiques, structurelles et cliniques du jeu de donnees du Bobathon ESILV conformement aux questions de [docs/DAY1.md](file:///c:/Users/louis/OneDrive/Desktop/Esilv/A4/Machine%20Learning/Hackathon/bobathon-esilv/docs/DAY1.md).

---

## Reponses aux questions du workbook

### Question 1 : Nombre de visites et de patients uniques
- **Jeu d'entrainement (`X_train`)** :
  - Nombre de visites : 44 590
  - Nombre de patients uniques : 5 576
  - Nombre de visites par patient : minimum = 4, mediane = 7, moyenne = 8.00, maximum = 12
- **Jeu de test (`X_test`)** :
  - Nombre de visites : 11 013
  - Nombre de patients uniques : 1 395
- **Implication de modelisation** :
  - Les patients du jeu de test ne se chevauchent pas avec ceux de l'entrainement.
  - La validation croisee doit imperativement etre groupee par patient (`GroupKFold` sur `patient_id`). Un decoupage aleatoire par ligne provoquerait une fuite d'information (*data leakage*) et une sur-estimation severe des performances.

---

### Question 2 : Valeurs manquantes et disponibilite des scores moteurs
- **Top 3 des colonnes avec le plus de valeurs manquantes** :
  1. `time_since_intake_off` : 78.75% manquantes (35 116 visites)
  2. `time_since_intake_on` : 46.37% manquantes (20 678 visites)
  3. `off` : 42.42% manquantes (18 913 visites)
  *(Suivies par `ledd` avec 36.64%, `gene` avec 32.37% et `on` avec 29.65%)*
- **Fraction de visites sans score `off`** : 42.42%
- **Fraction de visites sans score `on`** : 29.65%
- **Visites ou les deux scores sont manquants** : **0 visite** (0.0%).
- **Implication clinique** :
  - Chaque visite dispose au minimum de l'une des deux mesures cliniques (`on` ou `off`).
  - L'examen en condition OFF etant desagreable pour le patient, il est plus frequemment omis. Le pattern de valeurs manquantes est informatif et doit etre exploite par les modeles.

---

### Question 3 : Distribution de la cible `target` (true OFF)
- **Intervalle observe** : min = 0.0, max = 109.5
- **Intervalle theorique MDS-UPDRS** : 0 a 132
- **Statistiques descriptives** :
  - Moyenne : 37.47
  - Mediane : 37.30
  - Ecart-type : 16.50
  - Asymetrie (*skewness*) : 0.051 (quasi parfaitement symetrique, allure de loi normale centree a 37.3)
- **Implication de modelisation** :
  - La cible ne presente pas d'asymetrie marquee ni de queue lourde necessitant une transformation logarithmique forcee. Le critere d'optimisation RMSE est directement aligne avec la moyenne empirique.

---

### Question 4 : Associations avec `target`
- **Correlation de Pearson** :
  - `off` : +0.871 (Spearman : +0.872)
  - `on` : +0.669 (Spearman : +0.672)
  - `age` : +0.310 (Spearman : +0.295)
  - `ledd` : +0.298 (Spearman : +0.280)
  - `age_at_diagnosis` : +0.133 (Spearman : +0.120)
- **Comparaison entre `off`, `on` et `target`** :
  - `off` n'est pas une copie exacte de `target` : bien que fortement correle (0.87), le score observe est bruite par la subjectivite du medecin (`rater_id`) et l'intervalle depuis la derniere prise.
  - `on` est egalement tres correle (0.67), mais attenué car le traitement attenue les symptomes moteurs de 50 a 100%.

---

### Question 5 : Analyse de `ledd` par cohorte
- **Cohorte A** : 39 636 visites, 35.90% manquantes, moyenne de dose = 647.1 mg
- **Cohorte B** : 4 954 visites, 42.59% manquantes, moyenne de dose = 555.3 mg
- **Signification clinique** :
  - La donnee `ledd` (Levodopa Equivalent Daily Dose) n'est pas manquante completement au hasard (non-MCAR).
  - La cohorte B presente un taux de non-reponse plus eleve et des posologies moyennes plus faibles, ce qui temoigne de protocoles de collecte ou de stades de maladie differents entre centres cliniques.

---

### Question 6 : Colonnes a ne pas injecter directement au modele
1. `Index` : Cle d'identification technique sans signification clinique.
2. `patient_id` : Identifiant de patient. Les patients du test etant differents de ceux de l'entrainement, introduire `patient_id` comme variable d'entree entrainerait un surapprentissage severe sans capacite de generalisation. Elle doit etre utilisee uniquement pour structurer les plis de validation croisee (`GroupKFold`).

---

## Synthese pour la phase de modelisation
- La moyenne empirique de `target` est d'environ 37.47, et son ecart-type est d'environ 16.50.
- Le modele de base naif (`DummyRegressor(strategy="mean")`) devrait donc obtenir un RMSE theorique proche de 16.50 sur un decoupage aleatoire.
