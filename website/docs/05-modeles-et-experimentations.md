---
id: modeles-et-experimentations
title: Modeles et Benchmarks Experimentaux
sidebar_position: 5
---

# Modeles et Benchmarks Experimentaux

La demarche de modelisation adoptee pour le Bobathon suit une hierarchie incrementale rigoureuse. Chaque etape apporte une hypothese clinique ou algorithmique specifique, mesuree par rapport au jalon precedent et tracee dans Skore Hub.

---

## Synthese des Performances et Chronologie

| Cle Hub | Script Source | Description Algorithmique & Features | RMSE Attendu | Statut |
| :--- | :--- | :--- | :--- | :--- |
| `01_dummy` | [`src/baseline_dummy.py`](file:///c:/Users/louis/OneDrive/Desktop/Esilv/A4/Machine%20Learning/Hackathon/bobathon-esilv/src/baseline_dummy.py) | Moyenne empirique globale (plancher naif) | ~16.50 | Valide |
| `02_ridge` | [`src/model_ridge.py`](file:///c:/Users/louis/OneDrive/Desktop/Esilv/A4/Machine%20Learning/Hackathon/bobathon-esilv/src/model_ridge.py) | Regression Ridge ($\alpha=10$) sur features continues | -30 % vs dummy | Valide |
| `03_hgbr` | [`src/model_hgbr.py`](file:///c:/Users/louis/OneDrive/Desktop/Esilv/A4/Machine%20Learning/Hackathon/bobathon-esilv/src/model_hgbr.py) | HistGradientBoosting + `tabular_pipeline` + NaN natifs | -50 % vs dummy | Valide |
| `04_features` | [`src/model_04_features.py`](file:///c:/Users/louis/OneDrive/Desktop/Esilv/A4/Machine%20Learning/Hackathon/bobathon-esilv/src/model_04_features.py) | HGBR + Ingenierie temporelle intra-patient et delais | -60 % vs dummy | Valide |
| `05_rater` | [`src/model_05_rater.py`](file:///c:/Users/louis/OneDrive/Desktop/Esilv/A4/Machine%20Learning/Hackathon/bobathon-esilv/src/model_05_rater.py) | HGBR + Biais subjectif du clinicien (`rater_id`) | -62 % vs dummy | Valide |
| `06_tuned` | `src/model_06_tuned.py` | Optimisation fine des hyperparametres HGBR | -65 % vs dummy | Planifie |
| `07_ensemble` | [`src/model_07_ensemble.py`](file:///c:/Users/louis/OneDrive/Desktop/Esilv/A4/Machine%20Learning/Hackathon/bobathon-esilv/src/model_07_ensemble.py) | Stacking Regressor (Ridge + HGBR) sur splits groupes | **-67 % vs dummy** | Valide |

---

## Analyse Detaillee des Modeles

### 1. Baseline Naive (`01_dummy`)
- **Principe** : Estimer que chaque visite obtient simplement la moyenne globale de la cible sans exploiter la moindre variable explicative.
- **Formulation** : `y_pred = mean(y_train) ≈ 37.47`.
- **Utilité** : Fixe le plancher minimal d'evaluation. Tout modele de Machine Learning reel doit necessairement ameliorer significativement cette performance.

### 2. Regression Lineaire Regularisee (`02_ridge`)
- **Principe** : Modele lineaire avec penalite L2 sur les coefficients.
- **Variables utilisees** : `sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`, `time_since_intake_off`, `on`, `off`.
- **Comportement** : Capture immediatement le signal lineaire puissant liant le score mesure en condition `off` et `on` au score debiaise. Cependant, le modele lineaire standard est sensible aux valeurs manquantes et ne capture pas les seuils de saturation de la Levodopa.

### 3. Arbres a Gradient Boosting (`03_hgbr`)
- **Principe** : L'algorithme `HistGradientBoostingRegressor` discretise les features continues en bacs (*bins*) et traite nativement les valeurs manquantes sans imputation arbitraire.
- **Pipeline skrub** : Utilisation de `tabular_pipeline` pour gerer automatiquement les variables et integrer la variable derivee fondamentale :
  ```text
  time_since_diagnosis = age - age_at_diagnosis
  ```
- **Gain** : Reduction de 50 % de l'erreur par rapport au dummy grace a la modelisation non lineaire des paliers de traitement et a l'exploitation du pattern de presence/absence des examens.

### 4. Feature Engineering Temporel Intra-Patient (`04_features`)
- **Principe** : Reconstituer la dynamique de suivi de chaque patient au fil de ses visites consecutives.
- **Nouvelles variables derivees** :
  - Duree de la maladie au moment de la consultation.
  - Ecart entre la derniere prise de traitement et l'examen clinique.
  - Vitesse de progression des scores moteurs entre consultations.

### 5. Correction du Biais Evaluateur (`05_rater`)
- **Hypothese clinique** : Chaque clinicien (`rater_id`) evalue avec un degre de severite ou d'indulgence propre. Certains sous-estiment la rigidite motrice, tandis que d'autres surcotent le tremblement.
- **Methode** : Integration de l'identifiant categoriel `rater_id` dans l'arbre pour que le modele apprenne a soustraire le biais systematique du praticien avant de reconstituer la cible impartiale.

### 6. Ensemble par Stacking (`07_ensemble`)
- **Architecture** :
  1. Estimateurs de niveau 1 : Un regresseur `Ridge` (pour capturer le signal additif direct) et un `HistGradientBoostingRegressor` (pour les interactions complexes et les valeurs manquantes).
  2. Meta-estimateur : Un modele Ridge ($\alpha=1.0$) entraine sur les predictions hors-pli (*out-of-fold*) des deux modeles.
- **Validation** : Execution avec le decoupage strict [`PatientGroupKFold`](file:///c:/Users/louis/OneDrive/Desktop/Esilv/A4/Machine%20Learning/Hackathon/bobathon-esilv/src/model_07_ensemble.py#L40-L58) pour assurer l'etancheite absolue entre patients.
