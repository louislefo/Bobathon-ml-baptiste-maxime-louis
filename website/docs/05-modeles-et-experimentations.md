---
id: modeles-et-experimentations
title: Modeles et Benchmarks Experimentaux
sidebar_position: 5
---

# Modeles et Benchmarks Experimentaux

La demarche de modelisation adoptee pour le Bobathon suit une hierarchie incrementale rigoureuse. Chaque etape apporte une hypothese clinique ou algorithmique specifique, mesuree par rapport au jalon precedent et tracee dans Skore Hub.

---

## Synthese des Performances et Chronologie

| Cle Hub | Script Source | Description Algorithmique & Features | RMSE CV | Statut |
| :--- | :--- | :--- | :--- | :--- |
| `01_dummy` | `src/baseline_dummy.py` | Moyenne empirique globale (plancher naif) | ~16.50 | Plancher initial |
| `02_ridge` | `src/model_ridge.py` | Regression Ridge sur features continues | ~11.50 (-30 %) | Valide |
| `03_hgbr` | `src/model_hgbr.py` | HistGradientBoosting + tabular_pipeline + NaN natifs | ~8.25 (-50 %) | Valide |
| `04_features` | `src/model_04_features.py` | HGBR + Ingenierie temporelle intra-patient et delais | ~6.60 (-60 %) | Valide |
| `05_rater` | `src/model_05_rater.py` | HGBR + Biais subjectif du clinicien (rater_id) | ~6.30 (-62 %) | Valide |
| `07_ensemble` | `src/model_07_ensemble.py` | Stacking Regressor (Ridge + HGBR) sur splits groupes | ~5.40 (-67 %) | Valide |
| `24_coef_blend` | `src/model_24_wls_multi_seed.py` | Regression des coefficients de parabole patient | 3.49 | Etape majeure |
| `32_ultimate` | `src/model_32_ultimate.py` | Multi-etages A, B, C1, C2 avec sigmas heteroscedastiques | 3.50 | Valide |
| `33_bayes_blend` | `src/model_33_bayes_blend.py` | 32 + BayesCurveEstimator (blend fixe 0.35 Bay) | 3.48 | Avancee bayesienne |
| `34_bayes_stack` | `src/model_34_bayes_stack.py` | **Stacking Bayesien adaptatif (incertitude + double residu)** | **3.38 - 3.40** | **Meilleur Modele Champion** |

---

## Analyse Detaillee des Paliers Cles

### 1. Baseline Naive (`01_dummy`)
- **Principe** : Estimer que chaque visite obtient simplement la moyenne globale de la cible sans exploiter la moindre variable explicative.
- **Formulation** : `y_pred = mean(y_train) ≈ 37.47`.
- **Utilité** : Fixe le plancher minimal d'evaluation.

### 2. Regression Lineaire Regularisee (`02_ridge`)
- **Principe** : Modele lineaire avec penalite L2 sur les coefficients.
- **Variables utilisees** : `sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`, `time_since_intake_off`, `on`, `off`.
- **Comportement** : Capture immediatement le signal lineaire puissant liant le score mesure en condition `off` et `on` au score debiaise.

### 3. Arbres a Gradient Boosting (`03_hgbr`)
- **Principe** : Discretisation des variables en bacs et gestion native des valeurs manquantes.
- **Gain** : Reduction de 50 % de l'erreur grace a la modelisation non lineaire des paliers de traitement.

### 4. Feature Engineering Temporel Intra-Patient (`04_features`)
- **Principe** : Reconstituer la dynamique de suivi de chaque patient au fil de ses visites consecutives (duree de la maladie, delais depuis la prise, derivees temporelles).

### 5. Correction du Biais Evaluateur (`05_rater`)
- **Hypothese clinique** : Chaque clinicien (`rater_id`) evalue avec un degre de severite ou d'indulgence propre.
- **Methode** : Integration categorielle de `rater_id` pour compenser les decalages systematiques de cotation.

### 6. Ensemble par Stacking Initial (`07_ensemble`)
- **Architecture** : Regresseur Ridge + HistGradientBoostingRegressor combines par un meta-modele Ridge sur decoupages stricts `PatientGroupKFold`.

### 7. Le Saut Qualitatif : Modeles 24 a 32 (Paraboles Longitudinales)
- Les modeles 24 et 32 ont transforme l'approche en modelisant la dynamique globale de la maladie par patient sous forme d'une courbe quadratique `y(t) = b0 + b1*t + b2*t^2`.
- L'etage A produit des estimations par visite et leurs sigmas de dispersion.
- L'etage B ajuste les trajectoires (t1 a t4) et regresse les coefficients patient (cf).

### 8. L'Aboutissement : Modele 34 Bayes Stack (Champion)
- Le **Modele 34** integre l'estimateur probabiliste `BayesCurveEstimator` non plus comme un melange fixe, mais comme des covariables explicatives enrichies pour les metamodeles finaux Ca et Cb.
- En injectant `bay_sd` (l'ecart-type a posteriori), l'algorithme apprend quand le bayesien est sur ou incertain, abaissant l'erreur a **~3.38 RMSE**.
- Consultez la page dediee : [Modele 34 - Stacking Bayesien](/doc/modele-34-bayes-stack).
