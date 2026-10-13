# Plan — Meilleur score Kaggle Bobathon ESILV

## Vue d'ensemble

**Objectif** : minimiser le RMSE visit-level sur le leaderboard privé Kaggle (holdout par `patient_id`).

**Situation de départ** :
- Baseline dummy : RMSE ≈ 16.50 (= écart-type de `target`)
- Scripts prêts mais non exécutés : `02_ridge`, `03_hgbr`
- Aucune submission déposée à ce jour (seul `01_dummy.csv` existe)

**Approche** : exécuter d'abord les scripts existants pour avoir une première soumission valide,
puis appliquer les améliorations dans l'ordre décroissant d'impact attendu.
Chaque expérience = nouveau script `src/`, nouveau rapport Hub, nouveau CSV `submissions/`.

**Contrainte toolchain** : tout le code doit tourner dans Bob + skore. Évaluation via `skore.evaluate`
avec `GroupKFold(n_splits=5)` sur `patient_id`. Chaque soumission Kaggle nécessite l'URL Hub dans la description.

---

## Résultats mesurés et benchmarks (actualisés)

| # | Modèle / Script | Validation | RMSE mesuré | Gain vs Dummy | Rapport Skore Hub | Statut |
|---|---|---|---|---|---|---|
| 0 | `01_dummy` (Baseline) | Standard holdout | 16.4779 | Réf. | [#64092](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/estimators/64092) | Validé |
| 1 | `02_ridge` (Régression linéaire) | Standard holdout | 10.4877 | −36.4 % | [#64093](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/estimators/64093) | Validé |
| 2 | `03_hgbr` (Arbre gradient boosting) | GroupKFold (5 plis) | ~7.45 | −54.8 % | [#64096](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64096) | Validé |
| 3 | `04_hgbr_features` (Features patient & temporelles) | GroupKFold (5 plis) | 4.805 | −70.8 % | [#64117](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64117) | Validé |
| 4 | `05_hgbr_rater` (Correction clinicien) | GroupKFold (5 plis) | 4.8106 ± 0.094 | −70.8 % | [#64136](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64136) | Validé |
| 5 | `06_hgbr_tuned` (Hyperparamètres optimaux) | GroupKFold (5 plis) | 4.7760 ± 0.106 | −71.0 % | [#64154](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64154) | Validé (MAE 3.63) |
| 6 | `07_ensemble` (Stacking Ridge + HGBR de base) | GroupKFold (5 plis) | 4.8003 ± 0.103 | −70.9 % | [#64167](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64167) | Validé |
| 7 | `08_hgbr_blend` (VotingRegressor 5 graines sur 06) | GroupKFold (5 plis) | 4.7459 ± 0.099 | −71.2 % | [#65509](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65509) | Validé |
| 8 | `09_longitudinal` (Cynétique Bateman + Trajectoires) | GroupKFold (5 plis) | 4.0795 ± 0.055 | −75.2 % | [#65528](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65528) | Rupture majeure |
| 9 | `10_super_blend` (Ensemble 5 graines longitudinales) | GroupKFold (5 plis) | 4.0474 ± 0.060 | −75.4 % | [#65559](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65559) | Validé |
| 10 | `11_meta_ensemble` (Hybride HGBR + Ridge pondéré) | GroupKFold (5 plis) | **4.0459 ± 0.064** | **−75.4 %** | [#65640](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65640) | **Meilleur absolu** |
| 11 | `12_lovo` (Trajectoires LOVO strictes) | GroupKFold (5 plis) | 4.1317 ± 0.054 | −74.9 % | [#65983](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65983) | **SKD001 PASSED (Zéro fuite)** |
| 12 | `13_regularized_lovo` (LOVO + Early Stopping) | GroupKFold (5 plis) | 4.1413 ± 0.061 | −74.9 % | [#66145](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/66145) | **SKD001 PASSED (Régularisé)** |
| 13 | `14_lovo_ensemble` (Méta-Ensemble LOVO + Ridge) | GroupKFold (5 plis) | **4.1189 ± 0.064** | **−75.0 %** | Soumission générée | **SKD001 PASSED (Meilleur LOVO)** |

---

## Sous-tâche 1 — Exécuter les scripts existants et soumettre sur Kaggle

**Intent** : obtenir les deux premières soumissions valides le plus vite possible.

**Expected Outcomes** :
- `submissions/02_ridge.csv` présent
- `submissions/03_hgbr.csv` présent
- Rapports `02_ridge` et `03_hgbr` visibles sur Skore Hub
- Deux soumissions Kaggle avec URL Hub dans la description

**Todo List** :
1. Exécuter `src/model_ridge.py` (génère `02_ridge.csv` + push `02_ridge` sur Hub)
2. Uploader `submissions/02_ridge.csv` sur Kaggle — coller l'URL Hub dans la description
3. Exécuter `src/model_hgbr.py` (génère `03_hgbr.csv` + push `03_hgbr` sur Hub)
4. Uploader `submissions/03_hgbr.csv` sur Kaggle — coller l'URL Hub dans la description
5. Noter le RMSE GroupKFold de `03_hgbr` comme référence pour les étapes suivantes

**Relevant Context** :
- `src/model_ridge.py` : Ridge alpha=10, holdout aléatoire, features numériques uniquement
- `src/model_hgbr.py` : HGBR + `tabular_pipeline` + GroupKFold + `time_since_diagnosis`
- `parkinson/hub.py` : `load_skore_credentials()` à appeler avant `login(mode="hub")`

**Status** : [x] completed — `02_ridge.csv` (RMSE 10.4877, Hub #64093) et `03_hgbr.csv` (GroupKFold, Hub #64096) générés et publiés avec succès.

---

## Sous-tâche 2 — Feature engineering temporel intra-patient

**Intent** : le modèle actuel traite chaque visite comme indépendante. Or le score OFF d'un patient
à la visite N est fortement corrélé à ses visites précédentes. Les features de contexte patient
(statistiques agrégées, numéro de visite, tendance) capturent la trajectoire de progression.
C'est l'axe d'amélioration avec le plus fort impact attendu.

**Expected Outcomes** :
- Nouveau script `src/model_04_features.py`
- RMSE GroupKFold strictement inférieur à `03_hgbr`
- `submissions/04_hgbr_features.csv` + rapport `04_hgbr_features` sur Hub

**Todo List** :
1. Créer `src/model_04_features.py` basé sur `model_hgbr.py`
2. Trier les visites par `patient_id` + `age` (proxy de l'ordre chronologique)
3. Ajouter ces features calculées **par patient** (groupby `patient_id`, transform) :
   - `visit_number` : rang de la visite pour ce patient (1, 2, 3…)
   - `mean_off_patient` : moyenne historique de `off` pour ce patient
   - `mean_on_patient` : moyenne historique de `on` pour ce patient
   - `std_off_patient` : écart-type de `off` pour ce patient (variabilité intra-patient)
   - `n_visits_patient` : nombre total de visites du patient dans le train
4. ⚠️ **Anti-leakage critique** : ces stats doivent être calculées sur tout le train (pas de look-ahead),
   puis appliquées à X_test via un join sur `patient_id`. Pour X_test les patients sont nouveaux —
   utiliser la moyenne globale du train comme fallback (fillna).
5. Garder `GroupKFold(n_splits=5)` sur `patient_id` pour l'évaluation
6. Utiliser `tabular_pipeline(HistGradientBoostingRegressor(...))` avec les mêmes paramètres que `03_hgbr`
7. Évaluer, comparer le RMSE avec `03_hgbr`, push Hub sous `04_hgbr_features`
8. Générer `submissions/04_hgbr_features.csv` et soumettre sur Kaggle

**Relevant Context** :
- EDA : min 4, médiane 7, max 12 visites par patient — assez pour des stats intra-patient
- `off` corrélé à 0.87 avec `target` → la moyenne de `off` par patient est un proxy fort
- Pour X_test : les patients sont nouveaux, donc utiliser `mean_off_patient` global train comme fallback

**Status** : [x] completed — RMSE GroupKFold = 4.8047 ± 0.0824 (gain de −35.5 % vs 03_hgbr). Rapport Hub #50648, soumission `submissions/04_hgbr_features.csv` prete.

---

## Sous-tâche 3 — Correction du biais clinicien (`rater_id`)

**Intent** : le CONTEXT.md cite explicitement la "subjectivité du clinicien" comme source de biais.
`rater_id` est présent dans les données mais ignoré par tous les scripts. Encoder `rater_id`
comme variable catégorielle permet au modèle d'apprendre et de corriger le biais fixe par clinicien.

**Expected Outcomes** :
- Script `src/model_05_rater.py` basé sur `04_features`
- `rater_id` inclus dans les features (catégoriel)
- RMSE GroupKFold ≤ `04_hgbr_features`
- `submissions/05_hgbr_rater.csv` + rapport `05_hgbr_rater` sur Hub

**Todo List** :
1. Créer `src/model_05_rater.py` basé sur `model_04_features.py`
2. Ajouter `rater_id` dans les features comme colonne `.astype("category")`
3. Ne **pas** supprimer `rater_id` dans `prepare_features()` (contrairement aux identifiants `patient_id`/`Index`)
4. `tabular_pipeline` gère automatiquement le OneHotEncoding sur les catégorielles basse cardinalité
5. Vérifier que `rater_id` est présent dans `X_test` (sinon traiter comme NaN catégoriel)
6. Évaluer, comparer RMSE, push `05_hgbr_rater`, soumettre

**Relevant Context** :
- CONTEXT.md : "Human subjectivity in scoring" est la 1ère source de biais listée
- Si la cardinalité de `rater_id` est haute (>50 cliniciens), `tabular_pipeline` utilisera
  `StringEncoder` au lieu de OneHot — toujours correct, pas besoin de changer manuellement

**Status** : [x] completed — `src/model_05_rater.py`, RMSE GroupKFold = 4.8106 ± 0.0937. Rapport Hub [#64136](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64136), soumission `submissions/05_hgbr_rater.csv` prête. Note : 3 prédictions légèrement négatives détectées sans clipping.

---

## Sous-tâche 4 — Optimisation des hyperparamètres HGBR

**Intent** : les paramètres de `03_hgbr` (max_iter=150, lr=0.08, min_samples_leaf=20)
ont été choisis à la main. Une recherche sur grille avec GroupKFold peut améliorer le score
de quelques dixièmes de RMSE sans changer la logique du modèle.

**Expected Outcomes** :
- Script `src/model_06_tuned.py` avec `RandomizedSearchCV` ou grille manuelle
- RMSE GroupKFold ≤ `05_hgbr_rater`
- `submissions/06_hgbr_tuned.csv` + rapport `06_hgbr_tuned` sur Hub

**Todo List** :
1. Créer `src/model_06_tuned.py` basé sur le meilleur script précédent (04 ou 05)
2. Définir la grille de recherche sur les hyperparamètres HGBR :
   - `max_iter` : [200, 300, 500]
   - `learning_rate` : [0.03, 0.05, 0.08, 0.1]
   - `max_leaf_nodes` : [31, 63, 127] (défaut sklearn = 31)
   - `min_samples_leaf` : [10, 20, 40]
   - `l2_regularization` : [0.0, 0.1, 1.0]
3. Utiliser `RandomizedSearchCV(estimator=pipeline, param_distributions=grid, n_iter=20,
   cv=GroupKFold(n_splits=5), scoring="neg_root_mean_squared_error", random_state=0)`
   avec `groups=patient_ids` passé dans `fit()`
4. ⚠️ `skore.evaluate` prend le modèle final (best_estimator_) — évaluer séparément ou
   utiliser directement les paramètres gagnants dans un script final propre
5. Entraîner le modèle final avec les meilleurs paramètres sur tout le train
6. Push `06_hgbr_tuned`, générer `submissions/06_hgbr_tuned.csv`, soumettre

**Paramètres recommandés de départ** (basés sur la littérature HGBR sur données médicales) :
- `max_iter=300`, `learning_rate=0.05`, `max_leaf_nodes=63`, `min_samples_leaf=20`,
  `l2_regularization=0.1`, `random_state=0`
- Ces valeurs favorisent un modèle lent à apprendre (lr bas) mais précis (arbres plus profonds)

**Relevant Context** :
- 44 590 lignes × ~15 features : HGBR reste rapide même avec max_iter=500
- `GroupKFold` doit être passé dans `cv=` du `RandomizedSearchCV` avec `groups`

**Status** : [x] completed — `src/model_06_tuned.py`, RMSE GroupKFold = 4.7760 ± 0.1059. Hyperparamètres gagnants : min_samples_leaf=10, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=1.0. Rapport Hub [#64154](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64154), soumission `submissions/06_hgbr_tuned.csv` prête.

---

## Sous-tâche 5 — Stacking (optionnel, si temps disponible)

**Intent** : combiner les prédictions de Ridge (bon signal linéaire via `off`/`on`) et du
meilleur HGBR (captures les interactions et NaN) via un méta-modèle Ridge simple.
Le stacking est souvent le dernier 1–2 % de gain sur les compétitions.

**Expected Outcomes** :
- Script `src/model_07_ensemble.py`
- RMSE GroupKFold ≤ meilleur modèle précédent
- `submissions/07_ensemble.csv` + rapport `07_ensemble` sur Hub

**Todo List** :
1. Utiliser `sklearn.ensemble.StackingRegressor` avec :
   - estimateurs de base : `[("ridge", ridge_pipeline), ("hgbr", best_hgbr_pipeline)]`
   - méta-modèle : `Ridge(alpha=1.0)` (simple, pas surapprentissage)
   - `cv=GroupKFold(n_splits=5)` dans le `StackingRegressor`
2. ⚠️ Le `StackingRegressor` de sklearn accepte `cv=` mais **pas** `groups=` directement —
   passer les splits précomputés : `cv=cv_splits`
3. Évaluer avec `skore.evaluate` + `splitter=cv_splits`, push `07_ensemble`, soumettre

**Status** : [x] completed — `src/model_07_ensemble.py`, RMSE GroupKFold = 4.8003 ± 0.1027 (poids méta-modèle : Ridge 0.096, HGBR 0.909). Rapport Hub [#64167](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64167), soumission `submissions/07_ensemble.csv` prête. Diagnostic : Le modèle actuel a empilé le HGBR par défaut (`max_iter=150, lr=0.08`) au lieu du HGBR tuné (`max_iter=500, lr=0.03, l2=1.0`), et 8 prédictions négatives sont à borner.

---

## Récapitulatif des paramètres HGBR optimaux mesurés (06_hgbr_tuned)

```python
HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.03,
    max_leaf_nodes=31,
    min_samples_leaf=10,
    l2_regularization=1.0,
    random_state=0,
)
```

---

## Analyse des fichiers de soumission générés (`submissions/`)

Une analyse approfondie des 14 fichiers de soumission a été effectuée :

| Fichier | Moyenne | Écart-type | Min | Max | Valeurs < 0 | Valeurs > 132 | Diagnostic qualité |
|---|---|---|---|---|---|---|---|
| `01_dummy.csv` | 37.47 | 0.00 | 37.47 | 37.47 | 0 | 0 | Constante baseline |
| `02_ridge.csv` | 37.12 | 12.55 | 3.64 | 103.26 | 0 | 0 | Linéaire, bornes respectées |
| `03_hgbr.csv` | 36.84 | 14.69 | 2.08 | 84.79 | 0 | 0 | Arbre standard |
| `04_hgbr_features.csv` | 36.67 | 15.79 | 0.00 | 88.76 | 0 | 0 | Clippé à [0, 132] |
| `05_hgbr_rater.csv` | 36.67 | 15.81 | 0.00 | 88.76 | 0 | 0 | Corrigé et clippé à [0, 132] |
| `06_hgbr_tuned.csv` | 36.67 | 15.81 | 0.00 | 89.58 | 0 | 0 | Clippé à [0, 132], meilleure dispersion |
| `07_ensemble.csv` | 36.66 | 15.86 | 0.00 | 87.95 | 0 | 0 | Corrigé et clippé à [0, 132] |
| `08_hgbr_blend.csv` | 36.68 | 15.80 | 0.00 | 89.70 | 0 | 0 | Blend multi-graines [0, 132] |
| `09_longitudinal.csv` | 36.66 | 15.84 | 0.00 | 86.71 | 0 | 0 | Rupture PKPD + trajectoires [0, 132] |
| `10_super_blend.csv` | 36.68 | 15.82 | 0.00 | 86.00 | 0 | 0 | Ensemble multi-graines longitudinal |
| `11_meta_ensemble.csv` | 36.67 | 15.82 | 0.00 | 86.76 | 0 | 0 | Hybride HGBR + Ridge pondéré (Meilleur brut) |
| `12_lovo.csv` | 36.67 | 15.84 | 0.17 | 86.41 | 0 | 0 | LOVO strict, SKD001 validé (zéro fuite) |
| `13_regularized_lovo.csv` | 36.68 | 15.80 | 0.00 | 84.66 | 0 | 0 | LOVO + Early Stopping, SKD001 validé |
| `14_lovo_ensemble.csv` | 36.67 | 15.83 | 0.00 | 85.81 | 0 | 0 | Méta-Ensemble LOVO + Ridge, SKD001 validé |

### Enseignements clés :
1. **Bornage clinique systématique** : Le score MDS-UPDRS est théoriquement borné dans `[0, 132]`. Toutes les 14 soumissions sans exception sont désormais strictement bornées et exemptes de valeurs négatives ou de NaNs.
2. **Découverte majeure sur le DGP (Data Generating Process)** :
   - `target` est **100 % strictement non-décroissante** avec l'âge pour chaque patient (0 diff négative sur 39 014 transitions).
   - L'ajustement linéaire intra-patient `target ~ age` explique **94,1 % de la variance intra-patient** (RMSE moyen résiduel de 1.19 points).
   - Cinétique de Bateman Levodopa : $C(t) = A \cdot (e^{-k_e t} - e^{-k_a t})$ avec $A=45.53$, $k_e=0.187$, $k_a=0.571$. Elle permet d'estimer `off` à partir de `on` quand `off` est manquant (réduisant le RMSE zéro-shot de 23.96 à 11.55).
   - Corrélation des biais clinicien : $r = 0.975$ entre le biais OFF et le biais ON par clinicien.
3. **Résolution du diagnostic SKD001 (Potential Overfitting)** :
   - Dans les modèles 09 à 11, l'auto-inclusion de la visite courante dans la régression intra-patient provoquait un gap Train/Test de ~25 %, déclenchant l'alerte rouge `[SKD001]`.
   - L'implémentation de la méthodologie LOVO (Leave-One-Visit-Out) dans les modèles 12, 13 et 14 exclut strictement la visite courante du calcul de la pente et de la moyenne du patient.
   - Résultat : Le contrôle `[SKD001] Potential overfitting` est désormais validé (**PASSED**) sur Skore Hub.

---

## Analyse des diagnostics Skore Hub (Rapports récents)

L'inspection détaillée des vérifications automatiques Skore (`report.checks.summarize()`) sur la série de modèles révèle :

1. **[SKD001] Potential Overfitting (Résolu)** :
   - Modèles 06 à 11 : En alerte (gap > 10 % provoqué par l'auto-inclusion des points de visite).
   - Modèles 12 à 14 (LOVO) : **PASSED**. Le gap est ramené à un niveau sain sans perte substantielle de performance en généralisation.
2. **[SKD008] Variables hautement corrélées** :
   - Présent sur les premiers pipelines (`visit_number` vs `time_since_first_visit`).
   - Supprimé dans 12-14 par élagage systématique.
3. **[SKD012] Variables inutiles à élaguer (Recommandation clé appliquée)** :
   - Les variables parasites `['cohort', 'gene', 'sexM', 'time_since_first_visit', 'time_since_intake_off']` ont été entièrement exclues de la matrice de conception des modèles LOVO.
4. **Anomalie MAPE constatée sur le dashboard** :
   - Dans le tableau Skore Hub, le MAPE affiche des valeurs astronomiques (~2.76e+13). C'est normal : plusieurs cibles `target == 0.0` provoquent des divisions par zéro dans le calcul du MAPE. Seuls le **RMSE** (métrique officielle Kaggle) et le **MAE** sont fiables.

---

## Analyse Kaggle CLI & Leaderboard

L'interrogation de l'API Kaggle via la CLI révèle l'état exact du classement et de la compétition :

### Soumissions de l'équipe (`bobathon-esilv`) :
| Fichier | Date de soumission | Score Public Leaderboard (RMSE) |
|---|---|---|
| `01_dummy.csv` | 07 Octobre | 16.421 |
| `03_hgbr.csv` | 07 Octobre | 7.190 |
| `04_hgbr_features.csv` | 07 Octobre | **4.625** |
| `06_hgbr_tuned.csv` | Prêt localement | *Estimé ~4.55 sur public LB* |
| `08_hgbr_blend.csv` | Prêt localement | *Estimé ~4.50 sur public LB* |
| `09_longitudinal.csv` | Prêt localement | *Estimé ~3.85 sur public LB* |
| `10_super_blend.csv` | Prêt localement | *Estimé ~3.80 sur public LB* |
| `11_meta_ensemble.csv` | Prêt localement | *Estimé ~3.79 sur public LB* |
| `12_lovo.csv` | Prêt localement | *Estimé ~3.90 sur public LB (Zéro surapprentissage)* |
| `13_regularized_lovo.csv` | Prêt localement | *Estimé ~3.92 sur public LB (Early stopping)* |
| `14_lovo_ensemble.csv` | Prêt localement | *Estimé ~3.88 sur public LB (Ensemble robuste)* |

### Leaderboard Kaggle :
1. **Legendaries** : `2.196`
2. **Totally Bob's** : `2.212`
3. **Jewbuzz** : `2.518`
4. **Sleep team** : `2.684`
...
23. **Notre équipe (Bobra l'exploratrice)** : `4.625` (avec 04_hgbr_features)

*Note API Kaggle* : Le workshop s'étant clôturé le 7 octobre à 17h00 (voir `docs/kaggleinfo.md`), la plateforme Kaggle a désactivé les nouveaux dépôts (`400 Submissions have been disabled for this competition`). Tous les nouveaux fichiers de soumission sont néanmoins produits, validés selon les règles de la compétition et répertoriés avec leurs URLs Skore Hub correspondantes.

---

## Travaux d'optimisation exécutés et validés

### Sous-tâche 6 — Multi-Seed Blending (08_hgbr_blend)
- **Objectif** : Lisser la variance stochastique sur les hyperparamètres de 06_hgbr_tuned via un `VotingRegressor` sur 5 graines (0, 42, 123, 456, 789).
- **Résultat** : RMSE GroupKFold = **4.7459 ± 0.0996** (gain de −0.0301 vs 06_hgbr_tuned).
- **Statut** : [x] completed — Rapport Hub [#65509](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65509), soumission `submissions/08_hgbr_blend.csv` prête.

---

### Sous-tâche 7 — Modèle Longitudinal & Pharmacodynamique (09_longitudinal)
- **Objectif** : Reconstruire la cinétique de Bateman pour Levodopa ($C(t) = 45.53 \cdot (e^{-0.187 t} - e^{-0.571 t})$), créer le substitut continu `surrogate = off.fillna(on_adjusted)`, projeter la pente intra-patient et interpoler les mesures `off` et `on` par âge exact.
- **Résultat** : RMSE GroupKFold = **4.0795 ± 0.0552** (gain massif de **−0.6664** vs 08_hgbr_blend, −75.2 % vs Dummy).
- **Statut** : [x] completed — Rapport Hub [#65528](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65528), soumission `submissions/09_longitudinal.csv` prête.

---

### Sous-tâche 8 — Super Blend Longitudinal Multi-Graines (10_super_blend)
- **Objectif** : Construire un ensemble diversifié combinant 5 architectures HGBR (variations sur `max_iter`, `learning_rate`, `min_samples_leaf`, `max_leaf_nodes`, `l2_regularization` et graines 0, 42, 123, 456, 789) sur les variables longitudinales.
- **Résultat** : RMSE GroupKFold = **4.0474 ± 0.0597** (nouveau palier franchi).
- **Statut** : [x] completed — Rapport Hub [#65559](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65559), soumission `submissions/10_super_blend.csv` prête.

---

### Sous-tâche 9 — Méta-Ensemble Hybride HGBR + Ridge Pondéré (11_meta_ensemble)
- **Objectif** : Assemblage pondéré par `VotingRegressor` alliant 4 modèles HGBR diversifiés (poids 0.22 chacun) et une régression linéaire Ridge régularisée (poids 0.12) pour capturer les scores extrêmes au-delà de 90 points.
- **Résultat** : RMSE GroupKFold = **4.0459 ± 0.0637** (Meilleur score brut absolu de l'équipe).
- **Statut** : [x] completed — Rapport Hub [#65640](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65640), soumission `submissions/11_meta_ensemble.csv` prête.

---

### Sous-tâche 10 — Trajectoires Leave-One-Visit-Out (12_lovo)
- **Objectif** : Supprimer l'auto-inclusion de la visite courante dans le calcul de la moyenne et de la régression intra-patient (schéma LOVO strict), avec élagage des variables parasites SKD012.
- **Résultat** : RMSE GroupKFold = **4.1317 ± 0.0536**, **[SKD001] Potential overfitting PASSED**.
- **Statut** : [x] completed — Rapport Hub [#65983](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65983), soumission `submissions/12_lovo.csv` prête.

---

### Sous-tâche 11 — Régularisation Anti-Overfitting & Early Stopping (13_regularized_lovo)
- **Objectif** : Renforcer la régularisation L2 (l2=4.0), augmenter les feuilles minimales (min_samples_leaf=25) et activer l'arrêt précoce (`early_stopping=True`, tolérance 25 itérations).
- **Résultat** : RMSE GroupKFold = **4.1413 ± 0.0614**, **[SKD001] Potential overfitting PASSED**.
- **Statut** : [x] completed — Rapport Hub [#66145](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/66145), soumission `submissions/13_regularized_lovo.csv` prête.

---

### Sous-tâche 12 — Méta-Ensemble Hybride LOVO + Ridge (14_lovo_ensemble)
- **Objectif** : Assemblage pondéré par `VotingRegressor` combinant 4 architectures HGBR régularisées LOVO (poids 0.22 chacun) et 1 régression linéaire Ridge (poids 0.12).
- **Résultat** : RMSE GroupKFold = **4.1189 ± 0.0643**, **[SKD001] Potential overfitting PASSED**.
- **Statut** : [x] completed — Soumission `submissions/14_lovo_ensemble.csv` prête.

---

## Anti-patterns à éviter absolument

1. **Oublier le clipping à [0, 132]** — toute prédiction hors limites dégrade inutilement le score RMSE.
2. **Imputer les NaN de `off`/`on` avant HGBR** — les NaN sont un signal clinique (OFF exam skipped), pas une erreur.
3. **Inclure `patient_id` comme feature** — leakage total sur X_test (patients jamais vus).
4. **Inclure `Index` comme feature** — identifiant technique sans signal.
5. **Utiliser un holdout aléatoire pour évaluer** — surestime le score ; utiliser uniquement `GroupKFold`.
6. **Réutiliser la même clé Hub** pour deux modèles différents — chaque soumission Kaggle doit avoir son propre URL Hub unique.
