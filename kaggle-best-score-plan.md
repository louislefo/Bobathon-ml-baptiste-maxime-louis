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
| 5 | `06_hgbr_tuned` (Hyperparamètres optimaux) | GroupKFold (5 plis) | **4.7760 ± 0.106** | **−71.0 %** | [#64154](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64154) | **Meilleur individuel** |
| 6 | `07_ensemble` (Stacking Ridge + HGBR de base) | GroupKFold (5 plis) | 4.8003 ± 0.103 | −70.9 % | [#64167](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64167) | Validé |

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

Une analyse approfondie des 7 fichiers de soumission a été effectuée :

| Fichier | Moyenne | Écart-type | Min | Max | Valeurs < 0 | Valeurs > 132 | Diagnostic qualité |
|---|---|---|---|---|---|---|---|
| `01_dummy.csv` | 37.47 | 0.00 | 37.47 | 37.47 | 0 | 0 | Constante baseline |
| `02_ridge.csv` | 37.12 | 12.55 | 3.64 | 103.26 | 0 | 0 | Linéaire, bornes respectées |
| `03_hgbr.csv` | 36.84 | 14.69 | 2.08 | 84.79 | 0 | 0 | Arbre standard |
| `04_hgbr_features.csv` | 36.67 | 15.79 | 0.00 | 88.76 | 0 | 0 | Clippé à [0, 132] |
| `05_hgbr_rater.csv` | 36.67 | 15.81 | **−0.56** | 88.76 | **3** | 0 | ⚠️ Absence de clipping (3 valeurs négatives) |
| `06_hgbr_tuned.csv` | 36.67 | 15.81 | 0.00 | 89.58 | 0 | 0 | Clippé à [0, 132], meilleure dispersion |
| `07_ensemble.csv` | 36.66 | 15.86 | **−0.74** | 87.95 | **8** | 0 | ⚠️ Absence de clipping (8 valeurs négatives) |

### Enseignements clés :
1. **Bornage clinique obligatoire** : Le score MDS-UPDRS est théoriquement borné dans `[0, 132]`. Dans `05` et `07`, l'absence de `np.clip` produit des prédictions négatives qui dégradent artificiellement le RMSE sur le test set privé.
2. **Sous-estimation de l'extrémité haute** : Le max observé dans le train est de `109.50` (cible), alors que le max prédit par les modèles arborescents plafonne à `89.58`. Un assemblage pondéré avec Ridge (dont le max est `103.26`) aide à mieux capturer les cas de sévérité motrice extrême.
3. **Corrélations élevées** : Les prédictions de `04`, `05` et `06` sont corrélées à > 0.998. Le gain supplémentaire viendra de la réduction de variance (blending de graines) et de nouvelles features dynamiques séquentielles.

---

## Analyse des diagnostics Skore Hub (Rapport #64154 — `06_hgbr_tuned`)

L'inspection détaillée des vérifications automatiques Skore (`report.checks.summarize()`) sur le meilleur modèle (MAE: 3.63, RMSE: 4.78) révèle 2 alertes majeures et 1 recommandation clé :

1. **[SKD001] Potential Overfitting (Alerte rouge)** :
   - Un écart significatif train/validation est détecté sur 3 métriques prédictives sur 4.
   - Cause : `max_iter=500` avec `lr=0.03` et `min_samples_leaf=10` commence à sur-apprendre le bruit des données d'entraînement.
   - Solution : Introduire un arrêt précoce (`early_stopping`), augmenter `min_samples_leaf` ou renforcer la régularisation `l2_regularization`.
2. **[SKD008] Variables hautement corrélées (Alerte rouge)** :
   - `visit_number` vs `time_since_first_visit` (Spearman = 0.965).
   - `age` vs `age_at_diagnosis` (Spearman = 0.937).
   - Ces quasi-doublons alourdissent les arbres sans apporter de nouveau signal clinique.
3. **[SKD012] Variables inutiles à élaguer (Recommandation clé)** :
   - Les variables `['age', 'cohort', 'gene', 'ledd', 'sexM', 'time_since_intake_off']` ont une importance par permutation dont l'intervalle de confiance englobe zéro.
   - Leur suppression permet de réduire la variance du modèle, d'éliminer le surapprentissage et d'améliorer la généralisation sur le test set.
4. **Anomalie MAPE constatée sur le dashboard** :
   - Dans le tableau Skore Hub, le MAPE affiche des valeurs astronomiques (~2.76e+13). C'est normal : plusieurs cibles `target == 0.0` provoquent des divisions par zéro dans le calcul du MAPE. Seuls le **RMSE** (métrique officielle Kaggle) et le **MAE** sont fiables.

---

## Analyse Kaggle CLI & Leaderboard

L'interrogation de l'API Kaggle via la CLI révèle l'état exact du classement :

### Soumissions actuelles de l'équipe (`bobathon-esilv`) :
| Fichier | Date de soumission | Score Public Leaderboard (RMSE) |
|---|---|---|
| `01_dummy.csv` | 07 Octobre | 16.421 |
| `03_hgbr.csv` | 07 Octobre | 7.190 |
| `04_hgbr_features.csv` | 07 Octobre | **4.625** |
| `06_hgbr_tuned.csv` | *Non soumise* | *Prête à soumettre (estimée ~4.55 sur public LB)* |

### Leaderboard Kaggle (Top équipes) :
1. **Legendaries** : `2.196`
2. **Totally Bob's** : `2.212`
3. **Jewbuzz** : `2.518`
4. **Sleep team** : `2.684`
...
23. **Notre équipe (Bobra l'exploratrice)** : `4.625`

### Comment combler l'écart vers le top (< 3.0) ?
D'après `CONTEXT.md` et l'analyse des données :
1. **Modélisation à deux régimes (OFF présent vs OFF absent)** :
   - Quand `off` est renseigné : `target ≈ off - bias(rater_id)`. Le biais par clinicien va de +2.5 à +10.5 points. Un modèle direct de correction de biais atteint un RMSE local < 2.0 sur ce sous-groupe.
   - Quand `off` est manquant : le modèle doit reconstruire l'état moteur à partir de `on` + l'effet dopaminergique attendu `(off - on)`.
2. **Nettoyage des variables parasites (SKD012)** : Ne conserver que le signal moteur pur et les identifiants cliniciens.
3. **Multi-seed ensemble** : Lisser la variance des arbres.

---

## Nouvelles pistes d'amélioration pour franchir un palier

### Sous-tâche 6 — Multi-Seed Blending (08_hgbr_blend)

**Intent** : Le gradient boosting sur 44k lignes comporte une part de variance stochastique selon la graine aléatoire. L'inférence en moyennant les prédictions de 5 à 10 graines aléatoires (`seeds = [0, 42, 123, 456, 789]`) lisse la variance d'échantillonnage sans changer l'architecture.

**Résultat du test préliminaire mesuré** :
- RMSE modèle unique (seed 0) : `4.7813`
- RMSE blend 5 graines : **`4.7471`**
- **Gain immédiat : −0.0343 RMSE** (équivalent à ~0.7 % d'amélioration nette sans risque)

**Todo List** :
1. Créer `src/model_08_blend.py` reprenant les hyperparamètres de `06_hgbr_tuned`
2. Boucler sur 5 ou 10 graines et moyenner les prédictions
3. Appliquer `np.clip(preds, 0.0, 132.0)`
4. Évaluer, générer `submissions/08_hgbr_blend.csv`, publier sur Skore Hub

**Status** : [ ] pending

---

### Sous-tâche 7 — Élagage des variables bruitées & Anti-Overfitting (SKD012 + SKD001)

**Intent** : Supprimer les colonnes inutiles identifiées par Skore (`['age', 'cohort', 'gene', 'ledd', 'sexM', 'time_since_intake_off']`) et la variable doublon `time_since_first_visit`, tout en augmentant légèrement `min_samples_leaf` (ex: 20 au lieu de 10) et `l2_regularization` (2.0) pour stopper le surapprentissage.

**Todo List** :
1. Modifier `prepare_features()` pour exclure les colonnes parasites
2. Ré-entraîner HGBR et mesurer le gain en GroupKFold
3. Vérifier que l'alerte SKD001 disparaît dans le rapport Skore

**Status** : [ ] pending

---

### Sous-tâche 8 — Séquençage Temporel & Features de Transition (Lags et Diffs)

**Intent** : Dans `X_test`, chaque patient possède entre 4 et 12 visites (médiane = 7). Puisque toutes les visites de chaque patient sont disponibles, on peut ordonner les visites par `age` et calculer des transitions directes visite à visite :
1. `off_prev` : valeur de `off` à la visite précédente du patient (capturant la dynamique immédiate)
2. `diff_off_prev` : `off - off_prev` (accélération ou stabilisation de l'atteinte)
3. `time_since_prev_visit` : temps écoulé depuis la visite antérieure
4. `on_over_off_ratio` : ratio de réponse au traitement `on / (off + 1e-4)`

**Todo List** :
1. Intégrer ces features dans `prepare_features()` en s'assurant qu'aucun leakage n'intervienne (groupby par patient, sort par age, `shift(1)`)
2. Vérifier la gestion native des NaN pour la première visite du patient
3. Évaluer l'impact en GroupKFold

**Status** : [ ] pending

---

### Sous-tâche 9 — Stacking Optimisé avec HGBR Tuné (09_ensemble_tuned)

**Intent** : `07_ensemble.py` a obtenu 4.8003 car il utilisait un HGBR basique non optimisé. En ré-entraînant le Stacking avec le HGBR tuné (`max_iter=500, lr=0.03, l2=1.0`) et un Ridge calibré avec clipping strict, le méta-modèle devrait combiner l'amplitude de Ridge avec la finesse du HGBR et passer sous la barre des `4.73`.

**Status** : [ ] pending

---

## Anti-patterns à éviter absolument

1. **Oublier le clipping à [0, 132]** — toute prédiction hors limites dégrade inutilement le score RMSE.
2. **Imputer les NaN de `off`/`on` avant HGBR** — les NaN sont un signal clinique (OFF exam skipped), pas une erreur.
3. **Inclure `patient_id` comme feature** — leakage total sur X_test (patients jamais vus).
4. **Inclure `Index` comme feature** — identifiant technique sans signal.
5. **Utiliser un holdout aléatoire pour évaluer** — surestime le score ; utiliser uniquement `GroupKFold`.
6. **Réutiliser la même clé Hub** pour deux modèles différents — chaque soumission Kaggle doit avoir son propre URL Hub unique.
