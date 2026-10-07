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

## Ordre d'exécution et impact attendu

| # | Script | RMSE attendu vs dummy | Priorité |
|---|---|---|---|
| 1 | `02_ridge` (existant) | −30 % | Obligatoire |
| 2 | `03_hgbr` (existant) | −50 % | Obligatoire |
| 3 | `04_hgbr_features` (features temporelles) | −60 % | Forte |
| 4 | `05_hgbr_rater` (biais clinicien) | −62 % | Moyenne |
| 5 | `06_hgbr_tuned` (hyperparamètres optimisés) | −65 % | Moyenne |
| 6 | `07_ensemble` (stacking Ridge + HGBR) | −67 % | Optionnel |

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

**Status** : [ ] pending

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

**Status** : [ ] pending

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

**Status** : [ ] pending

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

**Status** : [ ] pending

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

**Status** : [ ] pending

---

## Récapitulatif des paramètres HGBR optimaux recommandés

```
HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=0,
    # NaN natifs — ne PAS imputer avant
)
```

## Features recommandées pour le meilleur modèle (sous-tâche 2+3)

| Feature | Source | Note |
|---|---|---|
| `off` | Raw | Corrélation 0.87 avec target — feature #1 |
| `on` | Raw | Corrélation 0.67 |
| `age` | Raw | Proxy progression |
| `ledd` | Raw | Dose levodopa |
| `time_since_intake_on` | Raw | Timing médicament |
| `time_since_intake_off` | Raw | Timing médicament |
| `age_at_diagnosis` | Raw | Caractéristique patient |
| `sexM` | Raw | Démographie |
| `cohort` | Raw catégoriel | Différences de protocole |
| `gene` | Raw catégoriel | Mutation génétique |
| `rater_id` | Raw catégoriel | Biais clinicien — **à ajouter** |
| `time_since_diagnosis` | Dérivée (age - age_at_diagnosis) | Durée maladie |
| `mean_off_patient` | Agrégée groupby patient_id | Niveau de base patient |
| `mean_on_patient` | Agrégée groupby patient_id | Niveau ON moyen patient |
| `std_off_patient` | Agrégée groupby patient_id | Variabilité intra-patient |
| `n_visits_patient` | Agrégée groupby patient_id | Durée de suivi |
| `visit_number` | Rang par patient_id + age | Progression temporelle |

## Anti-patterns à éviter absolument

1. **Imputer les NaN de `off`/`on` avant HGBR** — les NaN sont un signal clinique (OFF exam skipped), pas une erreur.
2. **Inclure `patient_id` comme feature** — leakage total sur X_test (patients jamais vus).
3. **Inclure `Index` comme feature** — identifiant technique sans signal.
4. **Utiliser un holdout aléatoire pour évaluer** — surestime le score ; utiliser uniquement `GroupKFold`.
5. **Réutiliser la même clé Hub** pour deux modèles différents — chaque soumission Kaggle doit avoir son propre URL Hub unique.
