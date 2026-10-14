# Plan — Modèle 26 : Ultimate Blend — Plan final basé sur le leaderboard Skore Hub réel

## Vue d'ensemble

**Objectif** : Battre le RMSE 3.49 de `24_coef_blend` (meilleur score officiel Skore Hub)
en construisant un blend de **toutes les architectures complémentaires** déjà prouvées,
avec des poids OOF optimisés sur les composantes déjà calculées.

**Contrainte forte** : run complet ≤ 15 min.

**Clé Hub cible** : `26_ultimate_blend`

**Scope** : Un seul nouveau fichier `src/model_26_ultimate_blend.py`. Ne pas modifier les existants.

---

## Leaderboard Skore Hub — Ce que la capture d'écran révèle

| Clé Hub | RMSE | MAE | Fit time (s) | Meilleur score |
|---|---|---|---|---|
| **24_coef_blend** | **3.49** | 2.56 | 399.29 | ✅ Vrai meilleur |
| 32_ultimate | 3.50 | 2.58 | 422.58 | 2ème (équipe noailhac) |
| 25_tri_blend | 3.52 | 2.59 | 99.10 | 3ème (Louis) |
| 24_wls_multi_seed | 3.53 | 2.60 | 72.56 | 4ème |
| 24_wls_multi_seed | 3.53 | 2.60 | 58.15 | 4ème (maxime) |
| 23_blend | 3.54 | 2.59 | 265.70 | 5ème |
| 25_corrected_blend | 3.56 | 2.62 | 125.02 | En dessous du 24 |
| 22_debiased | 3.56 | 2.60 | 160.29 | — |

**Enseignements critiques** :
1. `24_coef_blend` (RMSE 3.49) utilise une **architecture différente** : régression des 3
   coefficients de la parabole patient (b0, b1, b2) + blend 22 (60%) / coef (30%) / 11 (10%).
2. `32_ultimate` (RMSE 3.50, équipe noailhac) est inconnu de nous — possiblement un blend des
   meilleurs composantes avec plus de graines.
3. `25_tri_blend` (RMSE 3.52) apporte une architecture JPI-CQT complémentaire mais est moins
   bon que `24_coef_blend`.
4. Le fix leakage (`25_corrected_blend`, 3.56) **dégrade** bien la performance — confirmé.
5. `22_debiased` (3.56) : architecture avec modèles de mesure débiaisés, base du `24_coef_blend`.

**Hypothèse centrale** : `24_coef_blend` (3.49) et `25_tri_blend` (3.52) font des erreurs
**différentes** sur des sous-ensembles distincts de patients. Un blend OOF de ces deux
composantes peut potentiellement atteindre 3.46–3.48.

---

## Architecture du modèle 26 : Triple Blend OOF

```
Pilier A : 24_coef_blend (CoefBlendRegressor)          → RMSE 3.49
Pilier B : 25_tri_blend  (TriBlendRegressor / JPI-CQT) → RMSE 3.52
Pilier C : 24_wls_multi_seed (SuperBlendWLSRegressor)  → RMSE 3.53

Blend final : α * A + β * B + γ * C   (α + β + γ = 1)
Post-traitement : smooth_wls_monotonic (identique au modèle 24)
```

Les poids α, β, γ sont **optimisés sur OOF GroupKFold** — pas choisis à la main.

---

## Sous-tâche 1 — Calcul des prédictions OOF des 3 piliers

**Intent** : Entraîner chaque pilier sur les K−1 folds et prédire le fold restant (OOF),
pour obtenir 3 vecteurs de prédictions OOF sur tout le train. Ces vecteurs servent ensuite
à trouver les poids optimaux sans jamais faire de fuite.

Le cache OOF est sauvegardé dans `oof/26_triple_components_oof.csv` pour éviter de
recalculer si le script est relancé.

**Expected Outcomes** :
- Fichier `oof/26_triple_components_oof.csv` créé (colonnes : Index, fold, pred_A, pred_B, pred_C, y)
- RMSE OOF individuel affiché pour chaque pilier (doit correspondre aux chiffres Skore Hub)

**Todo List** :
- [ ] Dans `model_26_ultimate_blend.py`, écrire `_compute_oof_components(X, y, cv_splits) -> pd.DataFrame`
- [ ] Pour chaque fold k : instancier les 3 modèles, `fit` sur train_k, `predict` sur val_k
- [ ] Sauvegarder dans `oof/26_triple_components_oof.csv` avec colonnes Index, fold, pred_A, pred_B, pred_C, y
- [ ] Si le fichier existe déjà, le charger directement (cache)
- [ ] Afficher le RMSE OOF de chaque composante individuelle

**Relevant Context** :
- `src/model_24_coef_blend.py` : `CoefBlendRegressor`, `quick_grid()` (pattern de cache OOF à imiter)
- `src/model_25_tri_blend.py` : `TriBlendRegressor`
- `src/model_24_wls_multi_seed.py` : `SuperBlendWLSRegressor`
- `oof/` : répertoire déjà prévu dans model_24_coef_blend (`OOF_DIR`)

**Status** : [ ] pending

---

## Sous-tâche 2 — Recherche des poids optimaux par grille OOF

**Intent** : Avec les 3 vecteurs OOF calculés, explorer une grille de poids convexes
(α + β + γ = 1) et sélectionner la combinaison minimisant le RMSE OOF moyen sur les 5 folds.
Cette opération est quasi-instantanée (pas de retrain, juste des calculs vectoriels).

On applique le lissage `smooth_wls_monotonic` **après** le blend lors de l'évaluation
de la grille, pour avoir une comparaison équitable avec le run officiel.

**Expected Outcomes** :
- Tableau des 15–20 meilleures combinaisons de poids, trié par RMSE OOF
- Poids (α*, β*, γ*) sélectionnés pour le modèle 26

**Todo List** :
- [ ] Générer une grille de triplés (α, β, γ) avec α ∈ {0.5, 0.55, 0.6, 0.65, 0.7},
  β ∈ {0.1, 0.15, 0.2, 0.25, 0.3}, γ = 1 − α − β (≥ 0)
- [ ] Pour chaque triplet : calculer `blend = α*pred_A + β*pred_B + γ*pred_C`,
  appliquer `smooth_wls_monotonic` sur OOF, calculer RMSE per-fold puis mean ± std
- [ ] Afficher le top-5 des triplets
- [ ] Encoder les poids optimaux dans les defaults de `UltimateBlendRegressor`

**Relevant Context** :
- `src/model_24_coef_blend.py` : `quick_grid()` lignes 154–184 (pattern exact à adapter)
- `src/model_24_wls_multi_seed.py` : `smooth_wls_monotonic()` à appliquer dans la grille

**Status** : [ ] pending

---

## Sous-tâche 3 — Assemblage de `UltimateBlendRegressor` et run officiel

**Intent** : Créer l'estimateur final sklearn-compatible qui encapsule les 3 piliers avec
les poids optimaux, lancer `skore.evaluate()` officiel 5-fold GroupKFold, publier sur Hub,
générer la soumission CSV.

**Attention** : `CoefBlendRegressor` est déjà lent (Fit time 399s). En multipliant par 5 folds,
le run complet atteint ~2000s. On utilise donc `skore.evaluate()` directement sans les OOF
(le cache de la sous-tâche 1 est utilisé uniquement pour optimiser les poids, pas pour le run
officiel Skore). Le run Skore sera long (~7 min) mais dans les 15 min.

**Expected Outcomes** :
- RMSE GroupKFold officiel < 3.49 (meilleur Skore Hub actuel)
- Rapport Skore publié sous `26_ultimate_blend`
- `submissions/26_ultimate_blend.csv` généré
- `kaggle-best-score-plan.md` mis à jour

**Todo List** :
- [ ] Créer `class UltimateBlendRegressor` avec les 3 piliers et les poids α*, β*, γ*
- [ ] `fit()` : entraîne les 3 piliers sur X_train complet
- [ ] `predict()` : prédit avec chaque pilier, blend pondéré, puis `smooth_wls_monotonic`
- [ ] `main()` : argparse, chargement données, `quick_grid` optionnel (`--quick`), évaluation
  skore, fit final, prédiction, soumission, publication Hub
- [ ] Vérifier RMSE affiché vs cible 3.49
- [ ] Publier sur Hub
- [ ] Mettre à jour `kaggle-best-score-plan.md`

**Relevant Context** :
- `src/model_24_coef_blend.py` : structure complète (argparse, main, quick_grid, OOF cache)
- `src/model_25_tri_blend.py` : `TriBlendRegressor` à importer directement
- `src/model_24_wls_multi_seed.py` : `SuperBlendWLSRegressor`, `smooth_wls_monotonic`

**Status** : [ ] pending

---

## Récapitulatif des gains attendus

| Combinaison | RMSE OOF estimé | Raisonnement |
|---|---|---|
| A seul (24_coef_blend) | 3.49 | Mesuré |
| B seul (25_tri_blend) | 3.52 | Mesuré |
| C seul (24_wls_multi_seed) | 3.53 | Mesuré |
| A + B blend 50/50 | ~3.48–3.50 | Erreurs complémentaires |
| **A + B + C blend optimal** | **~3.46–3.49** | Diversification maximale |

**Cible réaliste** : RMSE 3.47–3.49  
**Cible optimiste** : RMSE 3.44–3.47 si les architectures sont vraiment complémentaires

---

## Anti-patterns à éviter absolument

- **Ne PAS appliquer le fix leakage Stage-A** (model_25_corrected_blend prouve RMSE 3.56 > 3.49)
- Ne pas calculer les poids de blend sur les données de test (fuite)
- Ne pas utiliser les prédictions in-sample pour calculer les OOF (utiliser le cache par fold)
- Ne pas changer la CV strategy (GroupKFold 5-fold sur patient_id)
- Ne pas modifier les fichiers models 21–25 existants
- Ne pas lancer de grille complète 5-fold pour la recherche des poids (utiliser les OOF déjà calculés)

---

## Ordre d'implémentation (agent mode)

1. **ST1 (OOF cache)** — écrire `_compute_oof_components()` + boucle de calcul
2. **ST2 (grille poids)** — chercher α*, β*, γ* sur les OOF
3. **ST3 (assemblage + run officiel)** — `UltimateBlendRegressor` + skore.evaluate + publish
