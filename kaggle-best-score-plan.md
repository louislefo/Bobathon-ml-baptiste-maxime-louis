# Plan — Meilleur score Kaggle Bobathon ESILV

## Vue d'ensemble

**Objectif** : minimiser le RMSE visit-level sur le leaderboard privé Kaggle (holdout par `patient_id`).

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
| 8 | `09_longitudinal` (Cinétique Bateman + Trajectoires) | GroupKFold (5 plis) | 4.0795 ± 0.055 | −75.2 % | [#65528](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65528) | Rupture majeure |
| 9 | `10_super_blend` (Ensemble 5 graines longitudinales) | GroupKFold (5 plis) | 4.0474 ± 0.060 | −75.4 % | [#65559](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65559) | Validé |
| 10 | `11_meta_ensemble` (Hybride HGBR + Ridge pondéré) | GroupKFold (5 plis) | **4.0459 ± 0.064** | **−75.4 %** | [#65640](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65640) | **Meilleur brut absolu** |
| 11 | `12_lovo` (Trajectoires LOVO strictes) | GroupKFold (5 plis) | 4.1317 ± 0.054 | −74.9 % | [#65983](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65983) | **SKD001 PASSED (Zéro fuite)** |
| 12 | `13_regularized_lovo` (LOVO + Early Stopping) | GroupKFold (5 plis) | 4.1413 ± 0.061 | −74.9 % | [#66145](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/66145) | **SKD001 PASSED (Régularisé)** |
| 13 | `14_lovo_ensemble` (Méta-Ensemble LOVO + Ridge) | GroupKFold (5 plis) | **4.1189 ± 0.064** | **−75.0 %** | Soumission locale | **SKD001 PASSED (Ensemble LOVO)** |
| 14 | `15_trajectory` (Trajectoire 3 étages bayésienne) | GroupKFold (5 plis) | **2.80** | **−83.0 %** | PR #9 (merge d83b288) | **Rupture majeure absolue (Nouveau best)** |
| 15 | `15_bateman_off` (Pharmacodynamie OFF) | GroupKFold (5 plis) | 4.1891 ± 0.068 | −74.6 % | [#66879](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/66879) | Dégradation (+0.057 vs 12) |
| 16 | `16_cohort_features` (Strates démographiques) | GroupKFold (5 plis) | 4.1873 ± 0.065 | −74.6 % | [#66962](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/66962) | Dégradation (hérite de 15) |
| 17 | `17_two_stage` (Progression bio + résidu clinicien) | GroupKFold (5 plis) | 4.2966 ± 0.052 | −73.9 % | [#67121](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/67121) | Dégradation nette (+0.25 vs best) |
| 18 | `18_isotonic` (Calibration isotonique) | GroupKFold (5 plis) | 4.0467 ± 0.062 | −75.4 % | [#67178](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/67178) | Parité avec 11_meta |
| 19 | `19_ultimate` (Ensemble LOVO + Longitudinal) | GroupKFold (5 plis) | 4.0404 ± 0.065 | −75.5 % | [#67287](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/67287) | Ensemble de base |
| 21 | `21_trajectory_pkpd` (Trajectoire Huber bayésienne + PKPD) | GroupKFold (5 plis) | 3.7796 ± 0.054 | −77.1 % | [#68377](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/68377) | Trajectoire Huber bayésienne + PKPD |
| 22 | `21_smooth` (Lissage quadratique + monotonie) | GroupKFold (5 plis) | 3.6892 ± 0.075 | −77.6 % | [#67528](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/67528) | Lissage parabolique + monotonie (MAE 2.727) |
| 23 | `22_adaptive_trajectory` (Calibration clinicien + PKPD) | GroupKFold (5 plis) | 3.7373 ± 0.074 | −77.3 % | [#69986](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/69986) | **SKD001 PASSED (0 issue)**, débiaisage clinicien + PKPD |
| 24 | `23_super_blend` (Super Blend Orthogonal 21+22) | GroupKFold (5 plis) | 3.5492 ± 0.031 | −78.5 % | [#70303](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/70303) | Blend orthogonal 21+22 (MAE 2.611) |
| 25 | `24_wls_multi_seed` (Multi-Graines 2x2 + WLS) | GroupKFold (5 plis) | **3.5340 ± 0.035** | **−78.6 %** | [#70569](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/70569) | **Nouveau record absolu** (MAE 2.601, blend multi-graines + WLS) |

---

## Analyse critique de la situation actuelle

### Écart avec le top du leaderboard

| Équipe | Score public | Écart absolu vs notre best (15_trajectory) |
|---|---|---|
| Legendaries | 2.196 | −0.60 RMSE |
| Totally Bob's | 2.212 | −0.59 RMSE |
| Jewbuzz | 2.518 | −0.28 RMSE |
| **Notre équipe (15_trajectory)** | **~2.80 mesuré** | Réf. (écart considérablement réduit) |

L'introduction de la trajectoire quadratique bayésienne dans `15_trajectory` a réduit l'écart avec le top de 1.85 à seulement ~0.60 RMSE !

---

### Analyse de la rupture majeure `15_trajectory` (RMSE = 2.80)

**Ce qui fait la puissance de cette approche :**
1. **Alignement avec le processus générateur de données** : Le target synthétique suit quasiment une parabole en fonction de l'âge ($y(t) = b_0 + b_1 t + b_2 t^2$) par patient (résidu moyen de seulement 0.31 points MDS-UPDRS sur le target pur). Les arbres de décision purs découpaient des boîtes rigides sans pouvoir extrapoler cette continuité.
2. **Débruitage d'étape A + estimation d'incertitude ($\sigma$)** : L'étage A prédit la cible visite par visite et modélise son incertitude $\sigma$.
3. **Moindres carrés pondérés et robustes (Étage B)** : La parabole par patient est ajustée en pondérant chaque visite par $1/\sigma^2$, avec perte de Huber (immunité aux valeurs aberrantes de mesure) et régularisation bayésienne vers la moyenne de population $(\mu_{b_1}, \mu_{b_2})$.
4. **Apprentissage du résidu (Étage C)** : L'étage final n'a plus qu'à prédire la micro-déviation par rapport à la parabole (`target - traj_pred`).

---

### Diagnostic du `[SKD001] Potential overfitting` sur `15_trajectory`

Skore compare la performance sur les plis d'entraînement ($R^2_{\text{train}}$, $\text{RMSE}_{\text{train}}$) et les plis de validation ($R^2_{\text{val}}$, $\text{RMSE}_{\text{val}}$).
Plusieurs facteurs expliquent pourquoi Skore déclenche l'alerte sur `15_trajectory` :

1. **Incohérence Train / Predict de l'Étage A** :
   - Pendant `fit()`, l'étage B et l'étage C reçoivent des prédictions hors-fold `oof_est` (bruitées, fidèles à la réalité out-of-sample).
   - Pendant `predict(X)` sur les données d'entraînement (utilisé par Skore pour calculer le score train), `predict` appelle `self.visit_model_.predict(V)` qui est un modèle ajusté sur 100 % des données d'entraînement (in-sample, sans bruit).
   - Les entrées de la parabole sont donc artificiellement parfaites sur le train, donnant un RMSE train extrêmement bas (~0.6-0.8) face à un RMSE validation de 2.80. Ce ratio train/val élevé déclenche SKD001.
2. **Complexité de l'Étage C (600 itérations d'arbres sur le résidu)** :
   - L'étage C utilise `max_iter=600`, `learning_rate=0.03`, `min_samples_leaf=30`. Sur les résidus d'entraînement, un arbre à 600 itérations apprend par cœur le bruit résiduel intra-visite.
3. **Variables `lf_*` non LOVO** :
   - `add_longitudinal_columns` utilise `prepare_longitudinal_features(df)` de `model_09`, qui intègre la visite courante dans `mean_off_patient`, `patient_linear_surrogate`, etc., introduisant une auto-information résiduelle.

---

### Pistes concrètes pour passer sous 2.50 et viser ~2.20

1. **Intégrer la pharmacodynamie de Bateman dans l'Étage A** :
   - Actuellement, `visit_features` donne `time_since_intake_on` et `on` au HGBR sans variable PKPD explicite.
   - En intégrant explicitement `bateman_effect_on = 45.53 * (exp(-0.187*t) - exp(-0.571*t))` et `on_adjusted = on + bateman_effect_on`, l'étage A débiaise beaucoup mieux les visites avec ON, réduisant drastiquement l'incertitude $\sigma$ transmise à la parabole.
2. **Débiaisage additif explicite des cliniciens (`rater_id`)** :
   - Les raters ont un biais additif corrélé à 0.975 entre ON et OFF. Un encodage bayésien / effets fixes régularisés par rater au sein de l'étage A réduira le bruit de mesure de ~1.5 point.
3. **A priori bayésien conditionnel (Étage B)** :
   - Actuellement, $(\mu_{b_1}, \mu_{b_2})$ est un a priori global unique.
   - Conditionner la pente et la courbure aux covariables démographiques observables (`age_at_diagnosis`, `cohort`, `gene`) améliorera fortement la trajectoire des patients ayant peu de visites (1 à 3 visites).
4. **Régularisation stricte de l'Étage C (résolution de SKD001)** :
   - Réduire `max_iter` à 200–300, passer `min_samples_leaf` à 50–80, augmenter `l2_regularization` à 4.0–8.0, et activer `early_stopping=True`.
5. **Ensembling multi-graines sur la trajectoire** :
   - Faire un blend de 3 à 5 graines de l'architecture pour lisser la variance des splits de l'étage A.

---

**Ce que nos meilleurs modèles font bien :**
- Trajectoire intra-patient via Bateman PKPD (gain de −0.67 RMSE vs modèle sans trajectoire)
- Encodage du biais clinicien via `rater_id`
- Correction LOVO pour éviter la fuite intra-visite (SKD001 résolu)
- Bornage clinique [0, 132]

**Ce que nos meilleurs modèles ne font pas encore :**
1. **Pas de `time_since_intake_off`** — 78.75 % manquant mais présent pour 21.25 % des visites. Pour ces visites, la cinétique Bateman en condition OFF (non le ON) serait le signal le plus direct du score OFF mesuré.
2. **La pharmacodynamie Bateman est appliquée uniquement au signal ON** — or la physiopathologie indique qu'une correction symétrique sur OFF (quand disponible + avec `time_since_intake_off`) peut directement débiaisé la mesure observée.
3. **Pas de features inter-patients** (médiane de la cohorte par stade de maladie, variabilité par `gene`, tendance de cohorte à age=X…) — les patients de X_test sont inconnus, mais on peut utiliser des agrégats de train en stratifiant sur des covariables observables comme `age_at_diagnosis`, `cohort`, `gene`.
4. **Le modèle ne prédit pas le résidu clinicien séparément** — architectures à deux étapes possibles : 1er modèle prédit la progression biologique pure, 2ème modèle corrige le biais `rater_id` sur ce résidu.
5. **Pas de features d'interaction LEDD × temps** — la dose et son interaction avec la durée depuis le diagnostic capturent l'évolution de la fenêtre thérapeutique (Figure 2 du CONTEXT.md).
6. **Pas de calibration isotonique post-hoc** — les prédictions peuvent systématiquement sur/sous-estimer les scores extrêmes (< 10 et > 80). Une régression isotonique ou quantile sur les prédictions OOF peut corriger ce biais sans changer le modèle.

---

## Sous-tâche 15 — Correction Bateman OFF + `time_since_intake_off`

**Intent** : pour les 21.25 % des visites disposant de `time_since_intake_off`, calculer
un `off_adjusted` via la cinétique Bateman en miroir du calcul `on_adjusted` existant.
Quand `off` ET `time_since_intake_off` sont disponibles, `off_adjusted` est un meilleur
proxy du score OFF "vrai" que `off` seul. Ce signal direct n'est jamais exploité
actuellement.

**Base** : `src/model_12_lovo.py` (meilleur modèle sans fuite intra-patient)

**Expected Outcomes** :
- Nouveau script `src/model_15_bateman_off.py`
- RMSE GroupKFold < 4.1317 (amélioration vs 12_lovo)
- `submissions/15_bateman_off.csv` + rapport `15_bateman_off` sur Hub

**Todo List** :
1. Créer `src/model_15_bateman_off.py` basé sur `model_12_lovo.py`
2. Dans `prepare_lovo_features`, ajouter :
   ```python
   # Cinétique Bateman OFF (quand time_since_intake_off disponible)
   t_off = data["time_since_intake_off"].clip(lower=0.0)  # NaN restent NaN
   data["bateman_off"] = BATEMAN_A * (np.exp(-BATEMAN_KE * t_off) - np.exp(-BATEMAN_KA * t_off))
   # off_corrected = off mesuré - effet résiduel de levodopa
   data["off_corrected"] = data["off"] - data["bateman_off"]
   # Reconstruire le surrogate avec off_corrected en priorité
   data["surrogate"] = data["off_corrected"].fillna(data["off"]).fillna(data["on_adjusted"])
   ```
3. Ajouter `bateman_off` et `off_corrected` comme features explicites (en plus du nouveau `surrogate`)
4. Inclure `time_since_intake_off` comme feature directe (signal de manquance ET signal quantitatif)
   — NaN laissés tels quels, HGBR les traite nativement
5. Recalculer les statistiques LOVO (lovo_mean_surrogate, lovo_trajectory_slope…) sur le nouveau `surrogate`
6. Évaluer, push Hub `15_bateman_off`, soumettre

**Paramètres HGBR** : idem `12_lovo` (max_iter=400, lr=0.03, min_samples_leaf=15, max_leaf_nodes=31, l2=1.5, random_state=42)

**Status** : [ ] planned

---

## Sous-tâche 16 — Features inter-patients par strate démographique

**Intent** : les patients X_test sont inconnus en tant qu'individus, mais ils partagent
des caractéristiques démographiques avec les patients de train. Créer des features
d'agrégation cross-patient conditionnelles sur (`cohort`, `gene`, strate d'`age_at_diagnosis`)
pour que le modèle apprenne la "trajectoire typique" de chaque profil de patient.

**Base** : `src/model_15_bateman_off.py` (si meilleur) ou `src/model_12_lovo.py`

**Expected Outcomes** :
- Nouveau script `src/model_16_cohort_features.py`
- RMSE GroupKFold inférieur au script de base choisi
- `submissions/16_cohort_features.csv` + rapport sur Hub

**Todo List** :
1. Créer `src/model_16_cohort_features.py`
2. Dans `prepare_features`, calculer sur le jeu d'entraînement (sans leakage) :
   ```python
   # Agrégats par (cohort, gene_fillna_unknown, age_at_diagnosis_decile)
   strate_key = ["cohort", "gene_filled", "aad_decile"]
   data["aad_decile"] = pd.qcut(data["age_at_diagnosis"], q=10, labels=False, duplicates="drop")
   data["gene_filled"] = data["gene"].fillna("Unknown")
   # Sur train : calculer les stats par strate
   strate_stats = (
       train_data.groupby(strate_key)["target"]
       .agg(["mean", "std", "median"])
       .rename(columns={"mean": "strate_target_mean", "std": "strate_target_std", "median": "strate_target_median"})
       .reset_index()
   )
   # Joindre sur train ET test via les mêmes clés
   ```
3. ⚠️ Anti-leakage : calculer `strate_stats` sur le fold de train uniquement (dans la CV), puis appliquer au fold de val via un join. Pour le fit final, calculer sur tout le train.
4. Implémenter via un transformer sklearn custom (`BaseEstimator + TransformerMixin`) avec `fit` (calcule les stats) et `transform` (joint).
5. Ajouter également `ledd_strate_mean` et `off_strate_mean` pour avoir le "niveau typique" de la strate
6. Évaluer, push Hub `16_cohort_features`, soumettre

**Status** : [ ] planned

---

## Sous-tâche 17 — Architecture deux étapes : prédiction biologique + correction clinicien

**Intent** : le target = score OFF "vrai" est une somme de deux signaux :
1. **Progression biologique** : déterminée par `age`, `age_at_diagnosis`, trajectoire intra-patient, LEDD
2. **Biais clinicien** : additive par `rater_id`, corrélée entre OFF et ON (r=0.975 mesuré)

Séparer ces deux modèles peut réduire le RMSE car chaque sous-problème est plus simple.

**Base** : `src/model_12_lovo.py`

**Expected Outcomes** :
- Nouveau script `src/model_17_two_stage.py`
- RMSE GroupKFold inférieur à 4.0459 (meilleur absolu = 11_meta_ensemble)
- `submissions/17_two_stage.csv` + rapport sur Hub

**Todo List** :
1. Créer `src/model_17_two_stage.py`
2. **Stage 1 — Modèle biologique** : prédire `target` en EXCLUANT `rater_id` et tous les scores moteurs observés (`off`, `on`) — utiliser uniquement les features temporelles, démographiques et LEDD
   ```python
   bio_features = ["age", "age_at_diagnosis", "time_since_diagnosis", "ledd",
                   "visit_number", "lovo_trajectory_slope", "lovo_trajectory_pred",
                   "ledd_patient_mean", ...]
   model_bio = tabular_pipeline(HistGradientBoostingRegressor(...))
   ```
3. **Stage 2 — Modèle de résidu** : prédire le résidu `target - y_hat_bio` en utilisant `rater_id`, `off`, `on`, `bateman_*` — ce qui isole le signal clinicien du signal biologique
   ```python
   residual_features = ["rater_id", "off", "on", "off_corrected", "diff_from_lovo_pred", ...]
   model_resid = tabular_pipeline(HistGradientBoostingRegressor(...))
   ```
4. ⚠️ Dans la CV : entraîner Stage 1 sur le fold de train, générer les résidus OOF, entraîner Stage 2 sur ces résidus OOF, puis combiner. Utiliser `cross_val_predict` avec `GroupKFold` pour les résidus.
5. Prédiction finale = `model_bio.predict(X) + model_resid.predict(X)`
6. Évaluer avec `skore.evaluate` sur les prédictions finales, push Hub `17_two_stage`, soumettre

**Relevant Context** :
- CONTEXT.md : corrélation des biais cliniciens r=0.975 entre OFF et ON → isoler le biais rater est justifié
- `lovo_trajectory_pred` est déjà une estimation de la progression biologique pure

**Status** : [ ] planned

---

## Sous-tâche 18 — Calibration isotonique post-hoc des prédictions

**Intent** : les modèles HGBR peuvent systématiquement sur/sous-estimer aux extrêmes de la distribution.
La régression isotonique appliquée sur les prédictions OOF (Out-Of-Fold) corrige ce biais de calibration
sans changer l'architecture.

**Base** : `src/model_11_meta_ensemble.py` (meilleur score brut = 4.0459)

**Expected Outcomes** :
- Nouveau script `src/model_18_isotonic.py`
- RMSE GroupKFold ≤ 4.0459 (amélioration ou parité sur 11_meta_ensemble)
- `submissions/18_isotonic.csv` + rapport sur Hub

**Todo List** :
1. Créer `src/model_18_isotonic.py` basé sur `model_11_meta_ensemble.py`
2. Ajouter une étape de calibration isotonique post-hoc :
   ```python
   from sklearn.isotonic import IsotonicRegression
   from sklearn.model_selection import cross_val_predict

   # Obtenir les prédictions OOF (Out-Of-Fold) de l'ensemble
   oof_preds = cross_val_predict(ensemble, X, y, cv=cv_splits, groups=patient_ids)

   # Entraîner la calibration isotonique sur les prédictions OOF vs targets réelles
   iso = IsotonicRegression(out_of_bounds="clip")
   iso.fit(oof_preds, y)

   # Prédiction finale = iso.transform(ensemble.predict(X_test))
   ```
3. ⚠️ La calibration isotonique doit être ajustée sur les prédictions OOF (pas sur le train directement — ce serait du leakage).
4. Appliquer `np.clip(iso.transform(raw_preds), 0.0, 132.0)` pour la soumission
5. Évaluer via GroupKFold (re-générer les OOF en cours de CV), push Hub `18_isotonic`, soumettre

**Relevant Context** :
- La distribution de `target` est quasi normale (skewness = 0.051), mais les extrêmes sont difficiles à prédire pour HGBR (valeurs > 80 peu représentées)
- L'isotonic regression est non-paramétrique et monotone — elle ne peut pas créer d'extrapolations incohérentes

**Status** : [ ] planned

---

## Sous-tâche 19 — Méta-Ensemble Ultimate (LOVO + Longitudinal + Calibration)

**Intent** : combiner les meilleures architectures des deux familles (LOVO pour la robustesse,
Longitudinal pour la performance brute) dans un `VotingRegressor` final, puis appliquer
la calibration isotonique sur les prédictions OOF du méta-ensemble.

**Base** : `model_12_lovo.py` + `model_11_meta_ensemble.py` + sous-tâche 18

**Expected Outcomes** :
- Nouveau script `src/model_19_ultimate.py`
- RMSE GroupKFold < 4.0459 (dépasse le meilleur absolu)
- `submissions/19_ultimate.csv` + rapport sur Hub

**Todo List** :
1. Créer `src/model_19_ultimate.py`
2. Combiner dans un seul VotingRegressor :
   - 3 architectures HGBR longitudinales (features model_09/11, params variés, graines 0/42/123) — poids 0.20 chacun
   - 2 architectures HGBR LOVO (features model_12, params variés, graines 0/42) — poids 0.20 chacun
   - 1 Ridge sur features longitudinales — poids 0.10
   - 1 Ridge sur features LOVO — poids 0.10
3. Générer les prédictions OOF GroupKFold du méta-ensemble via `cross_val_predict`
4. Calibrer avec `IsotonicRegression` (voir sous-tâche 18)
5. Prédiction finale = `np.clip(iso.transform(meta_ensemble.predict(X_test)), 0.0, 132.0)`
6. Évaluer, push Hub `19_ultimate`, soumettre

**Pourquoi ça peut battre 11_meta_ensemble** :
- Le mélange LOVO + Longitudinal exploite deux inductive biases complémentaires : le Longitudinal modélise la trajectoire continue, le LOVO garantit l'absence de fuite intra-visite
- La calibration isotonique corrige le biais systématique aux extrêmes
- 7 modèles au lieu de 5, avec architectures plus diversifiées

**Status** : [ ] planned

---

## Sous-tâche 20 — Optimisation directe sur MAE puis conversion RMSE

**Intent** : le leaderboard optimise le RMSE, mais dans notre cross-val le MAE est
mesuré à 3.63 (modèle 06). RMSE = MAE × facteur de dispersion. Réduire le MAE via
`loss="absolute_error"` dans HGBR peut produire un meilleur RMSE sur le leaderboard
si les outliers sont rares (ce qui est le cas : target quasi normale).

**Base** : `src/model_09_longitudinal.py` (meilleur sans LOVO)

**Expected Outcomes** :
- Nouveau script `src/model_20_mae_loss.py`
- RMSE GroupKFold ≤ `09_longitudinal` (4.0795)
- `submissions/20_mae_loss.csv` + rapport sur Hub

**Todo List** :
1. Créer `src/model_20_mae_loss.py` basé sur `model_09_longitudinal.py`
2. Modifier le HGBR :
   ```python
   HistGradientBoostingRegressor(
       loss="absolute_error",  # MAE direct vs squared (MSE par défaut)
       max_iter=500,
       learning_rate=0.03,
       min_samples_leaf=10,
       max_leaf_nodes=31,
       l2_regularization=1.0,
       random_state=42,
   )
   ```
3. Évaluer RMSE ET MAE GroupKFold (les deux sont reportés par skore.evaluate)
4. Si le RMSE est ≤ 09_longitudinal → valider, push Hub `20_mae_loss`, soumettre
5. Essayer également `loss="huber"` (robuste aux outliers, hybride MAE/MSE)

**Relevant Context** :
- La distribution de target est quasi normale → MAE ≈ 0.798 × RMSE théoriquement
- Pour des distributions à queue lourde MAE loss est meilleur, ici peu de différence attendue — à confirmer expérimentalement

**Status** : [ ] planned

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
| `21_smooth.csv` | 36.68 | 15.80 | 0.00 | 86.10 | 0 | 0 | Lissage quadratique + monotonie (RMSE 3.689, MAE 2.727) |
| `22_adaptive_trajectory.csv` | 36.64 | 15.75 | 0.00 | 86.21 | 0 | 0 | Débiaisage clinicien + PKPD, SKD001 PASSED (RMSE 3.737) |
| `23_super_blend.csv` | 36.67 | 15.79 | 0.00 | 85.81 | 0 | 0 | Super Blend orthogonal 21+22 (RMSE 3.549, MAE 2.611) |
| `24_wls_multi_seed.csv` | 36.66 | 15.78 | 0.00 | 85.47 | 0 | 0 | Nouveau record absolu multi-graines 2x2 + WLS (RMSE 3.534, MAE 2.601) |

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
   - ⚠️ **Note pour les prochains scripts** : `time_since_intake_off` a été exclu à cause de sa corrélation élevée avec d'autres features DANS le contexte de 12_lovo. Pour les modèles 15+, il faudra re-évaluer si l'inclure via la correction Bateman OFF apporte un gain net.
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

## Travaux d'optimisation exécutés et validés (Sous-tâches 1–14)

### Sous-tâche 1 — Scripts de base (01–03)
- **Statut** : [x] completed — `01_dummy.csv` (RMSE 16.4779), `02_ridge.csv` (RMSE 10.4877, Hub #64093), `03_hgbr.csv` (GroupKFold, Hub #64096).

### Sous-tâche 2 — Feature engineering temporel intra-patient (04)
- **Résultat** : RMSE GroupKFold = 4.8047 ± 0.0824. Hub #50648.
- **Statut** : [x] completed

### Sous-tâche 3 — Correction du biais clinicien `rater_id` (05)
- **Résultat** : RMSE GroupKFold = 4.8106 ± 0.0937. Hub [#64136](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64136).
- **Statut** : [x] completed

### Sous-tâche 4 — Optimisation hyperparamètres HGBR (06)
- **Résultat** : RMSE GroupKFold = 4.7760 ± 0.1059. Hub [#64154](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64154).
- **Statut** : [x] completed

### Sous-tâche 5 — Stacking Ridge + HGBR (07)
- **Résultat** : RMSE GroupKFold = 4.8003 ± 0.1027. Hub [#64167](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64167).
- **Statut** : [x] completed

### Sous-tâche 6 — Multi-Seed Blending (08)
- **Résultat** : RMSE GroupKFold = 4.7459 ± 0.0996. Hub [#65509](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65509).
- **Statut** : [x] completed

### Sous-tâche 7 — Modèle Longitudinal & Pharmacodynamique (09)
- **Résultat** : RMSE GroupKFold = 4.0795 ± 0.0552 (rupture majeure). Hub [#65528](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65528).
- **Statut** : [x] completed

### Sous-tâche 8 — Super Blend Longitudinal Multi-Graines (10)
- **Résultat** : RMSE GroupKFold = 4.0474 ± 0.0597. Hub [#65559](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65559).
- **Statut** : [x] completed

### Sous-tâche 9 — Méta-Ensemble Hybride HGBR + Ridge Pondéré (11)
- **Résultat** : RMSE GroupKFold = 4.0459 ± 0.0637 (Meilleur score brut absolu). Hub [#65640](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65640).
- **Statut** : [x] completed

### Sous-tâche 10 — Trajectoires Leave-One-Visit-Out (12)
- **Résultat** : RMSE GroupKFold = 4.1317 ± 0.0536, **[SKD001] PASSED**. Hub [#65983](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65983).
- **Statut** : [x] completed

### Sous-tâche 11 — Régularisation Anti-Overfitting & Early Stopping (13)
- **Résultat** : RMSE GroupKFold = 4.1413 ± 0.0614, **[SKD001] PASSED**. Hub [#66145](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/66145).
- **Statut** : [x] completed

### Sous-tâche 12 — Méta-Ensemble Hybride LOVO + Ridge (14)
- **Résultat** : RMSE GroupKFold = 4.1189 ± 0.0643, **[SKD001] PASSED**. Soumission locale.
- **Statut** : [x] completed

---

## Priorisation des prochaines sous-tâches

| Priorité | Sous-tâche | Impact attendu | Risque | Complexité |
|---|---|---|---|---|
| 🔴 1 | **ST-15** Bateman OFF + `time_since_intake_off` | −0.05 à −0.15 RMSE | Faible (signal direct non exploité) | Facile |
| 🔴 2 | **ST-19** Ultimate Ensemble (LOVO + Longitudinal + Isotonic) | −0.10 à −0.25 RMSE | Moyen (complexité d'assemblage) | Moyenne |
| 🟡 3 | **ST-18** Calibration isotonique sur 11_meta_ensemble | −0.02 à −0.08 RMSE | Faible (ne peut pas dégrader si bien implémenté) | Facile |
| 🟡 4 | **ST-16** Features inter-patients par strate | −0.05 à −0.15 RMSE | Moyen (leakage CV si mal implémenté) | Moyenne |
| 🟢 5 | **ST-17** Architecture deux étapes bio + clinicien | −0.05 à −0.20 RMSE | Élevé (complexité CV nested) | Difficile |
| 🟢 6 | **ST-20** MAE loss dans HGBR | −0.01 à −0.05 RMSE | Faible | Facile |

**Recommandation** : commencer par ST-15 (rapide, signal direct), puis ST-18 (calibration facile sur le meilleur modèle existant), puis ST-19 (l'ensemble ultime qui combine tout).

---

## Anti-patterns à éviter absolument

1. **Oublier le clipping à [0, 132]** — toute prédiction hors limites dégrade inutilement le score RMSE.
2. **Imputer les NaN de `off`/`on` avant HGBR** — les NaN sont un signal clinique (OFF exam skipped), pas une erreur.
3. **Inclure `patient_id` comme feature** — leakage total sur X_test (patients jamais vus).
4. **Inclure `Index` comme feature** — identifiant technique sans signal.
5. **Utiliser un holdout aléatoire pour évaluer** — surestime le score ; utiliser uniquement `GroupKFold`.
6. **Réutiliser la même clé Hub** pour deux modèles différents — chaque soumission Kaggle doit avoir son propre URL Hub unique.
7. **Calculer les stats intra-patient AVEC la visite courante** — auto-leakage → SKD001 alert. Toujours utiliser la méthodologie LOVO pour les features de trajectoire.
8. **Appliquer la calibration isotonique sur le train directement** — leakage. La calibration doit être ajustée sur des prédictions OOF uniquement.
