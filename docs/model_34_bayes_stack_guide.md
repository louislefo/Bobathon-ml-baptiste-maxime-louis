# Guide Technique Complet : Modele 34 Bayes Stack (34_bayes_stack)

Projet : Bobathon ESILV (Probabl x IBM)  
Cible de prediction : Score moteur MDS-UPDRS Part III (continu, borne [0, 132]).

---

## 1. Synthese et Vue d'ensemble

Le modele `34_bayes_stack.py` represente l'aboutissement de la strategie de modelisation du projet. Il s'agit d'une architecture de **Stacking Bayesien a plusieurs etages**.

Contrairement aux approches classiques qui combinent des modeles par simple moyenne ponderee (blend lineaire fixe, comme dans le Modele 33), le Modele 34 injecte directement l'esperance et l'incertitude calculees par un estimateur bayesien de trajectoires au sein des etages de gradient boosting finaux (HGBR - HistGradientBoostingRegressor).

```
                      Donnees Brutes (X_train, y_train)
                                    |
            +-----------------------+-----------------------+
            |                                               |
      [Etage A : Visite]                            [KNN Voisins (800)]
      - est_visit, est_off, est_on                   - Courbes candidates
      - u_off, u_on, sigmas                          - Modelisation du bruit GMM
            |                                               |
      [Etage B : Trajectoire]                       [BayesCurveEstimator]
      - t1 (visit), t2, t3, t4                              |
      - cf (coefficients parabole)             Sorties Hors-Fold (OOF) :
            |                                  * bay_mean (esperance a posteriori)
      [Matrice F1 (C1 du 32)]                  * bay_sd (ecart-type / incertitude)
            |                                  * bay_minus_t1 (ecart bay - t1)
            |                                  * bay_pmean (moyenne patient)
            +-----------------------+-----------------------+
                                    |
                     [Matrice Augmentee F_aug]
                                    |
            +-----------------------+-----------------------+
            |                                               |
     [Metamodele Ca]                                 [Metamodele Cb]
     Cible : y - t1                                  Cible : y - bay_mean
     Pred : t1 + Ca(F_aug)                           Pred : bay_mean + Cb(F_aug)
            |                                               |
            +-----------------------+-----------------------+
                                    |
                        [Ensemble des 4 etages]
             Pred = 0.30*Ca + 0.40*Cb + 0.10*C1 + 0.20*C2
                                    |
                    [Lissage Longitudinal par Patient]
             Parabole + IsotonicRegression + Clipping [0, 132]
                                    |
                            Predictions Finales
```

---

## 2. Pourquoi le Stacking Bayesien surpasse le Blend Fixe

Dans le Modele 33 (`33_bayes_blend.py`), l'estimateur de courbe bayesienne produisait une prediction `bay` qui etait melangee a coefficients constants avec les etages $C_1$ et $C_2$ :
$$\text{pred}_{\text{blend}} = 0.30 \cdot C_1 + 0.30 \cdot C_2 + 0.05 \cdot CF + 0.35 \cdot \text{bay}$$

### Limite du blend fixe :
Le modele bayesien est tres performant lorsque le patient dispose de mesures fiables (off et on) et de visites regulieres. En revanche, pour un patient ayant des mesures manquantes ou tres bruitees, le bayesien est incertain. Un poids fixe de 0.35 force le modele a faire confiance au bayesien avec la meme intensite pour tous les patients et toutes les visites.

### Avantages decisifs du Stacking (Modele 34) :
1. **Prise en compte explicite de l'incertitude (`bay_sd`)** :
   En transmettant `bay_sd` comme variable explicative aux arbres HGBR, l'arbre de gradient boosting apprend a ajuster sa dependance. Si `bay_sd` est faible (courbe tres bien contrainte), l'arbre donne la priorite a la forme bayesienne. Si `bay_sd` est grand (incertitude elevee), l'arbre reduit son impact et s'appuie davantage sur les variables cliniques et de cohortes.
2. **Signal d'ecart inter-modeles (`bay_minus_t1`)** :
   Mesure la divergence entre la dynamique globale parametrique de la maladie ($t_1$) et la distribution bayesienne basee sur la vraisemblance des mesures.
3. **Double ancre de residus ($C_a$ et $C_b$)** :
   - $C_a$ conserve l'ancre $t_1$ et apprend les corrections en exploitant les variables bayesiennes.
   - $C_b$ utilise directement `bay_mean` comme point de depart et apprend les residus par rapport a cette base bayesienne deja tres proche de la realite clinique.

Ce mecanisme permet de reduire la variance residuelle et ameliore le score RMSE d'environ 0.10 point par rapport a un blend classique.

---

## 3. Architecture detaillee et Composants

### 3.1 Etage A : Imputation et Estimation au Niveau de la Visite
Herite de `UltimateRegressor` (Modele 32), l'etage A decompose les variables cliniques et les mesures en sous-ensembles :
- `_visit_full` : toutes les variables de la visite (cliniques + mesures motrices off/on).
- `_visit_no_meas` : uniquement les covariables (sans les mesures motrices) pour les cas ou les examens off/on sont manquants.
- `_visit_pop` : variables sociodemographiques generales.

Pour modeliser l'heteroscedasticite des examens cliniques, des modeles dedies (`_sigma_hgbr`) predisent l'ecart absolu $|y - \hat{y}|$, ce qui permet de calibrer la confiance accordee a chaque visite.

### 3.2 Etage B : Trajectoires Longitudinales et Coefficients
L'etage B construit 4 trajectoires parametrées par patient en combinant les estimations de l'etage A et un a priori de population :
- $t_1$ : trajectoire ponderee par la variance de chaque visite.
- $t_2$ : trajectoire debiaisee.
- $t_3$ : trajectoire conditionnelle.
- $t_4$ : trajectoire conditionnelle d'ordre 4.
- $cf$ : regression directe des coefficients de la parabole individuelle ($b_0, b_1, b_2$) du patient a partir de ses caracteristiques globales.

### 3.3 Sorties Bayesiennes : `BayesCurveEstimator`
Le coeur probabiliste du systeme repose sur un modele de vraisemblance avance :
1. **Candidats de trajectoires via K-NN** :
   Pour chaque patient, ses 800 plus proches voisins dans l'espace des covariables de reference fournissent un faisceau de trajectoires potentielles.
2. **Modele de bruit hétéroscédastique et dissymetrique** :
   Les residus de mesure $r = \log(\text{mesure} + 1) - \beta \log(\text{cible} + 1) - \delta_{\text{rater}} - \delta_{\text{delai}}$ sont modelises par des melanges de gaussiennes (GMM a 3 composantes) ajustes par tranche de severite clinique.
3. **Marginalisation de l'effet ON du patient ($\pi$)** :
   L'effet therapeutique de la levodopa varie selon les individus. L'estimateur integre une variable latente $\pi \sim \mathcal{N}(0, \sigma_\pi^2)$ marginalisee sur une grille discrete.
4. **Sorties calculees** :
   - `bay_mean` : esperance conditionnelle a posteriori du score MDS-UPDRS.
   - `bay_sd` : ecart-type de la distribution a posteriori (indicateur d'incertitude).

### 3.4 Etage C : Stacking HGBR Augmente
A la matrice standard de l'etage C1 ($F_1$), 4 colonnes sont concatenees :
1. `bay_mean` : prediction bayesienne a posteriori.
2. `bay_sd` : incertitude a posteriori.
3. `bay_minus_t1` : difference `bay_mean - t1_pred`.
4. `bay_pmean` : moyenne intra-patient des `bay_mean`.

Deux estimateurs HGBR sont ensuite ajustes sur cette matrice augmentee $F_{\text{aug}}$ :
- **Modele Ca** :
  - Cible d'apprentissage : $y - t_1$
  - Prediction : $\hat{y}_{C_a} = t_1 + C_a(F_{\text{aug}})$
- **Modele Cb** :
  - Cible d'apprentissage : $y - \text{bay\_mean}$
  - Prediction : $\hat{y}_{C_b} = \text{bay\_mean} + C_b(F_{\text{aug}})$

### 3.5 Assemblage et Lissage Final
La prediction brute est une combinaison convexe des etages :
$$\text{pred}_{\text{brut}} = 0.30 \cdot \hat{y}_{C_a} + 0.40 \cdot \hat{y}_{C_b} + 0.10 \cdot \hat{y}_{C_1} + 0.20 \cdot \hat{y}_{C_2}$$

Cette prediction passe ensuite par `smooth_predictions` (`src/model_21_smooth.py`) :
- Ajustement d'une parabole intra-patient en fonction de l'age.
- Application d'une regression isotonique pour garantir la coherence temporelle de l'evolution de la maladie.
- Clipping final strict dans l'intervalle valide de l'echelle : $[0.0, 132.0]$.

---

## 4. Prevention Stricte des Fuites de Donnees (Data Leakage)

Dans un stacking multi-niveaux, le risque majeur est la fuite d'information : si une variable explicative a ete entrainee sur l'ensemble complet incluant la cible $y$, le metamodele surajuste de maniere critique.

Le Modele 34 applique un protocole d'isolation rigoureux :
1. **Regroupement par patient (`patient_id`)** :
   Tous les decoupages (K-Fold et ShuffleSplit) sont groupes par patient. Aucune visite d'un meme patient ne peut se retrouver a la fois dans le jeu d'entrainement et dans le jeu de validation.
2. **Generation Hors-Fold Interne (Inner OOF)** :
   A l'interieur de chaque pli d'entrainement, un sous-decoupage `GroupKFold(n_splits=3)` est applique. `BayesCurveEstimator` est entraine sur le sous-train et predit sur la sous-validation. Les variables `bay_mean`, `bay_sd`, etc., sont donc strictement hors-echantillon par rapport a $y$.
3. **Isolation de l'Etage A** :
   L'etage A de `UltimateRegressor` est lui aussi reconstruit integralement hors-fold a l'interieur de chaque pli externe.
4. **Variables longitudinales `lf_`** :
   Les statistiques de series temporelles intra-patients sont calculees uniquement sur les attributs descriptifs de la matrice $X$ sans jamais faire intervenir la cible $y$.

---

## 5. Guide d'Utilisation en Ligne de Commande (CLI)

Le script propose plusieurs modes d'execution pour s'adapter aux contraintes de calcul et aux besoins d'analyse :

### 5.1 Mode Standard (Evaluation complete + Fit + Soumission)
Execute l'evaluation sur 1 pli `GroupShuffleSplit(test_size=0.2)`, affiche les RMSE par composante et globales, ajuste le modele sur 100% des donnees d'entrainement, genere le fichier de soumission Kaggle et publie le rapport sur Skore Hub.
```powershell
python src/model_34_bayes_stack.py
```
Options courantes :
- `--no-hub` : Desactive l'envoi vers la plateforme Skore Hub (utile en local ou hors-ligne).
- `--w-ca`, `--w-cb`, `--w-c1`, `--w-c2` : Permet de surcharger les poids du blend (par defaut : 0.30, 0.40, 0.10, 0.20).

### 5.2 Mode Rapide (--quick)
Utilise pour calibrer et optimiser la grille de poids OOF sans recalculer les modeles de base si les caches existent.
- Charge les predictions OOF prealables depuis `oof/33_components_oof.csv`.
- Calcule les composantes hors-fold $C_a$ et $C_b$ (avec sauvegarde automatique a chaque pli dans `oof/34_components_oof.csv` pour reprendre en cas de coupure).
- Execute une recherche sur grille en deux passes (passe 1 brute sur toutes les combinaisons a pas 0.05, passe 2 avec lissage uniquement sur le top 20).
- Affiche le tableau comparatif 32 / 33 / 34.
```powershell
python src/model_34_bayes_stack.py --quick
```

### 5.3 Mode Rapport Seul (--report-only)
Calcule uniquement l'evaluation de validation, sauvegarde le rapport Skore sous `oof/34_report.joblib` et le publie sur Skore Hub sans relancer l'entrainement final sur tout le dataset.
```powershell
python src/model_34_bayes_stack.py --report-only
```

### 5.4 Mode Publication Seule (--publish-only)
Recharge instantanement le rapport deja calcule (`oof/34_report.joblib`) et le publie directement sur Skore Hub. Zero calcul lourd.
```powershell
python src/model_34_bayes_stack.py --publish-only
```

---

## 6. Tableau Comparatif des Modeles

| Modele | Type d'architecture | Traitement du Bayesien | RMSE CV Estimatif | Robustesse Patient |
| :--- | :--- | :--- | :--- | :--- |
| **Modele 32 (Ultimate)** | Multi-etages (A, B, C1, C2) | Aucun (purement frequentiste HGBR + Trajectoires) | ~3.55 | Moyenne |
| **Modele 33 (Bayes Blend)** | 32 + BayesCurveEstimator | Blend lineaire fixe (0.35 Bay + 0.30 C1 + 0.30 C2) | ~3.48 | Bonne mais rigide |
| **Modele 34 (Bayes Stack)** | Stacking Bayesien dynamique | Injection de `bay_mean`, `bay_sd`, `bay_minus_t1` dans $C_a$ et $C_b$ | **~3.38 - 3.40** | **Optimale (pondere selon l'incertitude)** |

---

## 7. Structure des Fichiers et Caches

- `src/model_34_bayes_stack.py` : Script executable principal.
- `src/model_33_bayes_blend.py` : Fournit la classe `BayesCurveEstimator`.
- `src/model_32_ultimate.py` : Fournit le socle `UltimateRegressor` et ses etages A et B.
- `src/model_21_smooth.py` : Fournit la fonction `smooth_predictions` pour le post-traitement.
- `oof/34_components_oof.csv` : Cache OOF des composantes $C_a$ et $C_b$ avec checkpoint fold par fold.
- `oof/34_report.joblib` : Rapport d'evaluation Skore serialise.
- `submissions/34_bayes_stack.csv` : Fichier de soumission final au format Kaggle (`Index`, `target`).
