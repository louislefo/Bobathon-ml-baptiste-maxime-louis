<p align="center">
  <img src="docs/figures/probabl-logo.svg" alt="Probabl" height="100">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/figures/ibm-logo.svg" alt="IBM" height="60">
</p>
# Help science build better Parkinson's evaluation

*Predicting and correcting bias in motor score evaluation — for each visit, predict the unbiased (“true”) OFF MDS-UPDRS motor score.*

## Getting started

```bash
git clone https://github.com/probabl-ai/bobathon-esilv.git
cd bobathon-esilv
```

The lab has three steps. Do them **in order**: each one assumes the previous one is finished.


| Step | Guide                     | What you do                                                                                                |
| ---- | ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1    | [SETUP.md](docs/SETUP.md) | Join Kaggle, form a team, install Python, create the shared Skore Hub workspace, install skore and sign in |
| 2    | [DAY1.md](docs/DAY1.md)   | Download the data, explore it (EDA), push a dummy baseline to Skore Hub                                    |
| 3    | [DAY2.md](docs/DAY2.md)   | Build real models, evaluate them with patient-grouped cross-validation, submit to Kaggle                   |


### Step 1 — Setup ([SETUP.md](docs/SETUP.md))

Follow every section of [SETUP.md](docs/SETUP.md):

1. **Join Kaggle and form a team** (maximum four people). Remember the exact team name.
2. **Install Python 3.12 or newer** and check it with `python --version` (or `python3 --version`).
3. **Create one Skore Hub workspace per team** on [skore.probabl.ai](https://skore.probabl.ai), named like the Kaggle team. One teammate creates it and invites the others.
4. **Allow Bob to use the lab skills** (tick **Skill** in Bob's Permissions menu), then **install skore and sign in** from the repo root.

#### Check your setup before moving on

Run the setup tests, with the virtual environment still active:

```bash
python -m pytest
```

See [SETUP.md step 5](docs/SETUP.md#5-install-skore-and-check-your-setup) for details.

**You cannot proceed to Day 1 until the last line says** `all passed`**.** Without a working `.skore`, you cannot push reports to Skore Hub, and every Kaggle Submission needs a Hub report URL.

### Step 2 — Day 1 ([DAY1.md](docs/DAY1.md))

1. Download the CSVs from the [competition Data tab](https://www.kaggle.com/competitions/bobathon-esilv/data) into `data/`.
2. Explore the data in Bob (EDA) and go through the workbook questions.
3. Build a **dummy mean** baseline, evaluate it with `skore.evaluate`, and `put` it to Skore Hub as `01_dummy`. Every later model must beat this score.

### Step 3 — Day 2 ([DAY2.md](docs/DAY2.md))

1. Train a **Ridge** linear model, compare it with the dummy, and make your **first Kaggle Submission**.
2. Switch to **patient-grouped cross-validation** (`GroupKFold`) so the score matches the Kaggle split.
3. Try **HistGradientBoosting** (missing values as signal), skrub `**tabular_pipeline`** (categorical columns), and skrub **DataOps** (groups baked into the graph).
4. Fit your best model on all training visits and **submit again**.

## Documents

- [SETUP.md](docs/SETUP.md) - Kaggle team, Python, Skore Hub workspace, skore install and sign-in
- [DAY1.md](docs/DAY1.md) - data, exploratory data analysis (EDA), dummy baseline
- [DAY2.md](docs/DAY2.md) - models, grouped cross-validation, Kaggle Submissions
- [CONTEXT.md](docs/CONTEXT.md) - scientific background, goals, and modelling pitfalls for the Parkinson's dataset
- Kaggle competition: [bobathon-esilv](https://www.kaggle.com/competitions/bobathon-esilv/)

## Submission rules

All code must be written and run in **Bob** (IDE or CLI), with evaluation through **skore**. Every Kaggle upload must put a Skore Hub EstimatorReport URL (`https://skore.probabl.ai/…`) in the Submission Description, and each Submission needs a **new** report key (`01_dummy`, `02_ridge`, …). Each experiment writes its file to `submissions/<report-key>.csv` (e.g. `submissions/02_ridge.csv`), so the filename matches the report key and a later model never overwrites an earlier submission. See the [competition rules](https://www.kaggle.com/competitions/bobathon-esilv/rules) for eligibility and what makes a Submission valid.

## Modèles, benchmarks et soumissions de l'équipe

Toutes les expérimentations ont été rigoureusement évaluées par validation croisée groupée par patient (`GroupKFold` 5 plis sur `patient_id`) via `skore.evaluate`, publiées sur Skore Hub et exportées dans le dossier `submissions/` :

| # | Modèle / Script | Fichier de soumission | Validation | RMSE mesuré | Gain vs Dummy | Rapport Skore Hub | Statut / Diagnostic |
|---|---|---|---|---|---|---|---|
| 0 | `01_dummy` | `01_dummy.csv` | Holdout | 16.4779 | Réf. | [#64092](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/estimators/64092) | Baseline constante |
| 1 | `02_ridge` | `02_ridge.csv` | Holdout | 10.4877 | −36.4 % | [#64093](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/estimators/64093) | Modèle linéaire simple |
| 2 | `03_hgbr` | `03_hgbr.csv` | GroupKFold 5 plis | ~7.45 | −54.8 % | [#64096](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64096) | Arbre standard skrub |
| 3 | `04_hgbr_features` | `04_hgbr_features.csv` | GroupKFold 5 plis | 4.805 | −70.8 % | [#64117](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64117) | Features patient & temporelles |
| 4 | `05_hgbr_rater` | `05_hgbr_rater.csv` | GroupKFold 5 plis | 4.8106 ± 0.094 | −70.8 % | [#64136](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64136) | Encodage catégoriel rater_id |
| 5 | `06_hgbr_tuned` | `06_hgbr_tuned.csv` | GroupKFold 5 plis | 4.7760 ± 0.106 | −71.0 % | [#64154](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64154) | Tuned RandomizedSearchCV (MAE 3.63) |
| 6 | `07_ensemble` | `07_ensemble.csv` | GroupKFold 5 plis | 4.8003 ± 0.103 | −70.9 % | [#64167](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/64167) | Stacking Ridge + HGBR |
| 7 | `08_hgbr_blend` | `08_hgbr_blend.csv` | GroupKFold 5 plis | 4.7459 ± 0.099 | −71.2 % | [#65509](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65509) | VotingRegressor 5 graines sur 06 |
| 8 | `09_longitudinal` | `09_longitudinal.csv` | GroupKFold 5 plis | 4.0795 ± 0.055 | −75.2 % | [#65528](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65528) | Cinétique Bateman + Trajectoires |
| 9 | `10_super_blend` | `10_super_blend.csv` | GroupKFold 5 plis | 4.0474 ± 0.060 | −75.4 % | [#65559](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65559) | Blend multi-architectures |
| 10 | `11_meta_ensemble` | `11_meta_ensemble.csv` | GroupKFold 5 plis | **4.0459 ± 0.064** | **−75.4 %** | [#65640](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65640) | Hybride HGBR + Ridge (Meilleur brut) |
| 11 | `12_lovo` | `12_lovo.csv` | GroupKFold 5 plis | 4.1317 ± 0.054 | −74.9 % | [#65983](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/65983) | **SKD001 PASSED (Zéro fuite intra-patient)** |
| 12 | `13_regularized_lovo` | `13_regularized_lovo.csv` | GroupKFold 5 plis | 4.1413 ± 0.061 | −74.9 % | [#66145](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/66145) | **SKD001 PASSED (Early stopping)** |
| 13 | `14_lovo_ensemble` | `14_lovo_ensemble.csv` | GroupKFold 5 plis | **4.1189 ± 0.064** | **−75.0 %** | Soumission locale | **SKD001 PASSED (Ensemble LOVO + Ridge)** |
| 14 | `15_trajectory` | `15_trajectory.csv` | GroupKFold 5 plis | **2.80** | **−83.0 %** | PR #9 | **Rupture majeure (score Kaggle test)** |
| 15 | `15_bateman_off` | `15_bateman_off.csv` | GroupKFold 5 plis | 4.1891 ± 0.068 | −74.6 % | [#66879](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/66879) | Pharmacodynamie OFF (signe inversé) |
| 16 | `16_cohort_features` | `16_cohort_features.csv` | GroupKFold 5 plis | 4.1873 ± 0.065 | −74.6 % | [#66962](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/66962) | Strates démographiques cohort/age |
| 17 | `17_two_stage` | `17_two_stage.csv` | GroupKFold 5 plis | 4.2966 ± 0.052 | −73.9 % | [#67121](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/67121) | Découpage bio / clinicien disjoint |
| 18 | `18_isotonic` | `18_isotonic.csv` | GroupKFold 5 plis | 4.0467 ± 0.062 | −75.4 % | [#67178](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/67178) | Calibration isotonique out-of-fold |
| 19 | `19_ultimate` | `19_ultimate.csv` | GroupKFold 5 plis | 4.0404 ± 0.065 | −75.5 % | [#67287](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/67287) | Ensemble LOVO + Longitudinal |
| 20 | `20_mae_loss` | `20_mae_loss.csv` | GroupKFold 5 plis | 4.0777 ± 0.054 | −75.3 % | [#67350](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/67350) | Comparaison squared vs absolute |
| 21 | `21_trajectory_pkpd` | `21_trajectory_pkpd.csv` | GroupKFold 5 plis | 3.7796 ± 0.054 | −77.1 % | [#68377](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/68377) | Trajectoire Huber bayésienne + PKPD |
| 22 | `21_smooth` | `21_smooth.csv` | GroupKFold 5 plis | 3.6892 ± 0.075 | −77.6 % | [#67528](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/67528) | Lissage parabolique + monotonie (MAE 2.727) |
| 23 | `22_adaptive_trajectory` | `22_adaptive_trajectory.csv` | GroupKFold 5 plis | 3.7373 ± 0.074 | −77.3 % | [#69986](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/69986) | **SKD001 PASSED (0 issue)**, débiaisage clinicien + PKPD |
| 24 | `23_super_blend` | `23_super_blend.csv` | GroupKFold 5 plis | 3.5492 ± 0.031 | −78.5 % | [#70303](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/70303) | Blend orthogonal 21+22 (MAE 2.611) |
| 25 | `24_wls_multi_seed` | `24_wls_multi_seed.csv` | GroupKFold 5 plis | **3.5340 ± 0.035** | **−78.6 %** | [#70569](https://skore.probabl.ai/Bobra%20l'exploratrice/bobathon-esilv/cross-validations/70569) | **Nouveau record absolu** (MAE 2.601, blend multi-graines 2x2 + WLS) |

Pour plus de détails méthodologiques et l'analyse complète des diagnostics de surapprentissage, consulter [kaggle-best-score-plan.md](kaggle-best-score-plan.md).
