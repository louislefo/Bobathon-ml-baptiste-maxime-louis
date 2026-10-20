# Hackathon Delivery: Model 34 Bayes Stack

**Project**: Bobathon ESILV (Probabl x IBM)  
**Task**: Predict unbiased true OFF MDS-UPDRS Part III motor score in Parkinson's disease.  
**Team**: Louis Le Forestier, Maxime Laurent, Baptiste Noailhac (ESILV - Data & AI Major)  
**Champion Model**: `34_bayes_stack` (RMSE ~3.38, ~79.5% relative error reduction over baseline)  
**GitHub Repository**: [https://github.com/louislefo/Bobathon-ml-baptiste-maxime-louis](https://github.com/louislefo/Bobathon-ml-baptiste-maxime-louis)

---

## Evaluation Rubric Mapping

This delivery folder is organized explicitly to provide transparent, immediate verification across all grading criteria:

| Component | Weight | Key Evidence in this Folder |
| :--- | :--- | :--- |
| **Raw Kaggle Score** | 50% of Challenge | Champion Model 34 achieves **RMSE ~3.38**. Submission file ready in `submissions/34_bayes_stack.csv`. |
| **Code Documentation** | 30% of Challenge | Complete technical report in `DOCUMENTATION.md`, Mermaid architecture diagrams, and fully commented codebase. |
| **Model Originality** | 20% of Challenge | Multi-stage adaptive Bayesian Stacking with uncertainty propagation (`bay_sd`), Bateman PK/PD kinetics, GMM likelihood, and dual residual anchors. |
| **Video Presentation** | 30% of Overall | Video file location `video/presentation.mp4` with backup streaming links in `video/README.md`. |

---

## Directory Layout

```text
Le_Forestier_Laurent_Noailhac_v2/
├── README.md                      # Master Evaluation Guide (this document)
├── DOCUMENTATION.md               # Complete mathematical and technical specification
├── requirements.txt               # Minimal tested environment dependencies
├── model_34_bayes_stack.py        # Self-contained executable champion script
│
├── submissions/                   # Verified Kaggle submission outputs
│   └── 34_bayes_stack.csv         # Champion submission file (Index, target)
│
├── video/                         # Video presentation deliverable
│   ├── README.md                  # Video drop location and cloud streaming backup links
│   └── presentation.mp4           # Video file
│
├── src/                           # Self-contained algorithmic source modules
└── parkinson/                     # Tracking and configuration helpers
```

---

## One-Command Quickstart Reproduction

### Step 1: Install Dependencies
```bash
pip install -r Le_Forestier_Laurent_Noailhac_v2/requirements.txt
```

### Step 2: Ensure Competition Data is Present
Ensure `data/` contains:
- `data/X_train.csv`
- `data/y_train.csv`
- `data/X_test.csv`

### Step 3: Run Model Training & Submission
From the repository root:
```bash
python Le_Forestier_Laurent_Noailhac_v2/model_34_bayes_stack.py
```
Or directly inside the `Le_Forestier_Laurent_Noailhac_v2/` directory:
```bash
cd rendu
python model_34_bayes_stack.py
```

The script will:
1. Run strict patient-grouped validation (`GroupShuffleSplit`, 20% holdout patients).
2. Report RMSE by component ($C_a$, $C_b$, $C_1$, $C_2$, and smoothed ensemble).
3. Train the model across 100% of the training data.
4. Export the verified submission file to `submissions/34_bayes_stack.csv`.
5. Report evaluation metrics and traceability to Skore Hub.

---

## Key Experimental Milestones

| Milestone | Architecture | Validation RMSE | Gain vs Baseline |
| :--- | :--- | :--- | :--- |
| `01_dummy` | Global Empirical Mean Floor | 16.50 | Baseline |
| `02_ridge` | Linear L2 Regression on Numerical Features | 10.49 | -36.4% |
| `03_hgbr` | Native Histogram Gradient Boosting Regressor | 7.45 | -54.8% |
| `05_rater` | Clinician Rater Bias Correction (`rater_id`) | 4.81 | -70.8% |
| `09_longitudinal` | Bateman Pharmacokinetics for Drug Intake Delays | 4.08 | -75.2% |
| `15_trajectory` | Longitudinal Quadratic Curve Regression | 2.80 | -83.0% |
| `24_wls_multi_seed`| Weighted Least Squares Multi-Seed Parabola | 3.53 | -78.6% |
| `32_ultimate` | Multi-Stage Architecture (A, B, C1, C2) | 3.50 | -78.8% |
| `33_bayes_blend`| Fixed Bayesian Trajectory Blend | 3.48 | -78.9% |
| **`34_bayes_stack`** | **Adaptive Uncertainty Bayesian Stacking** | **~3.38** | **-79.5% (Champion)** |

---

## Core Originality & Innovations

1. **Adaptive Uncertainty Routing (`bay_sd`)**: Injects the posterior standard deviation from the Bayesian curve estimator directly into tree splitters, allowing the boosting model to adaptively choose whether to follow the Bayesian trajectory or fall back on population covariates.
2. **Dual Residual Anchoring ($C_a$ and $C_b$)**: Mitigates variance by learning residual corrections both on top of the parametric dynamic baseline $t_1$ and directly on the Bayesian posterior mean.
3. **Bateman Pharmacodynamics**: Models levodopa absorption and elimination as a continuous two-compartment kinetic curve to filter out drug-timing fluctuations from real disease state.
4. **Heteroskedastic GMM Likelihood**: Accommodates non-Gaussian observation errors and clinician subjectivity across score tiers.
5. **Strict Anti-Leakage Protocol**: 100% of meta-features and trajectory models are generated through patient-level out-of-fold validation (`GroupKFold` on `patient_id`).
6. **Monotonic Longitudinal Regularization**: Enforces biological progression constraints via quadratic parabola fitting and isotonic regression.

For detailed mathematical proofs, parameter configurations, and ablation experiments, refer to [DOCUMENTATION.md](DOCUMENTATION.md).
