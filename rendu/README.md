# Hackathon Delivery: Model 34 Bayes Stack

**Project**: Bobathon ESILV (Probabl x IBM)  
**Task**: Debias and predict motor score MDS-UPDRS Part III (true OFF) in Parkinson's disease.  
**Team**: Baptiste, Maxime, Louis (ESILV - Data & AI Major)  
**Champion Model**: `34_bayes_stack` (**Team Best: RMSE ~3.38**)

---

## 1. Directory Structure

```text
rendu/
├── README.md                          # Quickstart, overview, and reproduction guide
├── requirements.txt                   # Minimal Python package dependencies
├── model_34_bayes_stack.py            # Executable code, fully commented in English
└── documentation_34_bayes_stack.md    # Comprehensive technical guide with Mermaid diagram
```

---

## 2. Quick Start: Reproducing the Champion Submission

### Step 1: Environment Setup
Ensure Python 3.10+ is installed, then install required dependencies:
```bash
pip install -r rendu/requirements.txt
```

### Step 2: Competition Data
Place the competition CSV files inside the `data/` directory at the repository root:
- `data/X_train.csv`
- `data/y_train.csv`
- `data/X_test.csv`

### Step 3: Run Training and Submission Generation
Execute the champion model script:
```bash
python rendu/model_34_bayes_stack.py
```

The script will:
1. Run strict patient-grouped validation (`GroupShuffleSplit`, 20% holdout patients).
2. Train the full Bayesian Stacking model across 100% of the training dataset.
3. Generate the formatted Kaggle submission file: `submissions/34_bayes_stack.csv`.
4. Report evaluation metrics and quality checks via `skore`.

---

## 3. Alternative Execution Modes

The script provides modular command-line flags:

- **Quick Mode** (weight grid optimization without recomputing full models):
  ```bash
  python rendu/model_34_bayes_stack.py --quick
  ```

- **Report-Only Mode** (validation and Skore Hub publishing without final CSV export):
  ```bash
  python rendu/model_34_bayes_stack.py --report-only
  ```

- **Publish-Only Mode** (instantly republishes saved `.joblib` report):
  ```bash
  python rendu/model_34_bayes_stack.py --publish-only
  ```

- **Local Run Without Hub**:
  ```bash
  python rendu/model_34_bayes_stack.py --no-hub
  ```

---

## 4. Performance Progression

| Model | Architecture | CV RMSE | Progression |
| :--- | :--- | :--- | :--- |
| `01_dummy` | Global Empirical Mean | ~16.50 | Baseline |
| `02_ridge` | Linear L2 Regression on Continuous Features | ~11.50 | -30% error |
| `03_hgbr` | HistGradientBoosting with Native NaN Handling | ~8.25 | -50% error |
| `05_rater` | Clinician Bias Correction (`rater_id`) | ~6.30 | -62% error |
| `07_ensemble` | Stacking Regressor (Ridge + HGBR) | ~5.40 | -67% error |
| `24_coef_blend`| Patient Quadratic Trajectory Regression | 3.49 | Breakthrough |
| `32_ultimate` | Multi-Stage Architecture (A, B, C1, C2) | 3.50 | Validated |
| `33_bayes_blend`| Fixed Bayesian Blend (0.35 Bay + 0.30 C1 + 0.30 C2) | 3.48 | Bayesian Prior |
| **`34_bayes_stack`** | **Adaptive Bayesian Stacking (Uncertainty + Dual Anchors)** | **~3.38** | **Champion Score** |

---

## 5. Technical Highlights

- **Uncertainty-Aware Stacking**: Injects posterior standard deviation (`bay_sd`) directly into final HistGradientBoosting estimators so trees dynamically decide when to rely on Bayesian curves versus visit-level covariates.
- **Dual Residual Anchors ($C_a$ and $C_b$)**: Mitigates variance by learning residual corrections both on top of the parametric dynamic $t_1$ and directly on the Bayesian posterior mean.
- **Strict Anti-Leakage Protocol**: 100% of intermediate and Bayesian features are generated through patient-level out-of-fold cross-validation.
- **Longitudinal Smoothing**: Post-processing combines parabola fitting with monotonic isotonic regression and clipping to $[0, 132]$.

For in-depth mathematical formulations, data schemas, and the Mermaid architecture diagram, see [documentation_34_bayes_stack.md](file:///c:/Users/louis/OneDrive/Desktop/Esilv/A4/Machine%20Learning/Hackathon/bobathon-esilv/rendu/documentation_34_bayes_stack.md).
