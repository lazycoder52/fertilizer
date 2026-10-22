# Model Card: Production Fertilizer Recommendation Classifier

## 1. Model Summary & Architectural Framing

- **Model Architecture**: Ensemble Random Forest Classifier (`RandomForestClassifier`)
- **Dataset Framing**: **Synthetic benchmark dataset** with rule-generated target boundaries.
- **Model Selection Rationale & Trade-off**:
  Across evaluated architectures, high clean-input accuracy stems from learning the synthetic rule partitions, while robustness under missing inputs comes strictly from **feature mask augmentation**, not from model architecture. TreeSHAP attribution is natively supported across tree models including LightGBM and Decision Trees.
  When continuous inputs are omitted, unmasked models collapse to $21.77\%$ Top-1 accuracy with extreme overconfidence (mean confidence $96.7\% - 100\%$, ECE $> 0.74$).
  When trained with 35% feature masking:
  - **Masked LightGBM** achieves **$49.84\%$ Top-1** ($84.19\%$ Top-3, ECE $0.0352$).
  - **Masked Decision Tree** achieves **$48.87\%$ Top-1** ($82.26\%$ Top-3, ECE $0.0623$).
  - **Masked Random Forest (Production)** achieves **$46.29\%$ Top-1** ($79.84\%$ Top-3, ECE $0.0593$).
  Masked LightGBM slightly outperforms the Random Forest on missing inputs ($49.84\%$ vs $46.29\%$). In accordance with project policy, the Random Forest artifact is maintained as the production baseline pending architectural migration review.
- **Hyperparameter Optimization**: Optuna Bayesian TPE executed **strictly on the 80% development split** ($N=2,480$) with 5-fold Stratified Cross-Validation. The holdout set ($N=620$) remained untouched.
- **Uncertainty & Masking Training**: Trained with 35% random feature masking using development distribution medians, ensuring predictive confidence naturally tempers under missing inputs rather than outputting false certainty.
- **Production Artifact**: `model_files/model.joblib` (Compressed, ~600 KB)
- **Input Dimension**: 10 tabular agronomic parameters (8 numerical soil/environmental test values, 2 categorical)
- **Output Dimension**: 10 target fertilizer classes with Top-3 probability ranking and missing-feature uncertainty flagging
- **Explainability Engine**: Fast TreeSHAP (`shap.TreeExplainer`) for real-time feature attribution

---

## 2. Intended Use & Critical Disclaimers

### Intended Use
- **Advisory Decision Support**: Assisting farmers, agricultural extension workers, and agronomists in selecting suitable commercial fertilizer formulations based on user-provided soil test values.
- **Explainable Inference**: Providing clear, transparent attribution of which specific nutrient deficit (e.g., Phosphorus deficiency or low pH) drove the recommendation.

### Critical Disclaimers & Out-of-Scope Use
> [!CAUTION]
> **Dosage Disclaimer**: This model recommends **which** fertilizer formulation to apply based on relative nutrient balance. It does **NOT** calculate dosage (application rate in $kg/ha$ or $lbs/acre$) nor does it prescribe seasonal application timing.

- **Laboratory Ground Truth**: Machine learning predictions must complement, rather than substitute for, professional wet-chemistry soil lab tests.
- **Unvalidated Extreme Climates**: Soil test values outside trained distributions should be verified with local agronomic extension guidelines.

---

## 3. Evaluation Benchmarks & Baseline Comparison

Candidate models were evaluated under **5-Fold Stratified Cross-Validation** on the Development Split ($N=2,480$, mean $\pm$ standard deviation) and on the untouched **Holdout Test Set ($N=620$)** across clean and missing-input conditions:

| Model Candidate | 5-Fold CV Top-1 | 5-Fold CV Macro-F1 | Clean Holdout Top-1 | Clean Holdout ECE | Soil+Crop Top-1 | Soil+Crop Top-3 | Soil+Crop ECE | Soil+Crop Mean Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Rule Majority Baseline** | $33.99\% \pm 0.10\%$ | $0.0507$ | $34.03\%$ | $0.6597$ | $34.03\%$ | $60.81\%$ | $0.6597$ | $100.00\%$ |
| **Logistic Regression (Standardized)** | $80.36\% \pm 0.91\%$ | $0.6815$ | $79.84\%$ | $0.0759$ | $19.35\%$ | $66.29\%$ | $0.1880$ | $38.15\%$ |
| **Decision Tree (Clean)** | $99.28\% \pm 0.52\%$ | $0.9771$ | $99.19\%$ | $0.0081$ | $21.77\%$ | $29.84\%$ | $0.7823$ | $100.00\%$ |
| **Decision Tree (Mask-Augmented 35%)** | $94.27\% \pm 1.06\%$ | $0.8910$ | $96.77\%$ | $0.0323$ | $48.87\%$ | $82.26\%$ | $0.0623$ | $55.09\%$ |
| **LightGBM (Clean)** | $98.71\% \pm 0.35\%$ | $0.9725$ | $98.71\%$ | $0.0109$ | $21.77\%$ | $66.77\%$ | $0.7823$ | $100.00\%$ |
| **LightGBM (Mask-Augmented 35%)** | $97.79\% \pm 0.58\%$ | $0.9060$ | $97.74\%$ | $0.0082$ | $49.84\%$ | $84.19\%$ | $0.0352$ | $52.96\%$ |
| **Random Forest (Default Clean)** | $99.11\% \pm 0.33\%$ | $0.9683$ | $99.52\%$ | $0.0721$ | $21.77\%$ | $37.74\%$ | $0.7496$ | $96.73\%$ |
| **Random Forest (Tuned Clean, 75 trees)** | $99.23\% \pm 0.20\%$ | $0.9720$ | $99.35\%$ | $0.0733$ | $21.77\%$ | $55.32\%$ | $0.7537$ | $97.15\%$ |
| **Random Forest (Tuned + 35% Masking, Production)** | $98.63\% \pm 0.59\%$ | $0.9475$ | **$98.71\%$** | **$0.1209$** | **$46.29\%$** | **$79.84\%$** | **$0.0593$** | **$44.90\%$** |

### Hyperparameters (Selected via Optuna on Dev Split)
- `n_estimators`: `75`
- `max_depth`: `13`
- `min_samples_split`: `5`
- `min_samples_leaf`: `1`
- `random_state`: `42`

---

## 4. Evaluation on Strict Untouched Holdout Test Set ($N=620$)

The production model was evaluated against an **untouched $20\%$ holdout split** ($N=620$, seed 42) that tuning never saw:

- **Top-1 Accuracy**: **$98.71\%$**
- **Top-3 Accuracy**: **$100.00\%$**
- **Macro-F1**: **$0.9431$**
- **Brier Score**: **$0.0552$**
- **Expected Calibration Error (ECE)**: **$0.1209$**
- **Mean Confidence**: **$86.62\%$**

### Per-Class Holdout Metrics

| Fertilizer Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Balanced NPK Fertilizer** | $0.9394$ | $1.0000$ | $0.9688$ | 31 |
| **Compost** | $0.9868$ | $1.0000$ | $0.9934$ | 75 |
| **DAP** | $1.0000$ | $1.0000$ | $1.0000$ | 211 |
| **General Purpose Fertilizer** | $1.0000$ | $1.0000$ | $1.0000$ | 6 |
| **Gypsum** | $1.0000$ | $0.3636$ | $0.5333$ | 11 |
| **Lime** | $1.0000$ | $1.0000$ | $1.0000$ | 36 |
| **Muriate of Potash** | $1.0000$ | $1.0000$ | $1.0000$ | 65 |
| **Organic Fertilizer** | $0.9048$ | $1.0000$ | $0.9500$ | 19 |
| **Urea** | $1.0000$ | $1.0000$ | $1.0000$ | 31 |
| **Water Retaining Fertilizer** | $0.9781$ | $0.9926$ | $0.9853$ | 135 |
| **Macro Average** | **$0.9809$** | **$0.9356$** | **$0.9431$** | **620** |
| **Weighted Average** | **$0.9877$** | **$0.9871$** | **$0.9846$** | **620** |

---

## 5. Confidence Behavior & Missing Input Dynamics

In agronomic applications, farmers frequently submit partial soil test records. Rather than emitting falsely confident scores (~99%) on imputed data, the model was trained with feature masking, producing appropriately tempered confidences:

| Input Scenario | Top-1 Accuracy | Top-3 Accuracy | Macro-F1 | Mean Confidence | Expected Calibration Error (ECE) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Soil Test Suite (10 features)** | **$98.71\%$** | **$100.0\%$** | **$0.9431$** | **$86.62\%$** | $0.1209$ | $0.0552$ |
| **NPK Core Only (5 features)** | **$75.16\%$** | **$96.94\%$** | **$0.5130$** | **$71.78\%$** | $0.0386$ | $0.3108$ |
| **Soil & Crop Only (2 features)** | **$46.29\%$** | **$79.84\%$** | **$0.1907$** | **$44.90\%$** | $0.0593$ | $0.6815$ |

### Reliability Analysis

The reliability diagram below plots mean predicted confidence against empirical accuracy across 10 equal-width bins:

![Reliability Diagram](images/reliability.png)

> [!NOTE]
> **Under-Confidence on Clean Inputs**:
> On clean inputs, the production model is **under-confident**: its mean predicted top-1 confidence is **$86.62\%$**, while its actual empirical accuracy is **$98.71\%$**. This conservative gap of $\sim 12\%$ accounts for the Expected Calibration Error of $\text{ECE} = 0.1209$.
>
> **Over-Confidence of Unmasked Models on Missing Inputs**:
> When soil test parameters are omitted, unmasked models suffer severe overconfidence breakdown: their accuracy collapses to $21.77\%$, but they continue to output $96.7\% - 100\%$ confidence ($\text{ECE} \approx 0.75 - 0.78$). Masked training successfully prevents false certainty by tempering confidence to $44.90\%$ ($\text{ECE} = 0.0593$).

---

## 6. Feature Ablation & Sensitivity Analysis

The unified ablation table compares Method A (deployed masked model with median imputation for omitted continuous inputs) against Method B (dedicated models retrained exclusively on each feature subset) on the untouched holdout set ($N=620$):

| Feature Subset | Feature Count | Method A Top-1 (Deployed + Imputed) | Method A Top-3 | Method B Top-1 (Retrained on Subset) | Method B Top-3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Soil + Crop** | 2 | **46.29%** | 79.84% | **48.39%** | 82.90% |
| **+N/P/K** | 5 | **75.16%** | 96.94% | **74.03%** | 96.45% |
| **{Soil, Crop, P, K, pH, Carbon}** | 6 | **85.97%** | 96.13% | **87.10%** | 99.19% |
| **All 10 Features** | 10 | **98.71%** | 100.00% | **99.35%** | 100.00% |

### Permutation Feature Importance (Holdout Set)

| Feature | Mean Accuracy Drop | Standard Deviation | Agronomic Role |
| :--- | :---: | :---: | :--- |
| **Phosphorous** | **$43.77\%$** | $\pm 1.45\%$ | Primary macro-nutrient; separates DAP from Balanced NPK |
| **Potassium** | **$20.48\%$** | $\pm 0.70\%$ | Separates MOP and potassium-deficient soils |
| **PH** | **$19.02\%$** | $\pm 1.17\%$ | Critical threshold separator for Lime ($\text{pH} < 6.0$) |
| **Carbon** | **$16.84\%$** | $\pm 1.36\%$ | Organic matter index separating Compost and Organic Fertilizer |
| **Moisture** | **$12.84\%$** | $\pm 0.81\%$ | Separates Water Retaining formulations |
| **Nitrogen** | **$9.37\%$** | $\pm 0.39\%$ | Separates Urea |
| **Soil Type** | **$4.89\%$** | $\pm 0.43\%$ | Textural/chemical baseline modifier |
| **Temperature** | **$0.26\%$** | $\pm 0.08\%$ | Minor climatic influence |
| **Rainfall** | **$0.02\%$** | $\pm 0.05\%$ | Minor climatic influence |
| **Crop Type** | **$0.00\%$** | $\pm 0.14\%$ | Synthetic labels assign fertilizer independently of crop |

### Leave-One-Feature-Out (LOFO) Ablation (Holdout Set)

Retraining separate models omitting one feature at a time reveals individual feature necessity:

| Omitted Feature | Resulting Top-1 | Accuracy Drop | Macro-F1 | Sensitivity Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Phosphorous** | $81.77\%$ | **$+16.94\%$** | $0.7888$ | Greatest individual necessity |
| **Carbon** | $87.58\%$ | **$+11.13\%$** | $0.7827$ | Essential for compost vs chemical split |
| **Potassium** | $90.32\%$ | **$+8.39\%$** | $0.8197$ | Essential for MOP identification |
| **PH** | $93.39\%$ | **$+5.32\%$** | $0.8037$ | Essential for Lime trigger |
| **Moisture** | $93.55\%$ | **$+5.16\%$** | $0.7949$ | Key for water retaining formulation |
| **Nitrogen** | $94.35\%$ | **$+4.35\%$** | $0.8435$ | Essential for Urea trigger |
| **Soil Type** | $98.55\%$ | $+0.16\%$ | $0.9400$ | Marginal individual impact |
| **Crop Type** | $99.35\%$ | $-0.65\%$ | $0.9695$ | Redundant with nutrient rules |
| **Temperature** | $99.52\%$ | $-0.81\%$ | $0.9700$ | Non-essential |
| **Rainfall** | $99.52\%$ | $-0.81\%$ | $0.9700$ | Non-essential |

---

## 7. Soil Test Sampling Noise Robustness

To evaluate model resilience to lab test measurement variations or sampling errors, Gaussian noise $\mathcal{N}(0, \sigma^2)$ was injected into continuous features:

| Noise Scale ($\sigma$) | Simulated Field Condition | Top-1 Accuracy | Macro-F1 | Relative Degradation |
| :---: | :--- | :---: | :---: | :---: |
| **$0\%$ (Clean)** | Certified lab soil test values | **$98.71\%$** | **$0.9431$** | Baseline |
| **$5\%$** | Standard field test sampling variance | **$90.00\%$** | **$0.8337$** | $-8.82\%$ |
| **$10\%$** | Moderate field testing error / variance | **$84.84\%$** | **$0.7995$** | $-14.05\%$ |
| **$20\%$** | High test kit variation or reading error | **$72.42\%$** | **$0.6274$** | $-26.63\%$ |

---

## 8. Continuous Model Quality Gate (CI)

A dedicated continuous integration test (`tests/test_model_quality.py`) guards the deployment pipeline:
- Loads the actual binary `model.joblib` and untouched `test_data.joblib` split.
- Asserts Holdout Top-1 Accuracy $\ge 95\%$ ($98.71\%$).
- Asserts Holdout Top-3 Accuracy $\ge 99\%$ ($100.0\%$).
- Asserts Holdout Macro-F1 $\ge 90\%$ ($94.31\%$).
- Asserts Multiclass Brier score $\le 0.10$ ($0.0552$).
- Asserts Expected Calibration Error $\le 0.15$ ($0.1209$).
- Asserts target leakage check: `Remark` strictly excluded from features and test matrix.
- Asserts masked inputs temper confidence ($\le 0.65$ mean confidence).
- Asserts that key numbers in `README.md` match `model_files/metrics.json` exactly.
- Fails the build if an inferior or overconfident model is committed.
