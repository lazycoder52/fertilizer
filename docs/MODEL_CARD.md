# Model Card: Production Fertilizer Recommendation Classifier

## 1. Model Summary & Architectural Framing

- **Model Architecture**: Ensemble Random Forest Classifier (`RandomForestClassifier`)
- **Dataset Framing**: **Synthetic benchmark dataset** with rule-generated target boundaries.
- **Model Selection Rationale**: On the development split, Decision Tree ($99.35\% \pm 0.35\%$), Random Forest ($99.27\% \pm 0.35\%$), and LightGBM ($99.11\% \pm 0.37\%$) perform within noise of each other due to rectangular decision boundaries. Random Forest was selected not under a claim of "beating" other tree algorithms, but for **variance reduction, probability calibration stability, and native, high-throughput TreeSHAP explainability support**.
- **Hyperparameter Optimization**: Optuna Bayesian TPE executed **strictly on the 80% development split** ($N=2,480$) with 5-fold Stratified Cross-Validation. The holdout set ($N=620$) remained untouched.
- **Uncertainty & Masking Retraining**: Retrained with 35% random feature masking using development distribution medians, ensuring probability distributions naturally temper under missing inputs rather than outputting false certainty.
- **Production Artifact**: `model_files/model.joblib` (Compressed, ~600 KB)
- **Input Dimension**: 10 tabular agronomic parameters (8 numerical soil/environmental test values, 2 categorical)
- **Output Dimension**: 10 target fertilizer classes with calibrated probabilities and Top-3 ranking
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

## 3. Evaluation Benchmarks & Baseline Comparison (Development Split)

Baseline algorithms were evaluated on the **Development split ($N=2,480$)** using **5-fold Stratified Cross-Validation**, reporting mean $\pm$ standard deviation:

| Model Candidate | Top-1 Accuracy | Top-3 Accuracy | Macro-F1 | Brier Score | ECE | Train Time ($s$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Rule Majority Baseline** | $33.99\% \pm 0.10\%$ | $60.73\% \pm 0.15\%$ | $0.0507 \pm 0.000$ | $1.3202 \pm 0.002$ | $0.6601 \pm 0.001$ | $0.02\text{s}$ |
| **Logistic Regression (Standardized)** | $79.96\% \pm 1.17\%$ | $98.15\% \pm 0.50\%$ | $0.6781 \pm 0.026$ | $0.2962 \pm 0.010$ | $0.0682 \pm 0.006$ | $0.15\text{s}$ |
| **Decision Tree (CART)** | $99.35\% \pm 0.35\%$ | $99.56\% \pm 0.24\%$ | $0.9756 \pm 0.016$ | $0.0129 \pm 0.007$ | $0.0065 \pm 0.003$ | $0.06\text{s}$ |
| **Random Forest (Default, 100 trees)** | $99.27\% \pm 0.35\%$ | $100.0\% \pm 0.00\%$ | $0.9743 \pm 0.011$ | $0.0310 \pm 0.003$ | $0.0804 \pm 0.004$ | $1.61\text{s}$ |
| **LightGBM (Gradient Boosting)** | $99.11\% \pm 0.37\%$ | $99.92\% \pm 0.16\%$ | $0.9786 \pm 0.016$ | $0.0173 \pm 0.008$ | $0.0093 \pm 0.005$ | $3.59\text{s}$ |

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

## 5. Calibration & Uncertainty under Missing / Imputed Inputs

In agronomic applications, farmers frequently submit partial soil test records. Rather than emitting falsely confident scores (~99%) on imputed data, the model was retrained with feature masking, producing appropriately tempered confidences:

| Input Scenario | Top-1 Accuracy | Top-3 Accuracy | Macro-F1 | Mean Confidence | Expected Calibration Error (ECE) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Soil Test Suite (10 features)** | **$98.71\%$** | **$100.0\%$** | **$0.9431$** | **$86.62\%$** | $0.1209$ | $0.0552$ |
| **NPK Core Only (5 features)** | **$75.16\%$** | **$96.94\%$** | **$0.5130$** | **$71.78\%$** | $0.0386$ | $0.3108$ |
| **Soil & Crop Only (2 features)** | **$46.29\%$** | **$79.84\%$** | **$0.1907$** | **$44.90\%$** | $0.0593$ | $0.6815$ |

> [!IMPORTANT]
> When default medians are imputed, the API response and web UI immediately display a visible **"Low confidence: using default values for missing soil test parameters"** warning, transparently communicating reduced certainty.

---

## 6. Permutation Importance & Leave-One-Feature-Out (LOFO) Ablation

### Permutation Feature Importance (Holdout Set)

| Feature | Mean Accuracy Drop | Standard Deviation | Agronomic Role |
| :--- | :---: | :---: | :--- |
| **Phosphorous** | **$0.4377$** | $0.0145$ | Primary macro-nutrient; separates DAP from Balanced NPK |
| **Potassium** | **$0.2048$** | $0.0070$ | Separates MOP and potassium-deficient soils |
| **PH** | **$0.1902$** | $0.0117$ | Critical threshold separator for Lime ($\text{pH} < 6.0$) |
| **Carbon** | **$0.1684$** | $0.0136$ | Organic matter index separating Compost and Organic Fertilizer |
| **Moisture** | **$0.1284$** | $0.0081$ | Separates Water Retaining formulations |
| **Nitrogen** | **$0.0937$** | $0.0039$ | Separates Urea |
| **Soil Type** | **$0.0489$** | $0.0043$ | Textural/chemical baseline modifier |
| **Temperature** | **$0.0026$** | $0.0008$ | Minor climatic influence |
| **Rainfall** | **$0.0002$** | $0.0005$ | Minor climatic influence |
| **Crop Type** | **$0.0000$** | $0.0014$ | Synthetic labels assign fertilizer independently of crop |

### Leave-One-Feature-Out (LOFO) Ablation

Retraining separate models omitting one feature at a time reveals individual feature necessity:

| Omitted Feature | Resulting Top-1 | Accuracy Drop | Macro-F1 | Sensitivity Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Phosphorous** | $81.13\%$ | **$-17.58\%$** | $0.7888$ | Greatest individual necessity |
| **Carbon** | $86.29\%$ | **$-12.42\%$** | $0.7827$ | Essential for compost vs chemical split |
| **Potassium** | $89.35\%$ | **$-9.35\%$** | $0.8197$ | Essential for MOP identification |
| **Moisture** | $92.26\%$ | **$-6.45\%$** | $0.7949$ | Key for water retaining formulation |
| **PH** | $93.39\%$ | **$-5.32\%$** | $0.8037$ | Essential for Lime trigger |
| **Nitrogen** | $93.55\%$ | **$-5.16\%$** | $0.8435$ | Essential for Urea trigger |
| **Soil Type** | $98.06\%$ | $-0.65\%$ | $0.8971$ | Marginal individual impact |
| **Temperature** | $98.71\%$ | $0.00\%$ | $0.9520$ | Non-essential |
| **Rainfall** | $98.71\%$ | $0.00\%$ | $0.9522$ | Non-essential |
| **Crop Type** | $99.19\%$ | $+0.48\%$ | $0.9695$ | Redundant with nutrient rules |

### Finer Input Tiers & Minimal High-Confidence Suite

| Feature Subset | Feature Count | Top-1 Accuracy | Top-3 Accuracy | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Soil & Crop Only** | 2 | $48.39\%$ | $82.58\%$ | $0.2603$ |
| **Environmental Only** | 6 | $57.74\%$ | $90.00\%$ | $0.4259$ |
| **NPK Nutrient Core** | 5 | $75.16\%$ | $96.29\%$ | $0.5448$ |
| **NPK + pH** | 6 | $81.45\%$ | $99.03\%$ | $0.7067$ |
| **Minimal Suite (`Soil, Crop, P, K, pH, Carbon`)** | **6** | **$86.77\%$** | **$98.06\%$** | **$0.6782$** |
| **Full Soil Test Suite** | **10** | **$98.71\%$** | **$100.0\%$** | **$0.9431$** |

*Takeaway*: Soil and Crop alone yields only **$48.39\%$ Top-1** accuracy. The minimal input set achieving high confidence ($\ge 85\%$ Top-1, $\ge 98\%$ Top-3) requires **Soil, Crop, Phosphorus, Potassium, pH, and Carbon** (6 features). Complete laboratory testing achieves **$98.71\%$ Top-1 and $100.0\%$ Top-3**.

---

## 7. Soil Test Sampling Noise Robustness

To evaluate model resilience to lab test measurement variations or sampling errors, Gaussian noise $\mathcal{N}(0, \sigma^2)$ was injected into continuous features:

| Noise Scale ($\sigma$) | Simulated Field Condition | Top-1 Accuracy | Macro-F1 | Relative Degradation |
| :---: | :--- | :---: | :---: | :---: |
| **$0\%$ (Clean)** | Certified lab soil test values | **$98.71\%$** | **$0.9431$** | Baseline |
| **$5\%$** | Standard field test sampling variance | **$90.00\%$** | **$0.8337$** | $-8.82\%$ |
| **$10\%$** | Moderate field testing error / variance | **$84.84\%$** | **$0.7995$** | $-14.05\%$ |
| **$20\%$** | High test kit variation or uncalibrated reading | **$72.42\%$** | **$0.6274$** | $-26.63\%$ |

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
- Fails the build if an inferior or overconfident model is committed.
