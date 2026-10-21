# Model Card: Production Fertilizer Recommendation Classifier

## 1. Model Summary

- **Model Architecture**: Ensemble Random Forest Classifier (`RandomForestClassifier`)
- **Hyperparameter Optimization**: Optuna Bayesian TPE (Tree-structured Parzen Estimator) across 5-fold Stratified Cross-Validation
- **Production Artifact**: `model_files/model.joblib` (Compressed, 699 KB)
- **Input Dimension**: 10 tabular agronomic features (8 numerical, 2 label-encoded categorical)
- **Output Dimension**: 10 target fertilizer classes with probability distribution and Top-3 ranking
- **Explainability Engine**: Fast TreeSHAP (`shap.TreeExplainer`) for real-time feature attribution

---

## 2. Intended Use & Critical Disclaimers

### Intended Use
- **Advisory Decision Support**: Assisting farmers, agronomists, and smart irrigation/fertilization systems in selecting appropriate commercial fertilizer formulations based on sensor telemetry.
- **Explainable Inference**: Providing clear, transparent attribution of which specific nutrient deficit (e.g., Phosphorus deficiency or low pH) drove the recommendation.

### Critical Disclaimers & Out-of-Scope Use
> [!CAUTION]
> **Dosage Disclaimer**: This model recommends **which** fertilizer formulation to apply based on relative nutrient balance. It does **NOT** calculate dosage (application rate in $kg/ha$ or $lbs/acre$) nor does it prescribe seasonal application timing.

- **Laboratory Ground Truth**: Machine learning predictions must complement, rather than substitute for, professional wet-chemistry soil lab tests.
- **Unvalidated Extreme Climates**: Sensor values outside trained distributions should be verified with local agronomic extension guidelines.

---

## 3. Evaluation Benchmarks & Baseline Comparison

All models were evaluated under identical conditions using **Stratified 5-Fold Cross-Validation** ($N=3,100$) and an independent $20\%$ holdout test split ($N=620$, seed 42):

| Model Candidate | Top-1 Accuracy | Top-3 Accuracy | Macro-F1 | Brier Score (Loss) | Train Time ($s$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Rule Majority Baseline** | $34.00\% \pm 0.06\%$ | $60.74\% \pm 0.08\%$ | $0.0507 \pm 0.000$ | $1.3200$ | $0.02\text{s}$ |
| **Logistic Regression (Standardized)** | $80.45\% \pm 0.73\%$ | $98.03\% \pm 0.34\%$ | $0.6903 \pm 0.019$ | $0.2929$ | $0.18\text{s}$ |
| **Decision Tree (CART)** | $99.39\% \pm 0.12\%$ | $99.61\% \pm 0.16\%$ | $0.9763 \pm 0.008$ | $0.0123$ | $0.10\text{s}$ |
| **Random Forest (Default, 100 trees)** | $99.35\% \pm 0.32\%$ | $100.0\% \pm 0.00\%$ | $0.9795 \pm 0.012$ | $0.0258$ | $2.04\text{s}$ |
| **LightGBM (Gradient Boosting)** | $99.16\% \pm 0.33\%$ | $99.97\% \pm 0.06\%$ | $0.9862 \pm 0.009$ | $0.0155$ | $3.60\text{s}$ |
| **Optuna Tuned Random Forest (Selected)** | **$99.52\%$** | **$100.0\%$** | **$0.9851$** | **$0.0298$** | $1.85\text{s}$ |

### Selected Hyperparameters
- `n_estimators`: `125`
- `max_depth`: `16`
- `min_samples_split`: `7`
- `min_samples_leaf`: `1`
- `random_state`: `42`

---

## 4. Per-Class Performance (Holdout Set)

| Fertilizer Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Balanced NPK Fertilizer** | $0.9688$ | $1.0000$ | $0.9841$ | 31 |
| **Compost** | $1.0000$ | $1.0000$ | $1.0000$ | 75 |
| **DAP** | $1.0000$ | $1.0000$ | $1.0000$ | 211 |
| **General Purpose Fertilizer** | $1.0000$ | $1.0000$ | $1.0000$ | 6 |
| **Gypsum** | $1.0000$ | $0.8182$ | $0.9000$ | 11 |
| **Lime** | $1.0000$ | $1.0000$ | $1.0000$ | 36 |
| **Muriate of Potash** | $1.0000$ | $1.0000$ | $1.0000$ | 65 |
| **Organic Fertilizer** | $0.9500$ | $1.0000$ | $0.9744$ | 19 |
| **Urea** | $1.0000$ | $1.0000$ | $1.0000$ | 31 |
| **Water Retaining Fertilizer** | $0.9926$ | $0.9926$ | $0.9926$ | 135 |
| **Macro Average** | **$0.9911$** | **$0.9811$** | **$0.9851$** | **620** |
| **Weighted Average** | **$0.9953$** | **$0.9952$** | **$0.9950$** | **620** |

---

## 5. Global Feature Importance & Explainability

### Gini / Mean Decrease in Impurity (MDI) Importance
```
Phosphorous   ████████████████████████████████ 32.14%
Carbon        ██████████████ 14.40%
PH            █████████████ 13.91%
Potassium     █████████████ 13.33%
Moisture      █████████ 9.22%
Nitrogen      ████████ 8.96%
Soil Type     ████ 4.35%
Rainfall      █ 1.43%
Crop Type     █ 1.17%
Temperature   █ 1.09%
```

### Local TreeSHAP Explainability
Every API prediction endpoint (`/api/v1/predict` and `/api/v1/explain`) executes TreeSHAP attribution:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$
The API computes exact marginal contributions for each sensor parameter, identifying top positive drivers and negative barriers that influenced the ranking.

---

## 6. Form Friction & Ablation Analysis

To quantify how model accuracy degrades when farmers cannot afford complete sensor telemetry, an ablation study was conducted across subsets:

| Feature Subset | Features Used | Top-1 Accuracy | Top-3 Accuracy | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Categorical Only (Frictionless)** | `Soil`, `Crop` (2) | $50.65\%$ | $82.00\%$ | $0.2727$ |
| **Environmental Only** | `Soil`, `Crop`, Temp, Moisture, Rain, pH (6) | $57.81\%$ | $91.39\%$ | $0.4825$ |
| **NPK Nutrient Core** | `Soil`, `Crop`, N, P, K (5) | $74.35\%$ | $95.23\%$ | $0.5523$ |
| **Full Telemetry Suite** | All 10 Features | **$99.35\%$** | **$100.0\%$** | **$0.9795$** |

*Takeaway*: Providing N-P-K nutrient values increases top-1 precision from $50.65\%$ to $74.35\%$, while the full suite reaches $99.35\%$. For partial payloads, the production backend transparently imputes missing continuous values with median training medians to maintain operational continuity.

---

## 7. Sensor Noise Robustness Analysis

To evaluate model resilience to field telemetry errors, Gaussian noise $\mathcal{N}(0, \sigma^2)$ was injected into continuous features:

| Noise Scale ($\sigma$) | Simulated Field Condition | Top-1 Accuracy | Macro-F1 | Relative Degradation |
| :---: | :--- | :---: | :---: | :---: |
| **$0\%$ (Clean)** | Laboratory-grade calibrated sensors | **$99.52\%$** | **$0.9851$** | Baseline |
| **$5\%$** | Low-cost IoT field sensors with slight drift | **$90.97\%$** | **$0.8771$** | $-8.59\%$ |
| **$10\%$** | Uncalibrated sensor in extreme weather | **$85.48\%$** | **$0.8327$** | $-14.10\%$ |
| **$20\%$** | Severely degraded / aging telemetry | **$73.06\%$** | **$0.6592$** | $-26.58\%$ |

---

## 8. Continuous Model Quality Gate (CI)

A dedicated continuous integration test (`tests/test_model_quality.py`) guards the deployment pipeline:
- Holdout Top-1 Accuracy must be $\ge 95\%$.
- Holdout Macro-F1 must be $\ge 95\%$.
- Multiclass Brier score must be $\le 0.10$.
- Model must beat the Zero-Rule majority baseline by $\ge 50\%$ absolute margin.
- Fails the build if an inferior or uncalibrated model is committed.
