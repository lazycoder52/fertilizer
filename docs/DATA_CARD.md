# Data Card: Fertilizer Recommendation Benchmark Dataset (Synthetic)

## 1. Dataset Overview & Provenance

- **Dataset Name**: Synthetic Agronomic Benchmark Dataset for Fertilizer Recommendation
- **Format**: Comma-Separated Values (`fertilizer_recommendation_dataset.csv`)
- **Total Records**: 3,100 instances
- **Total Features**: 10 input features (8 continuous numerical soil test parameters, 2 categorical variables)
- **Target Label**: `Fertilizer` (10 unique commercial formulation classes)
- **Auxiliary Column**: `Remark` (agronomic explanation / guidance note; strictly excluded from training features)
- **Dataset Nature**: **Synthetic / Rule-Generated Benchmark**. As documented in the empirical audit below, label assignments follow deterministic, axis-aligned rectangular rule boundaries.

---

## 2. Feature Schema & Agronomic Units

| Feature Name | Type | Unit / Scale | Agronomic Description | Physical Range |
| :--- | :--- | :--- | :--- | :--- |
| `Temperature` | Float | Celsius (°C) | Ambient field temperature | $-20.0^\circ\text{C} \text{ to } 60.0^\circ\text{C}$ |
| `Moisture` | Float | Ratio ($0.0 - 1.0$) | Volumetric soil moisture content | $0.0 \text{ to } 1.0$ |
| `Rainfall` | Float | Millimeters ($mm$) | Cumulative local precipitation | $0.0 \text{ to } 1000.0\text{ mm}$ |
| `PH` | Float | Logarithmic scale ($0 - 14$) | Soil acidity / alkalinity index | $0.0 \text{ to } 14.0$ |
| `Nitrogen` | Float | $kg/ha$ | Available elemental soil Nitrogen (N) | $0 \text{ to } 500\text{ kg/ha}$ |
| `Phosphorous` | Float | $kg/ha$ | Available elemental soil Phosphorus (P) | $0 \text{ to } 500\text{ kg/ha}$ |
| `Potassium` | Float | $kg/ha$ | Available elemental soil Potassium (K) | $0 \text{ to } 500\text{ kg/ha}$ |
| `Carbon` | Float | Percentage ($\%$) | Soil organic carbon matter content | $0.0\% \text{ to } 10.0\%$ |
| `Soil` | String | Categorical | Dominant soil textural / chemical class | 5 supported types |
| `Crop` | String | Categorical | Target cultivated agricultural crop | 31 supported crops |

### Supported Categorical Classes
- **Soil Types (5)**: `Acidic Soil`, `Alkaline Soil`, `Loamy Soil`, `Neutral Soil`, `Peaty Soil`.
- **Crops (31)**: `Adzuki Beans`, `Black gram`, `Chickpea`, `Coconut`, `Coffee`, `Cotton`, `Ground Nut`, `Jute`, `Kidney Beans`, `Lentil`, `Moth Beans`, `Mung Bean`, `Peas`, `Pigeon Peas`, `Rubber`, `Sugarcane`, `Tea`, `Tobacco`, `apple`, `banana`, `grapes`, `maize`, `mango`, `millet`, `muskmelon`, `orange`, `papaya`, `pomegranate`, `rice`, `watermelon`, `wheat`.

---

## 3. Target Class Distribution

The target variable `Fertilizer` contains 10 distinct commercial fertilizer formulations with moderate class imbalance:

| Fertilizer Class | Instance Count | Percentage | Primary Agronomic Indication |
| :--- | :---: | :---: | :--- |
| **DAP (Diammonium Phosphate)** | 1,054 | $34.00\%$ | High Phosphorus replenishment ($18\text{-}46\text{-}0$) for root development |
| **Water Retaining Fertilizer** | 675 | $21.77\%$ | Low moisture environments / drought risk mitigation |
| **Compost** | 375 | $12.10\%$ | Low organic matter and general soil structure conditioning |
| **Muriate of Potash (MOP)** | 325 | $10.48\%$ | Potassium deficiency ($0\text{-}0\text{-}60$) for fruiting and disease resistance |
| **Lime** | 180 | $5.81\%$ | Acidic soil correction ($\text{pH} < 6.0$) |
| **Urea** | 155 | $5.00\%$ | High Nitrogen supply ($46\text{-}0\text{-}0$) for rapid vegetative growth |
| **Balanced NPK Fertilizer** | 155 | $5.00\%$ | General maintenance with proportional N-P-K reserves |
| **Organic Fertilizer** | 95 | $3.06\%$ | Balanced low-impact organic nutrient enrichment |
| **Gypsum** | 55 | $1.77\%$ | Alkaline / sodic soil remediation |
| **General Purpose Fertilizer** | 31 | $1.00\%$ | Broad-spectrum multi-nutrient application |

---

## 4. Empirical Data Audit & Synthetic Artifact Findings

A statistical audit performed in `src/ml/audit.py` revealed that the labels in this dataset are **rule-generated and synthetic**:

1. **Physical Impossibility in Continuous Metrics**:
   - Negative minimum values were detected in continuous environmental variables that are strictly non-negative in real-world agronomy:
     - Minimum `Rainfall`: $-85.71\text{ mm}$
     - Minimum `Phosphorous`: $-37.65\text{ kg/ha}$
     - Minimum `Potassium`: $-31.42\text{ kg/ha}$
     - Minimum `Carbon`: $-0.95\%$
   - *Production Safeguard*: The production API enforces strict Pydantic v2 non-negative validation bounds ($x \ge 0.0$), rejecting physically impossible negative inputs at the API gateway.

2. **Deterministic Partitioning Boundaries**:
   - Class transitions occur along sharp axis-aligned decision planes at round thresholds:
     - Lime is assigned strictly when $\text{pH} < 6.0$.
     - Urea is assigned when $\text{Nitrogen} < 50$ and $\text{pH} \ge 6.0$.
     - DAP is assigned when $\text{Phosphorous} < 50$.
     - MOP is assigned when $\text{Potassium} < 50$.
   - This explains why an untuned single Decision Tree scores <!-- BEGIN:clean_dt_top1 -->$99.28\% \pm 0.52\%$<!-- END:clean_dt_top1 -->, comparable to Random Forest (<!-- BEGIN:clean_rf_top1 -->$99.11\% \pm 0.33\%$<!-- END:clean_rf_top1 -->) and LightGBM (<!-- BEGIN:clean_lgbm_top1 -->$98.71\% \pm 0.35\%$<!-- END:clean_lgbm_top1 -->). The algorithms have learned the underlying rule generator.

3. **Target Leakage Verification (`Remark`)**:
   - The auxiliary `Remark` column exhibits an exact 1-to-1 correspondence with `Fertilizer` (zero conditional entropy).
   - *Target Leakage Audit*: `Remark` is strictly excluded from feature vectors during preprocessing, cross-validation, and production inference. It is used solely as an agronomic display note for end users after inference is complete.

4. **Empirical Feature Sensitivity & Ablation**:
   - Omitting phosphorus drops model accuracy by <!-- BEGIN:lofo_phosphorus_drop -->$16.94\%$<!-- END:lofo_phosphorus_drop --> (down to <!-- BEGIN:lofo_phosphorus_top1 -->$81.77\%$<!-- END:lofo_phosphorus_top1 -->), confirming it as the primary chemical separator.
   - Omitting carbon drops accuracy by <!-- BEGIN:lofo_carbon_drop -->$11.13\%$<!-- END:lofo_carbon_drop -->.
   - The unified Method A vs Method B ablation and LOFO benchmarks are presented below:

<!-- BEGIN:ablation_table -->
| Feature Subset | Feature Count | Method A Top-1 (Deployed + Imputed) | Method A Top-3 | Method B Top-1 (Retrained on Subset) | Method B Top-3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Soil + Crop** | 2 | **46.29%** | 79.84% | **48.39%** | 82.90% |
| **+N/P/K** | 5 | **75.16%** | 96.94% | **74.03%** | 96.45% |
| **{Soil, Crop, P, K, pH, Carbon}** | 6 | **85.97%** | 96.13% | **87.10%** | 99.19% |
| **All 10 Features** | 10 | **98.71%** | 100.00% | **99.35%** | 100.00% |
<!-- END:ablation_table -->

<!-- BEGIN:lofo_table -->
| Omitted Feature | Resulting Top-1 | Accuracy Drop | Macro-F1 |
| :--- | :---: | :---: | :---: |
| **Phosphorous** | $81.77\%$ | **$+16.94\%$** | $0.8239$ |
| **Carbon** | $87.58\%$ | **$+11.13\%$** | $0.8169$ |
| **Potassium** | $90.32\%$ | $+8.39\%$ | $0.8435$ |
| **PH** | $93.39\%$ | $+5.32\%$ | $0.7999$ |
| **Moisture** | $93.55\%$ | $+5.16\%$ | $0.8304$ |
| **Nitrogen** | $94.35\%$ | $+4.35\%$ | $0.8558$ |
| **Soil Type (`Soil_encoded`)** | $98.55\%$ | $+0.16\%$ | $0.9332$ |
| **Crop Type (`Crop_encoded`)** | $99.35\%$ | $-0.65\%$ | $0.9832$ |
| **Temperature** | $99.52\%$ | $-0.81\%$ | $0.9851$ |
| **Rainfall** | $99.52\%$ | $-0.81\%$ | $0.9851$ |
<!-- END:lofo_table -->

---

## 5. Agronomic Limitations & Excluded Dimensions

When evaluating this dataset, several real-world farming considerations must be accounted for:

1. **No Application Dosage**: The dataset classifies *which* commercial fertilizer formulation to select, but does **not** calculate dosage (application rate in $kg/ha$ or $lbs/acre$).
2. **Missing Phenological Stages**: Real-world fertilization depends on crop growth stage (e.g., basal application at sowing vs. top-dressing during tillering or anthesis).
3. **Excluded Soil Chemistry Metrics**:
   - **Electrical Conductivity (EC)**: Vital for assessing soil salinity.
   - **Cation Exchange Capacity (CEC)**: Determines nutrient retention capacity.
   - **Micronutrient Profile**: Secondary and trace elements (Zinc, Boron, Sulfur, Iron, Magnesium) are unmeasured.
4. **Soil Texture Fractions**: Continuous proportions of sand, silt, and clay are categorized into broad types rather than continuous fractions.

---

## 6. Recommendations for Production Use

- **Input Validation**: Never pass raw unvalidated inputs directly to the model. Use Pydantic schemas to validate physical bounds.
- **Explainability**: Accompany all recommendations with TreeSHAP feature attributions so agronomists and farmers can verify the physical driver (e.g., *Low Soil Potassium*) before purchase.
- **Handling Incomplete Inputs**: When users omit optional soil test parameters, confidence naturally drops (~45-72%). The system must display a visible low-confidence notice and recommend full laboratory soil test analysis.
