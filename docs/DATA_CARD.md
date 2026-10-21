# Data Card: Fertilizer Recommendation Dataset

## 1. Dataset Overview

- **Dataset Name**: Fertilizer Recommendation Agronomic Dataset
- **Format**: Comma-Separated Values (`fertilizer_recommendation_dataset.csv`)
- **Total Records**: 3,100 instances
- **Total Features**: 10 predictive features (8 continuous numerical sensors, 2 categorical variables)
- **Target Label**: `Fertilizer` (10 unique classes)
- **Auxiliary Column**: `Remark` (deterministic explanation / recommendation notes)

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

A statistical audit performed in Phase 1 revealed that the labels in this dataset are **very likely synthetic or rule-generated**:

1. **Physical Impossibility in Continuous Metrics**:
   - Negative minimum values were detected in continuous environmental variables that are strictly non-negative in nature:
     - Minimum `Rainfall`: $-85.71\text{ mm}$
     - Minimum `Phosphorous`: $-37.65\text{ kg/ha}$
     - Minimum `Potassium`: $-31.42\text{ kg/ha}$
     - Minimum `Carbon`: $-0.95\%$
   - *Mitigation*: The production API enforces strict Pydantic v2 non-negative validation bounds ($x \ge 0.0$), discarding unphysical sensor input at the boundary.

2. **Deterministic Partitioning Boundaries**:
   - Class transitions occur along sharp axis-aligned decision planes at round thresholds:
     - Lime is assigned strictly when $\text{pH} < 6.0$.
     - Urea is assigned when $\text{Nitrogen} < 50$ and $\text{pH} \ge 6.0$.
     - DAP is assigned when $\text{Phosphorous} < 50$.
     - MOP is assigned when $\text{Potassium} < 50$.
   - This accounts for the near-perfect holdout accuracy ($>99.5\%$) observed across Decision Trees, Random Forests, and Gradient Boosting models.

3. **Exact 1-to-1 Correspondence with `Remark`**:
   - The auxiliary `Remark` text maps deterministically to the `Fertilizer` label with zero entropy, confirming rule-based label generation during dataset creation.

---

## 5. Agronomic Limitations & Excluded Dimensions

When utilizing this dataset for machine learning applications, several real-world farming considerations must be accounted for:

1. **No Application Dosage**: The dataset provides recommendations on *which* fertilizer type to select, but does **not** determine application rate (e.g., $kg/ha$ or $lbs/acre$).
2. **Missing Phenological Stages**: Real-world fertilization depends on crop growth stage (e.g., basal application at sowing vs. top-dressing during tillering or anthesis).
3. **Excluded Soil Chemistry Metrics**:
   - **Electrical Conductivity (EC)**: Vital for assessing soil salinity.
   - **Cation Exchange Capacity (CEC)**: Determines nutrient retention capacity.
   - **Micronutrient Profile**: Secondary and trace elements (Zinc, Boron, Sulfur, Iron, Magnesium) are unmeasured.
4. **Soil Texture Fractions**: Continuous proportions of sand, silt, and clay are categorized into broad types rather than continuous fractions.

---

## 6. Recommendations for Production Use

- **Input Validation**: Never pass raw unvalidated telemetry directly to the model. Use Pydantic schemas to validate sensor physics.
- **Explainability**: Accompany all recommendations with TreeSHAP feature attributions so agronomists and farmers can verify the physical driver (e.g., *Low Soil Potassium*) before purchase.
- **Safety Margin**: Where predictions have low confidence ($<60\%$), fallback to standard balanced NPK or recommend a laboratory chemical soil test.
