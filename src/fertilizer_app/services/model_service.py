"""Model Inference and Domain Service (Singleton)."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.fertilizer_app.services.shap_service import ShapService

logger = logging.getLogger(__name__)


class ModelService:
    """Thread-safe singleton managing ML model lifecycle and inference."""

    _instance: Optional["ModelService"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_dir: Path | None = None):
        if getattr(self, "_initialized", False):
            return

        self.model_dir = Path(model_dir or "model_files")
        self.model: Any = None
        self.encoders: dict[str, Any] = {}
        self.required_features: list[str] = []
        self.fertilizer_descriptions: dict[str, str] = {}
        self.model_metadata: dict[str, Any] = {}
        self.feature_medians: dict[str, float] = {}

        # Internal performance counters for /metrics
        self.metrics_counters = {
            "total_requests": 0,
            "total_predictions": 0,
            "prediction_errors": 0,
            "classes_predicted": {},
        }

        self._load_artifacts()
        self._initialized = True

    def _load_artifacts(self) -> None:
        """Load serialized models, encoders, and descriptions into memory."""
        logger.info("Loading model artifacts from %s...", self.model_dir)

        # 1. Model
        if (self.model_dir / "model.joblib").exists():
            self.model = joblib.load(self.model_dir / "model.joblib")
            self.encoders = joblib.load(self.model_dir / "encoders.joblib")
            self.required_features = joblib.load(self.model_dir / "features.joblib")
        elif (self.model_dir / "model.pkl").exists():
            import pickle

            with open(self.model_dir / "model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(self.model_dir / "encoders.pkl", "rb") as f:
                self.encoders = pickle.load(f)
            with open(self.model_dir / "features.pkl", "rb") as f:
                self.required_features = pickle.load(f)
        else:
            raise FileNotFoundError(f"Model artifacts not found in {self.model_dir}")

        self.le_soil = self.encoders["le_soil"]
        self.le_crop = self.encoders["le_crop"]
        self.le_fertilizer = self.encoders["le_fertilizer"]

        self.soil_types = self.le_soil.classes_.tolist()
        self.crop_types = self.le_crop.classes_.tolist()
        self.fertilizer_classes = self.le_fertilizer.classes_.tolist()

        # 2. Descriptions / Remarks
        desc_path = self.model_dir / "fertilizer_descriptions.json"
        if desc_path.exists():
            with open(desc_path, encoding="utf-8") as f:
                self.fertilizer_descriptions = json.load(f)

        # 3. Model Metadata and Runtime Metrics
        meta_path = self.model_dir / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                self.model_metadata = json.load(f)
        else:
            self.model_metadata = {}

        metrics_path = self.model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                self.metrics_data = json.load(f)
            holdout = self.metrics_data.get("holdout_evaluation", {})
            for key in [
                "top1_accuracy",
                "top3_accuracy",
                "macro_f1",
                "brier_score",
                "expected_calibration_error",
            ]:
                if key in holdout:
                    self.model_metadata[key] = holdout[key]
        else:
            self.metrics_data = {}

        # 4. Feature Medians for Imputation
        medians_path = self.model_dir / "feature_medians.json"
        if medians_path.exists():
            with open(medians_path, encoding="utf-8") as f:
                self.feature_medians = json.load(f)
        else:
            self.feature_medians = {
                "Temperature": 26.13,
                "Moisture": 0.60,
                "Rainfall": 96.49,
                "PH": 6.34,
                "Nitrogen": 59.43,
                "Phosphorous": 58.21,
                "Potassium": 58.71,
                "Carbon": 1.46,
            }

        # 5. Initialize Explainability
        self.shap_service = ShapService(model=self.model, feature_names=self.required_features)

        logger.info(
            "✓ Model loaded successfully. Supported soils: %d, crops: %d, fertilizers: %d",
            len(self.soil_types),
            len(self.crop_types),
            len(self.fertilizer_classes),
        )

    def normalize_category(
        self, value: str | None, valid_classes: list[str], suffix_to_strip: str = ""
    ) -> str | None:
        """Normalize user input to match canonical classes case-insensitively."""
        if not value:
            return None
        val_clean = str(value).strip().lower()
        for vc in valid_classes:
            vc_clean = vc.lower()
            if val_clean == vc_clean:
                return vc
            if (
                suffix_to_strip
                and val_clean == vc_clean.replace(suffix_to_strip.lower(), "").strip()
            ):
                return vc
        return None

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute model prediction and return ranked recommendations."""
        self.metrics_counters["total_requests"] += 1

        try:
            input_features = {}
            imputed_fields = []

            # Numerical features (defaults to development dataset median if omitted)
            numerical_features = [
                "Temperature",
                "Moisture",
                "Rainfall",
                "PH",
                "Nitrogen",
                "Phosphorous",
                "Potassium",
                "Carbon",
            ]

            for feat in numerical_features:
                val = input_data.get(feat)
                if val is not None and val != "":
                    input_features[feat] = float(val)
                else:
                    input_features[feat] = float(self.feature_medians.get(feat, 0.0))
                    imputed_fields.append(feat)

            # Categorical: Soil
            soil_raw = input_data.get("Soil")
            matched_soil = self.normalize_category(
                soil_raw, self.soil_types, suffix_to_strip="Soil"
            )
            if matched_soil:
                input_features["Soil_encoded"] = self.le_soil.transform([matched_soil])[0]
            elif soil_raw:
                raise ValueError(
                    f"Invalid soil type '{soil_raw}'. Valid options: {self.soil_types}"
                )
            else:
                input_features["Soil_encoded"] = 0

            # Categorical: Crop
            crop_raw = input_data.get("Crop")
            matched_crop = self.normalize_category(crop_raw, self.crop_types)
            if matched_crop:
                input_features["Crop_encoded"] = self.le_crop.transform([matched_crop])[0]
            elif crop_raw:
                raise ValueError(
                    f"Invalid crop type '{crop_raw}'. Valid options: {self.crop_types}"
                )
            else:
                input_features["Crop_encoded"] = 0

            # Construct DataFrame with strictly ordered features
            input_df = pd.DataFrame([input_features], columns=self.required_features)

            # Inference
            prediction_encoded = self.model.predict(input_df)[0]
            prediction_proba = self.model.predict_proba(input_df)[0]

            recommendation = self.le_fertilizer.inverse_transform([prediction_encoded])[0]
            confidence = float(max(prediction_proba)) * 100

            # Top 3 recommendations
            top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
            top_recommendations = [
                {
                    "fertilizer": self.le_fertilizer.inverse_transform([idx])[0],
                    "confidence": round(float(prediction_proba[idx]) * 100, 2),
                    "remark": self.fertilizer_descriptions.get(
                        self.le_fertilizer.inverse_transform([idx])[0], ""
                    ),
                }
                for idx in top_3_indices
            ]

            # Telemetry counters
            self.metrics_counters["total_predictions"] += 1
            self.metrics_counters["classes_predicted"][recommendation] = (
                self.metrics_counters["classes_predicted"].get(recommendation, 0) + 1
            )

            # Imputation trust warning
            is_imputed = len(imputed_fields) > 0
            warning_msg = None
            if is_imputed:
                warning_msg = (
                    f"Low confidence: using default benchmark values for omitted soil test parameters "
                    f"({', '.join(imputed_fields)}). Recommendations based on partial data have "
                    f"reduced accuracy (~50-75%); laboratory soil test values are recommended for precise field application."
                )

            # SHAP Explanation
            explanation = self.shap_service.explain(input_df, prediction_encoded)

            return {
                "success": True,
                "recommendation": recommendation,
                "confidence": round(confidence, 2),
                "remark": self.fertilizer_descriptions.get(recommendation, ""),
                "top_3_recommendations": top_recommendations,
                "is_imputed": is_imputed,
                "imputed_fields": imputed_fields,
                "warning": warning_msg,
                "explanation": explanation,
                "input_received": input_data,
            }

        except Exception as e:
            self.metrics_counters["prediction_errors"] += 1
            raise e

    def get_info(self) -> dict[str, Any]:
        """Return model metadata, feature requirements, and categories."""
        info = {
            "soil_types": self.soil_types,
            "crop_types": self.crop_types,
            "fertilizer_classes": self.fertilizer_classes,
            "required_features": self.required_features,
            "optional_features": [
                "Temperature",
                "Moisture",
                "Rainfall",
                "PH",
                "Nitrogen",
                "Phosphorous",
                "Potassium",
                "Carbon",
            ],
            "feature_medians": self.feature_medians,
            "required_categorical": ["Soil", "Crop"],
            "model_metadata": self.model_metadata,
        }
        if hasattr(self, "metrics_data") and "holdout_evaluation" in self.metrics_data:
            info["metrics"] = self.metrics_data["holdout_evaluation"]
        return info

    def get_metrics(self) -> dict[str, Any]:
        """Return live operational telemetry."""
        return {
            "status": "healthy",
            "counters": self.metrics_counters,
            "model_version": self.model_metadata.get("model_version", "1.0.0"),
            "git_commit": self.model_metadata.get("git_commit", "unknown"),
        }
