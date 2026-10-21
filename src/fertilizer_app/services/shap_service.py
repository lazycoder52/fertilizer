"""SHAP (SHapley Additive exPlanations) Explainability Service."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class ShapService:
    """Thread-safe singleton managing TreeSHAP explanations."""

    _instance: Optional["ShapService"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model: Any = None, feature_names: list[str] | None = None):
        if getattr(self, "_initialized", False):
            return

        self.model = model
        self.feature_names = feature_names or []
        self.explainer: shap.TreeExplainer | None = None

        if self.model is not None:
            self._init_explainer()

        self._initialized = True

    def _init_explainer(self) -> None:
        """Initialize TreeExplainer with model."""
        try:
            logger.info("Initializing TreeSHAP explainer...")
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("✓ TreeSHAP explainer initialized.")
        except Exception as e:
            logger.warning("TreeSHAP initialization failed: %s", e)
            self.explainer = None

    def explain(
        self,
        input_df: pd.DataFrame,
        predicted_class_idx: int,
        feature_display_names: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Compute local feature attribution for the given input and predicted class."""
        if self.explainer is None:
            return {"enabled": False, "reason": "Explainer not initialized"}

        try:
            shap_values = self.explainer.shap_values(input_df)

            # Handle shap_values output dimensions across shap versions
            # Format (samples, features, classes) or (classes, samples, features) or list
            if isinstance(shap_values, list):
                class_shap = np.array(shap_values[predicted_class_idx])[0]
            elif len(np.shape(shap_values)) == 3:
                shape = np.shape(shap_values)
                if shape[0] == 1:  # (1, num_features, num_classes)
                    class_shap = shap_values[0, :, predicted_class_idx]
                else:  # (num_classes, 1, num_features)
                    class_shap = shap_values[predicted_class_idx, 0, :]
            else:
                class_shap = np.array(shap_values)[0]

            display_names = feature_display_names or {
                "Soil_encoded": "Soil Type",
                "Crop_encoded": "Crop Type",
                "PH": "Soil pH",
            }

            contributions: list[dict[str, Any]] = []
            for feat_name, impact in zip(self.feature_names, class_shap, strict=False):
                raw_val = input_df[feat_name].iloc[0]
                readable_name = display_names.get(feat_name, feat_name)
                imp_val = round(float(impact), 4)

                contributions.append(
                    {
                        "feature": feat_name,
                        "display_name": readable_name,
                        "value": round(float(raw_val), 2)
                        if isinstance(raw_val, (int, float))
                        else str(raw_val),
                        "impact": imp_val,
                        "direction": "positive" if imp_val >= 0 else "negative",
                        "importance_abs": abs(imp_val),
                    }
                )

            # Sort by absolute impact
            contributions.sort(key=lambda x: x["importance_abs"], reverse=True)

            positive_drivers = [
                c
                for c in contributions
                if c["direction"] == "positive" and c["importance_abs"] > 0.001
            ][:3]
            negative_drivers = [
                c
                for c in contributions
                if c["direction"] == "negative" and c["importance_abs"] > 0.001
            ][:3]

            return {
                "enabled": True,
                "predicted_class_idx": int(predicted_class_idx),
                "top_contributions": contributions[:6],
                "positive_drivers": positive_drivers,
                "negative_drivers": negative_drivers,
            }

        except Exception as e:
            logger.exception("Error computing SHAP explanation: %s", e)
            return {"enabled": False, "error": str(e)}
