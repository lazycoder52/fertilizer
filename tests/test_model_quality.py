"""CI Model Quality Gate Tests.

Ensures that no degraded, uncalibrated, or sub-standard ML models
can ever pass CI or be deployed to production.
"""

import json
from pathlib import Path

import joblib
import pytest


@pytest.fixture
def metrics_data():
    """Load latest metrics artifact."""
    metrics_path = Path("model_files/metrics.json")
    assert metrics_path.exists(), "metrics.json must exist in model_files/"
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def metadata_data():
    """Load latest metadata artifact."""
    meta_path = Path("model_files/model_metadata.json")
    assert meta_path.exists(), "model_metadata.json must exist in model_files/"
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def test_model_holdout_accuracy_gate(metrics_data):
    """Quality Gate: Holdout Top-1 accuracy must meet or exceed 95%."""
    top1 = metrics_data["holdout_evaluation"]["top1_accuracy"]
    assert top1 >= 0.95, f"Model Top-1 accuracy {top1:.4f} is below 0.95 production threshold!"


def test_model_macro_f1_gate(metrics_data):
    """Quality Gate: Macro-F1 must meet or exceed 0.95 across all classes."""
    macro_f1 = metrics_data["holdout_evaluation"]["macro_f1"]
    assert macro_f1 >= 0.95, f"Model Macro-F1 {macro_f1:.4f} is below 0.95 production threshold!"


def test_model_calibration_brier_gate(metrics_data):
    """Quality Gate: Brier score must be <= 0.10 to guarantee probability calibration."""
    brier = metrics_data["holdout_evaluation"]["brier_score"]
    assert brier <= 0.10, (
        f"Model Brier score {brier:.4f} indicates poorly calibrated probabilities!"
    )


def test_model_beats_majority_baseline(metrics_data):
    """Quality Gate: Model must significantly outperform the majority class baseline."""
    model_top1 = metrics_data["holdout_evaluation"]["top1_accuracy"]
    baseline_top1 = metrics_data["baseline_comparison"]["Majority_Class_Baseline"][
        "top1_accuracy_mean"
    ]
    margin = model_top1 - baseline_top1
    assert margin > 0.50, f"Model performance margin {margin:.4f} over baseline is insufficient!"


def test_model_beats_logistic_regression(metrics_data):
    """Quality Gate: Model must outperform linear baseline."""
    model_top1 = metrics_data["holdout_evaluation"]["top1_accuracy"]
    lr_top1 = metrics_data["baseline_comparison"]["Logistic_Regression"]["top1_accuracy_mean"]
    assert model_top1 > lr_top1, "Model failed to beat Logistic Regression baseline!"


def test_artifacts_exist_and_loadable():
    """Quality Gate: All essential serialized artifacts must exist and be loadable."""
    model_dir = Path("model_files")
    model = joblib.load(model_dir / "model.joblib")
    encoders = joblib.load(model_dir / "encoders.joblib")
    features = joblib.load(model_dir / "features.joblib")

    assert model is not None
    assert "le_soil" in encoders
    assert "le_crop" in encoders
    assert "le_fertilizer" in encoders
    assert len(features) == 10
