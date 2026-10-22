"""CI Model Quality Gate Tests.

Ensures that no degraded, overconfident, or sub-standard ML models
can ever pass CI or be deployed to production.
Evaluates the actual serialized binary model directly against the untouched holdout split.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Compute top-label Expected Calibration Error (ECE) with equal-width bins.

    Formula:
        ECE = sum_{m=1}^{M} (|B_m| / N) * |acc(B_m) - conf(B_m)|
    where:
        - M = n_bins (10 equal-width bins spanning [0, 1]).
        - B_m is the set of samples whose top predicted confidence falls into bin m.
        - acc(B_m) is the empirical accuracy of predictions in bin m.
        - conf(B_m) is the mean predicted confidence in bin m.
    """
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray, num_classes: int) -> float:
    """Calculate Multi-Class Brier Score."""
    one_hot = np.zeros((len(y_true), num_classes))
    for i, label in enumerate(y_true):
        one_hot[i, int(label)] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


@pytest.fixture
def model_artifacts():
    """Load production model, encoders, feature list, and untouched test data."""
    model_dir = Path("model_files")
    assert (model_dir / "model.joblib").exists(), "model.joblib missing"
    assert (model_dir / "test_data.joblib").exists(), "test_data.joblib missing"
    assert (model_dir / "features.joblib").exists(), "features.joblib missing"
    assert (model_dir / "encoders.joblib").exists(), "encoders.joblib missing"

    model = joblib.load(model_dir / "model.joblib")
    test_data = joblib.load(model_dir / "test_data.joblib")
    features = joblib.load(model_dir / "features.joblib")
    encoders = joblib.load(model_dir / "encoders.joblib")

    return {
        "model": model,
        "X_test": test_data["X_test"],
        "y_test": test_data["y_test"],
        "features": features,
        "encoders": encoders,
    }


def test_saved_model_evaluates_on_untouched_holdout(model_artifacts):
    """Quality Gate: Evaluates serialized model directly on untouched holdout split."""
    model = model_artifacts["model"]
    X_test = model_artifacts["X_test"]
    y_test = model_artifacts["y_test"]
    num_classes = len(model_artifacts["encoders"]["le_fertilizer"].classes_)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    top1 = float(accuracy_score(y_test, y_pred))
    top3 = float(top_k_accuracy_score(y_test, y_proba, k=3, labels=np.arange(num_classes)))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    assert top1 >= 0.95, f"Holdout Top-1 accuracy {top1:.4f} is below 0.95 production threshold!"
    assert top3 >= 0.99, f"Holdout Top-3 accuracy {top3:.4f} is below 0.99 production threshold!"
    assert macro_f1 >= 0.90, f"Holdout Macro-F1 {macro_f1:.4f} is below 0.90 production threshold!"


def test_model_calibration_brier_and_ece_gates(model_artifacts):
    """Quality Gate: Asserts Brier score <= 0.10 and ECE <= 0.15 on untouched holdout."""
    model = model_artifacts["model"]
    X_test = model_artifacts["X_test"]
    y_test = model_artifacts["y_test"]
    num_classes = len(model_artifacts["encoders"]["le_fertilizer"].classes_)

    y_proba = model.predict_proba(X_test)
    brier = multiclass_brier_score(y_test.to_numpy(), y_proba, num_classes)
    ece = expected_calibration_error(y_test.to_numpy(), y_proba)

    assert brier <= 0.10, f"Model Brier score {brier:.4f} exceeds 0.10 production threshold!"
    assert ece <= 0.15, f"Model Expected Calibration Error {ece:.4f} exceeds 0.15 threshold!"


def test_zero_target_leakage_audit(model_artifacts):
    """Quality Gate: Verifies 'Remark' and target are strictly excluded from input vectors."""
    features = model_artifacts["features"]
    X_test = model_artifacts["X_test"]

    assert "Remark" not in features, "CRITICAL ERROR: 'Remark' target leakage detected in features!"
    assert "Fertilizer" not in features, "CRITICAL ERROR: 'Fertilizer' target leakage in features!"
    assert "Remark" not in X_test.columns, (
        "CRITICAL ERROR: 'Remark' found in test input feature columns!"
    )


def test_masked_input_reflects_uncertainty(model_artifacts):
    """Quality Gate: Missing inputs must not yield overconfident scores (~50% mean confidence)."""
    model = model_artifacts["model"]
    X_test = model_artifacts["X_test"].copy()
    num_cols = [
        "Temperature",
        "Moisture",
        "Rainfall",
        "PH",
        "Nitrogen",
        "Phosphorous",
        "Potassium",
        "Carbon",
    ]

    # Mask all numerical features with medians
    for col in num_cols:
        X_test[col] = float(X_test[col].median())

    y_proba_masked = model.predict_proba(X_test)
    mean_confidence = float(np.mean(np.max(y_proba_masked, axis=1)))

    assert mean_confidence <= 0.65, (
        f"Masked inputs produced overconfident score {mean_confidence:.4f}! "
        "Feature masking retrained model must temper confidence under missing inputs."
    )


def test_artifacts_exist_and_loadable():
    """Quality Gate: All required serialized artifacts exist and are loadable."""
    model_dir = Path("model_files")
    model = joblib.load(model_dir / "model.joblib")
    encoders = joblib.load(model_dir / "encoders.joblib")
    features = joblib.load(model_dir / "features.joblib")

    assert model is not None
    assert "le_soil" in encoders
    assert "le_crop" in encoders
    assert "le_fertilizer" in encoders
    assert len(features) == 10
    assert (model_dir / "feature_medians.json").exists()
    assert (model_dir / "model_metadata.json").exists()


def test_render_docs_check():
    """Quality Gate: Assert all markdown docs match metrics.json via render_docs --check."""
    from src.ml.render_docs import process_docs

    rc = process_docs(check=True)
    assert rc == 0, "Documentation check failed: docs out of sync with model_files/metrics.json"


def test_readme_row_and_line_level_metrics():
    """Quality Gate: Strict row-level and labelled-line metric verification in README.md."""
    metrics_path = Path("model_files/metrics.json")
    readme_path = Path("README.md")

    assert metrics_path.exists(), "metrics.json missing"
    assert readme_path.exists(), "README.md missing"

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    with open(readme_path, encoding="utf-8") as f:
        lines = f.readlines()

    holdout = metrics["holdout_evaluation"]
    top1_str = f"${holdout['top1_accuracy'] * 100:.2f}\\%$"
    top3_str = f"${holdout['top3_accuracy'] * 100:.2f}\\%$"
    f1_str = f"${holdout['macro_f1']:.4f}$"
    ece_str = f"${holdout['expected_calibration_error']:.4f}$"
    sc_top1_pct = (
        f"{metrics['masked_input_evaluations']['soil_and_crop_only']['top1_accuracy'] * 100:.2f}%"
    )

    # 1. Labelled line for Top-1
    top1_line = next((line for line in lines if "- **Top-1 Accuracy**:" in line), None)
    assert top1_line is not None and top1_str in top1_line, (
        f"Top-1 {top1_str} not in labelled line: {top1_line}"
    )

    # 2. Labelled line for Top-3
    top3_line = next((line for line in lines if "- **Top-3 Accuracy**:" in line), None)
    assert top3_line is not None and top3_str in top3_line, (
        f"Top-3 {top3_str} not in labelled line: {top3_line}"
    )

    # 3. Labelled line for Macro-F1
    f1_line = next((line for line in lines if "- **Macro-F1**:" in line), None)
    assert f1_line is not None and f1_str in f1_line, (
        f"Macro-F1 {f1_str} not in labelled line: {f1_line}"
    )

    # 4. Labelled line for ECE
    ece_line = next(
        (line for line in lines if "- **Expected Calibration Error (ECE)**:" in line), None
    )
    assert ece_line is not None and ece_str in ece_line, (
        f"ECE {ece_str} not in labelled line: {ece_line}"
    )

    # 5. Table row for Soil+Crop Top-1 under Method A ablation
    sc_ablation_line = next((line for line in lines if "| **Soil + Crop** |" in line), None)
    assert sc_ablation_line is not None and sc_top1_pct in sc_ablation_line, (
        f"Soil+Crop Top-1 {sc_top1_pct} not in ablation row: {sc_ablation_line}"
    )
