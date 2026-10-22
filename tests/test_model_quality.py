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


def test_cross_document_ablation_and_lofo_consistency():
    """Quality Gate: Asserts ablation and LOFO figures are identical across README, MODEL_CARD, and DATA_CARD."""
    metrics_path = Path("model_files/metrics.json")
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    doc_paths = [
        Path("README.md"),
        Path("docs/MODEL_CARD.md"),
        Path("docs/DATA_CARD.md"),
    ]

    doc_texts = {}
    for p in doc_paths:
        assert p.exists(), f"Doc missing: {p}"
        with open(p, encoding="utf-8") as f:
            doc_texts[p.name] = f.read()

    # 1. Method A & B Top-1 / Top-3 for all 4 subsets
    ablation_table = metrics.get("ablation_table", [])
    assert len(ablation_table) == 4, "Ablation table must have 4 subsets"

    for row in ablation_table:
        name = row["subset_name"]
        a_top1 = f"{row['method_a_deployed_imputed']['top1_accuracy'] * 100:.2f}%"
        a_top3 = f"{row['method_a_deployed_imputed']['top3_accuracy'] * 100:.2f}%"
        b_top1 = f"{row['method_b_retrained_subset']['top1_accuracy'] * 100:.2f}%"
        b_top3 = f"{row['method_b_retrained_subset']['top3_accuracy'] * 100:.2f}%"

        for doc_name, text in doc_texts.items():
            matching_lines = [line for line in text.splitlines() if f"**{name}**" in line]
            assert len(matching_lines) >= 1, f"Subset row for {name} missing in {doc_name}"
            row_line = matching_lines[0]
            assert a_top1 in row_line, f"Method A Top-1 {a_top1} missing in {doc_name} for {name}"
            assert a_top3 in row_line, f"Method A Top-3 {a_top3} missing in {doc_name} for {name}"
            assert b_top1 in row_line, f"Method B Top-1 {b_top1} missing in {doc_name} for {name}"
            assert b_top3 in row_line, f"Method B Top-3 {b_top3} missing in {doc_name} for {name}"

    # 2. LOFO phosphorus and carbon drops
    lofo = metrics.get("lofo_ablation", {})
    p_drop = lofo["Phosphorous"]["accuracy_drop"] * 100
    p_drop_str = f"+{p_drop:.2f}\\%"

    c_drop = lofo["Carbon"]["accuracy_drop"] * 100
    c_drop_str = f"+{c_drop:.2f}\\%"

    for doc_name, text in doc_texts.items():
        assert "<!-- BEGIN:lofo_table -->" in text, f"lofo_table marker missing in {doc_name}"
        lofo_block = text.split("<!-- BEGIN:lofo_table -->")[1].split("<!-- END:lofo_table -->")[0]

        # Check Phosphorous LOFO row
        p_lines = [line for line in lofo_block.splitlines() if "| **Phosphorous** |" in line]
        assert len(p_lines) == 1, f"Phosphorous LOFO row missing or duplicated in {doc_name}"
        assert p_drop_str in p_lines[0], (
            f"Phosphorous LOFO drop {p_drop_str} missing in {doc_name}: {p_lines[0]}"
        )

        # Check Carbon LOFO row
        c_lines = [line for line in lofo_block.splitlines() if "| **Carbon** |" in line]
        assert len(c_lines) == 1, f"Carbon LOFO row missing or duplicated in {doc_name}"
        assert c_drop_str in c_lines[0], (
            f"Carbon LOFO drop {c_drop_str} missing in {doc_name}: {c_lines[0]}"
        )
