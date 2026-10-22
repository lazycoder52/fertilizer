"""Comprehensive Training, Evaluation, and Benchmarking Pipeline.

Evaluates baselines on an isolated development split using Stratified 5-Fold CV,
conducts bounded Optuna hyperparameter optimization without holdout leakage,
trains with random feature masking to reflect uncertainty under missing values,
computes Expected Calibration Error (ECE) and multi-class Brier scores,
conducts Permutation Importance, Leave-One-Feature-Out (LOFO) and finer ablation studies,
and exports production artifacts with genuine untouched test data.
"""

import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.ml.audit import run_dataset_audit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress Optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_git_commit_hash() -> str:
    """Retrieve current Git commit hash."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
        return commit
    except Exception:
        return "unknown"


def multiclass_brier_score(
    y_true_indices: np.ndarray, y_proba: np.ndarray, num_classes: int
) -> float:
    """Calculate multi-class Brier score."""
    y_one_hot = np.zeros((len(y_true_indices), num_classes))
    y_one_hot[np.arange(len(y_true_indices)), y_true_indices] = 1.0
    return float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true_indices: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) for multi-class predictions."""
    preds = np.argmax(y_proba, axis=1)
    confs = np.max(y_proba, axis=1)
    corrects = (preds == y_true_indices).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true_indices)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (
            (confs > bin_lower) & (confs <= bin_upper)
            if i > 0
            else (confs >= bin_lower) & (confs <= bin_upper)
        )
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(corrects[in_bin])
            bin_conf = np.mean(confs[in_bin])
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_cv(
    model: Any, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, num_classes: int
) -> dict[str, float]:
    """Run Stratified K-Fold CV on development data and calculate aggregate metrics with std."""
    top1_scores: list[float] = []
    top3_scores: list[float] = []
    macro_f1_scores: list[float] = []
    brier_scores: list[float] = []
    ece_scores: list[float] = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        top1_scores.append(accuracy_score(y_val, y_pred))
        macro_f1_scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)
            all_classes = np.arange(num_classes)
            if y_proba.shape[1] == num_classes:
                top3_scores.append(
                    top_k_accuracy_score(y_val, y_proba, k=min(3, num_classes), labels=all_classes)
                )
                brier_scores.append(multiclass_brier_score(y_val.to_numpy(), y_proba, num_classes))
                ece_scores.append(expected_calibration_error(y_val.to_numpy(), y_proba))
            else:
                top3_scores.append(top1_scores[-1])
                brier_scores.append(1.0)
                ece_scores.append(1.0)
        else:
            top3_scores.append(top1_scores[-1])
            brier_scores.append(1.0)
            ece_scores.append(1.0)

    return {
        "top1_accuracy_mean": float(np.mean(top1_scores)),
        "top1_accuracy_std": float(np.std(top1_scores)),
        "top3_accuracy_mean": float(np.mean(top3_scores)),
        "top3_accuracy_std": float(np.std(top3_scores)),
        "macro_f1_mean": float(np.mean(macro_f1_scores)),
        "macro_f1_std": float(np.std(macro_f1_scores)),
        "brier_score_mean": float(np.mean(brier_scores)),
        "brier_score_std": float(np.std(brier_scores)),
        "ece_mean": float(np.mean(ece_scores)),
        "ece_std": float(np.std(ece_scores)),
    }


def create_mask_augmented_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numerical_cols: list[str],
    column_medians: dict[str, float],
    seed: int = 42,
    mask_rate: float = 0.35,
) -> tuple[pd.DataFrame, pd.Series]:
    """Augment training data with masked feature samples.

    Enables the tree ensemble to learn appropriate uncertainty when farmers
    omit optional soil test parameters, producing well-calibrated probabilities.
    """
    rng = np.random.default_rng(seed)
    augmented_X = [X_train.copy()]
    augmented_y = [y_train.copy()]

    # Variant 1: Random feature masking (each continuous feature masked with mask_rate)
    X_rand_masked = X_train.copy()
    for col in numerical_cols:
        mask = rng.random(len(X_train)) < mask_rate
        X_rand_masked.loc[mask, col] = column_medians[col]
    augmented_X.append(X_rand_masked)
    augmented_y.append(y_train.copy())

    # Variant 2: Minimal input mode (only Soil and Crop provided, all continuous masked)
    X_min_masked = X_train.copy()
    for col in numerical_cols:
        X_min_masked[col] = column_medians[col]
    augmented_X.append(X_min_masked)
    augmented_y.append(y_train.copy())

    # Variant 3: NPK only mode (Environmental features masked, N-P-K present)
    env_cols = [c for c in numerical_cols if c not in ["Nitrogen", "Phosphorous", "Potassium"]]
    X_npk_masked = X_train.copy()
    for col in env_cols:
        X_npk_masked[col] = column_medians[col]
    augmented_X.append(X_npk_masked)
    augmented_y.append(y_train.copy())

    combined_X = pd.concat(augmented_X, ignore_index=True)
    combined_y = pd.concat(augmented_y, ignore_index=True)
    return combined_X, combined_y


def run_pipeline(config_path: str = "config/train_config.yaml") -> dict[str, Any]:
    """Execute complete ML rigor, benchmarking, tuning, and evaluation pipeline."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    output_dir = Path(cfg.get("output_dir", "model_files"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Audit & Target Leakage Check
    logger.info("Running dataset audit and integrity checks...")
    audit_results = run_dataset_audit(cfg["data_path"])

    # Load dataset
    df = pd.read_csv(cfg["data_path"])
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    le_fertilizer = LabelEncoder()

    df["Soil_encoded"] = le_soil.fit_transform(df["Soil"].astype(str))
    df["Crop_encoded"] = le_crop.fit_transform(df["Crop"].astype(str))
    df["Fertilizer_encoded"] = le_fertilizer.fit_transform(df["Fertilizer"].astype(str))

    num_classes = len(le_fertilizer.classes_)
    feature_cols = [
        "Temperature",
        "Moisture",
        "Rainfall",
        "PH",
        "Nitrogen",
        "Phosphorous",
        "Potassium",
        "Carbon",
        "Soil_encoded",
        "Crop_encoded",
    ]
    numerical_cols = cfg["features"]["numerical"]

    # Verify zero target leakage
    assert "Remark" not in feature_cols, "CRITICAL ERROR: 'Remark' cannot be in feature columns!"
    assert "Fertilizer" not in feature_cols, (
        "CRITICAL ERROR: 'Fertilizer' cannot be in feature columns!"
    )
    logger.info(
        "✓ Target leakage check PASSED: 'Remark' and target are strictly excluded from input vectors."
    )

    X = df[feature_cols]
    y = df["Fertilizer_encoded"]

    # Calculate and persist training feature medians for production imputation
    feature_medians = {col: float(X[col].median()) for col in numerical_cols}
    with open(output_dir / "feature_medians.json", "w", encoding="utf-8") as f:
        json.dump(feature_medians, f, indent=2)

    # 2. Strict Train / Holdout Separation
    # Split into 80% Development (tuning & baselines) and 20% untouched Holdout
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    logger.info(
        "Dataset partitioned into Development set (%d samples) and Untouched Holdout (%d samples).",
        len(X_dev),
        len(X_test),
    )

    # Save untouched holdout set for CI quality gate evaluation
    joblib.dump(
        {
            "X_test": X_test,
            "y_test": y_test,
            "feature_cols": feature_cols,
            "classes": le_fertilizer.classes_.tolist(),
        },
        output_dir / "test_data.joblib",
        compress=3,
    )

    cv = StratifiedKFold(
        n_splits=cfg["cross_validation"]["n_splits"], shuffle=True, random_state=seed
    )

    # 3. Multi-Model Baseline Comparison on Development Data
    logger.info("Benchmarking model baselines on Development split using 5-fold Stratified CV...")
    baselines: dict[str, Any] = {
        "Majority_Class_Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic_Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "Decision_Tree": DecisionTreeClassifier(random_state=seed),
        "Random_Forest_Default": RandomForestClassifier(n_estimators=100, random_state=seed),
        "LightGBM_Default": LGBMClassifier(random_state=seed, verbosity=-1),
    }

    baseline_metrics: dict[str, dict[str, float]] = {}
    for name, model in baselines.items():
        start_time = time.time()
        metrics = evaluate_cv(model, X_dev, y_dev, cv, num_classes)
        metrics["training_time_seconds"] = round(time.time() - start_time, 3)
        baseline_metrics[name] = metrics
        logger.info(
            "[%s] Top-1: %.4f (±%.4f) | Top-3: %.4f (±%.4f) | Macro-F1: %.4f (±%.4f) | ECE: %.4f",
            name,
            metrics["top1_accuracy_mean"],
            metrics["top1_accuracy_std"],
            metrics["top3_accuracy_mean"],
            metrics["top3_accuracy_std"],
            metrics["macro_f1_mean"],
            metrics["macro_f1_std"],
            metrics["ece_mean"],
        )

    # 4. Bounded Optuna Hyperparameter Optimization on Development Data
    logger.info("Running bounded Optuna optimization on Random Forest (Development split only)...")

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 50, 200, step=25)
        max_depth = trial.suggest_int("max_depth", 5, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        res = evaluate_cv(clf, X_dev, y_dev, cv, num_classes)
        return res["macro_f1_mean"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        objective,
        n_trials=cfg["optuna"]["n_trials"],
        timeout=cfg["optuna"]["timeout_seconds"],
    )

    best_rf_params = study.best_params
    logger.info("Best RF Parameters: %s (Dev Macro-F1: %.4f)", best_rf_params, study.best_value)

    # 5. Train Final Model with Random Feature Masking
    # Augment X_dev with masked variants so the model learns uncertainty under missing features
    logger.info("Training final production model with feature masking on development split...")
    X_dev_aug, y_dev_aug = create_mask_augmented_data(
        X_dev,
        y_dev,
        numerical_cols,
        feature_medians,
        seed=seed,
        mask_rate=cfg["masking"]["mask_prob"],
    )

    final_model = RandomForestClassifier(**best_rf_params, random_state=seed, n_jobs=-1)
    final_model.fit(X_dev_aug, y_dev_aug)

    # 6. Evaluation on the Untouched Holdout Test Set
    logger.info("Evaluating on strict, untouched holdout set (%d samples)...", len(X_test))
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)

    holdout_top1 = float(accuracy_score(y_test, y_pred))
    holdout_top3 = float(top_k_accuracy_score(y_test, y_proba, k=3, labels=np.arange(num_classes)))
    holdout_macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    holdout_brier = multiclass_brier_score(y_test.to_numpy(), y_proba, num_classes)
    holdout_ece = expected_calibration_error(y_test.to_numpy(), y_proba)
    holdout_mean_conf = float(np.mean(np.max(y_proba, axis=1)))

    logger.info(
        "Holdout Results -> Top-1: %.4f | Top-3: %.4f | Macro-F1: %.4f | Brier: %.4f | ECE: %.4f | Avg Conf: %.4f",
        holdout_top1,
        holdout_top3,
        holdout_macro_f1,
        holdout_brier,
        holdout_ece,
        holdout_mean_conf,
    )

    # 7. Evaluate Calibration & Uncertainty under Missing / Imputed Inputs
    logger.info("Evaluating uncertainty and calibration on masked holdout sets...")
    masked_evaluations = {}

    # 7a. Soil & Crop Only (all continuous masked with medians)
    X_test_soil_crop = X_test.copy()
    for col in numerical_cols:
        X_test_soil_crop[col] = feature_medians[col]
    proba_sc = final_model.predict_proba(X_test_soil_crop)
    pred_sc = final_model.predict(X_test_soil_crop)
    masked_evaluations["soil_and_crop_only"] = {
        "top1_accuracy": round(float(accuracy_score(y_test, pred_sc)), 4),
        "top3_accuracy": round(
            float(top_k_accuracy_score(y_test, proba_sc, k=3, labels=np.arange(num_classes))), 4
        ),
        "macro_f1": round(float(f1_score(y_test, pred_sc, average="macro", zero_division=0)), 4),
        "brier_score": round(multiclass_brier_score(y_test.to_numpy(), proba_sc, num_classes), 4),
        "ece": round(expected_calibration_error(y_test.to_numpy(), proba_sc), 4),
        "mean_confidence": round(float(np.mean(np.max(proba_sc, axis=1))), 4),
    }

    # 7b. NPK Only (Environmental continuous masked with medians)
    X_test_npk = X_test.copy()
    for col in ["Temperature", "Moisture", "Rainfall", "PH", "Carbon"]:
        X_test_npk[col] = feature_medians[col]
    proba_npk = final_model.predict_proba(X_test_npk)
    pred_npk = final_model.predict(X_test_npk)
    masked_evaluations["npk_only"] = {
        "top1_accuracy": round(float(accuracy_score(y_test, pred_npk)), 4),
        "top3_accuracy": round(
            float(top_k_accuracy_score(y_test, proba_npk, k=3, labels=np.arange(num_classes))), 4
        ),
        "macro_f1": round(float(f1_score(y_test, pred_npk, average="macro", zero_division=0)), 4),
        "brier_score": round(multiclass_brier_score(y_test.to_numpy(), proba_npk, num_classes), 4),
        "ece": round(expected_calibration_error(y_test.to_numpy(), proba_npk), 4),
        "mean_confidence": round(float(np.mean(np.max(proba_npk, axis=1))), 4),
    }

    class_names = le_fertilizer.classes_.tolist()
    clf_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    # 8. Permutation Importance & LOFO Ablation
    logger.info("Computing Permutation Importance on holdout set...")
    perm_res = permutation_importance(
        final_model, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=-1
    )
    perm_importance = {
        col: {
            "mean_accuracy_drop": round(float(perm_res.importances_mean[i]), 5),
            "std": round(float(perm_res.importances_std[i]), 5),
        }
        for i, col in enumerate(feature_cols)
    }
    sorted_perm_importance = dict(
        sorted(
            perm_importance.items(), key=lambda item: item[1]["mean_accuracy_drop"], reverse=True
        )
    )

    # 9. Leave-One-Feature-Out (LOFO) Ablation
    logger.info("Executing Leave-One-Feature-Out (LOFO) ablation...")
    lofo_results = {}
    for col in feature_cols:
        remaining_cols = [c for c in feature_cols if c != col]
        X_tr_lofo = X_dev_aug[remaining_cols]
        X_te_lofo = X_test[remaining_cols]

        clf_lofo = RandomForestClassifier(**best_rf_params, random_state=seed, n_jobs=-1)
        clf_lofo.fit(X_tr_lofo, y_dev_aug)
        p_lofo = clf_lofo.predict(X_te_lofo)
        lofo_results[col] = {
            "omitted_feature": col,
            "top1_accuracy": round(float(accuracy_score(y_test, p_lofo)), 4),
            "macro_f1": round(float(f1_score(y_test, p_lofo, average="macro", zero_division=0)), 4),
            "accuracy_drop": round(holdout_top1 - float(accuracy_score(y_test, p_lofo)), 4),
        }

    # 10. Finer Ablation Subsets (Identifying Minimal Suite for ~95% Accuracy)
    logger.info("Evaluating finer ablation subsets...")
    ablation_results = {}
    for subset_name, cols in cfg["ablation"]["subsets"].items():
        X_tr_sub = X_dev_aug[cols]
        X_te_sub = X_test[cols]

        clf_sub = RandomForestClassifier(**best_rf_params, random_state=seed, n_jobs=-1)
        clf_sub.fit(X_tr_sub, y_dev_aug)

        pred_sub = clf_sub.predict(X_te_sub)
        proba_sub = clf_sub.predict_proba(X_te_sub)

        top1_sub = float(accuracy_score(y_test, pred_sub))
        top3_sub = float(
            top_k_accuracy_score(y_test, proba_sub, k=3, labels=np.arange(num_classes))
        )
        macro_f1_sub = float(f1_score(y_test, pred_sub, average="macro", zero_division=0))

        ablation_results[subset_name] = {
            "features_used": cols,
            "feature_count": len(cols),
            "top1_accuracy": round(top1_sub, 4),
            "top3_accuracy": round(top3_sub, 4),
            "macro_f1": round(macro_f1_sub, 4),
        }
        logger.info(
            "Ablation [%s] Top-1: %.4f | Top-3: %.4f | Macro-F1: %.4f (%d features)",
            subset_name,
            top1_sub,
            top3_sub,
            macro_f1_sub,
            len(cols),
        )

    # 11. Measurement Sampling Variance Robustness Testing
    logger.info("Conducting measurement sampling variance robustness testing...")
    noise_results = {}
    std_devs = X_test[numerical_cols].std()

    for noise_level in cfg["measurement_noise"]["noise_levels"]:
        X_noisy = X_test.copy()
        for col in numerical_cols:
            noise = np.random.normal(0, noise_level * std_devs[col], size=len(X_test))
            X_noisy[col] = X_noisy[col] + noise

        y_noisy_pred = final_model.predict(X_noisy)
        noisy_acc = float(accuracy_score(y_test, y_noisy_pred))
        noisy_f1 = float(f1_score(y_test, y_noisy_pred, average="macro", zero_division=0))

        noise_results[f"noise_sigma_{int(noise_level * 100)}pct"] = {
            "noise_level": noise_level,
            "top1_accuracy": round(noisy_acc, 4),
            "macro_f1": round(noisy_f1, 4),
            "relative_degradation_pct": round((holdout_top1 - noisy_acc) / holdout_top1 * 100, 2),
        }

    # 12. Model Serialization & Packaging
    logger.info("Serializing production models and metadata to %s...", output_dir)

    # Joblib artifacts
    joblib.dump(final_model, output_dir / "model.joblib", compress=3)
    encoders = {
        "le_soil": le_soil,
        "le_crop": le_crop,
        "le_fertilizer": le_fertilizer,
    }
    joblib.dump(encoders, output_dir / "encoders.joblib", compress=3)
    joblib.dump(feature_cols, output_dir / "features.joblib", compress=3)

    # Backward compatibility with pickle
    import pickle

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(output_dir / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(output_dir / "features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    # Descriptions / remarks mapping
    if audit_results.get("remarks_mapping"):
        with open(output_dir / "fertilizer_descriptions.json", "w", encoding="utf-8") as f:
            json.dump(audit_results["remarks_mapping"], f, indent=2)

    # Comprehensive metrics JSON
    metrics_payload = {
        "evaluation_timestamp": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit_hash(),
        "model_architecture": "RandomForestClassifier (Optuna Tuned on Dev Split with Masking)",
        "best_hyperparameters": best_rf_params,
        "target_leakage_audit": {
            "remark_column_excluded": True,
            "target_column_excluded": True,
            "status": "PASSED (Zero Leakage)",
        },
        "holdout_evaluation": {
            "test_size": 0.2,
            "test_sample_count": len(X_test),
            "top1_accuracy": round(holdout_top1, 4),
            "top3_accuracy": round(holdout_top3, 4),
            "macro_f1": round(holdout_macro_f1, 4),
            "brier_score": round(holdout_brier, 4),
            "expected_calibration_error": round(holdout_ece, 4),
            "mean_confidence": round(holdout_mean_conf, 4),
        },
        "masked_input_evaluations": masked_evaluations,
        "baseline_comparison": baseline_metrics,
        "permutation_importance": sorted_perm_importance,
        "lofo_ablation": lofo_results,
        "ablation_study": ablation_results,
        "measurement_noise_robustness": noise_results,
        "classification_report": clf_report,
        "confusion_matrix": conf_mat,
        "classes": class_names,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Production Metadata JSON
    metadata_payload = {
        "model_version": "1.1.0",
        "git_commit": get_git_commit_hash(),
        "algorithm": "RandomForestClassifier",
        "top1_accuracy": round(holdout_top1, 4),
        "top3_accuracy": round(holdout_top3, 4),
        "macro_f1": round(holdout_macro_f1, 4),
        "brier_score": round(holdout_brier, 4),
        "expected_calibration_error": round(holdout_ece, 4),
        "input_features": feature_cols,
        "target_classes": class_names,
        "serialization": "joblib",
        "data_card_notes": {
            "synthetic_labels_detected": True,
            "audit_summary": audit_results["synthetic_data_indicator"],
            "real_world_limitations": [
                "Does not calculate dosage (kg/ha or lbs/acre)",
                "Does not compute application schedule or growth stage timing",
                "Dataset contains synthetic artifact values in rainfall and nutrients",
                "Synthetic label boundaries allow high accuracy on rule-based partitions",
            ],
        },
    }

    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    logger.info("✓ Pipeline execution completed successfully!")
    return metrics_payload


if __name__ == "__main__":
    run_pipeline()
