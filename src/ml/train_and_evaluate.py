"""Comprehensive Training, Evaluation, and Benchmarking Pipeline.

Evaluates baselines, runs Stratified 5-Fold CV, conducts Optuna tuning,
performs ablation and noise robustness studies, and exports artifacts.
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def multiclass_brier_score(y_true_indices: np.ndarray, y_proba: np.ndarray, num_classes: int) -> float:
    """Calculate multi-class Brier score."""
    y_one_hot = np.zeros((len(y_true_indices), num_classes))
    y_one_hot[np.arange(len(y_true_indices)), y_true_indices] = 1.0
    return float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1)))


def evaluate_cv(
    model: Any, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, num_classes: int
) -> Dict[str, float]:
    """Run Stratified K-Fold CV and calculate aggregate metrics."""
    top1_scores: List[float] = []
    top3_scores: List[float] = []
    macro_f1_scores: List[float] = []
    brier_scores: List[float] = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        top1_scores.append(accuracy_score(y_val, y_pred))
        macro_f1_scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)
            # Ensure probabilities match total class count
            all_classes = np.arange(num_classes)
            if y_proba.shape[1] == num_classes:
                top3_scores.append(
                    top_k_accuracy_score(y_val, y_proba, k=min(3, num_classes), labels=all_classes)
                )
                brier_scores.append(multiclass_brier_score(y_val.to_numpy(), y_proba, num_classes))
            else:
                top3_scores.append(top1_scores[-1])
                brier_scores.append(1.0)
        else:
            top3_scores.append(top1_scores[-1])
            brier_scores.append(1.0)

    return {
        "top1_accuracy_mean": float(np.mean(top1_scores)),
        "top1_accuracy_std": float(np.std(top1_scores)),
        "top3_accuracy_mean": float(np.mean(top3_scores)),
        "top3_accuracy_std": float(np.std(top3_scores)),
        "macro_f1_mean": float(np.mean(macro_f1_scores)),
        "macro_f1_std": float(np.std(macro_f1_scores)),
        "brier_score_mean": float(np.mean(brier_scores)),
    }


def run_pipeline(config_path: str = "config/train_config.yaml") -> Dict[str, Any]:
    """Execute complete ML rigor, benchmarking, tuning, and evaluation pipeline."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    output_dir = Path(cfg.get("output_dir", "model_files"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Audit
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
        "Temperature", "Moisture", "Rainfall", "PH",
        "Nitrogen", "Phosphorous", "Potassium", "Carbon",
        "Soil_encoded", "Crop_encoded"
    ]

    X = df[feature_cols]
    y = df["Fertilizer_encoded"]

    cv = StratifiedKFold(n_splits=cfg["cross_validation"]["n_splits"], shuffle=True, random_state=seed)

    # 2. Multi-Model Baseline Comparison
    logger.info("Benchmarking model baselines using 5-fold Stratified CV...")
    baselines: Dict[str, Any] = {
        "Majority_Class_Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "Decision_Tree": DecisionTreeClassifier(random_state=seed),
        "Random_Forest_Default": RandomForestClassifier(n_estimators=100, random_state=seed),
        "LightGBM_Default": LGBMClassifier(random_state=seed, verbosity=-1),
    }

    baseline_metrics: Dict[str, Dict[str, float]] = {}
    for name, model in baselines.items():
        start_time = time.time()
        metrics = evaluate_cv(model, X, y, cv, num_classes)
        metrics["training_time_seconds"] = round(time.time() - start_time, 3)
        baseline_metrics[name] = metrics
        logger.info(
            "[%s] Top-1: %.4f (±%.4f) | Top-3: %.4f | Macro-F1: %.4f | Brier: %.4f",
            name,
            metrics["top1_accuracy_mean"],
            metrics["top1_accuracy_std"],
            metrics["top3_accuracy_mean"],
            metrics["macro_f1_mean"],
            metrics["brier_score_mean"],
        )

    # 3. Bounded Optuna Hyperparameter Optimization for Random Forest
    logger.info("Running bounded Optuna optimization on Random Forest...")

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
        res = evaluate_cv(clf, X, y, cv, num_classes)
        return res["macro_f1_mean"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        objective,
        n_trials=cfg["optuna"]["n_trials"],
        timeout=cfg["optuna"]["timeout_seconds"],
    )

    best_rf_params = study.best_params
    logger.info("Best RF Parameters: %s (Macro-F1: %.4f)", best_rf_params, study.best_value)

    # 4. Train Final Model and Conduct Holdout Evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    final_model = RandomForestClassifier(**best_rf_params, random_state=seed, n_jobs=-1)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)

    holdout_top1 = float(accuracy_score(y_test, y_pred))
    holdout_top3 = float(top_k_accuracy_score(y_test, y_proba, k=3, labels=np.arange(num_classes)))
    holdout_macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    holdout_brier = multiclass_brier_score(y_test.to_numpy(), y_proba, num_classes)

    class_names = le_fertilizer.classes_.tolist()
    clf_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    feature_importances = dict(
        zip(feature_cols, [round(float(v), 5) for v in final_model.feature_importances_])
    )
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    # 5. Form-Friction Ablation Study
    logger.info("Executing form-friction ablation studies...")
    ablation_results = {}
    for subset_name, cols in cfg["ablation"]["subsets"].items():
        X_sub = df[cols]
        clf_sub = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        sub_metrics = evaluate_cv(clf_sub, X_sub, y, cv, num_classes)
        ablation_results[subset_name] = {
            "features_used": cols,
            "feature_count": len(cols),
            "top1_accuracy": round(sub_metrics["top1_accuracy_mean"], 4),
            "top3_accuracy": round(sub_metrics["top3_accuracy_mean"], 4),
            "macro_f1": round(sub_metrics["macro_f1_mean"], 4),
        }
        logger.info(
            "Ablation [%s] Top-1: %.4f | Macro-F1: %.4f",
            subset_name,
            sub_metrics["top1_accuracy_mean"],
            sub_metrics["macro_f1_mean"],
        )

    # 6. Input Noise Robustness Testing
    logger.info("Conducting input noise robustness testing...")
    noise_results = {}
    numerical_cols = cfg["features"]["numerical"]
    std_devs = X_test[numerical_cols].std()

    for noise_level in cfg["noise_robustness"]["noise_levels"]:
        X_noisy = X_test.copy()
        for col in numerical_cols:
            noise = np.random.normal(0, noise_level * std_devs[col], size=len(X_test))
            X_noisy[col] = X_noisy[col] + noise

        y_noisy_pred = final_model.predict(X_noisy)
        noisy_acc = float(accuracy_score(y_test, y_noisy_pred))
        noisy_f1 = float(f1_score(y_test, y_noisy_pred, average="macro", zero_division=0))

        noise_results[f"noise_sigma_{int(noise_level*100)}pct"] = {
            "noise_level": noise_level,
            "top1_accuracy": round(noisy_acc, 4),
            "macro_f1": round(noisy_f1, 4),
            "relative_degradation_pct": round((holdout_top1 - noisy_acc) / holdout_top1 * 100, 2),
        }

    # 7. Model Serialization & Metadata Packaging
    logger.info("Saving models and metadata to %s...", output_dir)

    # Joblib artifacts (modern standard)
    joblib.dump(final_model, output_dir / "model.joblib", compress=3)
    encoders = {
        "le_soil": le_soil,
        "le_crop": le_crop,
        "le_fertilizer": le_fertilizer,
    }
    joblib.dump(encoders, output_dir / "encoders.joblib", compress=3)
    joblib.dump(feature_cols, output_dir / "features.joblib", compress=3)

    # Backward compatibility with pickle for current app.py
    import pickle
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(output_dir / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(output_dir / "features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    # Fertilizer descriptions / remarks mapping
    if audit_results.get("remarks_mapping"):
        with open(output_dir / "fertilizer_descriptions.json", "w", encoding="utf-8") as f:
            json.dump(audit_results["remarks_mapping"], f, indent=2)

    # Comprehensive metrics JSON
    metrics_payload = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "model_architecture": "RandomForestClassifier (Optuna Tuned)",
        "best_hyperparameters": best_rf_params,
        "holdout_evaluation": {
            "test_size": 0.2,
            "top1_accuracy": round(holdout_top1, 4),
            "top3_accuracy": round(holdout_top3, 4),
            "macro_f1": round(holdout_macro_f1, 4),
            "brier_score": round(holdout_brier, 4),
        },
        "baseline_comparison": baseline_metrics,
        "ablation_study": ablation_results,
        "noise_robustness": noise_results,
        "feature_importance": sorted_importances,
        "classification_report": clf_report,
        "confusion_matrix": conf_mat,
        "classes": class_names,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Production Metadata JSON
    metadata_payload = {
        "model_version": "1.0.0",
        "git_commit": get_git_commit_hash(),
        "algorithm": "RandomForestClassifier",
        "top1_accuracy": round(holdout_top1, 4),
        "top3_accuracy": round(holdout_top3, 4),
        "macro_f1": round(holdout_macro_f1, 4),
        "brier_score": round(holdout_brier, 4),
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
            ],
        },
    }

    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    logger.info("✓ Pipeline execution completed successfully!")
    return metrics_payload


if __name__ == "__main__":
    run_pipeline()
