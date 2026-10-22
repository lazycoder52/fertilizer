"""Reproducible Feature Ablation, Permutation Importance, and LOFO Analysis.

Generates ONE unified ablation table comparing two methods:
- Method A: Deployed masked production model with median imputation for omitted features.
- Method B: Dedicated models retrained exclusively on each feature subset.

Feature sets evaluated:
1. Soil + Crop (2 features)
2. +N/P/K (Soil, Crop, Nitrogen, Phosphorous, Potassium - 5 features)
3. {Soil, Crop, P, K, pH, Carbon} (6 features minimal suite)
4. All 10 features

Also computes:
- Leave-One-Feature-Out (LOFO) accuracy drops.
- Permutation Importance on the untouched holdout split.

Updates:
- model_files/metrics.json
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from src.ml.train_and_evaluate import get_git_commit_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_ablation_study(
    config_path: str = "config/train_config.yaml",
) -> dict[str, Any]:
    """Execute reproducible feature ablation study and persist results to metrics.json."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    output_dir = Path(cfg.get("output_dir", "model_files"))

    # Load test data and feature medians
    test_data = joblib.load(output_dir / "test_data.joblib")
    X_test: pd.DataFrame = test_data["X_test"]
    y_test: pd.Series = test_data["y_test"]
    feature_cols: list[str] = test_data["feature_cols"]
    class_names: list[str] = test_data["classes"]
    num_classes = len(class_names)
    y_test_np = y_test.to_numpy()

    with open(output_dir / "feature_medians.json", encoding="utf-8") as f:
        feature_medians = json.load(f)

    # Load raw dataset and reconstruct development split
    encoders = joblib.load(output_dir / "encoders.joblib")
    df = pd.read_csv(cfg["data_path"])
    df["Soil_encoded"] = encoders["le_soil"].transform(df["Soil"].astype(str))
    df["Crop_encoded"] = encoders["le_crop"].transform(df["Crop"].astype(str))
    df["Fertilizer_encoded"] = encoders["le_fertilizer"].transform(df["Fertilizer"].astype(str))

    X = df[feature_cols]
    y = df["Fertilizer_encoded"]
    X_dev = X.drop(index=X_test.index)
    y_dev = y.drop(index=X_test.index)

    # Load production model
    prod_model = joblib.load(output_dir / "model.joblib")

    # Hyperparameters
    best_rf_params = {
        "n_estimators": 75,
        "max_depth": 13,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
    }
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                existing_metrics = json.load(f)
            if "best_hyperparameters" in existing_metrics:
                best_rf_params = existing_metrics["best_hyperparameters"]
        except Exception:
            pass

    # Define the 4 canonical feature subsets
    subsets_def = [
        {
            "name": "Soil + Crop",
            "key": "soil_and_crop",
            "features": ["Soil_encoded", "Crop_encoded"],
        },
        {
            "name": "+N/P/K",
            "key": "soil_crop_npk",
            "features": ["Soil_encoded", "Crop_encoded", "Nitrogen", "Phosphorous", "Potassium"],
        },
        {
            "name": "{Soil, Crop, P, K, pH, Carbon}",
            "key": "soil_crop_pk_ph_carbon",
            "features": [
                "Soil_encoded",
                "Crop_encoded",
                "Phosphorous",
                "Potassium",
                "PH",
                "Carbon",
            ],
        },
        {
            "name": "All 10 Features",
            "key": "all_10_features",
            "features": feature_cols,
        },
    ]

    logger.info("Evaluating feature subsets across Method A (Imputed) and Method B (Retrained)...")
    ablation_table = []
    ablation_dict = {}

    for s in subsets_def:
        cols = s["features"]

        # --- Method A: Deployed Masked Model with Median Imputation ---
        X_test_imputed = X_test.copy()
        for c in feature_cols:
            if c not in cols and c in feature_medians:
                X_test_imputed[c] = feature_medians[c]

        pred_a = prod_model.predict(X_test_imputed)
        proba_a = prod_model.predict_proba(X_test_imputed)
        top1_a = round(float(accuracy_score(y_test_np, pred_a)), 4)
        top3_a = round(
            float(top_k_accuracy_score(y_test_np, proba_a, k=3, labels=np.arange(num_classes))), 4
        )
        f1_a = round(float(f1_score(y_test_np, pred_a, average="macro", zero_division=0)), 4)

        # --- Method B: Retrained Models on Specific Feature Subsets ---
        clf_b = RandomForestClassifier(**best_rf_params, random_state=seed, n_jobs=-1)
        clf_b.fit(X_dev[cols], y_dev)

        pred_b = clf_b.predict(X_test[cols])
        proba_b = clf_b.predict_proba(X_test[cols])
        top1_b = round(float(accuracy_score(y_test_np, pred_b)), 4)
        top3_b = round(
            float(top_k_accuracy_score(y_test_np, proba_b, k=3, labels=np.arange(num_classes))), 4
        )
        f1_b = round(float(f1_score(y_test_np, pred_b, average="macro", zero_division=0)), 4)

        row = {
            "subset_name": s["name"],
            "subset_key": s["key"],
            "feature_count": len(cols),
            "features": cols,
            "method_a_deployed_imputed": {
                "top1_accuracy": top1_a,
                "top3_accuracy": top3_a,
                "macro_f1": f1_a,
            },
            "method_b_retrained_subset": {
                "top1_accuracy": top1_b,
                "top3_accuracy": top3_b,
                "macro_f1": f1_b,
            },
        }
        ablation_table.append(row)
        ablation_dict[s["key"]] = row
        logger.info(
            "[%s] Method A -> Top-1: %.2f%%, Top-3: %.2f%% | Method B -> Top-1: %.2f%%, Top-3: %.2f%%",
            s["name"],
            top1_a * 100,
            top3_a * 100,
            top1_b * 100,
            top3_b * 100,
        )

    # --- Permutation Importance ---
    logger.info("Computing Permutation Importance on untouched holdout set...")
    perm_res = permutation_importance(
        prod_model, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=-1
    )
    perm_importance = {
        col: {
            "mean_accuracy_drop": round(float(perm_res.importances_mean[i]), 4),
            "std": round(float(perm_res.importances_std[i]), 4),
        }
        for i, col in enumerate(feature_cols)
    }
    sorted_perm_importance = dict(
        sorted(
            perm_importance.items(), key=lambda item: item[1]["mean_accuracy_drop"], reverse=True
        )
    )

    # --- Leave-One-Feature-Out (LOFO) Ablation ---
    logger.info("Computing Leave-One-Feature-Out (LOFO) ablation...")
    base_acc = float(accuracy_score(y_test_np, prod_model.predict(X_test)))
    lofo_results = {}

    for col in feature_cols:
        remaining_cols = [c for c in feature_cols if c != col]
        clf_lofo = RandomForestClassifier(**best_rf_params, random_state=seed, n_jobs=-1)
        clf_lofo.fit(X_dev[remaining_cols], y_dev)

        pred_lofo = clf_lofo.predict(X_test[remaining_cols])
        acc_lofo = float(accuracy_score(y_test_np, pred_lofo))
        f1_lofo = float(f1_score(y_test_np, pred_lofo, average="macro", zero_division=0))

        lofo_results[col] = {
            "omitted_feature": col,
            "top1_accuracy": round(acc_lofo, 4),
            "macro_f1": round(f1_lofo, 4),
            "accuracy_drop": round(base_acc - acc_lofo, 4),
        }
        logger.info(
            "LOFO omitting [%s]: Top-1 = %.2f%% (Drop: %+.2f%%)",
            col,
            acc_lofo * 100,
            (base_acc - acc_lofo) * 100,
        )

    # Update metrics.json
    metrics_data = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}

    metrics_data.update(
        {
            "ablation_timestamp": metrics_data.get(
                "ablation_timestamp", datetime.now(UTC).isoformat()
            ),
            "git_commit": metrics_data.get("git_commit", get_git_commit_hash()),
            "ablation_table": ablation_table,
            "ablation_study": ablation_dict,
            "permutation_importance": sorted_perm_importance,
            "lofo_ablation": lofo_results,
        }
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    logger.info("Successfully updated %s with unified ablation table.", metrics_path)
    return {
        "ablation_table": ablation_table,
        "permutation_importance": sorted_perm_importance,
        "lofo_ablation": lofo_results,
    }


if __name__ == "__main__":
    run_ablation_study()
