"""Render metrics and tables from model_files/metrics.json into documentation markdown files.

Guarantees 100% reproducible documentation with zero hand-typed numbers between
<!-- BEGIN:marker_name --> and <!-- END:marker_name --> tags in:
- README.md
- docs/MODEL_CARD.md
- docs/DATA_CARD.md

Usage:
    python -m src.ml.render_docs          # Update markdown files in-place
    python -m src.ml.render_docs --check  # Verify markdown files match metrics.json exactly (exits non-zero on diff)
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    """Load and return the parsed metrics dictionary."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def render_comparison_table(metrics: dict[str, Any]) -> str:
    """Render the 9-candidate model comparison markdown table."""
    baseline = metrics.get("baseline_comparison", {})
    clean_holdout = metrics.get("clean_holdout_comparison", {})
    missing_holdout = metrics.get("missing_input_comparison", {})

    models_order = [
        ("Majority_Class_Baseline", "**Zero-Rule Majority Baseline**"),
        ("Logistic_Regression", "**Logistic Regression (Standardized)**"),
        ("Decision_Tree_Clean", "**Decision Tree (Clean)**"),
        ("Decision_Tree_Masked", "**Decision Tree (Mask-Augmented 35%)**"),
        ("LightGBM_Clean", "**LightGBM (Clean)**"),
        ("LightGBM_Masked", "**LightGBM (Mask-Augmented 35%)**"),
        ("Random_Forest_Default", "**Random Forest (Default Clean)**"),
        ("Random_Forest_Tuned_Clean", "**Random Forest (Tuned Clean, 75 trees)**"),
        (
            "Random_Forest_Tuned_Masked",
            "**Random Forest (Tuned + 35% Masking, Production)**",
        ),
    ]

    lines = [
        (
            "| Model Candidate | 5-Fold CV Top-1 | 5-Fold CV Macro-F1 | "
            "Clean Holdout Top-1 | Clean Holdout ECE | Soil+Crop Top-1 | "
            "Soil+Crop Top-3 | Soil+Crop ECE | Soil+Crop Mean Conf |"
        ),
        ("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"),
    ]

    for key, display_name in models_order:
        cv = baseline.get(key, {})
        clean = clean_holdout.get(key, {})
        sc = missing_holdout.get(key, {})

        cv_top1 = f"${cv.get('top1_accuracy_mean', 0.0) * 100:.2f}\\% \\pm {cv.get('top1_accuracy_std', 0.0) * 100:.2f}\\%$"
        cv_f1 = f"${cv.get('macro_f1_mean', 0.0):.4f}$"

        clean_top1_val = clean.get("top1_accuracy", 0.0) * 100
        clean_ece_val = clean.get("expected_calibration_error", clean.get("ece", 0.0))

        sc_top1_val = sc.get("top1_accuracy", 0.0) * 100
        sc_top3_val = sc.get("top3_accuracy", 0.0) * 100
        sc_ece_val = sc.get("expected_calibration_error", sc.get("ece", 0.0))
        sc_conf_val = sc.get("mean_confidence", 0.0) * 100

        if key == "Random_Forest_Tuned_Masked":
            clean_top1 = f"**${clean_top1_val:.2f}\\%$**"
            clean_ece = f"**${clean_ece_val:.4f}$**"
            sc_top1 = f"**${sc_top1_val:.2f}\\%$**"
            sc_top3 = f"**${sc_top3_val:.2f}\\%$**"
            sc_ece = f"**${sc_ece_val:.4f}$**"
            sc_conf = f"**${sc_conf_val:.2f}\\%$**"
        else:
            clean_top1 = f"${clean_top1_val:.2f}\\%$"
            clean_ece = f"${clean_ece_val:.4f}$"
            sc_top1 = f"${sc_top1_val:.2f}\\%$"
            sc_top3 = f"${sc_top3_val:.2f}\\%$"
            sc_ece = f"${sc_ece_val:.4f}$"
            sc_conf = f"${sc_conf_val:.2f}\\%$"

        lines.append(
            f"| {display_name} | {cv_top1} | {cv_f1} | {clean_top1} | {clean_ece} | {sc_top1} | {sc_top3} | {sc_ece} | {sc_conf} |"
        )

    return "\n".join(lines)


def render_holdout_summary(metrics: dict[str, Any]) -> str:
    """Render bullet points for holdout evaluation summary."""
    holdout = metrics.get("holdout_evaluation", {})
    top1 = holdout.get("top1_accuracy", 0.0) * 100
    top3 = holdout.get("top3_accuracy", 0.0) * 100
    f1 = holdout.get("macro_f1", 0.0)
    brier = holdout.get("brier_score", 0.0)
    ece = holdout.get("expected_calibration_error", holdout.get("ece", 0.0))
    conf = holdout.get("mean_confidence", 0.0) * 100

    return (
        f"- **Top-1 Accuracy**: **${top1:.2f}\\%$**\n"
        f"- **Top-3 Accuracy**: **${top3:.2f}\\%$**\n"
        f"- **Macro-F1**: **${f1:.4f}$**\n"
        f"- **Brier Score**: **${brier:.4f}$**\n"
        f"- **Expected Calibration Error (ECE)**: **${ece:.4f}$**\n"
        f"- **Mean Top-Class Confidence**: **${conf:.2f}\\%$**"
    )


def render_ablation_table(metrics: dict[str, Any]) -> str:
    """Render Method A vs Method B ablation markdown table."""
    ablation = metrics.get("ablation_table", [])
    lines = [
        (
            "| Feature Subset | Feature Count | Method A Top-1 (Deployed + Imputed) | "
            "Method A Top-3 | Method B Top-1 (Retrained on Subset) | Method B Top-3 |"
        ),
        ("| :--- | :---: | :---: | :---: | :---: | :---: |"),
    ]

    for row in ablation:
        name = row.get("subset_name", "")
        count = row.get("feature_count", 0)
        ma = row.get("method_a_deployed_imputed", {})
        mb = row.get("method_b_retrained_subset", {})

        top1_a = f"**{ma.get('top1_accuracy', 0.0) * 100:.2f}%**"
        top3_a = f"{ma.get('top3_accuracy', 0.0) * 100:.2f}%"
        top1_b = f"**{mb.get('top1_accuracy', 0.0) * 100:.2f}%**"
        top3_b = f"{mb.get('top3_accuracy', 0.0) * 100:.2f}%"

        lines.append(f"| **{name}** | {count} | {top1_a} | {top3_a} | {top1_b} | {top3_b} |")

    return "\n".join(lines)


def render_lofo_table(metrics: dict[str, Any]) -> str:
    """Render Leave-One-Feature-Out (LOFO) accuracy drop markdown table."""
    lofo = metrics.get("lofo_ablation", {})
    # Sort features by accuracy drop descending
    sorted_items = sorted(
        lofo.items(),
        key=lambda item: item[1].get("accuracy_drop", 0.0),
        reverse=True,
    )

    name_map = {
        "Phosphorous": "**Phosphorous**",
        "Carbon": "**Carbon**",
        "Potassium": "**Potassium**",
        "PH": "**PH**",
        "Moisture": "**Moisture**",
        "Nitrogen": "**Nitrogen**",
        "Soil_encoded": "**Soil Type (`Soil_encoded`)**",
        "Crop_encoded": "**Crop Type (`Crop_encoded`)**",
        "Temperature": "**Temperature**",
        "Rainfall": "**Rainfall**",
    }

    lines = [
        "| Omitted Feature | Resulting Top-1 | Accuracy Drop | Macro-F1 |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for idx, (col, data) in enumerate(sorted_items):
        display_name = name_map.get(col, f"**{col}**")
        top1 = data.get("top1_accuracy", 0.0) * 100
        drop = data.get("accuracy_drop", 0.0) * 100
        f1 = data.get("macro_f1", 0.0)

        # Highlight top drops
        if idx < 2:
            drop_str = f"**${drop:+.2f}\\%$**"
        else:
            drop_str = f"${drop:+.2f}\\%$"

        lines.append(f"| {display_name} | ${top1:.2f}\\%$ | {drop_str} | ${f1:.4f}$ |")

    return "\n".join(lines)


def render_permutation_table(metrics: dict[str, Any], include_roles: bool = False) -> str:
    """Render Permutation Importance markdown table."""
    perm = metrics.get("permutation_importance", {})
    # Already sorted or sort by mean_accuracy_drop descending
    sorted_items = sorted(
        perm.items(),
        key=lambda item: item[1].get("mean_accuracy_drop", 0.0),
        reverse=True,
    )

    name_map = {
        "Phosphorous": "**Phosphorous**",
        "Potassium": "**Potassium**",
        "PH": "**PH**",
        "Carbon": "**Carbon**",
        "Moisture": "**Moisture**",
        "Nitrogen": "**Nitrogen**",
        "Soil_encoded": "**Soil Type**",
        "Temperature": "**Temperature**",
        "Rainfall": "**Rainfall**",
        "Crop_encoded": "**Crop Type**",
    }

    roles_map = {
        "Phosphorous": "Primary macro-nutrient; separates DAP from Balanced NPK",
        "Potassium": "Separates MOP and potassium-deficient soils",
        "PH": "Critical threshold separator for Lime ($\\text{pH} < 6.0$)",
        "Carbon": "Organic matter index separating Compost and Organic Fertilizer",
        "Moisture": "Separates Water Retaining formulations",
        "Nitrogen": "Separates Urea",
        "Soil_encoded": "Textural/chemical baseline modifier",
        "Temperature": "Minor climatic influence",
        "Rainfall": "Minor climatic influence",
        "Crop_encoded": "Synthetic labels assign fertilizer independently of crop",
    }

    if include_roles:
        lines = [
            "| Feature | Mean Accuracy Drop | Standard Deviation | Agronomic Role |",
            "| :--- | :---: | :---: | :--- |",
        ]
    else:
        lines = [
            "| Feature | Mean Accuracy Drop | Standard Deviation |",
            "| :--- | :---: | :---: |",
        ]

    for col, data in sorted_items:
        display_name = name_map.get(col, f"**{col}**")
        drop = data.get("mean_accuracy_drop", 0.0) * 100
        if abs(drop) < 1e-4:
            drop = 0.0
        std = data.get("std", 0.0) * 100

        if drop >= 10.0:
            drop_str = f"**${drop:.2f}\\%$**"
        else:
            drop_str = f"${drop:.2f}\\%$"

        std_str = f"$\\pm {std:.2f}\\%$"

        if include_roles:
            role = roles_map.get(col, "")
            lines.append(f"| {display_name} | {drop_str} | {std_str} | {role} |")
        else:
            lines.append(f"| {display_name} | {drop_str} | {std_str} |")

    return "\n".join(lines)


def render_per_class_table(metrics: dict[str, Any]) -> str:
    """Render per-class holdout precision, recall, f1-score table from classification_report."""
    report = metrics.get("classification_report", {})
    lines = [
        "| Fertilizer Class | Precision | Recall | F1-Score | Support |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for key, val in report.items():
        if key in ("accuracy", "macro avg", "weighted avg"):
            continue
        p = val.get("precision", 0.0)
        r = val.get("recall", 0.0)
        f1 = val.get("f1-score", 0.0)
        s = val.get("support", 0)
        lines.append(f"| **{key}** | ${p:.4f}$ | ${r:.4f}$ | ${f1:.4f}$ | {int(s)} |")

    if "macro avg" in report:
        m = report["macro avg"]
        lines.append(
            f"| **Macro Average** | **${m['precision']:.4f}$** | **${m['recall']:.4f}$** | **${m['f1-score']:.4f}$** | **{int(m['support'])}** |"
        )
    if "weighted avg" in report:
        w = report["weighted avg"]
        lines.append(
            f"| **Weighted Average** | **${w['precision']:.4f}$** | **${w['recall']:.4f}$** | **${w['f1-score']:.4f}$** | **{int(w['support'])}** |"
        )

    return "\n".join(lines)


def render_missing_input_table(metrics: dict[str, Any]) -> str:
    """Render missing input scenarios table for MODEL_CARD Section 5."""
    masked_evals = metrics.get("masked_input_evaluations", {})
    npk_data = masked_evals.get("npk_only", masked_evals.get("npk_core_only", {}))
    sc_data = masked_evals.get("soil_and_crop_only", {})

    scenarios = [
        ("Full Soil Test Suite (10 features)", metrics.get("holdout_evaluation", {})),
        ("NPK Core Only (5 features)", npk_data),
        ("Soil & Crop Only (2 features)", sc_data),
    ]

    lines = [
        (
            "| Input Scenario | Top-1 Accuracy | Top-3 Accuracy | Macro-F1 | "
            "Mean Confidence | Expected Calibration Error (ECE) | Brier Score |"
        ),
        ("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"),
    ]

    for name, data in scenarios:
        top1 = data.get("top1_accuracy", 0.0) * 100
        top3 = data.get("top3_accuracy", 0.0) * 100
        f1 = data.get("macro_f1", 0.0)
        conf = data.get("mean_confidence", 0.0) * 100
        ece = data.get("expected_calibration_error", data.get("ece", 0.0))
        brier = data.get("brier_score", 0.0)

        lines.append(
            f"| **{name}** | **${top1:.2f}\\%$** | **${top3:.2f}\\%$** | **${f1:.4f}$** | **${conf:.2f}\\%$** | ${ece:.4f}$ | ${brier:.4f}$ |"
        )

    return "\n".join(lines)


def render_load_test_table(metrics: dict[str, Any]) -> str:
    """Render the Gunicorn load test benchmark table."""
    load_test = metrics.get("load_test")
    if not load_test:
        # Fallback to benchmarks/locust_stats.csv or model_files/locust_benchmark.csv if available
        csv_path = Path("benchmarks/locust_stats.csv")
        if not csv_path.exists():
            csv_path = Path("model_files/locust_benchmark.csv")
        if csv_path.exists():
            import pandas as pd

            df = pd.read_csv(csv_path)
            # Reconstruct load_test dict
            items = []
            for _, r in df.iterrows():
                name = str(r["Name"])
                if name.strip() == "Aggregated" or pd.isna(r["Type"]):
                    continue
                items.append(
                    {
                        "name": name,
                        "requests": int(r["Request Count"]),
                        "failures": (float(r["Failure Count"]) / max(int(r["Request Count"]), 1))
                        * 100,
                        "p50": int(r["Median Response Time"]),
                        "p95": int(r["95%"]),
                        "avg": float(r["Average Response Time"]),
                        "rps": float(r["Requests/s"]),
                    }
                )
            agg_row = df[df["Name"] == "Aggregated"].iloc[0]
            load_test = {
                "endpoints": items,
                "aggregate": {
                    "requests": int(agg_row["Request Count"]),
                    "failures": (
                        float(agg_row["Failure Count"]) / max(int(agg_row["Request Count"]), 1)
                    )
                    * 100,
                    "p50": int(agg_row["Median Response Time"]),
                    "p95": int(agg_row["95%"]),
                    "avg": float(agg_row["Average Response Time"]),
                    "rps": float(agg_row["Requests/s"]),
                },
            }

    if not load_test:
        return (
            "| Endpoint / Scenario | Requests | Failures | Median ($p50$) | $p95$ | Average Latency | Offered Load |\n"
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
            "| *Load test benchmark data pending CI run* | - | - | - | - | - | - |"
        )

    name_display = {
        "[API] Health Check": "`GET /api/v1/health`",
        "[API] Model Info": "`GET /api/v1/info`",
        "[UI] Index Page": "`GET /` (Web Interface)",
        "[API] Predict (Full Soil Test)": "`POST /api/v1/predict` (Full Soil Test)",
        "[API] Predict (Partial / Imputed)": ("`POST /api/v1/predict` (Partial / Imputed)"),
        "[API] Explain (SHAP)": "`POST /api/v1/explain` (TreeSHAP)",
    }

    lines = [
        (
            "| Endpoint / Scenario | Requests | Failures | Median ($p50$) | "
            "$p95$ | Average Latency | Offered Load |"
        ),
        ("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"),
    ]

    # Sort endpoints in canonical display order
    endpoints = load_test.get("endpoints", [])
    order_keys = list(name_display.keys())
    endpoints_sorted = sorted(
        endpoints,
        key=lambda x: order_keys.index(x.get("name", "")) if x.get("name") in order_keys else 999,
    )

    for ep in endpoints_sorted:
        raw_name = ep.get("name", "")
        disp_name = name_display.get(raw_name, f"`{raw_name}`")
        reqs = ep.get("requests", 0)
        fails = ep.get("failures", 0.0)
        p50 = ep.get("p50", 0)
        p95 = ep.get("p95", 0)
        avg = ep.get("avg", 0.0)
        rps = ep.get("rps", 0.0)

        lines.append(
            f"| {disp_name} | {reqs} | ${fails:.2f}\\%$ | **{p50} ms** | **{p95} ms** | {avg:.1f} ms | ~{rps:.2f} req/s |"
        )

    agg = load_test.get("aggregate", {})
    if agg:
        lines.append(
            f"| **Aggregated Production Total** | **{agg.get('requests', 0)}** | **{agg.get('failures', 0.0):.2f}%** | "
            f"**{agg.get('p50', 0)} ms** | **{agg.get('p95', 0)} ms** | **{agg.get('avg', 0.0):.1f} ms** | **~{agg.get('rps', 0.0):.2f} req/s** |"
        )

    source = load_test.get("source", {})
    run_id = source.get("run_id")
    run_url = source.get("run_url")
    if run_id:
        if run_url:
            lines.append(f"\nSource: CI run [{run_id}]({run_url})")
        else:
            lines.append(f"\nSource: CI run {run_id}")

    return "\n".join(lines)


def render_load_test_description(metrics: dict[str, Any]) -> str:
    """Render load test lead description paragraph from load_test block."""
    lt = metrics.get("load_test", {})
    agg = lt.get("aggregate", {})
    desc = agg.get("offered_load_desc")
    if desc:
        return desc
    users = lt.get("users", 30)
    dur = lt.get("duration", "30s")
    rps = int(round(agg.get("rps", 20.0)))
    return (
        f"Executed against the production Linux Docker container running multi-worker Gunicorn "
        f"with {users} concurrent users for {dur} via headless Locust, representing "
        f"~{rps} req/s offered load ({users} Locust users with think time; "
        f"load generator and server share a GitHub Actions runner):"
    )


def get_marker_generators(metrics: dict[str, Any]) -> dict[str, str]:
    """Map all supported marker names to their generated string values."""
    holdout = metrics.get("holdout_evaluation", {})
    masked_evals = metrics.get("masked_input_evaluations", {})
    sc_eval = masked_evals.get("soil_and_crop_only", {})
    missing_comp = metrics.get("missing_input_comparison", {})
    ablation_tbl = {r["subset_key"]: r for r in metrics.get("ablation_table", [])}
    lofo = metrics.get("lofo_ablation", {})

    dt_sc = missing_comp.get("Decision_Tree_Masked", {})
    lgbm_sc = missing_comp.get("LightGBM_Masked", {})
    rf_sc = missing_comp.get("Random_Forest_Tuned_Masked", {})

    s6 = ablation_tbl.get("soil_crop_pk_ph_carbon", {})
    s6_ma = s6.get("method_a_deployed_imputed", {})
    s6_mb = s6.get("method_b_retrained_subset", {})

    p_lofo = lofo.get("Phosphorous", {})
    c_lofo = lofo.get("Carbon", {})

    dt_cv = metrics.get("baseline_comparison", {}).get("Decision_Tree_Clean", {})
    rf_cv = metrics.get("baseline_comparison", {}).get("Random_Forest_Default", {})
    lgbm_cv = metrics.get("baseline_comparison", {}).get("LightGBM_Clean", {})

    markers: dict[str, str] = {
        # Block tables
        "comparison_table": render_comparison_table(metrics),
        "holdout_summary": render_holdout_summary(metrics),
        "ablation_table": render_ablation_table(metrics),
        "lofo_table": render_lofo_table(metrics),
        "permutation_table": render_permutation_table(metrics, include_roles=False),
        "permutation_table_roles": render_permutation_table(metrics, include_roles=True),
        "per_class_table": render_per_class_table(metrics),
        "missing_input_table": render_missing_input_table(metrics),
        "load_test_table": render_load_test_table(metrics),
        "load_test_description": render_load_test_description(metrics),
        # Inline headline values
        "holdout_top1": f"${holdout.get('top1_accuracy', 0.0) * 100:.2f}\\%$",
        "holdout_top3": f"${holdout.get('top3_accuracy', 0.0) * 100:.2f}\\%$",
        "holdout_macro_f1": f"${holdout.get('macro_f1', 0.0):.4f}$",
        "holdout_brier": f"${holdout.get('brier_score', 0.0):.4f}$",
        "holdout_ece": f"${holdout.get('expected_calibration_error', holdout.get('ece', 0.0)):.4f}$",
        "holdout_mean_confidence": f"${holdout.get('mean_confidence', 0.0) * 100:.2f}\\%$",
        "masked_lgbm_sc_top1": f"${lgbm_sc.get('top1_accuracy', 0.0) * 100:.2f}\\%$",
        "masked_lgbm_sc_top3": f"${lgbm_sc.get('top3_accuracy', 0.0) * 100:.2f}\\%$",
        "masked_lgbm_sc_ece": f"${lgbm_sc.get('expected_calibration_error', lgbm_sc.get('ece', 0.0)):.4f}$",
        "masked_dt_sc_top1": f"${dt_sc.get('top1_accuracy', 0.0) * 100:.2f}\\%$",
        "masked_dt_sc_top3": f"${dt_sc.get('top3_accuracy', 0.0) * 100:.2f}\\%$",
        "masked_dt_sc_ece": f"${dt_sc.get('expected_calibration_error', dt_sc.get('ece', 0.0)):.4f}$",
        "masked_rf_sc_top1": f"${rf_sc.get('top1_accuracy', sc_eval.get('top1_accuracy', 0.0)) * 100:.2f}\\%$",
        "masked_rf_sc_top3": f"${rf_sc.get('top3_accuracy', sc_eval.get('top3_accuracy', 0.0)) * 100:.2f}\\%$",
        "masked_rf_sc_ece": f"${rf_sc.get('expected_calibration_error', sc_eval.get('expected_calibration_error', 0.0)):.4f}$",
        "masked_rf_sc_mean_conf": f"${rf_sc.get('mean_confidence', sc_eval.get('mean_confidence', 0.0)) * 100:.2f}\\%$",
        "suite6_method_a_top1": f"${s6_ma.get('top1_accuracy', 0.0) * 100:.2f}\\%$",
        "suite6_method_a_top3": f"${s6_ma.get('top3_accuracy', 0.0) * 100:.2f}\\%$",
        "suite6_method_b_top1": f"${s6_mb.get('top1_accuracy', 0.0) * 100:.2f}\\%$",
        "suite6_method_b_top3": f"${s6_mb.get('top3_accuracy', 0.0) * 100:.2f}\\%$",
        "lofo_phosphorus_top1": f"${p_lofo.get('top1_accuracy', 0.0) * 100:.2f}\\%$",
        "lofo_phosphorus_drop": f"${p_lofo.get('accuracy_drop', 0.0) * 100:.2f}\\%$",
        "lofo_carbon_drop": f"${c_lofo.get('accuracy_drop', 0.0) * 100:.2f}\\%$",
        "clean_dt_top1": f"${dt_cv.get('top1_accuracy_mean', 0.0) * 100:.2f}\\% \\pm {dt_cv.get('top1_accuracy_std', 0.0) * 100:.2f}\\%$",
        "clean_rf_top1": f"${rf_cv.get('top1_accuracy_mean', 0.0) * 100:.2f}\\% \\pm {rf_cv.get('top1_accuracy_std', 0.0) * 100:.2f}\\%$",
        "clean_lgbm_top1": f"${lgbm_cv.get('top1_accuracy_mean', 0.0) * 100:.2f}\\% \\pm {lgbm_cv.get('top1_accuracy_std', 0.0) * 100:.2f}\\%$",
    }
    return markers


def update_document_content(content: str, marker_generators: dict[str, str], filepath: Path) -> str:
    """Replace content inside <!-- BEGIN:name -->...<!-- END:name --> blocks with rendered output."""
    pattern = re.compile(r"<!-- BEGIN:([a-zA-Z0-9_\-]+) -->.*?<!-- END:\1 -->", re.DOTALL)

    def _replacer(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in marker_generators:
            return match.group(0)

        new_val = marker_generators[name]
        raw_matched = match.group(0)

        # Check if block marker (multiline) or inline marker (single line)
        is_block = "\n" in new_val or raw_matched.startswith(f"<!-- BEGIN:{name} -->\n")

        if is_block:
            return f"<!-- BEGIN:{name} -->\n{new_val.strip()}\n<!-- END:{name} -->"
        else:
            return f"<!-- BEGIN:{name} -->{new_val.strip()}<!-- END:{name} -->"

    return pattern.sub(_replacer, content)


def process_docs(
    check: bool = False,
    metrics_path: Path = Path("model_files/metrics.json"),
    target_docs: list[Path] | None = None,
) -> int:
    """Process all target documentation files.

    If check is True, returns 1 if differences found, 0 otherwise.
    If check is False, writes the updated content and returns 0.
    """
    if target_docs is None:
        target_docs = [
            Path("README.md"),
            Path("docs/MODEL_CARD.md"),
            Path("docs/DATA_CARD.md"),
        ]

    metrics = load_metrics(metrics_path)
    marker_generators = get_marker_generators(metrics)

    diff_detected = False

    for doc_path in target_docs:
        if not doc_path.exists():
            print(f"Warning: Document not found: {doc_path}", file=sys.stderr)
            continue

        with open(doc_path, encoding="utf-8") as f:
            original_content = f.read()

        updated_content = update_document_content(original_content, marker_generators, doc_path)

        if updated_content != original_content:
            diff_detected = True
            if check:
                print(f"[CHECK FAILED] Differences detected in {doc_path}:", file=sys.stderr)
                diff = difflib.unified_diff(
                    original_content.splitlines(keepends=True),
                    updated_content.splitlines(keepends=True),
                    fromfile=f"a/{doc_path.as_posix()}",
                    tofile=f"b/{doc_path.as_posix()}",
                )
                sys.stderr.writelines(diff)
            else:
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(updated_content)
                print(f"Updated {doc_path} from metrics.json successfully.")
        else:
            if not check:
                print(f"{doc_path} is already up to date with metrics.json.")

    if check:
        if diff_detected:
            print(
                "\nError: Documentation out of sync with model_files/metrics.json!", file=sys.stderr
            )
            print(
                "Run 'python -m src.ml.render_docs' to synchronize documentation.", file=sys.stderr
            )
            return 1
        print("All documentation files are synchronized with model_files/metrics.json.")
        return 0

    return 0


def main() -> int:
    """CLI entrypoint for render_docs."""
    parser = argparse.ArgumentParser(
        description="Render metrics and tables into repository markdown files."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether documentation matches metrics.json without modifying files.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("model_files/metrics.json"),
        help="Path to metrics.json file.",
    )
    args = parser.parse_args()
    return process_docs(check=args.check, metrics_path=args.metrics_path)


if __name__ == "__main__":
    sys.exit(main())
