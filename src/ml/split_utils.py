"""Shared dataset splitting utilities to guarantee consistent dev/holdout row order across ML workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_dataset_splits(
    data_path: Path | str = "fertilizer_recommendation_dataset.csv",
    output_dir: Path | str = "model_files",
    seed: int = 42,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Load dataset, fit or load encoders, and produce dev and holdout splits with identical row ordering.

    Uses the saved-split (drop-index) construction:
        X_dev = X.drop(index=X_test.index)
        y_dev = y.drop(index=X_test.index)
    This guarantees identical row order across train_and_evaluate, robustness_eval, and ablation.
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)

    df = pd.read_csv(data_path)

    encoders_path = output_dir / "encoders.joblib"
    test_data_path = output_dir / "test_data.joblib"

    if encoders_path.exists():
        encoders = joblib.load(encoders_path)
        le_soil = encoders["le_soil"]
        le_crop = encoders["le_crop"]
        le_fertilizer = encoders["le_fertilizer"]
    else:
        le_soil = LabelEncoder()
        le_crop = LabelEncoder()
        le_fertilizer = LabelEncoder()
        le_soil.fit(df["Soil"].astype(str))
        le_crop.fit(df["Crop"].astype(str))
        le_fertilizer.fit(df["Fertilizer"].astype(str))
        encoders = {
            "le_soil": le_soil,
            "le_crop": le_crop,
            "le_fertilizer": le_fertilizer,
        }

    df["Soil_encoded"] = le_soil.transform(df["Soil"].astype(str))
    df["Crop_encoded"] = le_crop.transform(df["Crop"].astype(str))
    df["Fertilizer_encoded"] = le_fertilizer.transform(df["Fertilizer"].astype(str))

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
    numerical_cols = [c for c in feature_cols if c not in ["Soil_encoded", "Crop_encoded"]]

    # Verify zero target leakage
    assert "Remark" not in feature_cols, "CRITICAL ERROR: 'Remark' cannot be in feature columns!"
    assert "Fertilizer" not in feature_cols, (
        "CRITICAL ERROR: 'Fertilizer' cannot be in feature columns!"
    )

    X = df[feature_cols]
    y = df["Fertilizer_encoded"]

    if test_data_path.exists():
        test_data = joblib.load(test_data_path)
        X_test: pd.DataFrame = test_data["X_test"]
        y_test: pd.Series = test_data["y_test"]
    else:
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

    # Saved-split (drop-index) construction: preserves original DataFrame index order
    X_dev = X.drop(index=X_test.index)
    y_dev = y.drop(index=X_test.index)

    class_names = [str(c) for c in le_fertilizer.classes_]

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_dev": X_dev,
        "y_dev": y_dev,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "numerical_cols": numerical_cols,
        "class_names": class_names,
        "encoders": encoders,
        "le_soil": le_soil,
        "le_crop": le_crop,
        "le_fertilizer": le_fertilizer,
    }
