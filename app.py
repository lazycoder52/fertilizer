import json
import os
import pickle
from pathlib import Path
from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

app = Flask(__name__)

# Load model and encoders (prefer joblib, fallback to pickle)
model_dir = Path("model_files")

if joblib and (model_dir / "model.joblib").exists():
    model = joblib.load(model_dir / "model.joblib")
    encoders = joblib.load(model_dir / "encoders.joblib")
    required_features = joblib.load(model_dir / "features.joblib")
else:
    with open(model_dir / "model.pkl", "rb") as file:
        model = pickle.load(file)
    with open(model_dir / "encoders.pkl", "rb") as file:
        encoders = pickle.load(file)
    with open(model_dir / "features.pkl", "rb") as file:
        required_features = pickle.load(file)

le_soil = encoders["le_soil"]
le_crop = encoders["le_crop"]
le_fertilizer = encoders["le_fertilizer"]

# Get available options for categorical features
soil_types = le_soil.classes_.tolist()
crop_types = le_crop.classes_.tolist()

# Load fertilizer descriptions if available
descriptions_path = model_dir / "fertilizer_descriptions.json"
fertilizer_descriptions = {}
if descriptions_path.exists():
    with open(descriptions_path, "r", encoding="utf-8") as f:
        fertilizer_descriptions = json.load(f)

# Load metadata if available
metadata_path = model_dir / "model_metadata.json"
model_metadata = {}
if metadata_path.exists():
    with open(metadata_path, "r", encoding="utf-8") as f:
        model_metadata = json.load(f)


def normalize_category(value, valid_classes, suffix_to_strip=""):
    if not value:
        return None
    val_clean = str(value).strip().lower()
    for vc in valid_classes:
        vc_clean = vc.lower()
        if val_clean == vc_clean:
            return vc
        if suffix_to_strip and val_clean == vc_clean.replace(suffix_to_strip.lower(), "").strip():
            return vc
    return None


@app.route("/")
def home():
    return render_template("index.html", soil_types=soil_types, crop_types=crop_types)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        # Extract and validate input
        input_features = {}

        # Numerical features
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

        for feature in numerical_features:
            if feature in data and data[feature] is not None and data[feature] != "":
                input_features[feature] = float(data[feature])
            else:
                input_features[feature] = 0.0

        # Categorical features with resilient normalization
        if "Soil" in data and data["Soil"]:
            matched_soil = normalize_category(data["Soil"], soil_types, suffix_to_strip="Soil")
            if matched_soil:
                input_features["Soil_encoded"] = le_soil.transform([matched_soil])[0]
            else:
                return jsonify({"error": f"Invalid soil type: {data['Soil']}"}), 400
        else:
            input_features["Soil_encoded"] = 0

        if "Crop" in data and data["Crop"]:
            matched_crop = normalize_category(data["Crop"], crop_types)
            if matched_crop:
                input_features["Crop_encoded"] = le_crop.transform([matched_crop])[0]
            else:
                return jsonify({"error": f"Invalid crop type: {data['Crop']}"}), 400
        else:
            input_features["Crop_encoded"] = 0

        # Create input dataframe with correct feature order
        input_df = pd.DataFrame([input_features], columns=required_features)

        # Make prediction
        prediction_encoded = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Decode prediction
        fertilizer_recommendation = le_fertilizer.inverse_transform([prediction_encoded])[0]
        confidence = float(max(prediction_proba)) * 100

        # Get top 3 recommendations with probabilities and remarks
        top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
        top_recommendations = [
            {
                "fertilizer": le_fertilizer.inverse_transform([idx])[0],
                "confidence": round(float(prediction_proba[idx]) * 100, 2),
                "remark": fertilizer_descriptions.get(
                    le_fertilizer.inverse_transform([idx])[0], ""
                ),
            }
            for idx in top_3_indices
        ]

        primary_remark = fertilizer_descriptions.get(fertilizer_recommendation, "")

        return jsonify({
            "success": True,
            "recommendation": fertilizer_recommendation,
            "confidence": round(confidence, 2),
            "remark": primary_remark,
            "top_3_recommendations": top_recommendations,
            "input_received": {k: v for k, v in data.items()},
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/info", methods=["GET"])
def info():
    """Endpoint to get available options and model metadata"""
    return jsonify({
        "soil_types": soil_types,
        "crop_types": crop_types,
        "required_features": required_features,
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
        "required_categorical": ["Soil", "Crop"],
        "model_metadata": model_metadata,
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)