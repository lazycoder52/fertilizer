from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and encoders
with open('model_files/model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model_files/encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

with open('model_files/features.pkl', 'rb') as file:
    required_features = pickle.load(file)

le_soil = encoders['le_soil']
le_crop = encoders['le_crop']
le_fertilizer = encoders['le_fertilizer']

# Get available options for categorical features
soil_types = le_soil.classes_.tolist()
crop_types = le_crop.classes_.tolist()

@app.route('/')
def home():
    return render_template('index.html', 
                         soil_types=soil_types, 
                         crop_types=crop_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract and validate input
        input_features = {}
        
        # Numerical features
        numerical_features = ['Temperature', 'Moisture', 'Rainfall', 'PH', 
                            'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon']
        
        for feature in numerical_features:
            if feature in data:
                input_features[feature] = float(data[feature])
            else:
                # Use median values as defaults if not provided
                input_features[feature] = 0.0
        
        # Categorical features
        if 'Soil' in data:
            try:
                input_features['Soil_encoded'] = le_soil.transform([data['Soil']])[0]
            except ValueError:
                return jsonify({'error': f'Invalid soil type: {data["Soil"]}'}), 400
        else:
            input_features['Soil_encoded'] = 0
        
        if 'Crop' in data:
            try:
                input_features['Crop_encoded'] = le_crop.transform([data['Crop']])[0]
            except ValueError:
                return jsonify({'error': f'Invalid crop type: {data["Crop"]}'}), 400
        else:
            input_features['Crop_encoded'] = 0
        
        # Create input dataframe with correct feature order
        input_df = pd.DataFrame([input_features], columns=required_features)
        
        # Make prediction
        prediction_encoded = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Decode prediction
        fertilizer_recommendation = le_fertilizer.inverse_transform([prediction_encoded])[0]
        confidence = float(max(prediction_proba)) * 100
        
        # Get top 3 recommendations with probabilities
        top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
        top_recommendations = [
            {
                'fertilizer': le_fertilizer.inverse_transform([idx])[0],
                'confidence': float(prediction_proba[idx]) * 100
            }
            for idx in top_3_indices
        ]
        
        return jsonify({
            'success': True,
            'recommendation': fertilizer_recommendation,
            'confidence': round(confidence, 2),
            'top_3_recommendations': top_recommendations,
            'input_received': {k: v for k, v in data.items()}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def info():
    """Endpoint to get available options"""
    return jsonify({
        'soil_types': soil_types,
        'crop_types': crop_types,
        'required_features': required_features,
        'optional_features': ['Temperature', 'Moisture', 'Rainfall', 'PH', 
                            'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon'],
        'required_categorical': ['Soil', 'Crop']
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)