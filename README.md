# Fertilizer Recommendation System

An ML-powered web application that recommends optimal fertilizers based on soil and crop conditions.

## Features

- ✅ Accepts partial input (only Soil and Crop type required)
- ✅ Uses intelligent defaults for missing numerical features
- ✅ Provides confidence scores for predictions
- ✅ Shows top 3 fertilizer recommendations
- ✅ Beautiful, responsive UI
- ✅ Ready for deployment on Render

## Project Structure
```
fertilizer-recommendation/
├── app.py                              # Flask application
├── train_model.py                      # Model training script
├── fertilizer_recommendation_dataset.csv
├── requirements.txt
├── Procfile                           # Render configuration
├── render.yaml                        # Render build settings
├── .gitignore
├── model_files/
│   ├── model.pkl                      # Trained model
│   ├── encoders.pkl                   # Label encoders
│   └── features.pkl                   # Feature list
└── templates/
    └── index.html                     # Frontend UI
```

## Local Setup

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd fertilizer-recommendation
```

### 2. Create Virtual Environment
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python train_model.py
```

This will:
- Load the dataset
- Train the Random Forest model
- Save model, encoders, and features to `model_files/`
- Display accuracy and feature importance

### 5. Run Locally
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## API Endpoints

### GET `/`
Returns the web interface

### POST `/predict`
Makes fertilizer predictions

**Request Body:**
```json
{
  "Soil": "Loamy",
  "Crop": "Wheat",
  "Temperature": 25.5,
  "Moisture": 60.0,
  "Rainfall": 150.0,
  "PH": 6.5,
  "Nitrogen": 40,
  "Phosphorous": 30,
  "Potassium": 20,
  "Carbon": 1.5
}
```

**Note:** Only `Soil` and `Crop` are required. All other fields are optional.

**Response:**
```json
{
  "success": true,
  "recommendation": "Urea",
  "confidence": 95.67,
  "top_3_recommendations": [
    {"fertilizer": "Urea", "confidence": 95.67},
    {"fertilizer": "DAP", "confidence": 3.21},
    {"fertilizer": "NPK", "confidence": 1.12}
  ],
  "input_received": {...}
}
```

### GET `/api/info`
Returns available options and required features

### GET `/health`
Health check endpoint (for monitoring)

## Deployment on Render

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy on Render

1. Go to [render.com](https://render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml` and configure everything
5. Click "Create Web Service"

### Step 3: Wait for Deployment

Deployment takes 2-5 minutes. You'll get a URL like:
```
https://fertilizer-recommendation-xxxx.onrender.com
```

## Model Performance

- **Accuracy:** ~99.7%
- **Features:** 10 (8 numerical + 2 categorical)
- **Algorithm:** Random Forest Classifier
- **Classes:** 7 fertilizer types

## Technologies Used

- **Backend:** Flask, Python
- **ML:** scikit-learn, pandas, numpy
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Render, Gunicorn

## License

MIT License

## Author

Created by Aditya