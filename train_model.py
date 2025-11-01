import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle
import os
sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("Numpy", np.__version__)
print("Pandas", pd.__version__)
print("Matplotlib", plt.matplotlib.__version__)
print("Seaborn", sns.__version__)
print("Sklearn", sklearn.__version__)

data_path = 'fertilizer_recommendation_dataset.csv'
df = pd.read_csv(data_path, encoding='ascii', delimiter=',')
print("Data loaded successfully. Dataset shape:", df.shape)

le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

df['Soil_encoded'] = le_soil.fit_transform(df['Soil'].astype(str))
df['Crop_encoded'] = le_crop.fit_transform(df['Crop'].astype(str))
df['Fertilizer_encoded'] = le_fertilizer.fit_transform(df['Fertilizer'].astype(str))

features = ['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon', 'Soil_encoded', 'Crop_encoded']
target = 'Fertilizer_encoded'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_fertilizer.classes_))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Create model_files directory if it doesn't exist
if not os.path.exists('model_files'):
    os.makedirs('model_files')

# Save model
with open('model_files/model.pkl', 'wb') as file:
    pickle.dump(clf, file)

# Save encoders
encoders = {
    'le_soil': le_soil,
    'le_crop': le_crop,
    'le_fertilizer': le_fertilizer
}
with open('model_files/encoders.pkl', 'wb') as file:
    pickle.dump(encoders, file)

# Save feature list
with open('model_files/features.pkl', 'wb') as file:
    pickle.dump(features, file)

print("\n✓ Model saved to model_files/model.pkl")
print("✓ Encoders saved to model_files/encoders.pkl")
print("✓ Features saved to model_files/features.pkl")
