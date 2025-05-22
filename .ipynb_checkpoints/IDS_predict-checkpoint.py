import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Ask for dataset
file_name = input(" Enter dataset filename (CSV): ")

if not os.path.exists(file_name):
    print("❌ File not found!")
    exit()

# Load dataset
print(f" Loading {file_name}...")
df = pd.read_csv(file_name, header=None)

# Preprocessing
X = df.drop(columns=[41])  # Remove label
cat_columns = [1, 2, 3]

for col in cat_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

X_scaled = scaler.transform(X)

# Predict
predictions = model.predict(X_scaled)

# Print results
print("\n Predictions:")
for idx, pred in enumerate(predictions):
    label = "Normal" if pred == 0 else "Attack"
    print(f"Row {idx+1}: {label}")

# Summary
attack_count = sum(predictions)
normal_count = len(predictions) - attack_count
print("\n Summary:")
print(f"Normal: {normal_count}")
print(f"Attack: {attack_count}")
