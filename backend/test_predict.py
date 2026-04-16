import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODELS_PATH = os.path.join(MODEL_DIR, "models.pkl")
UPLOAD_PATH = os.path.join(BASE_DIR, "data", "uploaded_dataset.csv")

try:
    models = joblib.load(MODELS_PATH)
    rf_model = models["Random Forest"]
    xg_model = models["XGBoost"]
    
    df = pd.read_csv(UPLOAD_PATH)
    df.columns = df.columns.str.lower()
    
    if "potability" in df.columns:
        df = df.rename(columns={"potability": "quality"})
        
    feature_cols = ["ph", "hardness", "solids", "chloramines", "sulfate"]
    X = df[feature_cols]
    y = df["quality"]
    
    print("Testing Random Forest on Training Data:")
    rf_preds = rf_model.predict(X)
    print("RF Predictions distribution:", np.unique(rf_preds, return_counts=True))
    
    print("\nTesting XGBoost on Training Data:")
    xg_preds = xg_model.predict(X)
    print("XG Predictions distribution:", np.unique(xg_preds, return_counts=True))
    
    # Let's find some safe water values (where true y == 1 and prediction == 1)
    safe_indices = np.where((y == 1) & (rf_preds == 1))[0]
    if len(safe_indices) > 0:
        print("\nExample Safe Water Values:")
        safe_val = X.iloc[safe_indices[0]]
        print(safe_val.to_dict())
    else:
        print("\nNo Safe Water predictions match actual Safe Water!")

except Exception as e:
    print(f"Error: {e}")
