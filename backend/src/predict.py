import pickle
import os
import numpy as np

# Get absolute path to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)


def predict_water_quality(features):
    features = np.array([features])
    result = model.predict(features)[0]

    if result == 1:
        return "Water is Safe to Drink"
    else:
        return "Water is Not Safe"