import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/water_quality.csv")

X = df[["ph", "Hardness", "Solids", "Turbidity"]]
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Trained and Saved!")