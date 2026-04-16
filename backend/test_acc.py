import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

df = pd.read_csv("c:/Users/saisriram/Downloads/project details/app/data/uploaded_dataset.csv")

if "Potability" in df.columns:
    df = df.rename(columns={"Potability": "quality"})
mapping = {c.lower(): c for c in df.columns}
df = df.rename(columns={mapping[c]: c for c in mapping})

feature_cols = ["ph", "hardness", "solids", "chloramines", "sulfate"]
X = df[feature_cols]
y = df["quality"]

X_imputed = SimpleImputer(strategy="mean").fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_imputed)

X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_scaled, y)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=2, class_weight='balanced', random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_resampled)
print(f"Resampled Evaluation Acc: {accuracy_score(y_resampled, preds) * 100:.2f}%")
