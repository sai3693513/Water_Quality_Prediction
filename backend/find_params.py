import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv("c:/Users/saisriram/Downloads/project details/project details/app/data/uploaded_dataset.csv")

if "Potability" in df.columns:
    df = df.rename(columns={"Potability": "quality"})

mapping = {c.lower(): c for c in df.columns}
df = df.rename(columns={
    mapping["ph"]: "ph",
    mapping["hardness"]: "hardness",
    mapping["solids"]: "solids",
    mapping["chloramines"]: "chloramines",
    mapping["sulfate"]: "sulfate",
    mapping["quality"]: "quality",
})

feature_cols = [
    "ph", "hardness", "solids", "chloramines", "sulfate"
]

X = df[feature_cols]
y = df["quality"]

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# Test constraints
for depth in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    rf = RandomForestClassifier(n_estimators=500, max_depth=depth, min_samples_split=2, min_samples_leaf=1, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_train, rf.predict(X_train)) * 100
    print(f"RF Depth {depth}: {acc:.2f}%")
