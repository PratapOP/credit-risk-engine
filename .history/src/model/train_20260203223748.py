import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
from pathlib import Path

# ---------------------------
# Paths
# ---------------------------
DATA_PATH = Path("data/processed/model_input.csv")
MODEL_PATH = Path("models/random_forest.pkl")

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop("defaulted", axis=1)
y = df["defaulted"]

# ---------------------------
# Train / Test split (bank style)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------------------
# Random Forest Model
# ---------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------
probs = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, probs)

print("\nROC-AUC:", round(roc, 4))
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))

# ---------------------------
# Save model
# ---------------------------
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
