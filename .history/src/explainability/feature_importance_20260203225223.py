import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/random_forest.pkl")
DATA_PATH = Path("data/processed/model_input.csv")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

X = df.drop("defaulted", axis=1)

importance = model.feature_importances_
features = X.columns

imp_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print("\nGlobal Feature Importance:")
print(imp_df)
