import shap
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/random_forest.pkl")
DATA_PATH = Path("data/processed/model_input.csv")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

X = df.drop("defaulted", axis=1)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Show explanation for first borrower
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0])
