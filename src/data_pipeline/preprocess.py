import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------
# Paths
# ---------------------------
RAW_PATH = Path("data/raw/loan_applications.csv")
PROCESSED_PATH = Path("data/processed/model_input.csv")

# ---------------------------
# Load raw data
# ---------------------------
df = pd.read_csv(RAW_PATH)

# ---------------------------
# Data validation (bank style)
# ---------------------------
df = df[df["annual_income"] > 0]
df = df[df["total_debt"] >= 0]
df = df[df["credit_score"].between(300, 850)]
df = df[df["credit_utilization"].between(0, 1)]
df = df[df["loan_term_months"].between(6, 120)]

# ---------------------------
# Feature Engineering
# ---------------------------

# Debt to Income
df["dti"] = df["total_debt"] / df["annual_income"]

# Monthly loan EMI estimate (simple)
df["monthly_income"] = df["annual_income"] / 12
df["emi_estimate"] = df["loan_amount"] / df["loan_term_months"]

# Cash left after EMI
df["income_after_emi"] = df["monthly_income"] - df["emi_estimate"]

# Liquidity buffer
df["savings_to_income"] = df["savings"] / df["annual_income"]

# Credit pressure (stress indicator)
df["credit_pressure"] = df["credit_utilization"] * df["missed_payments"]

# Stability index
df["stability_index"] = df["employment_length"] / (df["dependents"] + 1)

# Affordability ratio
df["affordability_ratio"] = df["emi_estimate"] / df["monthly_income"]

# ---------------------------
# Select model features
# ---------------------------
model_df = df[
    [
        "credit_score",
        "dti",
        "credit_utilization",
        "missed_payments",
        "employment_length",
        "savings_to_income",
        "credit_pressure",
        "stability_index",
        "affordability_ratio",
        "defaulted"
    ]
]

# Drop invalid rows
model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

# ---------------------------
# Save
# ---------------------------
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
model_df.to_csv(PROCESSED_PATH, index=False)

print("Processed dataset saved to:", PROCESSED_PATH)
print("Final rows:", len(model_df))
