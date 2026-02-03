from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/random_forest.pkl")

def compute_features(data):
    income = float(data["annual_income"])
    debt = float(data["total_debt"])
    savings = float(data["savings"])
    dependents = int(data["dependents"])
    credit_score = float(data["credit_score"])
    credit_util = float(data["credit_utilization"])
    missed = float(data["missed_payments"])
    emp = float(data["employment_length"])
    loan = float(data["loan_amount"])
    term = float(data["loan_term_months"])

    dti = debt / income
    monthly_income = income / 12
    emi = loan / term
    income_after = monthly_income - emi
    savings_ratio = savings / income
    credit_pressure = credit_util * missed
    stability = emp / (dependents + 1)
    affordability = emi / monthly_income

    return np.array([[credit_score, dti, credit_util, missed, emp,
                      savings_ratio, credit_pressure, stability, affordability]])

def risk_band(pd):
    if pd < 0.05:
        return "Low"
    elif pd < 0.20:
        return "Medium"
    else:
        return "High"

@app.route("/score", methods=["POST"])
def score():
    data = request.json
    X = compute_features(data)
    pd_value = model.predict_proba(X)[0][1]

    risk = risk_band(pd_value)
    decision = "Approve" if risk == "Low" else "Review" if risk == "Medium" else "Reject"

    return jsonify({
        "probability_of_default": round(float(pd_value), 4),
        "risk_tier": risk,
        "decision": decision
    })

if __name__ == "__main__":
    app.run(debug=True)