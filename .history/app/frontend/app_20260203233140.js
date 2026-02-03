function evaluate() {
    fetch("http://127.0.0.1:5000/score", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            annual_income: document.getElementById("income").value,
            total_debt: document.getElementById("debt").value,
            savings: document.getElementById("savings").value,
            dependents: document.getElementById("dependents").value,
            credit_score: document.getElementById("credit").value,
            credit_utilization: document.getElementById("util").value,
            missed_payments: document.getElementById("missed").value,
            employment_length: document.getElementById("emp").value,
            loan_amount: document.getElementById("loan").value,
            loan_term_months: document.getElementById("term").value
        })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("pd").innerText = "PD: " + (data.probability_of_default * 100).toFixed(2) + "%";
        document.getElementById("risk").innerText = "Risk: " + data.risk_tier;
        document.getElementById("decision").innerText = "Decision: " + data.decision;
    });
}
