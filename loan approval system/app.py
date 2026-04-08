from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("best_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = pd.DataFrame([{
        "ApplicantIncome": float(request.form.get("ApplicantIncome", 0)),
        "CoapplicantIncome": float(request.form.get("CoapplicantIncome", 0)),
        "LoanAmount": float(request.form.get("LoanAmount", 0)),
        "Loan_Amount_Term": float(request.form.get("Loan_Amount_Term", 0)),
        "Credit_Score": float(request.form.get("Credit_Score", 0)),
        "Credit_History": float(request.form.get("Credit_History", 0)),
        "Age": float(request.form.get("Age", 0)),
        "Education": request.form.get("Education", ""),
        "Employment_Status": request.form.get("Employment_Status", ""),
        "Property_Area": request.form.get("Property_Area", "")
    }])

    prediction = model.predict(data)

    result = "Loan Approved ✅" if prediction[0] == 1 else "Loan Rejected ❌"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)