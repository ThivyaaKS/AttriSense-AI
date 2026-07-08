from flask import Flask, render_template, request
import joblib
import pandas as pd
from flask import send_file
from reportlab.pdfgen import canvas
from datetime import datetime

app = Flask(__name__)
print(app.template_folder)

model = joblib.load("models/employee_attrition_model.pkl")
history = []
from flask import redirect, url_for

@app.route("/clear_history")
def clear_history():
    history.clear()
    return redirect(url_for("home"))
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    risk = ""
    age = ""
    income = ""
    experience = ""

    if request.method == "POST":
        age = int(request.form["Age"])
        income = int(request.form["MonthlyIncome"])
        experience = int(request.form["TotalWorkingYears"])

        data = pd.DataFrame([[age, income, experience]],
                            columns=["Age", "MonthlyIncome", "TotalWorkingYears"])

        result = model.predict(data)

        if result[0] == 1:
            prediction = "❌ Employee is likely to Leave"
            risk = "High"
        else:
            prediction = "✅ Employee is likely to Stay"
            risk = "Low"
        
    global report_age, report_income, report_experience, report_prediction, report_risk

    report_age = age
    report_income = income
    report_experience = experience
    report_prediction = prediction
    report_risk = risk  
    history.append({
        "Age": age,
        "Income": income,
        "Experience": experience,
        "Prediction": prediction,
        "DateTime": datetime.now().strftime("%d-%m-%Y %I:%M %p")
    })

    if len(history) > 10:
        history.pop(0) 
    total_predictions = len(history)

    stay_count = sum(
        1 for item in history
        if "Stay" in item["Prediction"]
    )

    leave_count = sum(
        1 for item in history
        if "Leave" in item["Prediction"]
    )
        
    return render_template(
        "index.html",
        prediction=prediction,
        risk=risk,
        age=age,
        income=income,
        experience=experience,
        history=history,
    total_predictions=len(history),
        stay_count=stay_count,
        leave_count=leave_count 
    )

@app.route("/download_report")
def download_report():

    pdf = canvas.Canvas("Employee_Report.pdf")

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(170, 800, "Employee Attrition Report")

    pdf.drawString(50, 770, f"Age : {report_age}")
    pdf.drawString(50, 750, f"Monthly Income : {report_income}")
    pdf.drawString(50, 730, f"Experience : {report_experience}")
    pdf.drawString(50, 710, f"Prediction : {report_prediction}")
    pdf.drawString(50, 690, f"Risk Level : {report_risk}")

    pdf.save()

    return send_file("Employee_Report.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
