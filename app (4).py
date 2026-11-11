"""Streamlit app: predict churn risk for one customer.  Run: streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
bundle = joblib.load("outputs/churn_model.joblib")
model, threshold = bundle["model"], bundle["threshold"]

st.title("📉 Customer Churn Predictor")
st.caption("Enter customer details to estimate churn risk.")

c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
with c2:
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

total = monthly * max(tenure, 1)
row = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
    "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
    "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly,
    "TotalCharges": total,
}])
row["AvgMonthlySpend"] = row["TotalCharges"] / row["tenure"].clip(lower=1)
row["NumServices"] = (row[["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                           "TechSupport", "StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)
row["TenureGroup"] = pd.cut(row["tenure"], bins=[-1, 12, 24, 48, 72],
                            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]).astype(str)

if st.button("Predict churn risk", type="primary"):
    p = model.predict_proba(row)[0, 1]
    st.metric("Churn probability", f"{p:.1%}")
    st.progress(float(p))
    if p >= threshold:
        st.error("⚠️ High risk: recommend a retention offer (e.g. contract upgrade discount).")
    else:
        st.success("✅ Low risk: customer likely to stay.")
