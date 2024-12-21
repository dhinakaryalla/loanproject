import streamlit as st
import requests

# Set the FastAPI endpoint
API_URL = "https://loanapproval-app-ja5c.onrender.com/predict"

# Streamlit App Title
st.title("Loan Approval Prediction")

# Form for user input
st.header("Enter Applicant Details")

# Input fields
person_age = st.number_input("Age", min_value=18, max_value=100, step=1)
person_gender = st.selectbox("Gender", ["male", "female"])
person_education = st.selectbox("Education", ["highschool", "bachelors", "masters", "phd"])
person_income = st.number_input("Income ($)", min_value=0, step=1000)
person_emp_exp = st.number_input("Years of Employment", min_value=0, step=1)
person_home_ownership = st.selectbox("Home Ownership", ["own", "rent", "mortgage", "other"])
loan_amnt = st.number_input("Loan Amount ($)", min_value=0, step=1000)
loan_intent = st.selectbox("Loan Intent", ["personal", "education", "medical", "business", "homeimprovement"])
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=1.0, step=0.01)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, step=1)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["yes", "no"])

# Submit button
if st.button("Predict Loan Status"):
    # Create payload
    payload = {
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
    }

    # Make a POST request to the FastAPI app
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Loan Status: {result['loan_status']}")
            st.info(f"Approval Probability: {result['probability']:.2f}")
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
