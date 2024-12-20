from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model_path = "final_model.joblib"
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load the model: {e}")
    raise

# Define FastAPI app
app = FastAPI(title="Loan Default Prediction API", description="Predict loan default status using a trained model.", version="1.0")

# Define input schema
class LoanApplication(BaseModel):
    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: float
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: float
    previous_loan_defaults_on_file: str

# Define output schema
class Prediction(BaseModel):
    loan_status: str
    probability: float

@app.post("/predict", response_model=Prediction)
async def predict(application: LoanApplication):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([application.dict()])

        # Make predictions
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0].max()

        # Map prediction to label (if necessary)
        loan_status = "Approved" if prediction == 1 else "Denied"

        return Prediction(loan_status=loan_status, probability=probability)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Loan Default Prediction API!"}