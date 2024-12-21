from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
model_path = "final_model.joblib"
try:
    model = joblib.load(model_path)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not have necessary methods (predict/predict_proba).")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load the model: {e}")
    raise RuntimeError(f"Model loading failed: {e}")

# Define FastAPI app
app = FastAPI(
    title="Loan Default Prediction API",
    description="Predict loan default status using a trained model.",
    version="1.0",
)

# Define input schema
class LoanApplication(BaseModel):
    person_age: float = Field(..., example=35)
    person_gender: str = Field(..., example="male")
    person_education: str = Field(..., example="bachelors")
    person_income: float = Field(..., example=50000)
    person_emp_exp: float = Field(..., example=5)
    person_home_ownership: str = Field(..., example="rent")
    loan_amnt: float = Field(..., example=15000)
    loan_intent: str = Field(..., example="education")
    loan_int_rate: float = Field(..., example=12.5)
    loan_percent_income: float = Field(..., example=0.3)
    cb_person_cred_hist_length: float = Field(..., example=10)
    credit_score: float = Field(..., example=700)
    previous_loan_defaults_on_file: str = Field(..., example="no")

# Define output schema
class Prediction(BaseModel):
    loan_status: str
    probability: float

@app.post("/predict", response_model=Prediction)
async def predict(application: LoanApplication):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([application.dict()])
        logger.info(f"Input data received: {input_data}")

        # Make predictions
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0].max()

        # Map prediction to label
        loan_status = "Approved" if prediction == 1 else "Denied"

        logger.info(f"Prediction: {loan_status}, Probability: {probability}")
        return Prediction(loan_status=loan_status, probability=probability)
    except ValueError as ve:
        logger.error(f"Value error during prediction: {ve}")
        raise HTTPException(status_code=400, detail=f"Value error: {str(ve)}")
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Loan Default Prediction API!"}
