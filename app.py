# Import necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pandas as pd
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = "final_model.joblib"
DATA_PATH = "your_dataset.csv"  # Replace with your actual dataset file

# Train and Save Model Function
def train_and_save_model():
    try:
        # Load dataset
        logger.info("Loading dataset...")
        data = pd.read_csv(DATA_PATH)
        X = data.drop(columns=["target"])  # Replace 'target' with the name of your target column
        y = data["target"]

        # Split the data
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        logger.info("Training the XGBoost model...")
        model = XGBClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.2f}")

        # Save the trained model
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Model saved at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error training or saving the model: {e}")
        raise RuntimeError(f"Failed to train and save model: {e}")

# Train and save the model if it does not already exist
try:
    logger.info("Checking for existing model...")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.warning("Model not found or failed to load. Training a new model...")
    train_and_save_model()
    model = joblib.load(MODEL_PATH)

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

        logger.info(f"Prediction: {loan_status}, Probability: {probability:.2f}")
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
