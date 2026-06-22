import joblib
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Load model from exported file instead of MLflow registry.
# joblib is Python's standard serialization library for ML objects.
# This removes the dependency on MLflow at serving time — the container
# only needs the model file itself, not the entire tracking infrastructure.
# MODEL_PATH checks an environment variable first, falls back to default.
# This lets us override the path easily in different environments
# (local, Docker, cloud) without changing code.

MODEL_PATH = os.getenv("MODEL_PATH", "model_store/churn_model.pkl")

print(f"Loading model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise


FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

class CustomerData(BaseModel):

    tenure: int = Field(ge=0, le=72, description="Months as customer (0-72)")
    MonthlyCharges: float = Field(ge=0, le=200, description="Monthly bill amount")
    TotalCharges: float = Field(ge=0, description="Total amount paid")

    gender: int = Field(ge=0, le=1, description="1=Male, 0=Female")
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: int = Field(ge=0, le=1)
    Dependents: int = Field(ge=0, le=1)
    PhoneService: int = Field(ge=0, le=1)
    PaperlessBilling: int = Field(ge=0, le=1)

    MultipleLines_No_phone_service: int = Field(ge=0, le=1)
    MultipleLines_Yes: int = Field(ge=0, le=1)
    InternetService_Fiber_optic: int = Field(ge=0, le=1)
    InternetService_No: int = Field(ge=0, le=1)
    OnlineSecurity_No_internet_service: int = Field(ge=0, le=1)
    OnlineSecurity_Yes: int = Field(ge=0, le=1)
    OnlineBackup_No_internet_service: int = Field(ge=0, le=1)
    OnlineBackup_Yes: int = Field(ge=0, le=1)
    DeviceProtection_No_internet_service: int = Field(ge=0, le=1)
    DeviceProtection_Yes: int = Field(ge=0, le=1)
    TechSupport_No_internet_service: int = Field(ge=0, le=1)
    TechSupport_Yes: int = Field(ge=0, le=1)
    StreamingTV_No_internet_service: int = Field(ge=0, le=1)
    StreamingTV_Yes: int = Field(ge=0, le=1)
    StreamingMovies_No_internet_service: int = Field(ge=0, le=1)
    StreamingMovies_Yes: int = Field(ge=0, le=1)
    Contract_One_year: int = Field(ge=0, le=1)
    Contract_Two_year: int = Field(ge=0, le=1)
    PaymentMethod_Credit_card_automatic: int = Field(ge=0, le=1)
    PaymentMethod_Electronic_check: int = Field(ge=0, le=1)
    PaymentMethod_Mailed_check: int = Field(ge=0, le=1)

    @field_validator('TotalCharges')
    @classmethod
    def total_must_exceed_monthly(cls, v, info):
        if 'MonthlyCharges' in info.data and v < info.data['MonthlyCharges']:
            raise ValueError(
                "TotalCharges cannot be less than MonthlyCharges"
            )
        return v
    
    

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts telecom customer churn probability",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return{
        "status": "healthy",
        "model": "RandomForest",
        "stage": "Production"
    }

@app.post("/predict")
def predict(customer: CustomerData):

    try:

        data = {
            'gender': customer.gender,
            'SeniorCitizen': customer.SeniorCitizen,
            'Partner': customer.Partner,
            'Dependents': customer.Dependents,
            'tenure': customer.tenure,
            'PhoneService': customer.PhoneService,
            'PaperlessBilling': customer.PaperlessBilling,
            'MonthlyCharges': customer.MonthlyCharges,
            'TotalCharges': customer.TotalCharges,
            'MultipleLines_No phone service': customer.MultipleLines_No_phone_service,
            'MultipleLines_Yes': customer.MultipleLines_Yes,
            'InternetService_Fiber optic': customer.InternetService_Fiber_optic,
            'InternetService_No': customer.InternetService_No,
            'OnlineSecurity_No internet service': customer.OnlineSecurity_No_internet_service,
            'OnlineSecurity_Yes': customer.OnlineSecurity_Yes,
            'OnlineBackup_No internet service': customer.OnlineBackup_No_internet_service,
            'OnlineBackup_Yes': customer.OnlineBackup_Yes,
            'DeviceProtection_No internet service': customer.DeviceProtection_No_internet_service,
            'DeviceProtection_Yes': customer.DeviceProtection_Yes,
            'TechSupport_No internet service': customer.TechSupport_No_internet_service,
            'TechSupport_Yes': customer.TechSupport_Yes,
            'StreamingTV_No internet service': customer.StreamingTV_No_internet_service,
            'StreamingTV_Yes': customer.StreamingTV_Yes,
            'StreamingMovies_No internet service': customer.StreamingMovies_No_internet_service,
            'StreamingMovies_Yes': customer.StreamingMovies_Yes,
            'Contract_One year': customer.Contract_One_year,
            'Contract_Two year': customer.Contract_Two_year,
            'PaymentMethod_Credit card (automatic)': customer.PaymentMethod_Credit_card_automatic,
            'PaymentMethod_Electronic check': customer.PaymentMethod_Electronic_check,
            'PaymentMethod_Mailed check': customer.PaymentMethod_Mailed_check,
        }

        input_df = pd.DataFrame([data])[FEATURE_COLUMNS]

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        if probability < 0.3:
            risk_level = "Low"

        elif probability < 0.6:
            risk_level = "Medium"

        else:
            risk_level = "High"

        return{
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "message": "Will churn" if prediction == 1 else "Will not churn"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)