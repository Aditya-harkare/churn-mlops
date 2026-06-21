import requests

test_customer = {
    "tenure": 2, "MonthlyCharges": 85.0, "TotalCharges": 170.0,
    "gender": 1, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
    "PhoneService": 1, "PaperlessBilling": 1,
    "MultipleLines_No_phone_service": 0, "MultipleLines_Yes": 1,
    "InternetService_Fiber_optic": 1, "InternetService_No": 0,
    "OnlineSecurity_No_internet_service": 0, "OnlineSecurity_Yes": 0,
    "OnlineBackup_No_internet_service": 0, "OnlineBackup_Yes": 0,
    "DeviceProtection_No_internet_service": 0, "DeviceProtection_Yes": 0,
    "TechSupport_No_internet_service": 0, "TechSupport_Yes": 0,
    "StreamingTV_No_internet_service": 0, "StreamingTV_Yes": 1,
    "StreamingMovies_No_internet_service": 0, "StreamingMovies_Yes": 1,
    "Contract_One_year": 0, "Contract_Two_year": 0,
    "PaymentMethod_Credit_card_automatic": 0,
    "PaymentMethod_Electronic_check": 1,
    "PaymentMethod_Mailed_check": 0
}

response = requests.post("http://127.0.0.1:8000/predict", json=test_customer)
print(response.json())