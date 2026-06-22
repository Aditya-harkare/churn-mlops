import mlflow
import joblib
import os

# Load the Production model from MLflow
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath('mlflow.db')}")
model = mlflow.sklearn.load_model("models:/RandomForest@production")

# Save it as a plain file that doesn't need MLflow to load
os.makedirs("model_store", exist_ok=True)
joblib.dump(model, "model_store/churn_model.pkl")
print("Model exported to model_store/churn_model.pkl")