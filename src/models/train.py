import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

def load_processed_data(path: str):
    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop('Churn', axis = 1)
    y = df['Churn']
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    return X, y

def split_data(X,y, test_size = 0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
        )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def build_pipeline(model, continuous_cols: list) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), continuous_cols)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),('classifier',model)
    ])

    return pipeline


def evaluate_model(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return{
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

def train_and_log(pipeline, model_name: str, params: dict, X_train, X_test, y_train, y_test):
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name=model_name):

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline, name="pipeline",
            registered_model_name=model_name
        )

        print(f"\n-- {model_name} --")
        for metric_name, value in metrics.items():
            print(f" {metric_name}: {value:.4f}")

    return metrics

if __name__ == "__main__":
    PROCESSED_PATH = os.path.join("data", "processed", "churn_processed.csv")

    X, y = load_processed_data(PROCESSED_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    continuous_cols = ['tenure','MonthlyCharges','TotalCharges']

    #Experiment1: Logistic Regression
    lr_params = {
        "models" : "LogisticRegression",
        "max_iter" : 1000,
        "class_weight" : "balanced",
        "scaling": "StandardScaler on continuous cols",
        "test_size" : 0.2,
        "random_state": 42
    }

    lr_pipeline = build_pipeline(
        LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        continuous_cols
    )
    train_and_log(lr_pipeline, "LogisticRegression", lr_params, X_train, X_test, y_train, y_test)

    #Experiment2: Random Forest

    rf_params = {
        "model": "RandomForest",
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "scaling": "StandardScaler on continuous cols",
        "test_size": 0.2,
        "random_state": 42
    }

    rf_pipeline = build_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ),
        continuous_cols
    )
    train_and_log(rf_pipeline, "RandomForest", rf_params, X_train, X_test, y_train, y_test)

    print("\nAll experiments logged. Run 'mlflow ui' to view results.")
