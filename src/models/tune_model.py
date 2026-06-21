import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier


# ─────────────────────────────────────────
# STEP 1 — Load and split data (same as before)
# ─────────────────────────────────────────

def load_and_split_data(path: str):
    df = pd.read_csv(path)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# STEP 2 — Build the pipeline (same pattern as train.py)
# ─────────────────────────────────────────

def build_pipeline(continuous_cols: list) -> Pipeline:
    """
    Same Pipeline pattern as before — scaling + classifier.
    We don't fix the classifier's hyperparameters here.
    GridSearchCV will inject different values during search.

    Note: we use XGBClassifier with placeholder defaults.
    GridSearchCV will override these during the search process.
    """
    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), continuous_cols)],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            random_state=42,
            eval_metric='logloss'   # required by XGBoost to suppress a warning
        ))
    ])

    return pipeline


# ─────────────────────────────────────────
# STEP 3 — Run GridSearchCV
# ─────────────────────────────────────────

def run_grid_search(pipeline, X_train, y_train):
    """
    param_grid:
      Keys use the 'classifier__paramname' format explained above.
      We keep this grid intentionally SMALL at first — exactly
      the lesson from your Question 1 answer. 3 x 3 x 2 = 18
      hyperparameter combinations.

    max_depth: [3, 5, 7]
      Controls how complex each tree can get. Shallow trees (3)
      underfit, deep trees (7+) risk overfitting. We test a range.

    n_estimators: [50, 100, 200]
      Number of sequential trees XGBoost builds. More trees can
      improve performance but increase training time and overfitting risk.

    scale_pos_weight: [1, 2.77]
      1 = no class imbalance correction (baseline)
      2.77 = the exact ratio we calculated together (5174/1869)
      We test both to see if imbalance correction actually helps
      THIS specific dataset and algorithm combination.

    StratifiedKFold(n_splits=5):
      Same concept discussed earlier — 5-fold cross-validation.
      'Stratified' ensures each fold maintains the original 26.54%
      churn ratio, just like we did with train_test_split's stratify=y.
      Without stratification, some folds could randomly end up
      with very few churn examples, making scores unreliable.

    scoring='f1':
      As we agreed — optimizing for F1 prevents the model from
      gaming recall by over-predicting churn for everyone.

    n_jobs=-1:
      Use ALL available CPU cores to train models in parallel.
      Since we're training 18 x 5 = 90 models total, parallelization
      meaningfully speeds this up.

    verbose=2:
      Prints progress as each combination is tested — useful
      feedback since this step takes a few minutes.
    """
    param_grid = {
        'classifier__max_depth': [3, 5, 7],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__scale_pos_weight': [1, 2.77]
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=cv_strategy,
        n_jobs=-1,
        verbose=2
    )

    print("\nStarting grid search... this may take a few minutes")
    grid_search.fit(X_train, y_train)

    print(f"\nBest F1 score (cross-validated): {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")

    return grid_search


# ─────────────────────────────────────────
# STEP 4 — Evaluate the best model on test set
# ─────────────────────────────────────────

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Same evaluation function as train.py — kept identical so
    results are directly comparable to our earlier models.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob)
    }


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    PROCESSED_PATH = os.path.join("data", "processed", "churn_processed.csv")
    continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Load data
    X_train, X_test, y_train, y_test = load_and_split_data(PROCESSED_PATH)

    # Build pipeline and run grid search
    pipeline = build_pipeline(continuous_cols)
    grid_search = run_grid_search(pipeline, X_train, y_train)

    # grid_search.best_estimator_ is the pipeline already refit
    # on the FULL training set using the best hyperparameters found
    best_pipeline = grid_search.best_estimator_

    # Evaluate on held-out test set (data the search never touched)
    metrics = evaluate_model(best_pipeline, X_test, y_test)

    print(f"\n── XGBoost (Tuned) — Test Set Results ──")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # ── Log to MLflow ──
    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath('mlflow.db')}")
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name="XGBoost_Tuned"):
        # Log the best hyperparameters found by grid search
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")

        # Log both CV score and test set metrics
        mlflow.log_metric("cv_best_f1", grid_search.best_score_)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            best_pipeline,
            name="pipeline",
            registered_model_name="XGBoost"
        )

    print("\nLogged to MLflow. Run 'mlflow ui' to compare with previous models.")