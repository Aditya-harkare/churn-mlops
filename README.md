# Customer Churn Prediction — End-to-End MLOps Pipeline

An end-to-end MLOps system that takes raw telecom customer data through data validation, preprocessing, experiment tracking, model serving, and production monitoring. Built as a hands-on implementation of MLOps concepts from DeepLearning.AI's *Machine Learning Engineering for Production* course.

---

## Architecture

```
Raw Data (CSV)
      │
      ▼
┌─────────────────┐
│  Data Validation │  ── schema checks, null detection, target validation
└────────┬─────────┘
         ▼
┌─────────────────┐
│   Preprocessing  │  ── type fixing, encoding, DVC versioning
└────────┬─────────┘
         ▼
┌─────────────────┐
│ Model Training   │  ── scikit-learn Pipeline (scaler + classifier)
│ + MLflow Tracking│  ── 3 models compared, best promoted to Production
└────────┬─────────┘
         ▼
┌─────────────────┐
│   FastAPI Server │  ── Pydantic validation, loads Production model
└────────┬─────────┘
         ▼
┌─────────────────┐
│ Drift Monitoring │  ── Evidently AI, reference vs. production comparison
└──────────────────┘
```

---

## Tech Stack

| Purpose | Tool |
|---|---|
| Data versioning | DVC |
| Experiment tracking & model registry | MLflow |
| Modeling | scikit-learn, XGBoost |
| API serving | FastAPI + Pydantic |
| Drift monitoring | Evidently AI |
| Language | Python 3.13 |

---

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 raw columns, binary churn label (26.5% churn rate).

---

## Key Design Decisions

**Why a scikit-learn `Pipeline` instead of manual scaling?**
Manually fitting a scaler and applying it separately to train/test data is a common source of *data leakage* — accidentally letting test-set statistics influence training. Wrapping `ColumnTransformer` (scaling) and the classifier into a single `Pipeline` makes leakage structurally impossible: the scaler is only ever fit on training folds, and the same fitted pipeline is later saved and served as one artifact — no manual preprocessing step to forget at inference time.

**Why optimize for F1, not recall, despite recall mattering more to the business?**
Optimizing purely for recall lets a model "cheat" by predicting churn for everyone, which destroys precision and wastes business resources on false retention offers. F1 forces a balance between catching churners and not crying wolf, while recall is still tracked and reported separately during evaluation since it carries more business weight than precision in this context.

**Why `class_weight='balanced'` / `scale_pos_weight`?**
With a 73.5% / 26.5% class split, an unweighted model is biased toward predicting "no churn." Reweighting penalizes mistakes on the minority (churn) class more heavily during training, without altering the underlying data via oversampling/undersampling.

**Why load the model via MLflow's `@production` alias instead of a hardcoded version number?**
This decouples *serving* from *training*. Promoting a newly trained model to production only requires updating an alias in the MLflow registry — no code changes or redeployment of the API logic itself.

**Why drop `customerID` before training?**
A unique identifier carries no generalizable signal and risks the model latching onto ID-specific patterns (data leakage via memorization) rather than learning real churn drivers.

---

## Model Results

Three models were trained and compared via MLflow, all sharing the same preprocessing pipeline and evaluation protocol (80/20 stratified split):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 73.81% | 50.43% | 78.34% | 61.36% | 84.18% |
| **Random Forest (Production)** | **76.58%** | **54.38%** | 73.00% | **62.33%** | **84.28%** |
| XGBoost (GridSearchCV-tuned, 5-fold CV) | 73.81% | 50.43% | 78.07% | 61.28% | 84.20% |

**Random Forest was promoted to production** — it had the best F1 and ROC-AUC of the three. XGBoost was tuned via `GridSearchCV` (`max_depth`, `n_estimators`, `scale_pos_weight`; 5-fold stratified CV, optimized for F1) but did not surpass Random Forest on this dataset — a result that was logged and kept rather than discarded, since negative experiment results are still useful project history.

---

## Honest Limitations

This project prioritizes pipeline correctness and reproducibility over squeezing out the last percentage points of model performance. Specifically:

- **No nested cross-validation on the final reported metrics** — the 80/20 test split is single-shot; CV was only used inside the XGBoost hyperparameter search.
- **No feature engineering beyond raw encoding** — no derived features (e.g., charges-per-tenure ratio) were tested.
- **Drift monitoring was validated against simulated drift**, not real production traffic, since the API has not yet been exposed to live users.
- **The XGBoost hyperparameter grid was intentionally small** (18 combinations) for tractability; a wider search (including `learning_rate`, `subsample`) was not exhausted.

These are documented deliberately rather than hidden — identifying and stating a model's limitations is treated here as part of the deliverable, not a weakness to omit.

---

## Project Structure

```
churn-mlops/
├── data/
│   ├── raw/                  # original dataset (DVC-tracked)
│   └── processed/            # cleaned, encoded dataset (DVC-tracked)
├── src/
│   ├── data/
│   │   ├── data_validation.py
│   │   └── data_preprocessing.py
│   ├── models/
│   │   ├── train.py          # baseline LR + RF training, MLflow logging
│   │   └── tune_model.py     # XGBoost + GridSearchCV
│   └── serving/
│       └── api.py            # FastAPI inference service
├── monitoring/
│   ├── monitor.py            # Evidently AI drift detection + alerting
│   └── reports/               # generated HTML drift reports
├── mlflow.db                  # MLflow tracking store (SQLite)
└── README.md
```

---

## Running the Project

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Validate and preprocess data
python src\data\data_validation.py
python src\data\data_preprocessing.py

# 3. Train and track experiments
python src\models\train.py
python src\models\tune_model.py
mlflow ui   # view at http://127.0.0.1:5000

# 4. Promote the best model to the "production" alias in MLflow UI, then serve it
python src\serving\api.py   # API docs at http://127.0.0.1:8000/docs

# 5. Run drift monitoring
python monitoring\monitor.py
```

---

## Next Steps

- Containerize the API with Docker for portable deployment
- Add a CI/CD workflow (GitHub Actions) to automate validation and retraining
- Deploy the API to a cloud provider for public accessibility
- Log real API prediction traffic to replace simulated drift data with genuine production data
