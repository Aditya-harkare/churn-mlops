import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def validate_data(df: pd.DataFrame) -> bool:
    errors = []

    required_cols = [
        'customerID','gender','tenure','MonthlyCharges','TotalCharges','Churn'
    ]
    missing = [c for c in required_cols if c not in df.columns]

    nulls = df[required_cols].isnull().sum()
    if nulls.any():
        errors.append(f"Null values found:\n{nulls[nulls>0]}")

    if not set(df['Churn'].unique()).issubset({'Yes','No'}):
        errors.append(f"Unexpected Churn values: {df['Churn'].unique()}")

    if errors:
        for e in errors: print(e)
        return False
    return True

def basic_eda(df: pd.DataFrame):
    print("\n--- Basic EDA ---")
    print(f"Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
    print(f"\nNumerical summary:\n{df.describe()}")

if __name__ == "__main__":
    RAW_PATH = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = load_data(RAW_PATH)
    validate_data(df)
    basic_eda(df)        


