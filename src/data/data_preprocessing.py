import pandas as pd
import numpy as np
import os

def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
    n_nulls = df['TotalCharges'].isnull().sum()
    print(f"TotalCharges: found {n_nulls} hidden nulls after conversion")

    df.loc[df['TotalCharges'].isnull(), 'TotalCharges'] = df['MonthlyCharges']
    return df

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    print(f"Churn encoding: {df['Churn'].value_counts().to_dict()}")
    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['customerID'])
    print(f"Dropped customerID. Shape now: {df.shape}")
    return df

def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    binary_cols = {
        'gender' : {'Male' : 1, 'Female' : 0},
        'Partner' : {'Yes' : 1, 'No':0},
        'Dependents' : {'Yes' : 1, 'No' : 0},
        'PhoneService' : {'Yes':1, 'No':0},
        'PaperlessBilling' : {'Yes':1, 'No':0},
    }

    for col, mapping in binary_cols.items():
        df[col] = df[col].map(mapping)
        print(f" Encoded {col}")

    return df

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    print(f"After one-hot encoding, Shape: {df.shape}")
    return df

def save_processed_data(df: pd.DataFrame, path: str):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")

if __name__ == "__main__":
    RAW_PATH = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    PROCESSED_PATH = os.path.join("data", "processed", "churn_processed.csv")
    
    # Load
    df = pd.read_csv(RAW_PATH)
    print(f"Original shape: {df.shape}")
    
    df = fix_total_charges(df)
    df = drop_unnecessary_columns(df)
    df = encode_target(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)

    save_processed_data(df, PROCESSED_PATH)

    print(f"\nFinal shape: {df.shape}")
    print(f"Final columns:\n{list(df.columns)}")
    print(f"\nSample (first 2 rows):\n{df.head(2)}")
