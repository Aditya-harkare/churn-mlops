import pandas as pd
import os

#___Loading Reference Data

def load_reference_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop('Churn', axis=1)
    print(f"Reference data loaded: {df.shape}")
    print(f"Reference tenure mean: {df['tenure'].mean():.2f}")
    print(f"Reference MonthlyCharges mean: {df['MonthlyCharges'].mean():.2f}")
    return df



import numpy as np

#__Simulating current production data

def simulate_production_data(reference_df: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:

    rng = np.random.default_rng(seed = 42)

    production = {}

    production['tenure'] = rng.normal(loc=8, scale=5, size=n_samples).clip(0,72).astype(int)
    production['MonthlyCharges'] = rng.normal(loc=90, scale=15, size=n_samples).clip(18,200)
    production['TotalCharges'] = (production['tenure']*production['MonthlyCharges'])

    binary_cols = [c for c in reference_df.columns if c not in 
                   ['tenure', 'MonthlyCharges', 'TotalCharges']]

    for col in binary_cols:
        production[col] = rng.choice(
           reference_df[col].values,
           size=n_samples,
           replace=True 
        )

    production_df = pd.DataFrame(production)[reference_df.columns]

    print(f"\nProduction data simulated: {production_df.shape}")
    print(f"Production tenure mean: {production_df['tenure'].mean():.2f}")
    print(f"Production MonthlyCharges mean:"
          f"{production_df['MonthlyCharges'].mean():.2f}")
    
    return production_df




from evidently import Report
from evidently.presets import DataDriftPreset

def generate_drift_report(reference_df: pd.DataFrame,
                           production_df: pd.DataFrame,
                           report_path: str) -> dict:
    
    report = Report(metrics=[
        DataDriftPreset()
    ])

    snapshot = report.run(reference_data=reference_df, current_data=production_df)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    snapshot.save_html(report_path)
    print(f"\nDrift report saved to: {report_path}")

    result = snapshot.dict()

    return result

import json
#__ Alerting 

def check_drift_and_alert(drift_result: dict, drift_threshold: float = 0.5):

    metrics = drift_result.get('metrics', [])

    drifted_count = 0
    drifted_share = 0.0
    drifted_features = []

    for entry in metrics:
        if 'ValueDrift' not in entry['metric_name']:
            continue
        value = entry['value']
        feature_threshold = entry['config']['threshold']
        try:
            drift_score = float(value)
        except (TypeError, ValueError):
            drift_score = 0.0
        if drift_score > feature_threshold:
            drifted_count += 1

    drifted_share = drifted_count / 30

    print(f"\n{'='*55}")
    print(f"DRIFT MONITORING REPORT")
    print(f"{'='*55}")
    print(f"Total features:   30")
    print(f"Drifted features: {int(drifted_count)}")
    print(f"Drift share:      {drifted_share:.1%}")

    if drifted_share > drift_threshold:
        print(f"\n[CRITICAL ALERT] {drifted_share:.1%} of features drifted")
        print(f"  Exceeds threshold of {drift_threshold:.1%}")
        print(f"  ACTION REQUIRED: Retrain the model on fresh data")
    else:
        print(f"\n[OK] Drift share {drifted_share:.1%} is within")
        print(f"  threshold of {drift_threshold:.1%} - Continue monitoring")

    print(f"\n{'-'*55}")
    print(f"PER-FEATURE DRIFT SCORES")
    print(f"{'-'*55}")
    print(f"{'Feature':<45} {'Score':>8}  {'Status'}")
    print(f"{'-'*55}")

    for entry in metrics:
        if 'ValueDrift' not in entry['metric_name']:
            continue

        col_name = entry['config']['column']
        feature_threshold = entry['config']['threshold']

        # Handle both float and dict value structures
        value = entry['value']
        if isinstance(value, dict):
            drift_score = value.get('drift_score', value.get('value', 0.0))
        else:
            try:
                drift_score = float(value)
            except (TypeError, ValueError):
                drift_score = 0.0

        is_drifted = drift_score > feature_threshold
        # ASCII status instead of emoji — avoids Windows encoding issues
        status = "[DRIFTED]" if is_drifted else "[OK]"

        print(f"{col_name:<45} {drift_score:>8.4f}  {status}")

        if is_drifted:
            drifted_features.append({
                'feature': col_name,
                'score': drift_score,
                'threshold': feature_threshold
            })

    # ── Drifted features summary ──
    if drifted_features:
        print(f"\n{'-'*55}")
        print("DRIFTED FEATURES SUMMARY")
        print(f"{'-'*55}")
        for f in drifted_features:
            severity = 'HIGH' if f['score'] > 0.5 else 'MEDIUM'
            print(f"  Feature:   {f['feature']}")
            print(f"  Score:     {f['score']:.4f}")
            print(f"  Threshold: {f['threshold']:.4f}")
            print(f"  Severity:  {severity}")
            print()

    return drifted_features
            



if __name__ == "__main__":
    PROCESSED_PATH = os.path.join("data", "processed", "churn_processed.csv")
    REPORT_PATH = os.path.join("monitoring", "reports", "drift_report.html")
    reference_df = load_reference_data(PROCESSED_PATH)
    production_df = simulate_production_data(reference_df)
    drift_result = generate_drift_report(reference_df, production_df, REPORT_PATH)
    drifted_features = check_drift_and_alert(
        drift_result, drift_threshold=0.5
    )

print(f"\nMonitoring complete")
print(f"Full report: {os.path.abspath(REPORT_PATH)}")
    