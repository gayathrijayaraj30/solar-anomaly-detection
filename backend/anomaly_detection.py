import pandas as pd
import numpy as np
import joblib
import os

FEATURE_COLUMNS = [
    'Self-used Ratio(%)', 'Anticipated Yield(INR)', 'Consumption(kWh)',
    'Grid Feed-in(kWh)', 'Electricity Purchasing(kWh)',
    'temp_max_C', 'temp_min_C', 'precipitation_mm',
    'temp_range', 'rain_flag', 'days_since_rain',
    'day_of_week', 'month', 'day_of_week_sin', 'day_of_week_cos',
    'month_sin', 'month_cos', 'prod_lag_1', 'cons_lag_1',
    'temp_max_lag_1', 'temp_min_lag_1', 'precip_lag_1', 'rain_flag_lag_1',
    'prod_lag_2', 'cons_lag_2', 'temp_max_lag_2', 'temp_min_lag_2',
    'precip_lag_2', 'rain_flag_lag_2', 'prod_lag_3', 'cons_lag_3',
    'temp_max_lag_3', 'temp_min_lag_3', 'precip_lag_3', 'rain_flag_lag_3',
    'prod_lag_7', 'cons_lag_7', 'temp_max_lag_7', 'temp_min_lag_7',
    'precip_lag_7', 'rain_flag_lag_7', 'prod_rollmean_3', 'prod_rollstd_3',
    'tempmax_rollmean_3', 'tempmax_rollstd_3', 'prod_rollmean_7',
    'prod_rollstd_7', 'tempmax_rollmean_7', 'tempmax_rollstd_7'
]

def detect_anomalies(df, model, threshold_std=2.0):
    X = df[FEATURE_COLUMNS]
    y_true = df['Production(kWh)']
    y_pred = model.predict(X)
    residuals = y_true - y_pred

    mean_res = residuals.mean()
    std_res = residuals.std()

    anomaly_mask = abs(residuals) > threshold_std * std_res
    anomalies = df[anomaly_mask].copy()
    anomalies['residual'] = residuals[anomaly_mask]

    return anomalies

def run_anomaly_detection(data_path, model_path, output_path=None, threshold_std=2.0):
    print("Loading data and model...")
    df = pd.read_csv(data_path, parse_dates=['date']).sort_values('date')
    model = joblib.load(model_path)

    print("Detecting anomalies...")
    anomalies = detect_anomalies(df, model, threshold_std=threshold_std)
    print(f"Total anomalies detected: {len(anomalies)}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        anomalies.to_csv(output_path, index=False)
        print(f"Saved anomalies to {output_path}")

    return anomalies

if __name__ == "__main__":
    run_anomaly_detection(
        data_path='output/merged_featured_data.csv',
        model_path='models/xgb_model.pkl',
        output_path='output/anomaly_results.csv'
    )
