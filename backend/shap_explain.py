import pandas as pd
import numpy as np
import shap
import joblib
import os
import matplotlib.pyplot as plt

DATA_PATH = 'output/merged_featured_data.csv'
MODEL_PATH = 'models/xgb_model.pkl'
ANOMALY_PATH = 'output/anomaly_results.csv'
OUTPUT_PATH = 'output/anomaly_results_with_shap.csv'
SHAP_SAVE_PATH = 'output/shap_anomaly_values.npy'
SHAP_CSV_PATH = 'output/shap_values_per_anomaly.csv'

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

def load_data():
    df_all = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df_anomaly = pd.read_csv(ANOMALY_PATH, parse_dates=['date'])
    df_all = df_all.sort_values('date')
    df_anomaly = df_anomaly.sort_values('date')
    
    merged = pd.merge(df_anomaly, df_all, on='date', suffixes=('', '_full'))
    
    X_anomaly = merged[FEATURE_COLUMNS]
    return merged, X_anomaly

def explain_shap(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap_values.values

def main():
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    merged_df, X_anomaly = load_data()

    feature_names = model.get_booster().feature_names if hasattr(model, 'get_booster') else FEATURE_COLUMNS
    print(f"Feature names from model: {feature_names}")

    shap_vals = explain_shap(model, X_anomaly)
    print(f"SHAP values shape: {shap_vals.shape}")
    print(f"Number of features: {len(feature_names)}")

    os.makedirs(os.path.dirname(SHAP_SAVE_PATH), exist_ok=True)
    np.save(SHAP_SAVE_PATH, shap_vals)

    merged_df.to_csv(OUTPUT_PATH, index=False)

    shap_df_all = pd.DataFrame(shap_vals, columns=feature_names)

    shap_df_all['date'] = merged_df['date'].values

    shap_df_all.to_csv(SHAP_CSV_PATH, index=False)

    print(f"Saved SHAP values to {SHAP_SAVE_PATH}")
    print(f"Saved merged anomaly data to {OUTPUT_PATH}")
    print(f"Saved SHAP values per anomaly row to {SHAP_CSV_PATH}")

    idx = 0
    sample_shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_vals[idx]
    })
    print(f"Sample SHAP DataFrame for idx={idx}:\n{sample_shap_df.head()}")

    shap.summary_plot(shap_vals, X_anomaly)
    plt.show()

if __name__ == "__main__":
    main()
