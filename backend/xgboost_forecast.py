import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
TARGET_COLUMN = 'Production(kWh)'

def load_data(data_path):
    df = pd.read_csv(data_path, parse_dates=['date'])
    df = df.sort_values('date')
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return df

def train_xgb_model(X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=True
    )
    return model

def evaluate_model(model, X_test, y_test, test_dates, plot=True):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Test RMSE: {rmse:.2f} kWh")
    print(f"R² Score: {r2:.3f}")

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, y_test, label="Actual")
        plt.plot(test_dates, predictions, label="Predicted")
        plt.title("XGBoost - Actual vs Predicted Production")
        plt.xlabel("Date")
        plt.ylabel("Production (kWh)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        residuals = y_test - predictions
        plt.figure(figsize=(12, 4))
        plt.plot(test_dates, residuals, marker='o', linestyle='-')
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Residuals (Actual - Predicted)")
        plt.xlabel("Date")
        plt.ylabel("Error (kWh)")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    return predictions, rmse, r2


def run_training_pipeline(data_path, model_path, plot=True):
    df = load_data(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    dates = df['date']

    X_train, X_test, y_train, y_test, train_dates, test_dates = train_test_split(
        X, y, dates, test_size=0.2, shuffle=False
    )

    model = train_xgb_model(X_train, y_train, X_test, y_test)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    predictions, rmse, r2 = evaluate_model(model, X_test, y_test, test_dates, plot=plot)
    return model, predictions, rmse, r2

if __name__ == "__main__":
    model, predictions, rmse, r2 = run_training_pipeline(
        data_path='output/merged_featured_data.csv',
        model_path='models/xgb_model.pkl'
    )
