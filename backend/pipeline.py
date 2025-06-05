import os
from .data_preprocessing import preprocess
from .xgboost_forecast import run_training_pipeline
from .anomaly_detection import run_anomaly_detection
from .shap_explain import main as shap_explanation_main
from .fetch_weather import fetch_weather_data

def full_pipeline(
    inverter_path,
    plant_path,
    weather_start,
    weather_end,
    weather_path,
    featured_output_path='output/merged_featured_data.csv',
    model_output_path='models/xgb_model.pkl',
    anomaly_output_path='output/anomaly_results.csv'
):
    print("\n--- Step 1: Fetching Weather Data ---")
    weather_df = fetch_weather_data(weather_start, weather_end)
    os.makedirs(os.path.dirname(weather_path), exist_ok=True)
    weather_df.to_csv(weather_path, index=False)

    print("\n--- Step 2: Preprocessing ---")
    df = preprocess(inverter_path, plant_path, weather_path)
    os.makedirs(os.path.dirname(featured_output_path), exist_ok=True)
    df.to_csv(featured_output_path, index=False)

    print("\n--- Step 3: Training Model ---")
    run_training_pipeline(data_path=featured_output_path, model_path=model_output_path)

    print("\n--- Step 4: Anomaly Detection ---")
    run_anomaly_detection(data_path=featured_output_path, model_path=model_output_path, output_path=anomaly_output_path)

    print("\n--- Step 5: SHAP Explanation ---")
    shap_explanation_main()

    print("\n✅ Full pipeline executed successfully.")
