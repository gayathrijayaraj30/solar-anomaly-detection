# Solar Production Anomaly Dashboard

A comprehensive Python-based solution for detecting and explaining anomalies in solar power production using advanced machine learning and explainability techniques. This project integrates data preprocessing, weather data fetching, XGBoost forecasting, anomaly detection, and SHAP-based interpretability into a cohesive pipeline, alongside an interactive dashboard for visual exploration.

---

## Project Overview

Solar energy production is inherently variable and subject to external factors such as weather changes and equipment issues. This project addresses the critical need to monitor and detect anomalies in solar production data, enabling timely intervention and improved forecasting accuracy.

**Key highlights:**

* **Robust Data Processing:** Cleans, merges, and engineers features from solar inverter and plant datasets combined with weather data.
* **Forecasting Model:** Implements an XGBoost model to forecast solar production and consumption.
* **Anomaly Detection:** Identifies deviations in production patterns indicating potential issues.
* **Explainability:** Uses SHAP (SHapley Additive exPlanations) to interpret model predictions and pinpoint feature contributions driving anomalies.
* **Interactive Dashboard:** (If applicable, mention Streamlit or any UI) Allows stakeholders to visualize anomalies and explanations intuitively.
* **End-to-end Pipeline:** A single script (`pipeline.py`) automates the entire workflow from raw data to actionable insights.

---

## Project Structure

```
solar-production-anomaly/
├── backend/                   # Core processing and modeling scripts
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── fetch_weather.py       # Weather data retrieval and integration
│   ├── xgboost_forecast.py   # Training and forecasting using XGBoost
│   ├── anomaly_detection.py  # Detect anomalies based on forecasting residuals
│   ├── shap_explain.py       # Generate SHAP explanations for anomaly cases
│   ├── pipeline.py           # Orchestrates the full pipeline execution
│   └── __pycache__/          # Python cache files
├── input/                    # Raw input data files (e.g., inverter.csv, plant.csv)
├── models/                   # Trained model files (e.g., xgb_model.pkl)
├── output/                   # Processed data, anomaly results, SHAP outputs
├── app.py                    # (Optional) Dashboard application entrypoint
├── requirements.txt          # Required Python packages
├── README.md                 # This documentation file
└── .DS_Store                 # MacOS system files (can be ignored)
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/solar-production-anomaly.git
   cd solar-production-anomaly
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your input data:**

   Place your `inverter.csv` and `plant.csv` inside the `input/` folder.

---

## ▶️ Running the Pipeline

To run the entire anomaly detection and explanation pipeline end-to-end, simply execute:

```bash
python backend/pipeline.py
```

This script will:

* Preprocess raw input data
* Fetch and integrate weather data
* Train or load the XGBoost forecasting model
* Detect anomalies based on forecasting residuals
* Generate SHAP values to explain anomalies
* Save outputs to the `output/` directory

---

## Module Details

* **data\_preprocessing.py**: Cleans and merges datasets, creates lag and rolling statistical features to improve model accuracy.

* **fetch\_weather.py**: Downloads and processes weather data to enrich the feature set.

* **xgboost\_forecast.py**: Contains code for training and using an XGBoost regression model to predict solar production.

* **anomaly\_detection.py**: Detects anomalies by comparing actual production with model forecasts using residual thresholds.

* **shap\_explain.py**: Uses SHAP to interpret the anomaly detection model, providing insight into which features contribute most to anomalies.

* **pipeline.py**: Integrates all steps into one seamless workflow.

---

## Outputs

* `output/merged_featured_data.csv`: Preprocessed dataset with engineered features.

* `output/anomaly_results.csv`: Records of detected anomalies with timestamps.

* `output/anomaly_results_with_shap.csv`: Anomalies enriched with SHAP values.

* `output/shap_anomaly_values.npy` & `output/shap_values_per_anomaly.csv`: Raw and tabular SHAP values for further analysis.

---

## Key Technologies & Libraries

* **Python 3.x**
* **XGBoost** – High-performance gradient boosting for regression
* **SHAP** – Model explainability using Shapley values
* **Pandas & NumPy** – Data manipulation and numerical computations
* **Matplotlib** – Visualization of SHAP values
* **Joblib** – Model serialization and loading


---

Future enhancements planned:

* Integration of LSTM-based models for sequence forecasting
* Automated alerts for detected anomalies
* Expanded weather data features and sources

