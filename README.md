# Solar Anomaly Detection

---

## Project Overview

This project implements an end-to-end pipeline for detecting anomalies in solar power generation using machine learning techniques. It combines inverter telemetry data, plant metadata, and weather data to forecast expected solar power output with XGBoost, detect anomalies via residual analysis, and provide explainability with SHAP values. An interactive Streamlit dashboard allows users to visualize results and insights intuitively.

---

## Key Features

* **Data Preprocessing:** Cleans, merges, and engineers features from inverter and weather datasets.
* **Weather Data Integration:** Automates fetching and incorporating weather data relevant to solar plant location.
* **Power Output Forecasting:** Utilizes XGBoost regression to predict solar power generation.
* **Anomaly Detection:** Identifies deviations from forecasted values signaling possible faults or irregularities.
* **Explainability:** Employs SHAP to interpret feature impacts on model predictions.
* **Interactive Dashboard:** Visualizes forecasts, anomalies, and feature importance dynamically for user exploration.

---

## Project Structure

```
solar-anomaly-detection/
├── app.py                        # Streamlit dashboard main app
├── backend/                      # Core scripts and pipeline modules
│   ├── anomaly_detection.py      # Detect anomalies in solar output
│   ├── data_preprocessing.py     # Data cleaning and feature engineering
│   ├── fetch_weather.py          # Weather data fetching utilities
│   ├── pipeline.py               # Orchestrates the entire processing pipeline
│   ├── shap_explain.py           # SHAP explainability code
│   └── xgboost_forecast.py       # XGBoost training and prediction code
├── input/                       # Raw input data files (inverter, weather)
├── models/                      # Saved model artifacts
├── output/                      # Generated output data and visualizations
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Dataset Information

The dataset used in this project comes from a **local solar inverter system** located in **Thrissur, Kerala, India**.

- **Location**: Thrissur, Kerala  
- **Coordinates**: Latitude `10.52385`, Longitude `76.21313`
- **Source**: Privately owned solar inverter (not publicly released)
- **Contents**:
  - Daily solar energy **production** and **consumption** (kWh)
  - **Grid feed-in**, **electricity purchasing**, and **anticipated yield (INR)**
  - Weather data: temperature, precipitation, rain flags, etc.
- **Time Period**: *(e.g., Jan 2022 – Mar 2023)*  
- **Note**: Original data not shared due to privacy, but structure-matching samples are used.

## Model Performance

The production forecast model is based on an **XGBoost Regressor** trained on feature-engineered historical data.

- **Test RMSE**: `0.11 kWh`
- **R² Score**: `0.999`

These metrics are calculated on a hold-out test set and indicate a **very high accuracy**, with predictions closely matching actual production values.

Residual plots and SHAP value explanations are included to further analyze prediction reliability and feature impact.


## Getting Started

### Prerequisites

* Python 3.7 or higher
* Recommended: Use a virtual environment (venv, conda, etc.)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/gayathrijayaraj30/solar-anomaly-detection.git
   cd solar-anomaly-detection
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Ensure any necessary API keys or config files are set if your pipeline uses external APIs for weather data.

---

## Usage

### Running the Full Pipeline

The core processing pipeline can be executed via:

```bash
python backend/pipeline.py
```

This runs data preprocessing, weather fetching, forecasting, anomaly detection, and explainability steps end-to-end.

### Launching the Dashboard

To start the interactive Streamlit dashboard, run:

```bash
streamlit run app.py
```

The dashboard enables you to:

* Visualize solar power forecasts vs actuals
* Explore detected anomalies
* Investigate feature importance with SHAP plots

---

## Scripts Overview

| Script                  | Description                                         |
| ----------------------- | --------------------------------------------------- |
| `data_preprocessing.py` | Data cleaning, merging, and feature engineering     |
| `fetch_weather.py`      | Downloads weather data for the plant location       |
| `xgboost_forecast.py`   | Trains and predicts solar power output with XGBoost |
| `anomaly_detection.py`  | Detects anomalies using residual errors             |
| `shap_explain.py`       | Generates SHAP values for model interpretation      |
| `pipeline.py`           | Runs the entire workflow in sequence                |
| `app.py`                | Streamlit dashboard frontend                        |

---


## Live Demo

Try out the interactive solar anomaly detection dashboard live at:

🔗 [https://solar-anomaly-detection.streamlit.app/](https://solar-anomaly-detection.streamlit.app/)

Explore solar power forecasts, anomaly detection results, and SHAP explainability visuals directly in your browser—no setup required!

---

## Contributing

Contributions are welcome! Please fork the repo, create feature branches, and submit pull requests with clear descriptions.


