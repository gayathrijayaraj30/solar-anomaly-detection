import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from backend.pipeline import full_pipeline

st.set_page_config(layout="wide")

st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #111111 !important;
            color: #E0C097 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        h1, h2, h3 {
            color: #FFB347 !important;
            text-align: center;
        }
        .stButton>button {
            background-color: #FFB347 !important;
            color: #1E1E1E !important;
            font-weight: bold;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #FFA500 !important;
        }
    </style>
""", unsafe_allow_html=True)


INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
WEATHER_PATH = os.path.join(OUTPUT_FOLDER, "weather.csv")
FEATURED_PATH = os.path.join(OUTPUT_FOLDER, "merged_featured_data.csv")
ANOMALY_PATH = os.path.join(OUTPUT_FOLDER, "anomaly_results.csv")
SHAP_PATH = os.path.join(OUTPUT_FOLDER, "shap_anomaly_values.npy")


SHAP_CSV_PATH = os.path.join(OUTPUT_FOLDER, "shap_values_per_anomaly.csv")

if os.path.exists(SHAP_CSV_PATH):
    shap_df_all = pd.read_csv(SHAP_CSV_PATH)
    FEATURE_NAMES = shap_df_all.columns.drop('date').tolist()
else:
    FEATURE_NAMES = []
    st.warning("SHAP feature names file not found. SHAP plots may not render properly.")


st.title("Solar Production Anomaly Dashboard")

# Sidebar: Upload files or use input folder ===
st.sidebar.header("Upload Source Files")
inv_file = st.sidebar.file_uploader("Upload inverter.csv", type=["csv"])
plant_file = st.sidebar.file_uploader("Upload plant.csv", type=["csv"])

if inv_file is not None:
    inverter_path = os.path.join(INPUT_FOLDER, "inverter.csv")
    with open(inverter_path, "wb") as f:
        f.write(inv_file.read())
elif os.path.exists(os.path.join(INPUT_FOLDER, "inverter.csv")):
    inverter_path = os.path.join(INPUT_FOLDER, "inverter.csv")
else:
    st.error("inverter.csv not uploaded or missing in input/")
    st.stop()

if plant_file is not None:
    plant_path = os.path.join(INPUT_FOLDER, "plant.csv")
    with open(plant_path, "wb") as f:
        f.write(plant_file.read())
elif os.path.exists(os.path.join(INPUT_FOLDER, "plant.csv")):
    plant_path = os.path.join(INPUT_FOLDER, "plant.csv")
else:
    st.error("plant.csv not uploaded or missing in input/")
    st.stop()

# Sidebar: Date Range for Weather Fetching 
st.sidebar.header("Select Weather Data Range")
today = datetime.today()
start_date = st.sidebar.date_input("Start Date", datetime(today.year, 1, 1))
end_date = st.sidebar.date_input("End Date", today)

# Run Full Pipeline  
if st.sidebar.button("Run Full Pipeline"):
    progress = st.progress(0, "Starting...")
    progress.progress(10, "Fetching weather & preparing input...")

    try:
        full_pipeline(
            inverter_path=inverter_path,
            plant_path=plant_path,
            weather_start=start_date.strftime("%Y-%m-%d"),
            weather_end=end_date.strftime("%Y-%m-%d"),
            weather_path=WEATHER_PATH,
            featured_output_path=FEATURED_PATH,
            model_output_path="models/xgb_model.pkl",
            anomaly_output_path=ANOMALY_PATH
        )
        progress.progress(100, "Pipeline complete!")
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

# Load Output Data  
@st.cache_data
def load_outputs():
    data = pd.read_csv(FEATURED_PATH, parse_dates=["date"])
    anomalies = pd.read_csv(ANOMALY_PATH, parse_dates=["date"])
    shap_vals = np.load(SHAP_PATH, allow_pickle=True)
    return data, anomalies, shap_vals

if os.path.exists(FEATURED_PATH) and os.path.exists(ANOMALY_PATH) and os.path.exists(SHAP_PATH):
    data, anomalies, shap_values = load_outputs()
else:
    st.warning("⚠️ Run the pipeline first to generate results.")
    st.stop()

# Main Dashboard  
min_d, max_d = anomalies['date'].min().date(), anomalies['date'].max().date()
selected_range = st.date_input("Select Date Range", [min_d, max_d], min_value=min_d, max_value=max_d)
filtered_data = data[(data['date'].dt.date >= selected_range[0]) & (data['date'].dt.date <= selected_range[1])]
print(selected_range)
filtered_anoms = anomalies[(anomalies['date'].dt.date >= selected_range[0]) & (anomalies['date'].dt.date <= selected_range[1])]

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='text-align: left;'>Production Summary</h3>", unsafe_allow_html=True)
    prod_stats = {
        "Mean (kWh)": filtered_data['Production(kWh)'].mean(),
        "Max (kWh)": filtered_data['Production(kWh)'].max(),
        "Min (kWh)": filtered_data['Production(kWh)'].min(),
        "Std Dev (kWh)": filtered_data['Production(kWh)'].std()
    }
    st.dataframe(pd.DataFrame(prod_stats, index=["Stats"]).T)

with col2:
    st.markdown("<h3 style='text-align: left;'>Anomaly Summary</h3>", unsafe_allow_html=True)
    anomaly_stats = {
        "Total Anomalies": len(filtered_anoms),
        "Max Residual": filtered_anoms['residual'].max(),
        "Mean Residual": filtered_anoms['residual'].mean(),
        "Std Dev Residual": filtered_anoms['residual'].std()
    }
    st.dataframe(pd.DataFrame(anomaly_stats, index=["Anomaly Stats"]).T)

# Anomaly Display
if not filtered_anoms.empty:
    st.subheader("Anomaly Details")
    options = [f"{row['date'].date()} | Residual: {row['residual']:.2f}" for _, row in filtered_anoms.iterrows()]
    selected = st.selectbox("Select anomaly", options)
    selected_row = filtered_anoms.iloc[options.index(selected)]

    idx = anomalies.index.get_loc(selected_row.name)

    st.markdown("### Production Plot with Anomaly")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered_data['date'], filtered_data['Production(kWh)'], label='Production', color='orange')
    ax.axvline(selected_row['date'], color='red', linestyle='--', label='Anomaly')
    ax.legend()
    st.pyplot(fig)

    st.markdown("### SHAP Feature Importance")
    shap_df = pd.DataFrame({'feature': FEATURE_NAMES, 'shap_value': shap_values[idx]})
    shap_df = shap_df.reindex(shap_df.shap_value.abs().sort_values(ascending=False).index).head(15)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(shap_df['feature'], shap_df['shap_value'], color='#FFB347')
    ax.invert_yaxis()
    ax.set_title("Top SHAP Contributors")
    st.pyplot(fig)

else:
    st.warning("No anomalies found in selected range.")
