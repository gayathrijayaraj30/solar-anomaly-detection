import pandas as pd
import numpy as np

def load_raw_data(inverter_path, plant_path, weather_path):
    inverter = pd.read_csv(inverter_path, parse_dates=['Updated Time'])
    plant = pd.read_csv(plant_path, parse_dates=['Updated Time'])
    weather = pd.read_csv(weather_path, parse_dates=['date'])

    inverter.rename(columns={'Updated Time': 'date'}, inplace=True)
    plant.rename(columns={'Updated Time': 'date'}, inplace=True)

    return inverter, plant, weather

def preprocess(inverter_path, plant_path, weather_path):
    inverter, plant, weather = load_raw_data(inverter_path, plant_path, weather_path)

    inv_agg = inverter.groupby('date').agg({
        'Production(kWh)': 'sum',
        'Consumption(kWh)': 'sum',
        'Grid Feed-in(kWh)': 'sum',
        'Electricity Purchasing(kWh)': 'sum'
    }).reset_index()

    df = pd.merge(plant, inv_agg, on='date', how='inner', suffixes=('_plant', '_inverter'))
    df = pd.merge(df, weather, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)

    df['temp_range'] = df['temp_max_C'] - df['temp_min_C']
    df['rain_flag'] = (df['precipitation_mm'] > 0).astype(int)

    days_since_rain = []
    count = 1000
    for rain in df['rain_flag']:
        count = 0 if rain == 1 else count + 1
        days_since_rain.append(count)
    df['days_since_rain'] = days_since_rain

    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    lags = [1, 2, 3, 7]
    for lag in lags:
        df[f'prod_lag_{lag}'] = df['Production(kWh)'].shift(lag)
        df[f'cons_lag_{lag}'] = df['Consumption(kWh)'].shift(lag)
        df[f'temp_max_lag_{lag}'] = df['temp_max_C'].shift(lag)
        df[f'temp_min_lag_{lag}'] = df['temp_min_C'].shift(lag)
        df[f'precip_lag_{lag}'] = df['precipitation_mm'].shift(lag)
        df[f'rain_flag_lag_{lag}'] = df['rain_flag'].shift(lag)

    windows = [3, 7]
    for window in windows:
        df[f'prod_rollmean_{window}'] = df['Production(kWh)'].rolling(window).mean()
        df[f'prod_rollstd_{window}'] = df['Production(kWh)'].rolling(window).std()
        df[f'tempmax_rollmean_{window}'] = df['temp_max_C'].rolling(window).mean()
        df[f'tempmax_rollstd_{window}'] = df['temp_max_C'].rolling(window).std()

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.dropna(inplace=True)

    return df
