import sys
sys.path.append('/Users/ignasivalles/Oceanography/IEO/projects/pom-cost.github.io/src')
from data_functions import *
import pandas as pd
import numpy as np
import glob
import os

# === Load Historical Data ===
folder_path = "/Users/ignasivalles/Library/CloudStorage/GoogleDrive-ignasi.valles@gmail.com/La meva unitat/POM/data/raw/"
historical_data = "/Users/ignasivalles/Library/CloudStorage/GoogleDrive-ignasi.valles@gmail.com/La meva unitat/POM/data/muestreoCubo1970_78.xlsx"

# Load and preprocess historical data
df = pd.read_excel(historical_data).rename(columns={
    'año': 'year', 'mes': 'month', 'dia': 'day', 'temperatura agua': 'temperatura'
})
df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])

# === 1. Climatology Calculation ===
climatology = df.groupby('month')['temperatura'].median().reset_index()
climatology['fractional_time'] = climatology['month'] - 1  # 0 for Jan, 11 for Dec

# === 2. Load Observational Data ===
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
print(f"Found {len(csv_files)} CSV files")

all_data = pd.concat([get_data_from_temp_sensors(file) for file in csv_files], ignore_index=True)

# Ensure date format and extract fractional time
all_data['Date'] = pd.to_datetime(all_data['Date'])
all_data['Temperature'] = all_data['Temperature'].round(2)

# === 3. Calculate Fractional Time ===
all_data['fractional_time'] = (
    all_data['Date'].dt.month - 1 + (all_data['Date'].dt.day - 1) / all_data['Date'].dt.days_in_month
)

# === 4. Interpolate Climatology & Calculate Anomalies ===
all_data['climatology_temp'] = np.interp(
    all_data['fractional_time'],
    climatology['fractional_time'],
    climatology['temperatura']
)

all_data['Temperature_Anomaly'] = all_data['Temperature'] - all_data['climatology_temp']

# === 5. Save the Result ===
out_path = '/Users/ignasivalles/Oceanography/IEO/projects/pom-cost.github.io/data/individual_data.csv'
all_data.to_csv(out_path, index=False)
print(f"Saved {len(all_data)} rows to {out_path}")
