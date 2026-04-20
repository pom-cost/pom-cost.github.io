"""
process_all.py — Process all raw sensor files into individual_data.csv

Run from the repo root:
    python src/process_all.py

Or with a custom raw data directory:
    python src/process_all.py --raw-dir /path/to/data/raw

What it does:
    1. Reads every CSV in data/raw/ (EnvLogger format)
    2. Applies the minimum-variance algorithm to extract the water temperature
    3. Computes fractional_time, climatology_temp, Temperature_Anomaly
    4. Saves results to data/individual_data.csv
"""

import sys
import glob
import pathlib
import argparse

import numpy as np
import pandas as pd

# Make sure data_functions is importable from src/
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data_functions import get_data_from_temp_sensors

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).parent.parent
DATA_XLS = ROOT / 'data' / 'muestreoCubo1970_78.xlsx'
OUT_CSV  = ROOT / 'data' / 'individual_data.csv'


def load_climatology():
    """Monthly median temperatures from the Sardinero 1970–1978 reference."""
    df = pd.read_excel(DATA_XLS)
    df = df.rename(columns={
        'año': 'year', 'mes': 'month', 'dia': 'day', 'temperatura agua': 'temperatura'
    })
    clim = df.groupby('month')['temperatura'].median().reset_index()
    clim['fractional_time'] = clim['month'] - 1  # 0 = Jan, 11 = Dec
    return clim


def process_raw_files(raw_dir):
    """Process all CSVs in raw_dir, return a combined DataFrame."""
    csv_files = sorted(glob.glob(str(raw_dir / '*.csv')))
    print(f"Raw files found: {len(csv_files)}")

    rows = []
    skipped = 0
    for filepath in csv_files:
        name = pathlib.Path(filepath).name
        try:
            df = get_data_from_temp_sensors(filepath)
            # Validate result: skip rows with nan temperature or missing date
            if df.empty or df['Temperature'].isna().all():
                print(f"  ⚠ No valid water temperature found, skipping: {name}")
                skipped += 1
                continue
            rows.append(df)
        except Exception as e:
            print(f"  ✗ Error processing {name}: {e}")
            skipped += 1

    print(f"Processed: {len(rows)} files, skipped: {skipped}")

    if not rows:
        print("ERROR: No data could be processed.")
        return None

    all_data = pd.concat(rows, ignore_index=True)
    all_data['Date'] = pd.to_datetime(all_data['Date'], errors='coerce')
    all_data = all_data.dropna(subset=['Date', 'Temperature'])
    all_data['Temperature'] = all_data['Temperature'].round(2)

    # Fractional time: 0.0 = Jan 1, 11.97 = Dec 31
    all_data['fractional_time'] = (
        all_data['Date'].dt.month - 1 +
        (all_data['Date'].dt.day - 1) / all_data['Date'].dt.days_in_month
    )

    # Interpolate climatology and compute anomaly
    clim = load_climatology()
    all_data['climatology_temp'] = np.interp(
        all_data['fractional_time'],
        clim['fractional_time'].values,
        clim['temperatura'].values
    )
    all_data['Temperature_Anomaly'] = (
        all_data['Temperature'] - all_data['climatology_temp']
    ).round(3)

    # ── Outlier filter ────────────────────────────────────────────────────────
    before = len(all_data)

    # 1. Physical bounds: impossible sea surface temperatures
    all_data = all_data[
        (all_data['Temperature'] >= 5) & (all_data['Temperature'] <= 27)
    ]

    # 2. Monthly IQR: catch algorithm failures (e.g. air temp in summer)
    month = all_data['Date'].dt.month
    keep = pd.Series(True, index=all_data.index)
    for m in month.unique():
        idx = all_data.index[month == m]
        grp = all_data.loc[idx, 'Temperature']
        if len(grp) < 4:
            continue
        q1, q3 = grp.quantile(0.25), grp.quantile(0.75)
        iqr = q3 - q1
        keep.loc[idx] = (grp >= q1 - 2 * iqr) & (grp <= q3 + 2 * iqr)
    all_data = all_data[keep].reset_index(drop=True)

    print(f"After outlier filter: {len(all_data)} rows (removed {before - len(all_data)})")

    # Final column order (matches existing individual_data.csv)
    cols = ['Date', 'Latitude', 'Longitude', 'Temperature',
            'fractional_time', 'Team', 'climatology_temp', 'Temperature_Anomaly']
    all_data = all_data[cols].sort_values('Date').reset_index(drop=True)

    return all_data


def main():
    parser = argparse.ArgumentParser(description='Process EnvLogger raw files into individual_data.csv')
    parser.add_argument(
        '--raw-dir', type=pathlib.Path,
        default=ROOT / 'data' / 'raw',
        help='Directory containing raw EnvLogger CSV files (default: data/raw/)'
    )
    args = parser.parse_args()

    if not args.raw_dir.exists():
        print(f"ERROR: raw-dir does not exist: {args.raw_dir}")
        sys.exit(1)

    all_data = process_raw_files(args.raw_dir)
    if all_data is None:
        sys.exit(1)

    all_data.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Saved {len(all_data)} rows → {OUT_CSV}")
    print(f"  Date range : {all_data['Date'].min().date()} → {all_data['Date'].max().date()}")
    print(f"  Temp range : {all_data['Temperature'].min():.1f} – {all_data['Temperature'].max():.1f} °C")
    print(f"  Anomaly    : {all_data['Temperature_Anomaly'].min():.1f} – {all_data['Temperature_Anomaly'].max():.1f} °C")


if __name__ == '__main__':
    main()
