"""
generate_miguel_bravo.py — Process Miguel Bravo school sensor files and generate miguel_plot.html

Run from the repo root:
    python src/generate_miguel_bravo.py

What it does:
    1. Copies files from data/raw/ whose names start with 'MB' or 'Colegio MB'
       into data/Miguel_Bravo/ (skips files already there)
    2. Processes them with the minimum-variance water temperature algorithm
    3. Saves miguel_plot.html at the repo root
"""

import sys
import glob
import shutil
import pathlib

import numpy as np
import pandas as pd
import holoviews as hv
from bokeh.models import HoverTool

hv.extension('bokeh')

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data_functions import get_data_from_temp_sensors

ROOT      = pathlib.Path(__file__).parent.parent
RAW_DIR   = ROOT / 'data' / 'raw'
MB_DIR    = ROOT / 'data' / 'Miguel_Bravo'
OUT_HTML  = ROOT / 'miguel_plot.html'

MB_PREFIXES = ('MB', 'Colegio MB', 'Miguel')


def sync_mb_files():
    """Copy MB files from data/raw/ to data/Miguel_Bravo/ (never overwrites)."""
    MB_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(RAW_DIR.glob('*.csv'))
    copied = 0
    for f in raw_files:
        if f.name.startswith(MB_PREFIXES):
            dest = MB_DIR / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                copied += 1
    print(f"Synced {copied} new file(s) to data/Miguel_Bravo/")


def process_mb_files():
    csv_files = sorted(MB_DIR.glob('*.csv'))
    print(f"Miguel Bravo files: {len(csv_files)}")
    rows = []
    for f in csv_files:
        try:
            df = get_data_from_temp_sensors(str(f))
            if df.empty or df['Temperature'].isna().all():
                print(f"  ⚠ No valid water temp, skipping: {f.name}")
                continue
            rows.append(df)
        except Exception as e:
            print(f"  ✗ Error processing {f.name}: {e}")

    if not rows:
        print("ERROR: No Miguel Bravo data could be processed.")
        return None

    all_data = pd.concat(rows, ignore_index=True)
    all_data['Date'] = pd.to_datetime(all_data['Date'], errors='coerce')
    all_data = all_data.dropna(subset=['Date', 'Temperature'])
    all_data['Temperature'] = all_data['Temperature'].round(2)
    return all_data.sort_values('Date').reset_index(drop=True)


def make_plot(df):
    hover = HoverTool(tooltips=[
        ('Date',        '@Date{%F}'),
        ('Temperature', '@Temperature{0.1f} °C'),
    ], formatters={'@Date': 'datetime'})

    line = hv.Curve(
        df, kdims=['Date'], vdims=['Temperature'], label='Temperature'
    ).opts(color='blue', line_width=2)

    scatter = hv.Scatter(
        df, kdims=['Date'], vdims=['Temperature']
    ).opts(
        tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
        color='blue', size=6, alpha=0.7,
    )

    return (line * scatter).opts(
        title='Sea Temperature — Miguel Bravo School',
        xlabel='Date', ylabel='Temperature [°C]',
        show_grid=True,
        responsive=True, height=200,
    )


def main():
    sync_mb_files()
    df = process_mb_files()
    if df is None:
        sys.exit(1)

    print(f"  Date range : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Temp range : {df['Temperature'].min():.1f} – {df['Temperature'].max():.1f} °C")

    hv.save(make_plot(df), OUT_HTML, backend='bokeh')
    print(f"✓ Saved → {OUT_HTML}")


if __name__ == '__main__':
    main()
