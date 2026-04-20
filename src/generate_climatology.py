"""
generate_climatology.py — Regenerate the climatological year plots for pom-cost.

Run from the repo root:
    python src/generate_climatology.py

Outputs:
    climatological-year/climatological_plot.html      (desktop)
    climatological-year/climatological_plot_mbl.html  (mobile)
"""

import sys
import pathlib

import pandas as pd
import holoviews as hv
from bokeh.models import HoverTool

hv.extension('bokeh')

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data_functions import plot_climatological_year, plot_rolling_by_year

ROOT     = pathlib.Path(__file__).parent.parent
DATA     = ROOT / 'data' / 'individual_data.csv'
DATA_XLS = ROOT / 'data' / 'muestreoCubo1970_78.xlsx'
OUT_DIR  = ROOT / 'climatological-year'

LON_MIN, LON_MAX = -5.0, -3.0
LAT_MIN, LAT_MAX = 43.2, 43.9

MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MONTH_TICKS  = [(i, l) for i, l in enumerate(MONTH_LABELS)]


def load_data():
    df = pd.read_csv(DATA)
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (
        (df['Longitude'] >= LON_MIN) & (df['Longitude'] <= LON_MAX) &
        (df['Latitude']  >= LAT_MIN) & (df['Latitude']  <= LAT_MAX)
    )
    df = df[mask].copy().sort_values('Date').reset_index(drop=True)
    print(f"Points after bbox filter: {len(df)}")
    return df


def make_plot(df, height):
    hover = HoverTool(tooltips=[
        ('Date',        '@Date{%F}'),
        ('Temperature', '@Temperature{0.1f} °C'),
        ('Anomaly',     '@Temperature_Anomaly{+0.1f} °C'),
    ], formatters={'@Date': 'datetime'})

    climato = plot_climatological_year(str(DATA_XLS))

    year_colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                   '#e67e22', '#1abc9c', '#e91e63', '#ff5722']
    df = df.copy()
    df['year'] = pd.to_datetime(df['Date']).dt.year
    years = sorted(df[df['year'] >= 2025]['year'].unique())
    scatter_opts = dict(tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
                        size=6, alpha=0.5, xlabel='Month', ylabel='Temperature [°C]',
                        responsive=True)
    scatters = []
    pre = df[df['year'] < 2025]
    if len(pre):
        scatters.append(hv.Scatter(pre, kdims=['fractional_time'],
                                   vdims=['Temperature', 'Temperature_Anomaly', 'Date'])
                        .opts(color='grey', **scatter_opts))
    for i, year in enumerate(years):
        yr = df[df['year'] == year]
        scatters.append(hv.Scatter(yr, kdims=['fractional_time'],
                                   vdims=['Temperature', 'Temperature_Anomaly', 'Date'],
                                   label=str(year))
                        .opts(color=year_colors[i % len(year_colors)], **scatter_opts))

    return (climato * hv.Overlay(scatters) * plot_rolling_by_year(df)).opts(
        title='Sea Surface Temperature — Climatological View',
        xticks=MONTH_TICKS, xlim=(0, 12),
        show_grid=True, legend_position='top_left',
        responsive=True, height=height,
    )


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_data()

    print("Generating desktop plot...")
    hv.save(make_plot(df, height=500), OUT_DIR / 'climatological_plot.html', backend='bokeh')
    print("  → climatological-year/climatological_plot.html")

    print("Generating mobile plot...")
    hv.save(make_plot(df, height=300), OUT_DIR / 'climatological_plot_mbl.html', backend='bokeh')
    print("  → climatological-year/climatological_plot_mbl.html")

    print("\nDone.")


if __name__ == '__main__':
    main()
