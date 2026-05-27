"""
generate_plots.py — Generate Plotly dashboard HTML plots for pom-cost.

Run from the repo root:
    python src/generate_plots.py

Outputs:
    plots/timeseries_plot.html
    plots/climatology_plot.html
"""

import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).parent.parent
DATA     = ROOT / 'data' / 'individual_data.csv'
DATA_XLS = ROOT / 'data' / 'muestreoCubo1970_78.xlsx'
PLOTS    = ROOT / 'plots'

# ── Spatial filter ────────────────────────────────────────────────────────────
LON_MIN, LON_MAX = -5.0, -3.0
LAT_MIN, LAT_MAX = 43.2, 43.9

# ── Year colours ──────────────────────────────────────────────────────────────
YEAR_COLORS = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
               '#e67e22', '#1abc9c', '#e91e63', '#ff5722']

# ── Shared layout ─────────────────────────────────────────────────────────────
LAYOUT = dict(
    font=dict(family='Open Sans, sans-serif', size=13, color='#2c3e50'),
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(l=60, r=24, t=24, b=56),
    legend=dict(bgcolor='rgba(255,255,255,0.85)', borderwidth=0),
    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)', zeroline=False,
               linecolor='rgba(0,0,0,0.12)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)', zeroline=False,
               linecolor='rgba(0,0,0,0.12)', title_standoff=12),
    hoverlabel=dict(bgcolor='white', bordercolor='rgba(0,0,0,0.15)',
                    font_size=12, font_family='Open Sans, sans-serif'),
)


# ── Smoothing helper ──────────────────────────────────────────────────────────
def _smooth_spline(x, y, n_out=300, s_factor=1.5):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 6:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x, inv = np.unique(x, return_inverse=True)
    unique_y = np.array([y[inv == i].mean() for i in range(len(unique_x))])
    if len(unique_x) < 6:
        return unique_x, unique_y
    spl = UnivariateSpline(unique_x, unique_y, s=len(unique_x) * s_factor, k=3)
    x_fine = np.linspace(unique_x.min(), unique_x.max(), n_out)
    return x_fine, spl(x_fine)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA)
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (
        (df['Longitude'] >= LON_MIN) & (df['Longitude'] <= LON_MAX) &
        (df['Latitude']  >= LAT_MIN) & (df['Latitude']  <= LAT_MAX)
    )
    df = df[mask].copy().sort_values('Date').reset_index(drop=True)
    print(f"Points after bbox filter: {len(df)}")
    print(f"Date range : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"Temp range : {df['Temperature'].min():.1f} – {df['Temperature'].max():.1f} °C")
    return df


# ── Time series ───────────────────────────────────────────────────────────────
def make_timeseries(df):
    df = df[df['Date'] >= '2025-01-01'].copy().sort_values('Date').reset_index(drop=True)

    t_lo = float(df['Temperature'].quantile(0.05))
    t_hi = float(df['Temperature'].quantile(0.95))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Temperature'],
        mode='markers',
        marker=dict(
            color=df['Temperature'],
            colorscale='RdBu_r',
            cmin=t_lo, cmax=t_hi,
            size=7, opacity=0.65,
            colorbar=dict(title='°C', thickness=12, len=0.55, x=1.01),
        ),
        hovertemplate='<b>%{x|%d %b %Y}</b><br>%{y:.1f} °C<extra></extra>',
        showlegend=False,
    ))

    x_num = (df['Date'] - df['Date'].min()).dt.days.values.astype(float)
    x_fine, y_fine = _smooth_spline(x_num, df['Temperature'].values, s_factor=0.4)
    if len(x_fine) > 1:
        dates_fine = df['Date'].min() + pd.to_timedelta(x_fine, unit='D')
        fig.add_trace(go.Scatter(
            x=dates_fine,
            y=y_fine,
            mode='lines',
            line=dict(color='#0077b6', width=2.5),
            hoverinfo='skip',
            name='Trend',
            showlegend=False,
        ))

    fig.update_layout(**LAYOUT, yaxis_title='Temperature [°C]')
    return fig


# ── Climatology helpers ───────────────────────────────────────────────────────
def _clim_bands():
    df = pd.read_excel(DATA_XLS, header=0)
    df = df.rename(columns={'año': 'year', 'mes': 'month', 'dia': 'day',
                             'temperatura agua': 'temperatura'})
    g  = df.groupby('month')['temperatura']
    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    iqr = q3 - q1
    lower, upper, medians = q1 - 1.5 * iqr, q3 + 1.5 * iqr, g.median()

    x = np.linspace(0, 12, 84)
    sm = lambda arr: savgol_filter(np.interp(x, np.arange(12), arr), 7, 2)
    return x, sm(lower), sm(upper), sm(medians)


def _rolling_curve(year_df, color, year):
    year_df = year_df.sort_values('fractional_time').reset_index(drop=True)
    if len(year_df) < 5:
        return None
    x_fine, y_fine = _smooth_spline(
        year_df['fractional_time'].values,
        year_df['Temperature'].values,
        s_factor=0.3,
    )
    if len(x_fine) < 2:
        return None
    return go.Scatter(
        x=x_fine, y=y_fine,
        mode='lines',
        line=dict(color=color, width=2.5),
        hoverinfo='skip',
        legendgroup=str(year),
        showlegend=False,
    )


# ── Climatology plot ──────────────────────────────────────────────────────────
def make_climatology(df):
    x_idx, lower, upper, medians = _clim_bands()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([x_idx, x_idx[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill='toself',
        fillcolor='rgba(173,216,230,0.35)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        name='IQR 1970–78',
    ))

    fig.add_trace(go.Scatter(
        x=x_idx, y=medians,
        mode='lines',
        line=dict(color='#2980b9', width=2),
        hoverinfo='skip',
        name='Median 1970–78',
    ))

    df = df.copy()
    df['year'] = pd.to_datetime(df['Date']).dt.year
    df['date_str'] = df['Date'].dt.strftime('%d %b %Y')
    years_2025plus = sorted(df[df['year'] >= 2025]['year'].unique())

    pre = df[df['year'] < 2025]
    if len(pre):
        fig.add_trace(go.Scatter(
            x=pre['fractional_time'], y=pre['Temperature'],
            mode='markers',
            marker=dict(color='#aaaaaa', size=6, opacity=0.4),
            customdata=pre[['date_str', 'Temperature_Anomaly']],
            hovertemplate='<b>%{customdata[0]}</b><br>%{y:.1f} °C  Δ%{customdata[1]:+.1f} °C<extra></extra>',
            name='Pre-2025',
        ))

    for i, year in enumerate(years_2025plus):
        color = YEAR_COLORS[i % len(YEAR_COLORS)]
        yr = df[df['year'] == year]
        fig.add_trace(go.Scatter(
            x=yr['fractional_time'], y=yr['Temperature'],
            mode='markers',
            marker=dict(color=color, size=6, opacity=0.65),
            customdata=yr[['date_str', 'Temperature_Anomaly']],
            hovertemplate='<b>%{customdata[0]}</b><br>%{y:.1f} °C  Δ%{customdata[1]:+.1f} °C<extra></extra>',
            name=str(year),
            legendgroup=str(year),
        ))
        curve = _rolling_curve(yr, color, year)
        if curve:
            fig.add_trace(curve)

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig.update_layout(**LAYOUT, yaxis_title='Temperature [°C]')
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(12)),
        ticktext=month_labels,
        range=[-0.3, 12],
        showgrid=True, gridcolor='rgba(0,0,0,0.06)',
        zeroline=False, linecolor='rgba(0,0,0,0.12)',
    )
    return fig


# ── Save ──────────────────────────────────────────────────────────────────────
def save_fig(fig, path):
    fig.update_layout(height=520)
    fig.write_html(
        str(path),
        include_plotlyjs='cdn',
        full_html=False,
        config={'displayModeBar': 'hover', 'scrollZoom': False},
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def update_index_stats(df):
    """Update hardcoded stats in index.html from latest data."""
    import re

    index_path = ROOT / 'index.html'
    html = index_path.read_text(encoding='utf-8')

    last_date  = df['Date'].dt.date.max()
    last_day   = df[df['Date'].dt.date == last_date]
    last_temp  = round(last_day['Temperature'].mean(), 1)
    last_anom  = last_day['Temperature_Anomaly'].mean()
    last_clim  = round(last_day['climatology_temp'].mean(), 1)
    total_obs  = len(df)

    # Format values
    anom_str   = f"{last_anom:+.1f}"
    date_label = last_date.strftime('%-d %b %Y').lower()
    month_name = last_date.strftime('%B').lower()

    # Replace stat blocks using regex
    html = re.sub(
        r'(<div class="stat-value">)\s*[\d.]+&thinsp;°C\s*(</div>\s*<div class="stat-label">Última medición · )[\w\s]+?(</div>)',
        lambda m: f'{m.group(1)}{last_temp}&thinsp;°C{m.group(2)}{date_label}{m.group(3)}',
        html
    )
    html = re.sub(
        r'(<div class="stat-value anomaly">)\s*[+\-][\d.]+&thinsp;°C\s*(</div>)',
        f'\\g<1>{anom_str}&thinsp;°C\\g<2>',
        html
    )
    html = re.sub(
        r'(<div class="stat-value">)\s*[\d.]+&thinsp;°C\s*(</div>\s*<div class="stat-label">Referencia histórica · )[\w]+?(</div>)',
        lambda m: f'{m.group(1)}{last_clim}&thinsp;°C{m.group(2)}{month_name}{m.group(3)}',
        html
    )
    html = re.sub(
        r'(<div class="stat-value">)\s*\d+\s*(</div>\s*<div class="stat-label">Observaciones totales)',
        f'\\g<1>{total_obs}\\g<2>',
        html
    )

    index_path.write_text(html, encoding='utf-8')
    print(f"  → index.html stats updated ({last_temp} °C · {anom_str} °C · {last_clim} °C · {total_obs} obs · {date_label})")


def main():
    PLOTS.mkdir(exist_ok=True)
    df = load_data()

    print("\nGenerating time series plot...")
    save_fig(make_timeseries(df), PLOTS / 'timeseries_plot.html')
    print("  → plots/timeseries_plot.html")

    print("Generating climatology plot...")
    save_fig(make_climatology(df), PLOTS / 'climatology_plot.html')
    print("  → plots/climatology_plot.html")

    print("Updating index.html stats...")
    update_index_stats(df)

    print("\nDone.")


if __name__ == '__main__':
    main()
