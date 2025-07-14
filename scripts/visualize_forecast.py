#!/usr/bin/env python
"""
HTML interactivo con el último pronóstico de 7 días y meteo asociada.
Requiere: plotly (`pip install plotly`)
"""

from pathlib import Path
import sys, webbrowser
import pandas as pd
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

ROOT = Path(".")
OUT_DIR = ROOT / "outputs"
PROC_PARQUET = ROOT / "data" / "processed" / "dataset.parquet"
HRES_DIR = ROOT / "data" / "raw" / "hres"
OUT_HTML = OUT_DIR / "visualizaciones" / "forecast_last_run.html"
OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

# ───────── forecast (carpeta YYYYMMDDHH) ─────────
runs = sorted(
    [p for p in OUT_DIR.iterdir()
     if p.is_dir() and p.name.isdigit() and len(p.name) == 10],
    key=lambda p: p.name
)
if not runs:
    sys.exit("❌ No se encontró ninguna carpeta outputs/<YYYYMMDDHH>/")

run_dir = runs[-1]
fc_path = run_dir / "forecast_7d.csv"
fc_df = pd.read_csv(fc_path, parse_dates=["timestamp"])
target_cols = [c for c in fc_df.columns if c.endswith("_pred")]
targets = [c.replace("_pred", "") for c in target_cols]
print(f"• Forecast utilizado: {fc_path}")

# ───────── históricos (últimos 7 días) ─────────
hist_df = None
t0_pred = fc_df["timestamp"].iloc[0]
if PROC_PARQUET.exists():
    hist_df = pd.read_parquet(PROC_PARQUET)
    if "timestamp" in hist_df.columns:
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
        mask = (hist_df["timestamp"] >= t0_pred - timedelta(days=7)) & \
               (hist_df["timestamp"] <  t0_pred)
        hist_df = hist_df.loc[mask]

# ───────── meteorología (NetCDF) ─────────
nc_files = sorted(HRES_DIR.glob("ifs_hres_*.nc"))
if not nc_files:
    sys.exit("❌ No se encontró ifs_hres_*.nc en data/raw/hres/")
nc = nc_files[-1]
ds = xr.open_dataset(nc).mean(dim=("latitude", "longitude"), keep_attrs=True)
run_ts = pd.to_datetime(nc.stem[-10:], format="%Y%m%d%H", utc=True)
valid = run_ts + ds["step"].astype("timedelta64[h]")
met_df = pd.DataFrame({
    "timestamp": valid,
    "wind_speed": np.sqrt(ds["u10"]**2 + ds["v10"]**2).values,
    "t2m": ds["t2m"].values,
    "ssrd": ds["ssrd"].values / 3600.0,
})

# ───────── figura interactiva ─────────
rows = len(targets) + ("wind" in targets) + ("solar" in targets) + 1
fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True,
    subplot_titles=[
        *(f"{t.capitalize()} real vs pred" for t in targets),
        *(["Wind-speed vs eólica"] if "wind" in targets else []),
        *(["SSRD vs solar"] if "solar" in targets else []),
        "Correlación meteo ↔ pred"
    ],
    vertical_spacing=0.04
)

row = 1
for tgt in targets:
    # histórico
    if hist_df is not None and tgt in hist_df:
        fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df[tgt],
                                 name=f"{tgt} real",
                                 line=dict(color="steelblue", width=1.2)),
                      row=row, col=1)
    # predicción
    fig.add_trace(go.Scatter(x=fc_df["timestamp"], y=fc_df[f"{tgt}_pred"],
                             name=f"{tgt} pred",
                             line=dict(color="firebrick", width=1.4)),
                  row=row, col=1)
    row += 1

# dispersión
merged = pd.merge(fc_df, met_df, on="timestamp", how="inner")
if "wind" in targets:
    fig.add_trace(go.Scatter(
        x=merged["wind_speed"], y=merged["wind_pred"],
        mode="markers", marker=dict(size=5, color="darkgreen", opacity=0.6)),
        row=row, col=1)
    row += 1
if "solar" in targets:
    fig.add_trace(go.Scatter(
        x=merged["ssrd"], y=merged["solar_pred"],
        mode="markers", marker=dict(size=5, color="orange", opacity=0.6)),
        row=row, col=1)
    row += 1

# heat-map
corr_cols = ["wind_speed", "t2m", "ssrd"] + [f"{t}_pred" for t in targets]
corr = merged[corr_cols].corr().round(2)
fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                         colorscale="RdBu", zmid=0, showscale=True),
              row=row, col=1)

fig.update_layout(
    height=350 + 250 * (rows - 1),
    title=f"Predicción & meteorología – run {run_dir.name}",
    template="plotly_white",
    showlegend=False,
    margin=dict(t=60, l=60, r=30, b=40),
    hovermode="closest"
)

fig.write_html(str(OUT_HTML), include_plotlyjs="cdn")
print(f"✅ HTML generado en {OUT_HTML.resolve()}")
webbrowser.open(OUT_HTML.resolve().as_uri())
