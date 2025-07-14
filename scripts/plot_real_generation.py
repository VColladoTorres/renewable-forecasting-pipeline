#!/usr/bin/env python
"""
Genera un HTML interactivo con la evolución diaria del último año para:
    • producción eólica   (columna 'wind')
    • producción solar PV (columna 'solar_pv')
    • demanda             (columna 'demand')
El HTML se guarda en outputs/visualizaciones/real_generation_last_year.html
"""

from pathlib import Path
import sys, webbrowser
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

# ─────────── configuración ───────────
PARQUET = Path(
    r"C:\Users\Vicente Collado\Desktop\Master Thesis\XGBoost\renewable-forecasting-pipeline\data\processed\dataset.parquet"
)

COLS = {"wind": "Eólica (MW)",
        "solar_pv": "Solar PV (MW)",
        "demand": "Demanda (MW)"}   # ajusta si tus columnas tienen otros nombres

OUT_HTML = (
    Path("outputs") / "visualizaciones" / "real_generation_last_year.html"
)
OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

# ─────────── carga de datos ───────────
if not PARQUET.exists():
    sys.exit(f"❌ No se encontró el parquet en {PARQUET}")

df = pd.read_parquet(PARQUET)

if "timestamp" not in df.columns:
    sys.exit("❌ El parquet no contiene la columna 'timestamp'.")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

# filtrar último año
end = df.index.max()
start = end - timedelta(days=365)
df_year = df.loc[start:end, COLS.keys()].copy()

if df_year.empty:
    sys.exit("❌ No hay datos para el último año en el parquet.")

# ─────────── figura interactiva ───────────
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=[COLS[c] for c in COLS]
)

for i, (col, label) in enumerate(COLS.items(), start=1):
    if col not in df_year.columns:
        fig.add_annotation(
            text=f"'{col}' no encontrado", xref="paper", yref="paper",
            x=0.5, y=1 - (i - 1) / 3, showarrow=False, font=dict(color="red")
        )
        continue

    fig.add_trace(
        go.Scatter(
            x=df_year.index,
            y=df_year[col],
            mode="lines",
            name=label,
            line=dict(width=1.2)
        ),
        row=i, col=1
    )

fig.update_layout(
    height=900,
    title_text="Producción real – último año",
    template="plotly_white",
    showlegend=False,
    hovermode="x unified",
    margin=dict(t=70, l=60, r=30, b=40)
)

# guardar y abrir
fig.write_html(str(OUT_HTML), include_plotlyjs="cdn")
print(f"✅ HTML generado: {OUT_HTML.resolve()}")
webbrowser.open(OUT_HTML.resolve().as_uri())
