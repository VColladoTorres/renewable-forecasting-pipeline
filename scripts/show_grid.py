#!/usr/bin/env python
"""
Muestra la malla 0.25° × 0.25° usada para extraer meteorología HRES
sobre la Península Ibérica.
"""

from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import webbrowser

# ─────────── parámetros de dominio ───────────
LON_MIN, LON_MAX = -10.0, 4.0
LAT_MIN, LAT_MAX =  36.0, 44.0
RESOLUTION       = 0.25    # grados

# ─────────── generar la cuadrícula ───────────
lons = np.arange(LON_MIN, LON_MAX + RESOLUTION, RESOLUTION)
lats = np.arange(LAT_MIN, LAT_MAX + RESOLUTION, RESOLUTION)
lon_grid, lat_grid = np.meshgrid(lons, lats)
lon_flat = lon_grid.ravel()
lat_flat = lat_grid.ravel()

print(f"• Nodos generados: {len(lon_flat)}")

# ─────────── figura interactiva ───────────
fig = go.Figure()

# puntos de la malla
fig.add_trace(go.Scattergeo(
    lon = lon_flat,
    lat = lat_flat,
    mode = "markers",
    marker = dict(size=4, color="red", opacity=0.7),
    name = "Nodos 0.25°"
))

# estilo general
fig.update_geos(
    projection_type="mercator",
    showcountries=True,  countrycolor="black",
    showcoastlines=True, coastlinecolor="gray",
    lataxis=dict(range=[LAT_MIN-1, LAT_MAX+1]),
    lonaxis=dict(range=[LON_MIN-1, LON_MAX+1]),
    resolution=50
)
fig.update_layout(
    title="Malla ECMWF-HRES (0.25°) – Península Ibérica",
    height=700,
    template="plotly_white",
    margin=dict(t=60, l=30, r=30, b=30)
)

# ─────────── guardar y abrir ───────────
out_html = Path("outputs") / "visualizaciones" / "grid_ifs_hres.html"
out_html.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(out_html), include_plotlyjs="cdn")
print(f"✅ HTML guardado en {out_html.resolve()}")
webbrowser.open(out_html.resolve().as_uri())
