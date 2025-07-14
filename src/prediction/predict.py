#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pronóstico horario a 7 días a partir del último IFS-HRES (0.25°).
• Guarda CSV + JSON + informe Plotly en:
  outputs/<run>/<model_name>/
"""

from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr
from pvlib.solarposition import get_solarposition

from src.features.build_features import engineer_feature_matrix
from src.utils.capacity import build_capacity_series
from src.utils.yaml_cfg import load

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("predict")

# ───────── helpers ──────────────────────────────────────────────────────────
def _latest_file(folder: Path) -> Path:
    nc = sorted(p for p in folder.glob("ifs_hres_*.nc") if ".idx" not in p.suffixes)
    g2 = sorted(folder.glob("ifs_hres_*.grib2"))
    if nc:
        return nc[-1]
    if g2:
        return g2[-1]
    raise FileNotFoundError("No ifs_hres_*.nc / *.grib2 en data/raw/hres")


def _open_nwp(path: Path) -> xr.Dataset:
    """Abre NetCDF pseudo-GRIB o GRIB2 y fusiona u10,v10,t2m,ssrd."""
    try:
        if path.suffix == ".nc":
            return xr.open_dataset(path, engine="netcdf4")
    except Exception as e_nc:  # noqa: BLE001
        LOG.debug("netcdf4 falló → cfgrib (%s)", e_nc)

    def _load(short: str) -> xr.Dataset:
        return xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"shortName": short},
                "indexpath": "",
            },
        ).drop_vars("heightAboveGround", errors="ignore")

    return xr.merge([_load(s) for s in ("10u", "10v", "2t", "ssrd")],
                    compat="override")


def _postprocess(pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Reescala CF→MW con capacidad dinámica y PV nocturno = 0."""
    caps = build_capacity_series(cfg["capacity_timeline"], pred.index)
    pred["wind"]      *= caps["wind"]
    pred["solar_pv"]  *= caps["solar_pv"]

    elev = get_solarposition(pred.index.tz_convert("Europe/Madrid"),
                             cfg["site"]["lat"], cfg["site"]["lon"])["elevation"]
    pred.loc[elev <= 0, "solar_pv"] = 0.0
    return pred


def _save_report(pred: pd.DataFrame, out_dir: Path) -> None:
    """Genera informe Plotly interactivo (HTML)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred.index, y=pred["wind"],
        mode="lines", name="Wind (MW)"))
    fig.add_trace(go.Scatter(
        x=pred.index, y=pred["solar_pv"],
        mode="lines", name="Solar PV (MW)"))

    fig.update_layout(
        title="7-Day Ahead Power Forecast",
        xaxis_title="Time (Europe/Madrid)",
        yaxis_title="Power (MW)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    pio.write_html(fig, file=str(out_dir / "forecast_report.html"),
                   auto_open=False)


# ───────── main ─────────────────────────────────────────────────────────────
def main(cfg_path: str) -> None:
    cfg = load(cfg_path)

    hres_dir  = Path(cfg["paths"].get("hres_dir", "data/raw/hres"))
    model_dir = Path(cfg["paths"]["best_model_dir"])
    outputs   = Path(cfg["paths"].get("outputs", "outputs"))

    nwp_file = _latest_file(hres_dir)
    LOG.info("Usando NWP %s", nwp_file.name)
    ds = _open_nwp(nwp_file)

    targets   = cfg["model"]["targets"]
    feat_used = joblib.load(model_dir / targets[0] / "features_used.pkl")
    models    = {t: joblib.load(model_dir / t / f"{t}.joblib") for t in targets}

    feats_all = engineer_feature_matrix(ds, cfg, future=True)
    feats = feats_all.reindex(columns=feat_used, fill_value=0.0)

    preds = pd.DataFrame({t: m.predict(feats) for t, m in models.items()},
                         index=feats.index)
    preds = _postprocess(preds, cfg)

    run = pd.Timestamp.utcnow().strftime("%Y%m%d%H")
    out_dir = outputs / run / model_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    preds.to_csv(out_dir / "forecast_7d.csv")
    preds.to_json(out_dir / "forecast_7d.json",
                  orient="table", date_format="iso")
    _save_report(preds, out_dir)

    LOG.info("Pronóstico e informe guardados → %s", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    try:
        main(ap.parse_args().config)
    except Exception as exc:          # noqa: BLE001
        LOG.error("❌  %s", exc)
        sys.exit(1)
