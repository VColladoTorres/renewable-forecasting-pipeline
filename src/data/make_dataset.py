# src/data/make_dataset.py
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np                   # CAMBIO: para derivadas
import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm

from src.data.cds_client import CDSClient
from src.data.esios_client import EsiosClient
from src.utils.logger import init_logger

init_logger()
_LOGGER = logging.getLogger(__name__)

FREQ = "15T"                         # CAMBIO: frecuencia objetivo 15 min

def _parse_dates(cfg: dict) -> tuple[datetime, datetime]:
    start = pd.Timestamp(cfg["data"]["start_date"], tz="UTC").to_pydatetime()
    end   = pd.Timestamp(cfg["data"]["end_date"],   tz="UTC").to_pydatetime()
    return start, end

def _open_many_nc(nc_paths: list[Path]) -> xr.Dataset:
    def _drop_expver(ds: xr.Dataset) -> xr.Dataset:
        if "expver" in ds.dims:
            ds = ds.sel(expver=ds.expver.max()).drop_vars("expver")
        return ds

    return xr.open_mfdataset(
        [str(p) for p in nc_paths],
        combine="by_coords",
        preprocess=_drop_expver,
        compat="override",
        coords="minimal",
        parallel=True,
        engine="h5netcdf",
    )

def _derive_vars(ds: xr.Dataset) -> xr.Dataset:
    """Añade campos derivados inmediatamente tras abrir ERA5."""
    # CAMBIO: velocidades del viento y radiación normalizada
    ds = ds.copy()
    ds["wind_speed10"]  = np.hypot(ds["u10"],  ds["v10"])
    ds["wind_speed100"] = np.hypot(ds["u100"], ds["v100"])
    # ssrd está en J m⁻² h⁻¹ → W m⁻²
    if "ssrd" in ds:
        ds["ssrd_ghi"] = ds["ssrd"] / 3600.0
    # logaritmo para XGB/GBDT
    ds["log_wind_speed"] = np.log(ds["wind_speed10"].clip(min=0.1))
    return ds

def main(config: str) -> None:
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    start, end = _parse_dates(cfg)

    # ── ESIOS ─────────────────────────
    esios = EsiosClient()
    tech_cols = ["wind", "solar_pv", "demand"]
    dfs: list[pd.DataFrame] = []

    for tech in tqdm(tech_cols, desc="ESIOS", unit="tech"):
        _LOGGER.info("Descargando %s …", tech)
        df_tech = esios.fetch_series(start, end, technology=tech)
        dfs.append(df_tech.rename(columns={"mw": tech}))

    df_gen = pd.concat(dfs, axis=1)
    # CAMBIO: resampleo a 15 min
    df_gen = (
        df_gen
        .resample(FREQ)
        .interpolate(method="time")
    )

    # ── ERA5 ─────────────────────────
    cds = CDSClient(Path(cfg["paths"]["raw_dir"]) / "era5")
    nc_paths = cds.download(cfg["variables"]["era5"], start, end)
    _LOGGER.info("Abriendo %d NetCDF ERA5…", len(nc_paths))
    ds = _open_many_nc(nc_paths)
    ds = _derive_vars(ds)            # CAMBIO: derivadas

    # Media peninsular (bounding-box) y DataFrame horario
    spain = ds.sel(latitude=slice(44, 36), longitude=slice(-10, 5)).mean(["latitude", "longitude"])
    df_met = spain.to_dataframe()

    # Índice con zona horaria y re-muestra 15 min
    df_met.index = (
        df_met.index
        .tz_localize(timezone.utc, nonexistent="shift_forward")
        .tz_convert("Europe/Madrid")
        .round(FREQ)                 # CAMBIO: redondeo seguro
        .to_period(FREQ).to_timestamp()
    )
    df_met = (
        df_met
        .resample(FREQ)
        .interpolate("time")
    )

    # ── Merge y guardado ─────────────────────────
    df = df_gen.join(df_met, how="inner")

    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "initial_dataset.parquet"
    df.to_parquet(out_file)
    _LOGGER.info(
        "Dataset 15-min completo hasta %s: %d filas, guardado en %s",
        end.strftime('%d %b %Y'), len(df), out_file
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 15-min dataset until cutoff date.")
    parser.add_argument("--config", "-c", required=True, help="Ruta del YAML de configuración.")
    main(**vars(parser.parse_args()))
