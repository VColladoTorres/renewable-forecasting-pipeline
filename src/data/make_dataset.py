# src/data/make_dataset.py
# MIT License
# Copyright (c) 2025 MSc Candidate
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction …

"""
ETL principal : descarga ESIOS (+ fallback), ERA5/ERA5-Land, alinea series
horarias y guarda un único parquet listo para modelar.

Uso
───
    python -m src.data.make_dataset --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm

from src.data.cds_client import CDSClient
from src.data.esios_client import EsiosClient
from src.utils.logger import init_logger

# ──────────────────────────── logging ──────────────────────────────
init_logger()
_LOGGER = logging.getLogger(__name__)
# ───────────────────────────────────────────────────────────────────


# ------------------------------------------------------------------
def _parse_dates(cfg: dict) -> tuple[datetime, datetime]:
    """Devuelve (start UTC, end UTC) convertidos desde YAML."""
    start = pd.Timestamp(cfg["data"]["start_date"], tz="UTC").to_pydatetime()
    end = pd.Timestamp(cfg["data"]["end_date"], tz="UTC").to_pydatetime()
    return start, end


def _open_many_nc(nc_paths: list[Path]) -> xr.Dataset:
    """
    Abre decenas de NetCDF ERA5 asegurando que solo se use
    la versión más reciente (expver) y evitando conflictos.
    """
    import xarray as xr

    def _drop_expver(ds: xr.Dataset) -> xr.Dataset:
        # Selecciona la expver más alta (normalmente 5) y elimina la variable.
        if "expver" in ds.dims:
            ds = ds.sel(expver=ds.expver.max())
            ds = ds.drop_vars("expver")
        return ds

    return xr.open_mfdataset(
        [str(p) for p in nc_paths],
        combine="by_coords",
        preprocess=_drop_expver,   # ← clave
        compat="override",         # evita conflictos menores
        coords="minimal",
        parallel=True,
        engine="h5netcdf",
    )


# ------------------------------------------------------------------
def main(config: str) -> None:  # noqa: D401
    # ---------- lee YAML ----------
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    start, end = _parse_dates(cfg)

    # ---------- descarga ESIOS ----------
    esios = EsiosClient()
    tech_cols = ["wind", "solar_pv"]
    dfs: list[pd.DataFrame] = []

    for tech in tqdm(tech_cols, desc="ESIOS", unit="tech"):
        _LOGGER.info("Download %s generation…", tech)
        df_tech = esios.fetch_series(start, end, technology=tech).rename(columns={"mw": tech})
        dfs.append(df_tech)

    demand = esios.fetch_series(start, end, technology="demand").rename(columns={"mw": "demand"})
    df_gen = pd.concat(dfs + [demand], axis=1)

    # ---------- descarga ERA5 ----------
    cds = CDSClient(raw_dir / "era5")
    nc_paths = cds.download(cfg["variables"]["era5"], start, end)  # lista[Path]
    _LOGGER.info("Abriendo %d ficheros NetCDF ERA5…", len(nc_paths))
    ds = _open_many_nc(nc_paths)

    # media espacial (bounding-box España peninsular)
    spain = ds.sel(latitude=slice(44, 36), longitude=slice(-10, 5)).mean(
        ["latitude", "longitude"]
    )
    df_met = spain.to_dataframe()
    df_met.index = (
        df_met.index.tz_localize(timezone.utc, nonexistent="shift_forward")
        .tz_convert("Europe/Madrid")
    )

    # ---------- merge & persist ----------
    df = df_gen.join(df_met, how="inner")
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dataset.parquet"
    df.to_parquet(out_file)
    _LOGGER.info("Dataset final: %d filas, guardado en %s", len(df), out_file)


# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build raw dataset (ESIOS + ERA5).")
    parser.add_argument(
        "--config", "-c", required=True, help="Ruta del YAML de configuración."
    )
    main(**vars(parser.parse_args()))