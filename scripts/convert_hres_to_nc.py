#!/usr/bin/env python
"""
Convierte los GRIB2 HRES descargados (0.25°) a NetCDF, fusionando variables
10u 10v 2t ssrd.  Omite ficheros ya convertidos o NetCDF directos.

Uso:
    python convert_hres_to_nc.py
"""
from __future__ import annotations
from pathlib import Path

import xarray as xr

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "hres"

for grib in RAW_DIR.glob("ifs_hres_*.grib2"):
    nc = grib.with_suffix(".nc")
    if nc.exists():
        continue

    def _load(short: str) -> xr.Dataset:
        """Abre una sola variable del GRIB2 con cfgrib."""
        return xr.open_dataset(
            grib,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"shortName": short},
                "indexpath": "",                 # desactiva index on-disk
            },
        ).drop_vars("heightAboveGround", errors="ignore")

    ds = xr.merge([_load(s) for s in ("10u", "10v", "2t", "ssrd")], compat="override")

    # elimina duplicados de la coordenada tiempo (a veces 0 h aparece dos veces)
    time = ds.indexes["time"]
    if time.duplicated().any():
        ds = ds.sel(time=~time.duplicated())

    ds.to_netcdf(nc, encoding={"time": {"calendar": "proleptic_gregorian"}})
    print(f"✓ creado {nc.name}")
