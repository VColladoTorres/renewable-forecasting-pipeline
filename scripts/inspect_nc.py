#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inspección detallada de un HRES Open-Data (.nc o .grib2):

$ python scripts/inspect_nc.py <path>

• Si engine='netcdf4' falla, abre cada shortName por separado con cfgrib
  y evita el conflicto heightAboveGround.
• Muestra resumen de variables, dimensiones y atributos.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

SHORTNAMES = ("10u", "10v", "2t", "ssrd")


def open_single_var(path: Path, short: str) -> xr.Dataset:
    """Carga sólo la variable `short` con cfgrib y elimina heightAboveGround."""
    return xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"shortName": short},
            "indexpath": "",          # no *.idx en disco
        },
    ).drop_vars("heightAboveGround", errors="ignore")


def open_nwp(path: Path) -> xr.Dataset:
    """Intenta netcdf4; si falla fusiona cada shortName vía cfgrib."""
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except Exception as e_nc:                        # noqa: BLE001
        print(f"[netcdf4] falló: {e_nc}  → cfgrib *por variable* …")
        parts = [open_single_var(path, s) for s in SHORTNAMES]
        return xr.merge(parts, compat="override")


def build_summary(ds: xr.Dataset) -> pd.DataFrame:
    rows = []
    for v in ds.data_vars:
        dims = " × ".join(f"{d}={ds.dims[d]}" for d in ds[v].dims)
        attrs = {k: ds[v].attrs.get(k) for k in ("units", "long_name", "short_name") if k in ds[v].attrs}
        rows.append({"variable": v, "dims": dims, "dtype": str(ds[v].dtype), **attrs})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="ruta al ifs_hres_*.nc / *.grib2")
    args = parser.parse_args()

    path = Path(args.file)
    ds = open_nwp(path)
    summary = build_summary(ds)

    pd.set_option("display.max_rows", None)
    print("\n─ VARIABLES ──────────────────────────────────────────")
    print(summary.to_string(index=False))
    print("\n─ COORDENADAS ────────────────────────────────────────")
    for c in ds.coords:
        print(f"{c:12s} dims={ds[c].dims} shape={ds[c].shape} attrs={list(ds[c].attrs.keys())}")


if __name__ == "__main__":
    main()
