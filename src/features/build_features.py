#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generador robusto de *features* para entrenamiento y predicción.

Novedades
─────────
• Variables cíclicas hora / día-año, dirección de viento, log_wind_speed.
• Interpolación bidireccional solo en predictores exógenos.
• Mantiene filas siempre que los *targets* no sean NaN.
"""
from __future__ import annotations
import argparse, logging, joblib
from math import atan2, pi
from pathlib import Path
from typing import Iterable, Sequence

from src.utils.capacity import build_capacity_series

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pvlib.solarposition import get_solarposition

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger(__name__)

DS_VAR_MAP = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature":          "t2m",
    "surface_solar_radiation_downwards": "ssrd",
}

# ───── helpers ───────────────────────────────────────────────────────────────
def _derive_physical(df: pd.DataFrame) -> pd.DataFrame:
    # módulo y dirección del viento 10 m
    if {"u10", "v10"}.issubset(df.columns):
        df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
        df["wind_dir"]   = (np.degrees(np.arctan2(df["u10"], df["v10"])) + 360) % 360
        df["log_wind_speed"] = np.log1p(df["wind_speed"])
    return df

def _add_solar_geom(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    loc = df.index.tz_convert("Europe/Madrid")
    sp  = get_solarposition(loc, latitude=lat, longitude=lon)
    df["solar_zenith"]    = sp["zenith"].values
    df["solar_elevation"] = sp["elevation"].values
    df["is_day"]          = (df["solar_elevation"] > 0).astype("int8")
    return df

def _add_time_cyc(df: pd.DataFrame) -> pd.DataFrame:
    hr   = df.index.hour
    doy  = df.index.dayofyear
    df["hour_sin"] = np.sin(2*pi*hr/24)
    df["hour_cos"] = np.cos(2*pi*hr/24)
    df["doy_sin"]  = np.sin(2*pi*doy/365)
    df["doy_cos"]  = np.cos(2*pi*doy/365)
    return df

def _add_lags_rolls(df: pd.DataFrame, cols: Iterable[str],
                    lags: Sequence[int], rolls: Sequence[int]) -> pd.DataFrame:
    for c in cols:
        if c not in df or not np.issubdtype(df[c].dtype, np.number):
            continue
        for k in lags:
            df[f"{c}_lag{k}"] = df[c].shift(k)
        for r in rolls:
            df[f"{c}_roll{r}"] = df[c].rolling(r).mean()
    return df

def _add_split(df: pd.DataFrame, split_cfg: dict) -> pd.DataFrame:
    tr_end = pd.to_datetime(split_cfg["train_end"], utc=True).tz_convert("Europe/Madrid")
    va_end = pd.to_datetime(split_cfg["val_end"],   utc=True).tz_convert("Europe/Madrid")
    df["split"] = np.where(df.index <= tr_end, "train",
                    np.where(df.index <= va_end, "val", "test"))
    return df

def engineer_feature_matrix(ds: xr.Dataset, cfg: dict, *, future=True) -> pd.DataFrame:
    rename = {o: n for o, n in DS_VAR_MAP.items() if o in ds}
    ds = ds.rename(rename)

    # 1) eje temporal único
    if "valid_time" in ds.coords:
        ds = ds.swap_dims({"step": "valid_time"})
    else:
        vt = (ds["time"] + ds["step"]).data      # usar .data
        ds = (ds.assign_coords(valid_time=("step", vt))
                .swap_dims({"step": "valid_time"})
                .drop_vars(["time", "step"]))

    # 2) seleccionar grid-point representativo (≈ 40 N, 4 W)
    ds = ds.sel(latitude=40.0, longitude=-4.0, method="nearest")

    # 3) DataFrame base
    df = ds.to_dataframe()
    df.index = (pd.to_datetime(df.index)
                  .tz_localize("UTC")
                  .tz_convert("Europe/Madrid"))

    # 4) derivadas
    df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
    df = _add_solar_geom(df, cfg["site"]["lat"], cfg["site"]["lon"])
    df = _add_time_cyc(df)
    fe = cfg["feature_engineering"]
    df = _add_lags_rolls(df, fe["lag_roll_cols"], fe["lags"], fe["rolling_means"])

    # 5) fase entrenamiento → normalizar a CF
    if not future:
        from src.utils.capacity import build_capacity_series
        caps = build_capacity_series(cfg["capacity_timeline"], df.index)
        for tgt in cfg["model"]["targets"]:
            df[tgt] = df[tgt] / caps[tgt]
        df = df.dropna(subset=cfg["model"]["targets"])
        num = df.select_dtypes(np.number).columns
        df[num] = df[num].interpolate(limit_direction="both")
    return df

# ───── construcción offline ─────────────────────────────────────────────────
def _main(cfg_file: str, in_parquet: str, out_parquet: str) -> None:
    cfg = yaml.safe_load(open(cfg_file, encoding="utf-8"))
    df  = pd.read_parquet(in_parquet)
    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df = df.tz_convert("Europe/Madrid").sort_index()

    missing = [t for t in cfg["model"]["targets"] if t not in df]
    if missing:
        raise RuntimeError(f"Faltan targets {missing} en {in_parquet}")

    df = _derive_physical(df)
    df = _add_solar_geom(df, cfg["site"]["lat"], cfg["site"]["lon"])
    df = _add_time_cyc(df)

    exog = [c for c in df.select_dtypes(np.number).columns
            if c not in cfg["model"]["targets"]]
    fe = cfg["feature_engineering"]
    df = _add_lags_rolls(df, exog, fe["lags"], fe["rolling_means"])

    df = df.dropna(subset=cfg["model"]["targets"])
    num = df.select_dtypes(np.number).columns
    df[num] = df[num].interpolate(limit_direction="both")

    # … después de montar `df`
    caps = build_capacity_series(cfg["capacity_timeline"], df.index)
    for tgt in cfg["model"]["targets"]:
        df[tgt] = df[tgt] / caps[tgt]

    df = _add_split(df, cfg["data"]["split_dates"])
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    joblib.dump(num.tolist(), Path(out_parquet).with_suffix(".pkl"))
    LOG.info("✅ Features → %s  shape=%s", out_parquet, df.shape)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", dest="cfg_file", required=True)
    p.add_argument("--in",  dest="in_parquet",  required=True)
    p.add_argument("--out", dest="out_parquet", required=True)
    _main(**vars(p.parse_args()))
