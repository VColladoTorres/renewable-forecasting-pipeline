 
# MIT License
# Copyright (c) 2025 MSc Candidate
#
# Permission is hereby granted, free of charge, to any person obtaining a copy …
"""
ETL script: downloads ESIOS + ERA5, aligns hourly and persists parquet.

Run:
    python -m src.data.make_dataset --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm

from src.data.cds_client import CDSClient
from src.data.esios_client import EsiosClient
from src.utils.logger import init_logger

init_logger()
_LOGGER = logging.getLogger(__name__)


def _parse_dates(cfg: dict) -> tuple[datetime, datetime]:
    start = pd.Timestamp(cfg["data"]["start_date"]).tz_convert("UTC")
    end   = pd.Timestamp(cfg["data"]["end_date"]).tz_convert("UTC")
    return start.to_pydatetime(), end.to_pydatetime()


def main(config: str) -> None:  # noqa: D401
    with open(config) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    start, end = _parse_dates(cfg)

    # ------------ download ESIOS series ------------ #
    esios = EsiosClient()
    dfs = []
    for tech in ("wind", "solar_pv"):
        _LOGGER.info("Download %s generation…", tech)
        dfs.append(esios.fetch_series(start, end, technology=tech).rename(columns={"mw": tech}))
    demand = esios.fetch_series(start, end, technology="demand").rename(columns={"mw": "demand"})
    df_gen = pd.concat(dfs + [demand], axis=1)

    # ------------ download ERA5-Land --------------- #
    cds = CDSClient(raw_dir / "era5")
    nc_path = cds.download(cfg["variables"]["era5"], start, end)
    _LOGGER.info("Open ERA5 dataset %s", nc_path.name)
    ds = xr.open_dataset(nc_path)  # lazily read

    # Spatial average over Spain (bounding box as rough proxy)
    spain = ds.sel(latitude=slice(44, 36), longitude=slice(-10, 5)).mean(
        ["latitude", "longitude"]
    )
    df_met = spain.to_dataframe()
    df_met.index = df_met.index.tz_localize(timezone.utc).tz_convert("Europe/Madrid")

    # ------------ merge & persist ------------------ #
    df = df_gen.join(df_met, how="inner")
    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "dataset.parquet")
    _LOGGER.info("Saved merged dataset with %d rows", len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build raw dataset.")
    parser.add_argument("--config", "-c", required=True, help="YAML config path.")
    main(**vars(parser.parse_args()))
