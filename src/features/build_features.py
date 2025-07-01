
# MIT License
# Copyright (c) 2025 MSc Candidate
#
# Permission is hereby granted, free of charge, to any person obtaining a copy …
"""
Create lag/rolling features and temporal splits.

Run:
    python -m src.features.build_features --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.utils.logger import init_logger

init_logger()
_LOGGER = logging.getLogger(__name__)


def main(config: str) -> None:  # noqa: D401
    with open(config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "dataset.parquet")

    # ---------------- feature engineering ---------------- #
    lags = cfg["feature_engineering"]["lags"]
    rolls = cfg["feature_engineering"]["rolling_means"]

    for col in ("wind", "solar_pv"):
        for k in lags:
            df[f"{col}_lag{k}"] = df[col].shift(k)
        for w in rolls:
            df[f"{col}_roll{w}"] = df[col].rolling(w).mean()

    # simple theoretical wind power from 100 m speed: P ~ v³
    df["wind_power_theory"] = (df["wind_speed_100m"] ** 3).clip(upper=30_000)

    # drop rows with NA introduced by lags
    df.dropna(inplace=True)

    # ---------------- splits ---------------- #
    train_end = pd.Timestamp(cfg["data"]["split"]["train_end"])
    val_end = pd.Timestamp(cfg["data"]["split"]["val_end"])

    df["split"] = np.where(
        df.index <= train_end,
        "train",
        np.where(df.index <= val_end, "val", "test"),
    )

    # save
    out_path = Path(cfg["paths"]["feature_table"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    _LOGGER.info("Features ready → %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features table.")
    parser.add_argument("--config", "-c", required=True)
    main(**vars(parser.parse_args()))
