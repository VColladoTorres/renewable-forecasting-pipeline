
# MIT License
# Copyright (c) 2025 MSc Candidate
#
# Permission is hereby granted, free of charge, to any person obtaining a copy …
"""
Evaluate trained model and create diagnostic plot.

Run:
    python -m src.models.evaluate --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import init_logger

init_logger()
_LOGGER = logging.getLogger(__name__)


def main(config: str) -> None:  # noqa: D401
    with open(config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(cfg["paths"]["feature_table"])
    test = df[df["split"] == "test"]

    model = joblib.load(Path(cfg["paths"]["models"]) / "xgb_model.joblib")
    scaler = joblib.load(Path(cfg["paths"]["models"]) / "scaler.joblib")
    feats = [c for c in test.columns if c not in ("mw", "split")]
    X_test = scaler.transform(test[feats])
    y_true = test["mw"].values
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    nmae = mae / y_true.max()
    r2 = r2_score(y_true, y_pred)
    _LOGGER.info("Test MAE %.2f | RMSE %.2f | nMAE %.2f | R² %.3f", mae, rmse, nmae, r2)

    # plot
    plt.figure(figsize=(10, 4))
    plt.plot(test.index, y_true, label="obs")
    plt.plot(test.index, y_pred, label="pred")
    plt.legend()
    plt.title("XGB – observed vs predicted")
    out_png = Path(cfg["paths"]["models"]) / "prediction_vs_obs.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    _LOGGER.info("Plot saved to %s", out_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    main(**vars(parser.parse_args()))
