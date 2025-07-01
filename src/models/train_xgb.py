
# MIT License
# Copyright (c) 2025 MSc Candidate
#
# Permission is hereby granted, free of charge, to any person obtaining a copy …
"""
Train XGBoost with Optuna.

Run:
    python -m src.models.train_xgb --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.utils.logger import init_logger

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
init_logger()
_LOGGER = logging.getLogger(__name__)


def _objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("eta", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("lambda", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "random_state": RANDOM_SEED,
        "n_jobs": os.cpu_count(),
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, pred, squared=False)
    return rmse


def main(config: str) -> None:  # noqa: D401
    with open(config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(cfg["paths"]["feature_table"])
    target = cfg["data"]["target"]
    feats = [c for c in df.columns if c not in (target, "split")]

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feats])
    y_train = train[target]
    X_val = scaler.transform(val[feats])
    y_val = val[target]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(lambda t: _objective(t, X_train, y_train, X_val, y_val), n_trials=cfg["optuna"]["n_trials"])

    best = study.best_params | {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": os.cpu_count(),
    }
    model = xgb.XGBRegressor(**best)
    model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

    out_dir = Path(cfg["paths"]["models"])
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "xgb_model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    json.dump(best, (out_dir / "best_params.json").open("w"))

    _LOGGER.info("Model + artefacts saved to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    main(**vars(parser.parse_args()))
