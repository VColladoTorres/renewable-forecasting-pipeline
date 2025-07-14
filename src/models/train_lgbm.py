# src/models/train_lgbm.py
"""
Entrena LightGBM para cada target definido en configs/*.yaml
• Optuna para hallar hiper-parámetros  •  escalado z-score
• Guarda modelo, scaler y best_params
Run:
    python -m src.models.train_lgbm --config configs/default.yaml
"""
from __future__ import annotations

import argparse, json, logging, os
from datetime import datetime
from math import sqrt
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.utils.logger import init_logger
from src.utils.yaml_cfg import load

def rmse(y_true, y_pred):
    # mean_squared_error sin squared arg, luego sqrt
    mse = mean_squared_error(y_true, y_pred)
    return sqrt(mse)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
init_logger()
_L = logging.getLogger(__name__)

def _objective(trial, Xtr, ytr, Xval, yval):
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "n_jobs": os.cpu_count(),
        "random_state": RANDOM_SEED,
        "num_leaves": trial.suggest_int("num_leaves", 31, 512),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(Xtr, ytr)
    preds = model.predict(Xval)
    return rmse(yval, preds)

def train_one_target(df: pd.DataFrame, feats: list[str], target: str,
                     cfg: dict, out_dir: Path) -> None:
    _L.info("▶ Entrenando LightGBM para %s", target)
    tr  = df.query("split=='train'").dropna(subset=feats + [target])
    val = df.query("split=='val'").dropna(subset=feats + [target])

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(tr[feats]);  ytr = tr[target].values
    Xval= scaler.transform(val[feats]);   yval = val[target].values

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    study.optimize(lambda t: _objective(t, Xtr, ytr, Xval, yval),
                   n_trials=cfg["optuna"]["n_trials"])

    best = study.best_params | {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "n_jobs": os.cpu_count(),
        "random_state": RANDOM_SEED,
    }
    booster = lgb.LGBMRegressor(**best)
    booster.fit(
        np.vstack([Xtr, Xval]),
        np.hstack([ytr, yval])
    )

    tgt_dir = out_dir / target
    tgt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(booster,    tgt_dir/"model.joblib")
    joblib.dump(scaler,     tgt_dir/"scaler.joblib")
    json.dump(best, (tgt_dir/"best_params.json").open("w"))
    _L.info("✔ Guardado LightGBM artefactos en %s", tgt_dir)

def main(config: str):
    cfg = load(config)
    df  = pd.read_parquet(cfg["paths"]["feature_table"])
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats   = [c for c in numeric if c not in (*cfg["model"]["targets"], "split")]

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(cfg["paths"]["models"]) / f"lgbm_{ts}"
    for tgt in cfg["model"]["targets"]:
        train_one_target(df, feats, tgt, cfg, out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="ruta YAML")
    main(**vars(ap.parse_args()))
