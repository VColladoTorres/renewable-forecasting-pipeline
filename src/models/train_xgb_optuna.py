#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrena un XGBoost por objetivo usando solo variables exógenas
(NWP + rasgos horarios) cuyo target ya está en *capacity factor* (0-1).

• Búsqueda de hiper-parámetros con Optuna (nMAE).
• Validación temporal con TimeSeriesSplit.
"""
from __future__ import annotations
import argparse, logging, yaml, joblib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)


# ───── utils ────────────────────────────────────────────────────────────────
def load_cfg(p: str) -> dict:
    return yaml.safe_load(open(p, encoding="utf-8"))


# ───── Optuna objective ─────────────────────────────────────────────────────
def make_objective(X: pd.DataFrame, y: pd.Series, splits: int):
    cv = TimeSeriesSplit(n_splits=splits, test_size=len(X) // (splits + 1))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
        }
        scores = []
        for tr_idx, val_idx in cv.split(X):
            pipe = Pipeline([
                ("sc", StandardScaler()),
                ("xgb", XGBRegressor(**params)),
            ])
            pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = pipe.predict(X.iloc[val_idx])
            scores.append(mean_absolute_error(y.iloc[val_idx], pred))
        return np.mean(scores)
    return objective


def train_final_model(X: pd.DataFrame, y: pd.Series, params: dict) -> Pipeline:
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("xgb", XGBRegressor(**params)),
    ])
    pipe.fit(X, y)
    return pipe


# ───── main ────────────────────────────────────────────────────────────────
def main(cfg_path: str) -> None:
    cfg = load_cfg(cfg_path)
    df = pd.read_parquet(cfg["paths"]["feature_table"])
    numf = pd.read_pickle(cfg["paths"]["feature_table"].replace(".parquet", ".pkl"))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    out_root = Path(cfg["paths"]["models"]) / f"xgb_optuna_{ts}"

    for tgt in cfg["model"]["targets"]:
        LOG.info("⚙️  Optimizando «%s» …", tgt)
        y = df[tgt]                     # ya es capacity factor 0-1
        X = df[numf]

        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(X, y, splits=5),
                       n_trials=cfg["optuna"]["n_trials"])
        best = study.best_params
        LOG.info("→ Mejor parámetro nMAE=%.4f  %s", study.best_value, best)

        model = train_final_model(X, y, best)

        # métricas en periodo de validación (cfg.data.split_dates.val_end)
        val_mask = df["split"] == "val"
        pred = model.predict(X[val_mask])
        mae = mean_absolute_error(y[val_mask], pred)
        rmse = np.sqrt(mean_squared_error(y[val_mask], pred))
        LOG.info("✓ %s | nMAE %.4f | nRMSE %.4f", tgt, mae, rmse)

        dst = out_root / tgt
        dst.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, dst / f"{tgt}.joblib")
        joblib.dump(numf, dst / "features_used.pkl")
        yaml.safe_dump(best, open(dst / "best_params.yaml", "w"))

    LOG.info("🏁 Modelos guardados en %s", out_root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    main(p.parse_args().config)
