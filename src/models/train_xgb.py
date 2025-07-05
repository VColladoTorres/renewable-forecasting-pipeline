# src/models/train_xgb.py
"""
Entrena XGBoost para cada target definido en configs/*.yaml
▪ Optuna hiper-parámetros ▪ escalado z-score ▪ salida con time-stamp

Run:
    python -m src.models.train_xgb --config configs/default.yaml
"""
from __future__ import annotations

# ───────────────────── import básicos ────────────────────────────────
import argparse, json, logging, os
from datetime import datetime
from math import sqrt
from pathlib import Path

import joblib, numpy as np, optuna, pandas as pd, xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.utils.logger import init_logger
from src.utils.yaml_cfg import load        # pequeña utilidad para YAML

# ───────────────────── helpers ───────────────────────────────────────
def rmse(y_true, y_pred):
    """RMSE compatible con cualquier versión de scikit-learn."""
    try:                                    # versiones ≥ 0.22
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:                       # versiones antiguas
        return sqrt(mean_squared_error(y_true, y_pred))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
init_logger()
_L = logging.getLogger(__name__)

# ───────────────────── Optuna objective ─────────────────────────────
def _objective(trial, Xtr, ytr, Xval, yval):
    p = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_estimators":     trial.suggest_int("n_estimators", 400, 2000),
        "learning_rate":    trial.suggest_float("eta", 5e-3, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 12),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda":       trial.suggest_float("lambda", 1e-2, 10.0, log=True),
        "reg_alpha":        trial.suggest_float("alpha",  1e-3, 10.0, log=True),
        "random_state": RANDOM_SEED,
        "n_jobs": os.cpu_count(),
    }
    mdl = xgb.XGBRegressor(**p)
    mdl.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    pred = mdl.predict(Xval)
    return rmse(yval, pred)

# ───────────────────── entrenamiento por target ─────────────────────
def train_one_target(df: pd.DataFrame, feats: list[str], target: str,
                     cfg: dict, out_dir: Path) -> None:
    _L.info("▶ Entrenando modelo para **%s**", target)

    tr  = df.query("split == 'train'")
    val = df.query("split == 'val'")

    scaler = StandardScaler()
    Xtr  = scaler.fit_transform(tr[feats]);  ytr  = tr[target].values
    Xval = scaler.transform(val[feats]);     yval = val[target].values

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(lambda t: _objective(t, Xtr, ytr, Xval, yval),
                   n_trials=cfg["optuna"]["n_trials"])

    best_params = study.best_params | {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": os.cpu_count(),
    }

    final = xgb.XGBRegressor(**best_params)
    final.fit(np.vstack([Xtr, Xval]), np.hstack([ytr, yval]))

    # ── persistencia ────────────────────────────────────────────────
    (out_dir / target).mkdir(parents=True, exist_ok=True)
    joblib.dump(final,  out_dir / target / "model.joblib")
    joblib.dump(scaler, out_dir / target / "scaler.joblib")
    json.dump(best_params, (out_dir / target / "best_params.json").open("w"))
    _L.info("✅ Artefactos guardados en %s", out_dir / target)

# ───────────────────── main ─────────────────────────────────────────
def main(config: str):
    cfg = load(config)
    df  = pd.read_parquet(cfg["paths"]["feature_table"])

    if "model" not in cfg or "targets" not in cfg["model"]:
        raise KeyError("El YAML debe incluir la clave model: targets: [...]")

    feats = [c for c in df.columns if c not in (*cfg["model"]["targets"], "split")]

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(cfg["paths"]["models"]) / f"xgb_{ts}"

    for tgt in cfg["model"]["targets"]:
        train_one_target(df, feats, tgt, cfg, out_dir)

# ───────────────────── CLI ──────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    main(**vars(ap.parse_args()))
