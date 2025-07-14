# src/models/train_mlp.py
"""
Entrena un MLPRegressor (scikit-learn) para cada target.
• Entrenamiento rápido sin Optuna  •  escalado z-score
• Guarda modelo y scaler
Run:
    python -m src.models.train_mlp --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.utils.logger import init_logger
from src.utils.yaml_cfg import load

def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)  # sin squared arg
    return math.sqrt(mse)

init_logger()
_L = logging.getLogger(__name__)

def train_one_target(df: pd.DataFrame, feats: list[str], target: str,
                     cfg: dict, out_dir: Path) -> None:
    _L.info("▶ Entrenando MLP para %s", target)

    # split y dropna
    tr  = df.query("split=='train'").dropna(subset=feats + [target])
    val = df.query("split=='val'").dropna(subset=feats + [target])

    # escalado
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(tr[feats]);  ytr = tr[target].values
    Xval= scaler.transform(val[feats]);   yval= val[target].values

    # modelo MLP sencillo
    mlp = MLPRegressor(
        hidden_layer_sizes=(100,100),
        activation="relu",
        solver="adam",
        early_stopping=True,
        n_iter_no_change=10,
        max_iter=200,
        random_state=42,
    )
    mlp.fit(Xtr, ytr)

    # evaluación en validación
    preds = mlp.predict(Xval)
    mae = mean_absolute_error(yval, preds)
    rm  = rmse(yval, preds)
    r2  = None
    try:
        from sklearn.metrics import r2_score
        r2 = r2_score(yval, preds)
    except ImportError:
        pass

    _L.info("Resultados %s → MAE: %.4f  RMSE: %.4f  R2: %s",
           target, mae, rm, f"{r2:.4f}" if r2 is not None else "N/A")

    # guardar artefactos
    tgt_dir = out_dir / target
    tgt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(mlp,    tgt_dir/"model.joblib")
    joblib.dump(scaler, tgt_dir/"scaler.joblib")
    _L.info("✔ Guardado MLP artefactos en %s", tgt_dir)


def main(config: str):
    cfg = load(config)
    df  = pd.read_parquet(cfg["paths"]["feature_table"])

    # seleccionar solo numéricas
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats   = [c for c in numeric if c not in (*cfg["model"]["targets"], "split")]

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(cfg["paths"]["models"]) / f"mlp_{ts}"
    for tgt in cfg["model"]["targets"]:
        train_one_target(df, feats, tgt, cfg, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="ruta YAML")
    main(**vars(ap.parse_args()))
