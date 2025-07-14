#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrena modelos XGBoost para cada *target* definido en el YAML.

Salida:
    models/xgb_<UTC-timestamp>/{wind,solar_pv}/
        ├─ <target>.joblib          ← Pipeline (Scaler + XGBRegressor)
        ├─ features_used.pkl        ← lista columnas numéricas de entrada

Dependencias:
    pip install xgboost scikit-learn pandas numpy joblib pyyaml
"""
from __future__ import annotations
import argparse, logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)


# ─────────────────── Utilidades ------------------------------------------------
def load_cfg(path: str) -> dict:
    return yaml.safe_load(open(path, encoding="utf-8"))


def save_artifacts(pipe: Pipeline, feats: list[str], dst: Path, tgt: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe,  dst / f"{tgt}.joblib")
    joblib.dump(feats, dst / "features_used.pkl")


def _ensure_splits(df: pd.DataFrame, tgt: str) -> pd.DataFrame:
    """
    Garantiza que existan filas `split=="train"` y `split=="val"`.
    Si la columna 'split' no existe —o la sección 'train' queda vacía—
    genera un *hold-out* 80 / 20 en orden temporal.
    """
    if "split" not in df.columns or df[df["split"] == "train"].empty:
        LOG.warning("Creando splits 80/20 automáticamente para «%s».", tgt)
        df = df.sort_index().copy()
        train_idx, val_idx = train_test_split(
            df.index, test_size=0.2, shuffle=False
        )
        df["split"] = "train"
        df.loc[val_idx, "split"] = "val"
    return df


def train_one(df: pd.DataFrame, tgt: str, cfg: dict, out_dir: Path) -> None:
    """Entrena y guarda el modelo para un *target*; compatible con scikit-learn <0.22."""
    if tgt not in df.columns or df[tgt].dropna().empty:
        LOG.error("❌  La columna «%s» está vacía o ausente → omitida.", tgt)
        return

    df = _ensure_splits(df.dropna(subset=[tgt]), tgt)
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]

    feats = [c for c in df.select_dtypes(np.number).columns
             if c not in cfg["model"]["targets"]]
    if not feats or train.empty or val.empty:
        LOG.error("❌  Split o features vacíos para «%s» → omitido.", tgt)
        return

    Xtr, ytr = train[feats], train[tgt]
    Xva, yva = val[feats],   val[tgt]

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("xgb",   XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )),
    ])
    pipe.fit(Xtr, ytr)

    pred  = pipe.predict(Xva)
    mae   = mean_absolute_error(yva, pred)
    rmse  = np.sqrt(mean_squared_error(yva, pred))   # ← sin squared=False
    LOG.info("✓  %s | MAE %.3f | RMSE %.3f | train=%d | val=%d",
             tgt, mae, rmse, len(Xtr), len(Xva))

    save_artifacts(pipe, feats, out_dir / tgt, tgt)


# ─────────────────── Main ------------------------------------------------------
def main(cfg_path: str) -> None:
    cfg = load_cfg(cfg_path)
    df  = pd.read_parquet(cfg["paths"]["feature_table"])

    ts  = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    out = Path(cfg["paths"]["models"]) / f"xgb_{ts}"

    for tgt in cfg["model"]["targets"]:
        LOG.info("Entrenando modelo para «%s» …", tgt)
        train_one(df.copy(), tgt, cfg, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="configs/default.yaml")
    main(parser.parse_args().config)
