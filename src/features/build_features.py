# src/features/build_features.py
# MIT © 2025 MSc Candidate
"""
Genera tabla de features:
    • lee dataset.parquet (salida de make_dataset.py)
    • añade retardos / medias móviles
    • añade columna 'split' (train / val / test) según fechas del YAML
    • guarda features.parquet
Ejemplo:
    python -m src.features.build_features \
           --config configs/default.yaml \
           --in  data/processed/dataset.parquet \
           --out data/processed/features.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)


def add_lags_rolls(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Crea retardos y medias móviles definidas en el YAML."""
    lags = cfg["feature_engineering"]["lags"]
    rolls = cfg["feature_engineering"]["rolling_means"]

    for col in ("wind", "solar_pv", "demand", "t2m", "u10", "v10", "ssrd"):
        if col not in df.columns:
            continue
        for l in lags:
            df[f"{col}_lag{l}"] = df[col].shift(l)
        for r in rolls:
            df[f"{col}_roll{r}"] = df[col].rolling(r).mean()
    return df


def add_split_column(df: pd.DataFrame, split_cfg: dict) -> pd.DataFrame:
    """Añade columna 'split' basándose en las fechas del YAML."""
    df["split"] = "test"  # por defecto todo es test
    df.loc[df.index <= split_cfg["train_end"], "split"] = "train"
    df.loc[
        (df.index > split_cfg["train_end"]) & (df.index <= split_cfg["val_end"]),
        "split",
    ] = "val"
    return df


def main(config: str, in_path: str, out_path: str) -> None:  # noqa: D401
    # ---------------- YAML -----------------
    cfg = yaml.safe_load(open(config, encoding="utf-8"))
    split_cfg = cfg["data"]["split_dates"]

    # -------------- DATASET ---------------
    df = pd.read_parquet(in_path)
    _LOGGER.info("Dataset cargado: %s → %s filas", Path(in_path).name, len(df))

    # ---------- FEATURES ENGINEERING -------
    df = add_lags_rolls(df, cfg)

    # ------------- SPLIT COLUMN ------------
    df = add_split_column(df, split_cfg)
    _LOGGER.info("Columna 'split' añadida (train/val/test)")

    # ------------ GUARDAR ------------------
    df.to_parquet(out_path)
    _LOGGER.info("Features guardadas en %s | shape=%s", out_path, df.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Ruta YAML")
    p.add_argument("--in", dest="in_path", required=True, help="dataset.parquet")
    p.add_argument(
        "--out", dest="out_path", required=True, help="Destino features.parquet"
    )
    args = p.parse_args()
    main(**vars(args))
