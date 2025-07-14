# src/data/describe_dataset.py
"""
Visualiza un resumen rápido del dataset procesado.

Uso
────
python -m src.data.describe_dataset -c configs/default.yaml [-n 10]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import indent

import pandas as pd
import yaml


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config: str, n_rows: int = 10) -> None:
    cfg = load_cfg(config)
    parq = Path(cfg["paths"]["processed_dir"]) / "dataset.parquet"

    if not parq.exists():
        raise FileNotFoundError(f"No se encontró {parq.resolve()}")

    df = pd.read_parquet(parq)

    # Asegura que 'timestamp' es datetime-UTC si está presente
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # ──────────── salida ────────────
    print("Resumen general".center(80, "─"))
    print(f"Filas totales : {len(df):,}")
    print(f"Columnas      : {df.shape[1]} → {', '.join(df.columns)}")
    if "timestamp" in df.columns:
        print(f"Rango fechas  : {df['timestamp'].min()}  →  {df['timestamp'].max()}")

    print("\nValores nulos por columna".center(80, "─"))
    print(df.isna().sum().to_string())

    print(f"\nPrimeras {n_rows} filas".center(80, "─"))
    print(df.head(n_rows).to_string(index=False))

    print("\nEstadísticos básicos (numéricos)".center(80, "─"))
    # Compatibilidad pandas 1.x / 2.x: datetime_is_numeric sólo existe en 1.x
    try:
        stats = df.describe(datetime_is_numeric=True)
    except TypeError:
        stats = df.select_dtypes("number").describe()
    print(indent(stats.to_string(), "  "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Ruta del YAML.")
    ap.add_argument("-n", "--n_rows", type=int, default=10, help="Filas a mostrar.")
    main(**vars(ap.parse_args()))
