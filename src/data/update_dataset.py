# src/data/update_dataset.py
"""
Incremental updater – añade datos desde initial_dataset.parquet hasta ahora.
"""
from __future__ import annotations

import os
import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv

from src.utils.yaml_cfg import load  # tu cargador de YAML

load_dotenv(find_dotenv())

# IDs ESIOS según default.yaml
IDS_ESIOS = {"wind": 551, "solar_pv": 2044, "demand": 1}


def latest_time(parquet_path: Path) -> pd.Timestamp:
    """Detecta la última hora ya descargada, examinando columna o índice."""
    df = pd.read_parquet(parquet_path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        return ts.max().tz_convert("UTC")
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        return idx.max().tz_convert("UTC")
    if "index" in df.columns:
        ts = pd.to_datetime(df["index"], utc=True)
        return ts.max().tz_convert("UTC")
    raise KeyError("No se encontró ni 'timestamp' ni DatetimeIndex en el parquet")


def fetch_esios(ind: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    token = os.getenv("ESIOS_TOKEN")
    if not token:
        raise EnvironmentError("Falta ESIOS_TOKEN en variables de entorno")
    url = (
        f"https://api.esios.ree.es/indicators/{ind}"
        f"?start_date={start:%Y-%m-%dT%H:%M:%SZ}"
        f"&end_date={end:%Y-%m-%dT%H:%M:%SZ}&time_trunc=hour"
    )
    headers = {
        "x-api-key": token,
        "Accept": "application/json; application/vnd.esios-api-v1+json",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    vals = r.json()["indicator"].get("values", [])
    if not vals:
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.Series(index=idx, dtype="float32", name=str(ind))

    df = pd.DataFrame(vals)
    dt_col = next(c for c in ("datetime_utc", "datetime", "date") if c in df.columns)
    ser = (
        df[[dt_col, "value"]]
          .assign(**{dt_col: lambda d: pd.to_datetime(d[dt_col], utc=True)})
          .set_index(dt_col)["value"]
          .astype("float32")
    )
    return ser


def main(config: str) -> None:
    cfg = load(config)
    parquet_path = Path(cfg["paths"]["processed_dir"]) / "initial_dataset.parquet"

    last = latest_time(parquet_path)
    now = pd.Timestamp.utcnow().floor("h")
    if last + timedelta(hours=1) > now:
        print("Dataset ya al día.")
        return

    start = last + timedelta(hours=1)
    print(f"Actualizando datos ESIOS desde {start} hasta {now}")

    # descarga incremental de ESIOS
    series: list[pd.Series] = []
    for col, ind in IDS_ESIOS.items():
        s = fetch_esios(ind, start, now).rename(col)
        series.append(s)

    df_new = (
        pd.concat(series, axis=1)
          .reset_index()
          .rename(columns={"index": "timestamp"})
    )

    # lee el parquet viejo y asegura columna timestamp
    df_old = pd.read_parquet(parquet_path)
    df_old = df_old.reset_index()
    if "timestamp" not in df_old.columns:
        df_old = df_old.rename(columns={"index": "timestamp"})

    # une, elimina duplicados y ordena
    df_full = pd.concat([df_old, df_new], axis=0, ignore_index=True)
    df_full = (
        df_full
          .drop_duplicates(subset="timestamp", keep="last")
          .sort_values("timestamp")
    )

    # guarda sin índice
    df_full.to_parquet(parquet_path, index=False)
    print(f"✓ initial_dataset.parquet actualizado: ahora {len(df_full)} filas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Ruta al YAML de configuración"
    )
    args = parser.parse_args()
    main(**vars(args))
