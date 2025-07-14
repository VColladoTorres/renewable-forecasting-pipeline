# src/analysis/error_breakdown.py
"""
Diagnóstico de error para un experimento XGBoost.

Genera para cada target:
  • <run>/diagnostics/<tgt>_<group>_rmse.csv
  • <run>/diagnostics/<tgt>_<group>_rmse.png

Grupos (--group / -g):
  hour | month | wind_bin
"""
from __future__ import annotations
import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from src.utils.yaml_cfg import load


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    return np.sqrt(mean_squared_error(a[m], b[m])) if m.any() else np.nan


def pred_for_target(run_dir: Path, tgt: str, test_df: pd.DataFrame, feats: list[str]) -> pd.Series:
    model  = joblib.load(run_dir / tgt / "model.joblib")
    scaler = joblib.load(run_dir / tgt / "scaler.joblib")
    X      = scaler.transform(test_df[feats])
    return pd.Series(model.predict(X), index=test_df.index, name="pred")


def _ensure_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert("Europe/Madrid")


def group_key(df: pd.DataFrame, group: str) -> pd.Series:
    if group == "hour":
        hours = _ensure_tz(df.index).hour
        return pd.Series(hours, index=df.index, name="group")
    if group == "month":
        months = _ensure_tz(df.index).month
        return pd.Series(months, index=df.index, name="group")
    if group == "wind_bin":
        if {"u10", "v10"}.issubset(df.columns):
            v = np.hypot(df["u10"], df["v10"])
            bins = [0, 4, 8, 12, 16, 20, 25, np.inf]
            labels = ["0–4", "4–8", "8–12", "12–16", "16–20", "20–25", "25+"]
            cats = pd.cut(v, bins=bins, labels=labels, right=False)
            return pd.Series(cats, index=df.index, name="group")
        warnings.warn("Columnas u10/v10 no encontradas; 'wind_bin' omitido.")
        return pd.Series(np.nan, index=df.index, name="group")
    raise ValueError(f"Grupo no soportado: {group}")


def breakdown(config: str, run: str, group: str) -> None:
    cfg      = load(config)
    run_dir  = Path(cfg["paths"]["models"]) / run
    df       = pd.read_parquet(cfg["paths"]["feature_table"])
    test_df  = df.query("split == 'test'").copy()

    # Solo features numéricas, excluyendo targets y split
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols    = [c for c in numeric_cols if c not in (*cfg["model"]["targets"], "split")]

    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tgt in cfg["model"]["targets"]:
        print(f"→ Diagnóstico {tgt.upper()} por {group}")
        y_true = test_df[tgt]
        y_pred = pred_for_target(run_dir, tgt, test_df, feat_cols)
        gkey   = group_key(test_df, group)

        df_grp = (
            pd.concat({"obs": y_true, "pred": y_pred, "key": gkey}, axis=1)
              .dropna(subset=["key"])
              .groupby("key")
              .apply(lambda x: rmse(x["obs"].values, x["pred"].values))
              .rename("rmse")
        )

        csv_path = out_dir / f"{tgt}_{group}_rmse.csv"
        df_grp.to_csv(csv_path, header=True)

        plt.figure(figsize=(8, 3))
        df_grp.plot.bar()
        plt.ylabel("RMSE")
        plt.xlabel(group)
        plt.title(f"{tgt.upper()} – RMSE por {group}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{tgt}_{group}_rmse.png", dpi=150)
        plt.close()

        print(f"  Guardado {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-r", "--run",    required=True)
    ap.add_argument("-g", "--group",  choices=["hour", "month", "wind_bin"], default="hour")
    args = ap.parse_args()
    breakdown(args.config, args.run, args.group)
