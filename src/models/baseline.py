# src/models/baseline.py
"""
Persistencia 24 h para cada target (benchmark rápido).
Run:
    python -m src.models.baseline --config configs/default.yaml
Genera:
    models/baseline/baseline_persist24.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils.yaml_cfg import load


def main(config: str):
    cfg   = load(config)
    df    = pd.read_parquet(cfg["paths"]["feature_table"])
    test  = df.query("split == 'test'")
    out_d = Path(cfg["paths"]["models"]) / "baseline"
    out_d.mkdir(parents=True, exist_ok=True)

    results = {}
    for tgt in cfg["model"]["targets"]:
        y_true = test[tgt]
        y_pred = y_true.shift(24).reindex_like(y_true)

        valid = y_true.notna() & y_pred.notna()          # filas sin NaN
        if valid.sum() == 0:
            print(f"[{tgt:<8}]  Sin datos válidos para persistencia-24 h")
            continue

        mae  = mean_absolute_error(y_true[valid], y_pred[valid])
        rmse = np.sqrt(mean_squared_error(y_true[valid], y_pred[valid]))
        nmae = mae / y_true[valid].mean()

        print(f"[{tgt:<8}] Persist24h  MAE={mae:,.2f}  RMSE={rmse:,.2f}  nMAE={nmae:,.3f}")

        results[f"{tgt}_obs"]  = y_true
        results[f"{tgt}_pred"] = y_pred

    pd.DataFrame(results).to_csv(out_d / "baseline_persist24.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default="configs/default.yaml")
    main(**vars(p.parse_args()))
