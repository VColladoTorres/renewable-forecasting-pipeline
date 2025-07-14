# src/analysis/compare_runs.py
"""
Compara un experimento XGBoost (u otro) frente a la persistencia-24h.
Genera:
  • compare_baseline.csv — métricas y % de mejora
  • improvement.png      — barras con ΔRMSE (%)
Uso:
  python -m src.analysis.compare_runs -c configs/default.yaml -r xgb_YYYYMMDD_HHMM
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from src.utils.yaml_cfg import load

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float, float]:
    valid = y_true.notna() & y_pred.notna()
    y_t, y_p = y_true[valid], y_pred[valid]
    mae  = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    nmae = mae / y_t.mean()
    return mae, rmse, nmae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--run", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    df = pd.read_parquet(cfg["paths"]["feature_table"])
    test = df.query("split=='test'")

    # definir features numéricas excluyendo targets y split
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric if c not in (*cfg["model"]["targets"], "split")]

    # cargar línea base
    base_csv = Path(cfg["paths"]["models"]) / "baseline" / "baseline_persist24.csv"
    base_df = pd.read_csv(base_csv, index_col=0, parse_dates=True)

    rows = []
    for tgt in cfg["model"]["targets"]:
        y_true = test[tgt]
        y_pred_b = base_df[f"{tgt}_pred"].reindex_like(y_true)
        mae_b, rmse_b, nmae_b = compute_metrics(y_true, y_pred_b)

        model_dir = Path(cfg["paths"]["models"]) / args.run / tgt
        model  = joblib.load(model_dir / "model.joblib")
        scaler = joblib.load(model_dir / "scaler.joblib")
        X = scaler.transform(test[feats])
        y_pred_m = pd.Series(model.predict(X), index=test.index)
        mae_m, rmse_m, nmae_m = compute_metrics(y_true, y_pred_m)

        rows.append({
            "target": tgt,
            "mae_base": mae_b, "mae_model": mae_m,
            "rmse_base": rmse_b, "rmse_model": rmse_m,
            "nmae_base": nmae_b, "nmae_model": nmae_m
        })

    res = pd.DataFrame(rows)
    res["mae_improv_%"]  = 100*(res["mae_base"] - res["mae_model"])/res["mae_base"]
    res["rmse_improv_%"] = 100*(res["rmse_base"] - res["rmse_model"])/res["rmse_base"]

    out_dir = Path(cfg["paths"]["models"]) / args.run
    res.to_csv(out_dir / "compare_baseline.csv", index=False)

    plt.figure(figsize=(6,3))
    plt.bar(res["target"], res["rmse_improv_%"])
    plt.ylabel("ΔRMSE vs persistencia (%)")
    plt.title("Mejora relativa del modelo")
    plt.tight_layout()
    plt.savefig(out_dir / "improvement.png", dpi=150)
    plt.close()

    print("\nComparativa vs persistencia-24h:\n")
    print(res.to_string(index=False))
