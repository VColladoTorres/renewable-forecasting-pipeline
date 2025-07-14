#!/usr/bin/env python
"""
Valida los modelos sobre el *hold-out* (split == 'test').

• Lee features.parquet con columna 'split'.
• Carga modelos + escaladores + features_used.pkl desde --model-dir/<target>/.
• Calcula MAE, RMSE, MAPE, R² para cada target y los guarda en outputs/metrics/.
"""

from pathlib import Path
from datetime import datetime
import argparse, json, joblib, logging, sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ─────────── logging ───────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)

# ─────────── métricas ───────────
def rmse(y, yhat):
    return mean_squared_error(y, yhat, squared=False)

def mape(y, yhat):
    mask = y != 0
    return (np.abs((yhat[mask] - y[mask]) / y[mask])).mean() * 100

# ─────────── validación ───────────
def validate(df: pd.DataFrame, model_root: Path, targets: list[str]) -> dict:
    results = {}
    for tgt in targets:
        tgt_dir = model_root / tgt
        try:
            scaler = joblib.load(tgt_dir / "scaler.joblib")
            model  = joblib.load(tgt_dir / "model.joblib")
            feats  = joblib.load(tgt_dir / "features_used.pkl")
        except FileNotFoundError:
            LOG.error("Faltan artefactos para %s en %s", tgt, tgt_dir)
            continue

        missing = set(feats) - set(df.columns)
        if missing:
            LOG.error("Features faltantes para %s: %s", tgt, missing)
            continue

        X = scaler.transform(df[feats])
        y = df[tgt].values
        yhat = model.predict(X)

        results[tgt] = {
            "MAE":  float(mean_absolute_error(y, yhat)),
            "RMSE": float(rmse(y, yhat)),
            "MAPE_pct": float(mape(y, yhat)),
            "R2":  float(r2_score(y, yhat)),
            "n":   int(len(y)),
        }
        LOG.info("%s  MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.3f",
                 tgt.upper(), results[tgt]["MAE"],
                 results[tgt]["RMSE"],
                 results[tgt]["MAPE_pct"],
                 results[tgt]["R2"])
    return results

# ─────────── CLI ───────────
def main(config: str, model_dir: str):
    import yaml
    cfg = yaml.safe_load(open(config, encoding="utf-8"))

    feat_path = Path(cfg["paths"]["feature_table"])
    df = pd.read_parquet(feat_path)
    if "split" not in df.columns:
        sys.exit("❌ La tabla de features no contiene columna 'split'.")

    test_df = df[df["split"] == "test"].dropna()
    if test_df.empty:
        sys.exit("❌ No hay filas con split=='test'.")

    targets = cfg["model"]["targets"]
    metrics = validate(test_df, Path(model_dir), targets)

    # guardar JSON
    out_dir = Path("outputs") / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_json = out_dir / f"validation_{ts}.json"
    json.dump(metrics, out_json.open("w"), indent=2)
    LOG.info("✅ Métricas guardadas en %s", out_json)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="YAML de configuración")
    ap.add_argument("-m", "--model-dir", default="models/current",
                    help="Directorio con modelos por target")
    main(**vars(ap.parse_args()))
