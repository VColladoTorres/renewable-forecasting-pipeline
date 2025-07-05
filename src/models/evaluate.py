# src/models/evaluate.py
"""
Evalúa los modelos entrenados (uno por target) y genera:
  ▸ métricas MAE / RMSE / R²
  ▸ gráfico obs-vs-pred en PNG

Uso:
    python -m src.models.evaluate \
           --config configs/default.yaml \
           --run    xgb_YYYYMMDD_HHMM
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

# ── ciencia de datos ──────────────────────────────────────────────────
import joblib
import matplotlib

matplotlib.use("Agg")                     # backend sin GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ── utilidades propias ────────────────────────────────────────────────
from src.utils.logger import init_logger
from src.utils.yaml_cfg import load

init_logger()
_LOG = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
def eval_target(run_dir: Path, df_test: pd.DataFrame, target: str, feats: list[str]) -> None:
    """Carga modelo + scaler, calcula métricas y guarda gráfico."""
    model = joblib.load(run_dir / target / "model.joblib")
    scaler = joblib.load(run_dir / target / "scaler.joblib")

    # ── elimina filas con NaN en target o features ────────────────────
    subset = feats + [target]
    df_ok = df_test.dropna(subset=subset)

    X = scaler.transform(df_ok[feats])
    y_true = df_ok[target].values.astype(float)
    y_pred = model.predict(X).astype(float)

    # métricas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5   # √MSE
    r2 = r2_score(y_true, y_pred)

    _LOG.info("%s → MAE %.2f | RMSE %.2f | R² %.3f",
              target.upper(), mae, rmse, r2)

    # ── gráfico obs vs pred ───────────────────────────────────────────
    plt.figure(figsize=(10, 3))
    plt.plot(df_ok.index, y_true, lw=1, label="obs")
    plt.plot(df_ok.index, y_pred, lw=1, label="pred")
    plt.legend()
    plt.title(f"{target.upper()} – obs vs pred")
    plt.tight_layout()
    plt.savefig(run_dir / target / "pred_vs_obs.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────
def main(config: str, run: str) -> None:
    cfg = load(config)
    df = pd.read_parquet(cfg["paths"]["feature_table"])
    test = df.query("split == 'test'")

    # mismas columnas predictoras usadas en entrenamiento
    feats = [c for c in df.columns if c not in (*cfg["model"]["targets"], "split")]

    run_dir = Path(cfg["paths"]["models"]) / run
    if not run_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de resultados: {run_dir}")

    for tgt in cfg["model"]["targets"]:
        eval_target(run_dir, test, tgt, feats)


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="ruta a YAML de configuración")
    parser.add_argument("--run", "-r", required=True,
                        help="subcarpeta dentro de models/ creada por train_xgb")
    main(**vars(parser.parse_args()))
