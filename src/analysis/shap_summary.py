# src/analysis/shap_summary.py
"""
Genera summary_plot de SHAP para cada target y guarda las figuras PNG en
models/<run>/shap/.

Uso (una sola línea):
  python -m src.analysis.shap_summary -c configs/default.yaml -r xgb_YYYYMMDD_HHMM
"""
from __future__ import annotations
import argparse
from pathlib import Path

# Parche para NumPy ≥2.0
from src.utils.numpy_patch import *
import matplotlib
matplotlib.use("Agg")  # backend sin GUI

import matplotlib.pyplot as plt
import joblib
import pandas as pd
import shap
from src.utils.yaml_cfg import load


def main(config: str, run: str) -> None:
    cfg   = load(config)
    df    = pd.read_parquet(cfg["paths"]["feature_table"])
    test  = df.query("split == 'test'")
    # Solo features numéricas, excluyendo targets y split
    numeric = df.select_dtypes(include=[float, int]).columns.tolist()
    feats   = [c for c in numeric if c not in (*cfg["model"]["targets"], "split")]

    out_dir = Path(cfg["paths"]["models"]) / run / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tgt in cfg["model"]["targets"]:
        print(f"→ Generando SHAP summary para {tgt}")
        mdir   = Path(cfg["paths"]["models"]) / run / tgt
        model  = joblib.load(mdir / "model.joblib")
        scaler = joblib.load(mdir / "scaler.joblib")
        X_test = scaler.transform(test[feats])

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test, check_additivity=False)

        shap.summary_plot(
            shap_vals,
            X_test,
            feature_names=feats,
            show=False,
            max_display=20,
            plot_size=(8, 3),
            title=f"{tgt.upper()} – SHAP global",
        )

        fig_path = out_dir / f"{tgt}_shap_summary.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figura guardada: {fig_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-r", "--run",    required=True)
    main(**vars(p.parse_args()))
