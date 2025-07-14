# src/analysis/shap_local.py
"""
Crea explicaciones locales (force_plot) de SHAP
para los N peores errores de un target.

Guarda HTML en models/<run>/<target>/shap_local/
Uso (una línea):
  python -m src.analysis.shap_local -c configs/default.yaml -r xgb_YYYYMMDD_HHMM -t wind -n 10
"""
from __future__ import annotations
import argparse
from pathlib import Path

# Parche NumPy≥2.0
from src.utils.numpy_patch import *  # debe ir antes de shap
import joblib
import pandas as pd
import shap
from src.utils.yaml_cfg import load

def main(config: str, run: str, target: str, top_n: int) -> None:
    cfg    = load(config)
    df     = pd.read_parquet(cfg["paths"]["feature_table"])
    test   = df.query("split == 'test'")
    # Solo features numéricas:
    numeric = df.select_dtypes(include=[float, int]).columns.tolist()
    feats   = [c for c in numeric if c not in (*cfg["model"]["targets"], "split")]

    model_dir = Path(cfg["paths"]["models"]) / run / target
    out_dir   = model_dir / "shap_local"
    out_dir.mkdir(parents=True, exist_ok=True)

    model  = joblib.load(model_dir / "model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    X_test = scaler.transform(test[feats])

    # calcula errores y selecciona worst
    preds     = model.predict(X_test)
    errors    = abs(preds - test[target].values)
    worst_idx = errors.argsort()[-top_n:][::-1]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test[worst_idx], check_additivity=False)

    for rank, idx in enumerate(worst_idx, start=1):
        fp = shap.force_plot(
            explainer.expected_value,
            shap_vals[rank-1],
            X_test[idx],
            feature_names=feats,
            matplotlib=False,
        )
        html_path = out_dir / f"case_{rank:02d}_idx{idx}.html"
        shap.save_html(str(html_path), fp)
        print(f"  Guardado: {html_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, help="Ruta YAML config")
    p.add_argument("-r", "--run",    required=True, help="ID del run (p.ej. xgb_YYYYMMDD_HHMM)")
    p.add_argument("-t", "--target", required=True, choices=["wind", "solar_pv", "demand"])
    p.add_argument("-n", "--top_n",  type=int, default=10, help="Número de casos a explicar")
    main(**vars(p.parse_args()))
