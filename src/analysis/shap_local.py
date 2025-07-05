# src/analysis/shap_local.py
"""
Crea explicaciones locales (force_plot) de SHAP
para los N peores errores de un target.

Guarda HTML en          models/<run>/<target>/shap_local/
Ejemplo de uso (una línea):
  python -m src.analysis.shap_local -c configs/default.yaml -r xgb_20250705_1117 -t wind -n 10
"""
from __future__ import annotations
import argparse
from pathlib import Path

# --- parche NumPy>=2.0 (np.obj2sctype ausente) ---------------------
from src.utils.numpy_patch import *        # debe ir antes de import shap
# -------------------------------------------------------------------

import joblib
import pandas as pd
import shap
from src.utils.yaml_cfg import load


def main(config: str, run: str, target: str, top_n: int) -> None:
    cfg   = load(config)
    df    = pd.read_parquet(cfg["paths"]["feature_table"])
    test  = df.query("split == 'test'")
    feats = [c for c in df.columns if c not in (*cfg["model"]["targets"], "split")]

    model_dir = Path(cfg["paths"]["models"]) / run / target
    out_dir   = model_dir / "shap_local"
    out_dir.mkdir(parents=True, exist_ok=True)

    model  = joblib.load(model_dir / "model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    X_test = scaler.transform(test[feats])

    # --- calcular errores absolutos ---
    pred   = model.predict(X_test)
    errors = abs(pred - test[target].values)
    worst_idx = errors.argsort()[-top_n:][::-1]        # descendente

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test[worst_idx])

    for i, idx in enumerate(worst_idx, start=1):
        fp = shap.force_plot(
            explainer.expected_value,
            shap_vals[i-1],
            X_test[worst_idx][i-1],
            feature_names=feats,
            matplotlib=False,
        )
        html_path = out_dir / f"case_{i:02d}_idx{idx}.html"
        shap.save_html(html_path, fp)
        print(f"  Guardado: {html_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, help="Ruta YAML config")
    p.add_argument("-r", "--run",    required=True, help="Carpeta de modelo (xgb_YYYYMMDD_HHMM)")
    p.add_argument("-t", "--target", required=True, choices=["wind", "solar_pv", "demand"])
    p.add_argument("-n", "--top_n",  type=int, default=10, help="Número de casos a explicar")
    main(**vars(p.parse_args()))
