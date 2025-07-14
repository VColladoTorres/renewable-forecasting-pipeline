#!/usr/bin/env python
"""
Evaluate a run (models/<run_id>/) on the test split and
generate an HTML report with PNGs (Matplotlib, headless).
"""

from __future__ import annotations
import argparse, logging, math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.yaml_cfg import load

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
LOG = logging.getLogger(__name__)

# ────────── helpers ──────────
def metrics_dict(y, yhat):
    mae  = mean_absolute_error(y, yhat)
    rmse = math.sqrt(mean_squared_error(y, yhat))
    nmae = mae / np.mean(np.abs(y))
    mape = np.mean(np.abs((y - yhat) / y)) * 100
    r2   = r2_score(y, yhat)
    return dict(MAE=mae, RMSE=rmse, nMAE=nmae, MAPE=mape, R2=r2)

def plot_scatter(y, yhat, path: Path, tgt: str):
    plt.figure(figsize=(6, 4.5))
    lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    plt.plot(lims, lims, "--", color="black", lw=1)
    plt.scatter(y, yhat, s=18, alpha=0.7, c="royalblue")
    plt.title(f"{tgt} – Observed vs Predicted")
    plt.xlabel("Observed"); plt.ylabel("Predicted")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()

def plot_residuals(resid, path: Path, tgt: str):
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=40, color="indianred", alpha=0.8)
    plt.axvline(resid.mean(), color="black", ls="--", lw=1)
    plt.title(f"{tgt} – Residual Histogram")
    plt.xlabel("Residual"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()

# ────────── evaluate target ──────────
def evaluate_target(run_dir: Path, df_test: pd.DataFrame, tgt: str):
    tgt_dir = run_dir / tgt
    model  = joblib.load(tgt_dir / "model.joblib")
    scaler = joblib.load(tgt_dir / "scaler.joblib")
    feats  = joblib.load(tgt_dir / "features_used.pkl")

    df_t = df_test.dropna(subset=feats + [tgt])
    if df_t.empty:
        LOG.warning("No test samples for %s – skipped.", tgt)
        return None

    X = scaler.transform(df_t[feats])
    y, yhat = df_t[tgt].values, model.predict(X)
    resid = y - yhat
    mets = metrics_dict(y, yhat)

    scatter_png = tgt_dir / "obs_vs_pred.png"
    resid_png   = tgt_dir / "residuals.png"
    plot_scatter(y, yhat, scatter_png, tgt)
    plot_residuals(resid, resid_png, tgt)

    # ruta relativa para el HTML
    scatter_rel = scatter_png.relative_to(run_dir).as_posix()
    resid_rel   = resid_png.relative_to(run_dir).as_posix()

    LOG.info("%s  MAE=%.2f  RMSE=%.2f  R²=%.3f", tgt.upper(), mets["MAE"], mets["RMSE"], mets["R2"])
    return dict(target=tgt, metrics=mets,
                scatter=scatter_rel, resid=resid_rel)

# ────────── HTML report ──────────
HEAD = """<!doctype html><html><head><meta charset="utf-8">
<title>Evaluation {run}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:20px;background:#fafafa}}
table{{border-collapse:collapse;margin-bottom:30px}}
th,td{{border:1px solid #999;padding:6px 10px;text-align:right}}
th{{background:#eee}}h2{{margin-top:40px}}
img{{max-width:100%;height:auto;border:1px solid #ddd;margin-bottom:20px}}
</style></head><body>
<h1>Evaluation report: {run}</h1>
<small>Generated: {ts}</small>"""
FOOT = "</body></html>"

def build_report(run_dir: Path, run: str, res):
    html = HEAD.format(run=run,
                       ts=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    html += "<h2>Metrics (test split)</h2><table><tr><th>Target</th>"
    for m in ["MAE", "RMSE", "nMAE", "MAPE", "R2"]:
        html += f"<th>{m}</th>"
    html += "</tr>"
    for r in res:
        html += f"<tr><td>{r['target']}</td>"
        for m in ["MAE", "RMSE", "nMAE", "MAPE", "R2"]:
            html += f"<td>{r['metrics'][m]:.3f}</td>"
        html += "</tr>"
    html += "</table>"
    for r in res:
        html += f"<h2>{r['target']}</h2>"
        html += f"<img src='{r['scatter']}' alt='scatter'>"
        html += f"<img src='{r['resid']}'   alt='residuals'>"
    (run_dir / "report.html").write_text(html, encoding="utf-8")
    LOG.info("Report saved to %s", run_dir / "report.html")

# ────────── main CLI ──────────
def main(config: str, run: str):
    cfg = load(config)
    df = pd.read_parquet(cfg["paths"]["feature_table"])
    df_test = df[df["split"] == "test"]
    run_dir = Path(cfg["paths"]["models"]) / run

    results = [r for tgt in cfg["model"]["targets"]
               if (r := evaluate_target(run_dir, df_test, tgt))]
    if results:
        build_report(run_dir, run, results)
    else:
        LOG.error("No results generated – check artefacts/data.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-r", "--run",    required=True)
    main(**vars(ap.parse_args()))
