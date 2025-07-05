"""
Extrae pronóstico NWP reciente, construye *features*, aplica modelo
y guarda CSV+JSON en outputs/<YYYYMMDD_HH>.  Uso:
  python -m src.prediction.predict \
         -c configs/default.yaml \
         -m models/current        \
         --era5-dir /data/nwp/latest
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd, joblib, xarray as xr
from src.features.build_features import engineer_feature_matrix
from src.utils.yaml_cfg import load

def main(cfg_p: str, model_dir: str, era5_dir: str):
    cfg   = load(cfg_p)
    # --- 1. leer últimos ficheros NetCDF (determinista o ensemble) ---
    ds    = xr.open_mfdataset(f"{era5_dir}/*.nc")
    # --- 2. transformar al mismo grid & variables que el entrenamiento ---
    feats = engineer_feature_matrix(ds, cfg)           # reutiliza tu función
    scaler= joblib.load(Path(model_dir)/"scaler.joblib")
    model = joblib.load(Path(model_dir)/"model.joblib")
    X     = scaler.transform(feats[cfg["model"]["full_feature_list"]])
    y_hat = model.predict(X)

    out_ts = feats.index[0].strftime("%Y%m%d_%H%M")
    odir   = Path("outputs")/out_ts; odir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": feats.index, "pred": y_hat}) \
      .to_csv(odir/"predictions.csv", index=False)
    json.dump({"timestamp": out_ts, "pred": y_hat.tolist()},
              (odir/"predictions.json").open("w"))
    print(f"Predicciones escritas en {odir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c","--config", required=True)
    p.add_argument("-m","--model-dir", required=True)
    p.add_argument("--era5-dir", required=True)
    main(**vars(p.parse_args()))
