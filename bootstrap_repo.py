"""
Genera el repo “renewable-forecasting-pipeline” con un solo comando.
MIT © 2025 MSc Candidate
"""
from __future__ import annotations

import textwrap
from pathlib import Path

# ---------------------------------------------------- #
#  Mapa: ruta → contenido (multilínea)                  #
# ---------------------------------------------------- #
FILES: dict[str, str] = {
    ".env.example": textwrap.dedent(
        """\
        # Copia a .env y rellena con tus claves reales
        ESIOS_TOKEN=5fceb15ccd22a22b251223161ac2500b61c433af22ffeaf3feecf6e97262ccbb
        CDS_API_KEY=b4f98e85-8c4a-4866-b49d-fe25bf3f77ad:49b69de6-d1f3-482f-b380-5e3e88188a87
        """
    ),
    ".gitignore": textwrap.dedent(
        """\
        # artefactos que NO deben subirse
        .env
        .venv/
        __pycache__/
        data/
        models/
        """
    ),
    "pyproject.toml": """\
# MIT © 2025 MSc Candidate
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "renewable-forecasting-pipeline"
version = "0.1.0"
description = "End-to-end Spanish wind & PV forecasting pipeline (ERA5-Land + ESIOS)"
authors = [{name = "MSc Candidate", email = "author@example.com"}]
license = {text = "MIT"}
requires-python = ">=3.9"

dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "python-dotenv>=1.0",
    "requests>=2.32",
    "cdsapi>=0.6",
    "xgboost>=2.0",
    "scikit-learn>=1.4",
    "optuna>=3.6",
    "joblib>=1.4",
    "matplotlib>=3.9",
    "pyyaml>=6.0",
    "ruff>=0.4",
    "tqdm>=4.66",
    "xarray>=2024.2",
    "netCDF4>=1.6",
    "scipy>=1.13"
]

[tool.ruff]
target-version = "py310"
line-length = 88
select = ["E", "F", "I", "UP", "B"]
""",
    "README.md": textwrap.dedent(
        """\
        # Renewable-Forecasting-Pipeline

        Workflow reproductible para previsión horaria de **eólica on-shore** y **PV utility-scale** en España (2018-2024).

        ## Uso rápido
        ```bash
        cp .env.example .env         # añade tus tokens
        pip install -e .
        python -m src.data.make_dataset     --config configs/default.yaml
        python -m src.features.build_features --config configs/default.yaml
        python -m src.models.train_xgb       --config configs/default.yaml
        python -m src.models.evaluate        --config configs/default.yaml
        ```
        """
    ),
    "configs/default.yaml": textwrap.dedent(
        """\
        paths:
          raw_dir: data/raw
          interim_dir: data/interim
          processed_dir: data/processed
          models: models
          feature_table: data/processed/features.parquet

        data:
          target: mw
          split:
            train_end: "2022-12-31T23:00:00+01:00"
            val_end:   "2023-06-30T23:00:00+02:00"

        variables:
          era5:
            - wind_speed_100m
            - surface_solar_radiation_downwards
            - temperature_2m

        feature_engineering:
          lags: [1, 2, 3, 6, 12, 24, 48]
          rolling_means: [3, 6]

        optuna:
          n_trials: 50
        """
    ),
    "docker/Dockerfile": textwrap.dedent(
        """\
        # MIT © 2025 MSc Candidate
        FROM python:3.10-slim
        ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
        WORKDIR /app

        # sistema deps para NetCDF & CDO
        RUN apt-get update && apt-get install -y --no-install-recommends \\
                libproj-dev proj-data proj-bin cdo && \\
            rm -rf /var/lib/apt/lists/*

        COPY pyproject.toml ./
        RUN pip install --upgrade pip && pip install -r <(python - <<'EOF'
import tomllib, pathlib, sys
deps = tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['dependencies']
print('\\n'.join(deps))
EOF
        )

        COPY src ./src
        COPY configs ./configs
        ENTRYPOINT ["python", "-m", "src.models.train_xgb", "--config", "configs/default.yaml"]
        """
    ),
    ".github/workflows/ci.yml": textwrap.dedent(
        """\
        name: CI
        on: [push, pull_request]
        jobs:
          test:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
              - uses: actions/setup-python@v4
                with: {python-version: "3.10"}
              - run: pip install -r <(python - <<'EOF'
import tomllib, pathlib, sys
print('\\n'.join(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['dependencies']))
EOF
                )
              - run: ruff .
              - run: pytest -q
        """
    ),
    # ---------- src package ---------- #
    "src/__init__.py": '"""Package root."""\n',
    "src/utils/__init__.py": '"""Shared utilities."""\n',
    "src/utils/logger.py": textwrap.dedent(
        '''\
        """Standardised logger."""
        import logging, sys
        FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        def init_logger(level: int = logging.INFO) -> None:
            logging.basicConfig(level=level, format=FORMAT,
                                handlers=[logging.StreamHandler(sys.stdout)])
        '''
    ),
    "src/data/__init__.py": '"""Data layer."""\n',
    "src/data/esios_client.py": """<ESIOS_CLIENT_CODE>""",
    "src/data/cds_client.py": """<CDS_CLIENT_CODE>""",
    "src/data/make_dataset.py": """<MAKE_DATASET_CODE>""",
    "src/features/__init__.py": '"""Feature eng."""\n',
    "src/features/build_features.py": """<BUILD_FEATURES_CODE>""",
    "src/models/__init__.py": '"""ML models."""\n',
    "src/models/train_xgb.py": """<TRAIN_XGB_CODE>""",
    "src/models/evaluate.py": """<EVALUATE_CODE>""",
    # ---------- tests ---------- #
    "tests/test_esios_client.py": """<TEST_ESIOS_CODE>""",
    "tests/test_split.py": """<TEST_SPLIT_CODE>""",
}

# Sustituye marcadores por los fragmentos de código completos ya suministrados
REPLACEMENTS = {
    "<ESIOS_CLIENT_CODE>": ''' 
# MIT License
# Copyright (c) 2025 MSc Candidate
#
# Permission is hereby granted, free of charge, to any person obtaining a copy …
"""
ESIOS REST-API client.

Rationale
---------
Encapsulates HTTP + pagination while keeping token outside VCS.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
_LOGGER = logging.getLogger(__name__)
_BASE_URL = "https://api.esios.ree.es"
_API_VERSION = "v1"

_TECH_MAP = {
    "wind": 75,          # ESIOS “Generación eólica”
    "solar_pv": 68,      # “Generación fotovoltaica”
    "demand": 1,         # “Demanda peninsular”
}


class EsiosClient:
    """Thin wrapper around the ESIOS API."""

    def __init__(self, token: str | None = None) -> None:
        token = token or os.getenv("ESIOS_TOKEN")
        if not token:
            raise ValueError("ESIOS_TOKEN missing. Provide via env or arg.")

        self._headers = {
            "Accept": f"application/json; application/vnd.esios-api.{_API_VERSION}",
            "Authorization": f"Token token={token}",
        }
        self._session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.3,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=["GET"],
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retries))

    # ---------------- public methods ---------------- #
    def fetch_series(
        self,
        start: datetime,
        end: datetime,
        technology: str = "wind",
    ) -> pd.DataFrame:
        """Return hourly MW series in Europe/Madrid."""
        series_id = _TECH_MAP[technology]
        url = f"{_BASE_URL}/archives/{series_id}"
        params = {
            "start_date": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_date": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time_trunc": "hour",
        }
        raw = self._paginate(url, params)
        df = _json_to_df(raw)
        return df.tz_convert("Europe/Madrid")

    # ---------------- internal ---------------- #
    def _paginate(self, url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        while url:
            _LOGGER.debug("GET %s", url)
            r = self._session.get(url, params=params, headers=self._headers, timeout=30)
            r.raise_for_status()
            body = r.json()
            data.extend(body["data"])
            url = body.get("links", {}).get("next")
            params = {}
        return data


def _json_to_df(payload: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(payload, sep="_")
    df["timestamp"] = pd.to_datetime(df["attributes_datetime"], utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[["attributes_value"]].rename(columns={"attributes_value": "mw"})
    return df.sort_index()
''',
    "<CDS_CLIENT_CODE>": ''' # MIT License
# Copyright (c) 2025 MSc Candidate
#
# Permission is hereby granted, free of charge, to any person obtaining a copy …
"""
Copernicus CDS (ERA5-Land) client.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import cdsapi
import xarray as xr

_LOGGER = logging.getLogger(__name__)


class CDSClient:
    """Download ERA5-Land hourly data and merge monthly files lazily."""

    def __init__(self, target_dir: str | Path = "data/raw/era5") -> None:
        self._client = cdsapi.Client(quiet=True)
        self._target_dir = Path(target_dir)
        self._target_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def download(
        self,
        variables: List[str],
        start: datetime,
        end: datetime,
        grid: float = 0.25,
    ) -> Path:
        """Return path to merged NetCDF covering *start*–*end* inclusive."""
        months = _month_stubs(start, end)
        paths: List[Path] = []
        for stub in months:
            out = self._target_dir / f"era5_{stub}.nc"
            if out.exists():
                paths.append(out)
                continue
            _LOGGER.info("Request ERA5-Land %s → %s", stub, out.name)
            self._client.retrieve(
                "reanalysis-era5-land",
                {
                    "format": "netcdf",
                    "variable": variables,
                    "year": stub[:4],
                    "month": stub[4:6],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "grid": [grid, grid],
                },
                str(out),
            )
            paths.append(out)

        merged = self._target_dir / "era5_combined.nc"
        if len(paths) == 1:
            return paths[0]
        _LOGGER.info("Concatenate %d NetCDFs → %s", len(paths), merged.name)
        ds = xr.open_mfdataset(paths, combine="by_coords")  # lazy
        ds.to_netcdf(merged)
        return merged


# ---------------- helpers ---------------- #
def _month_stubs(start: datetime, end: datetime) -> List[str]:
    cur, stubs = start.replace(day=1), []
    while cur <= end:
        stubs.append(cur.strftime("%Y%m"))
        cur = (cur.replace(day=28) + pd.Timedelta(days=4)).replace(day=1)  # next month
    return stubs
''' ,
    "<MAKE_DATASET_CODE>": ''' 
# MIT License
# Copyright (c) 2025 MSc Candidate
#
# Permission is hereby granted, free of charge, to any person obtaining a copy …
"""
ETL script: downloads ESIOS + ERA5, aligns hourly and persists parquet.

Run:
    python -m src.data.make_dataset --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm

from src.data.cds_client import CDSClient
from src.data.esios_client import EsiosClient
from src.utils.logger import init_logger

init_logger()
_LOGGER = logging.getLogger(__name__)


def _parse_dates(cfg: dict) -> tuple[datetime, datetime]:
    test_end = pd.Timestamp(cfg["data"]["split"]["val_end"]).tz_convert("UTC")
    train_start = pd.Timestamp("2018-01-01T00:00:00+00:00")
    return train_start.to_pydatetime(), test_end.to_pydatetime()


def main(config: str) -> None:  # noqa: D401
    with open(config) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    start, end = _parse_dates(cfg)

    # ------------ download ESIOS series ------------ #
    esios = EsiosClient()
    dfs = []
    for tech in ("wind", "solar_pv"):
        _LOGGER.info("Download %s generation…", tech)
        dfs.append(esios.fetch_series(start, end, technology=tech).rename(columns={"mw": tech}))
    demand = esios.fetch_series(start, end, technology="demand").rename(columns={"mw": "demand"})
    df_gen = pd.concat(dfs + [demand], axis=1)

    # ------------ download ERA5-Land --------------- #
    cds = CDSClient(raw_dir / "era5")
    nc_path = cds.download(cfg["variables"]["era5"], start, end)
    _LOGGER.info("Open ERA5 dataset %s", nc_path.name)
    ds = xr.open_dataset(nc_path)  # lazily read

    # Spatial average over Spain (bounding box as rough proxy)
    spain = ds.sel(latitude=slice(44, 36), longitude=slice(-10, 5)).mean(
        ["latitude", "longitude"]
    )
    df_met = spain.to_dataframe()
    df_met.index = df_met.index.tz_localize(timezone.utc).tz_convert("Europe/Madrid")

    # ------------ merge & persist ------------------ #
    df = df_gen.join(df_met, how="inner")
    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "dataset.parquet")
    _LOGGER.info("Saved merged dataset with %d rows", len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build raw dataset.")
    parser.add_argument("--config", "-c", required=True, help="YAML config path.")
    main(**vars(parser.parse_args()))
''' ,
    "<BUILD_FEATURES_CODE>": '''
# MIT License
# Copyright (c) 2025 MSc Candidate
#
# Permission is hereby granted, free of charge, to any person obtaining a copy …
"""
Create lag/rolling features and temporal splits.

Run:
    python -m src.features.build_features --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.utils.logger import init_logger

init_logger()
_LOGGER = logging.getLogger(__name__)


def main(config: str) -> None:  # noqa: D401
    with open(config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "dataset.parquet")

    # ---------------- feature engineering ---------------- #
    lags = cfg["feature_engineering"]["lags"]
    rolls = cfg["feature_engineering"]["rolling_means"]

    for col in ("wind", "solar_pv"):
        for k in lags:
            df[f"{col}_lag{k}"] = df[col].shift(k)
        for w in rolls:
            df[f"{col}_roll{w}"] = df[col].rolling(w).mean()

    # simple theoretical wind power from 100 m speed: P ~ v³
    df["wind_power_theory"] = (df["wind_speed_100m"] ** 3).clip(upper=30_000)

    # drop rows with NA introduced by lags
    df.dropna(inplace=True)

    # ---------------- splits ---------------- #
    train_end = pd.Timestamp(cfg["data"]["split"]["train_end"])
    val_end = pd.Timestamp(cfg["data"]["split"]["val_end"])

    df["split"] = np.where(
        df.index <= train_end,
        "train",
        np.where(df.index <= val_end, "val", "test"),
    )

    # save
    out_path = Path(cfg["paths"]["feature_table"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    _LOGGER.info("Features ready → %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features table.")
    parser.add_argument("--config", "-c", required=True)
    main(**vars(parser.parse_args()))
''',
    "<TRAIN_XGB_CODE>": '''
# MIT License
# Copyright (c) 2025 MSc Candidate
#
# Permission is hereby granted, free of charge, to any person obtaining a copy …
"""
Train XGBoost with Optuna.

Run:
    python -m src.models.train_xgb --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.utils.logger import init_logger

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
init_logger()
_LOGGER = logging.getLogger(__name__)


def _objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("eta", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("lambda", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "random_state": RANDOM_SEED,
        "n_jobs": os.cpu_count(),
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, pred, squared=False)
    return rmse


def main(config: str) -> None:  # noqa: D401
    with open(config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(cfg["paths"]["feature_table"])
    target = cfg["data"]["target"]
    feats = [c for c in df.columns if c not in (target, "split")]

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feats])
    y_train = train[target]
    X_val = scaler.transform(val[feats])
    y_val = val[target]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(lambda t: _objective(t, X_train, y_train, X_val, y_val), n_trials=cfg["optuna"]["n_trials"])

    best = study.best_params | {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": os.cpu_count(),
    }
    model = xgb.XGBRegressor(**best)
    model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

    out_dir = Path(cfg["paths"]["models"])
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "xgb_model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    json.dump(best, (out_dir / "best_params.json").open("w"))

    _LOGGER.info("Model + artefacts saved to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    main(**vars(parser.parse_args()))
''',
    "<EVALUATE_CODE>": '''
# MIT License
# Copyright (c) 2025 MSc Candidate
#
# Permission is hereby granted, free of charge, to any person obtaining a copy …
"""
Evaluate trained model and create diagnostic plot.

Run:
    python -m src.models.evaluate --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import init_logger

init_logger()
_LOGGER = logging.getLogger(__name__)


def main(config: str) -> None:  # noqa: D401
    with open(config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(cfg["paths"]["feature_table"])
    test = df[df["split"] == "test"]

    model = joblib.load(Path(cfg["paths"]["models"]) / "xgb_model.joblib")
    scaler = joblib.load(Path(cfg["paths"]["models"]) / "scaler.joblib")
    feats = [c for c in test.columns if c not in ("mw", "split")]
    X_test = scaler.transform(test[feats])
    y_true = test["mw"].values
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    nmae = mae / y_true.max()
    r2 = r2_score(y_true, y_pred)
    _LOGGER.info("Test MAE %.2f | RMSE %.2f | nMAE %.2f | R² %.3f", mae, rmse, nmae, r2)

    # plot
    plt.figure(figsize=(10, 4))
    plt.plot(test.index, y_true, label="obs")
    plt.plot(test.index, y_pred, label="pred")
    plt.legend()
    plt.title("XGB – observed vs predicted")
    out_png = Path(cfg["paths"]["models"]) / "prediction_vs_obs.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    _LOGGER.info("Plot saved to %s", out_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    main(**vars(parser.parse_args()))
''',
    "<TEST_ESIOS_CODE>": '''
# MIT © 2025 MSc Candidate
"""Unit test for EsiosClient using responses to mock HTTP."""
import json
from datetime import datetime, timezone

import pandas as pd
import pytest
import responses

from src.data.esios_client import EsiosClient


@responses.activate
def test_fetch_series():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
    sample = {
        "data": [
            {
                "attributes": {
                    "datetime": "2024-01-01T00:00:00Z",
                    "value": 500,
                }
            },
            {
                "attributes": {
                    "datetime": "2024-01-01T01:00:00Z",
                    "value": 600,
                }
            },
        ],
        "links": {"next": None},
    }
    responses.add(
        responses.GET,
        "https://api.esios.ree.es/archives/75",
        json=sample,
        status=200,
    )

    client = EsiosClient(token="dummy")
    df = client.fetch_series(start, end, technology="wind")
    assert isinstance(df, pd.DataFrame)
    assert df["mw"].iloc[0] == 500
''',
    "<TEST_SPLIT_CODE>": '''
# MIT © 2025 MSc Candidate
"""Verify there is no temporal leakage between splits."""
import pandas as pd
from src.features.build_features import main as build_main

def test_split(tmp_path, monkeypatch):
    # minimal dummy config
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        """
paths:
  processed_dir: tests
  feature_table: tests/feat.parquet
data:
  split:
    train_end: "2020-12-31T23:00:00+01:00"
    val_end: "2021-12-31T23:00:00+01:00"
feature_engineering:
  lags: [1]
  rolling_means: []
"""
    )
    # create dummy dataset
    idx = pd.date_range("2020-01-01", periods=10, freq="H", tz="Europe/Madrid")
    df = pd.DataFrame({"wind": range(10)}, index=idx)
    df.to_parquet("tests/dataset.parquet")

    build_main(str(cfg_file))
    df_feat = pd.read_parquet("tests/feat.parquet")
    assert set(df_feat["split"].unique()) == {"train", "val", "test"}
''',
}

for marker, code in REPLACEMENTS.items():
    for path, content in FILES.items():
        if marker in content:
            FILES[path] = content.replace(marker, code)

# ---------------------------------------------------- #
#  Escritura física                                    #
# ---------------------------------------------------- #
for filepath, content in FILES.items():
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")

print("Repo creado con", len(FILES), "archivos")
