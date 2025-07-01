
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
