#!/usr/bin/env python
"""
Crea la columna 'split' en features.parquet usando cortes temporales fijos:
    • train  → hasta 2023-12-31 23:00
    • val    → 2024-01-01 .. 2024-12-31
    • test   → 2025-01-01 .. fin
"""

from pathlib import Path
import pandas as pd

FEATURES = Path("data/processed/features.parquet")  # <- ajusta si es otro nombre
OUT      = FEATURES                                 # sobreescribe (o cambia)

CUT_TRAIN_END = "2023-12-31 23:00"
CUT_VAL_END   = "2024-12-31 23:00"

df = pd.read_parquet(FEATURES)
df["timestamp"] = pd.to_datetime(df["timestamp"])

df.loc[df["timestamp"] <= CUT_TRAIN_END, "split"] = "train"
df.loc[(df["timestamp"] > CUT_TRAIN_END) &
       (df["timestamp"] <= CUT_VAL_END), "split"] = "val"
df.loc[df["timestamp"] > CUT_VAL_END, "split"] = "test"

df.to_parquet(OUT)
print("✅ columna 'split' añadida y guardada en", OUT)
