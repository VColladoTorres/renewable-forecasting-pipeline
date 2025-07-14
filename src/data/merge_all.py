# src/data/merge_all.py
"""
Une el dataset inicial (initial_dataset.parquet) —que contiene ERA5+ESIOS hasta el corte—
con las actualizaciones hechas por update_dataset, guardando el resultado final en dataset.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.yaml_cfg import load  # tu util para cargar YAML

def main(config: str) -> None:
    # 1) carga la configuración
    cfg = load(config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    initial_file = processed_dir / "initial_dataset.parquet"
    final_file   = processed_dir / "dataset.parquet"

    # 2) lee el parquet inicial (que ya incluye las actualizaciones)
    df = pd.read_parquet(initial_file)

    # 3) simplemente lo vuelca como dataset.parquet final
    final_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(final_file, index=False)

    print(f"✓ Dataset final guardado en {final_file} ({len(df)} filas).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all processed data into one file")
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Ruta al YAML de configuración"
    )
    args = parser.parse_args()
    main(**vars(args))
