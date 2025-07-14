#!/usr/bin/env python
"""
Convierte cualquier fichero HRES (.nc o .grib2) a resolución 15 min
creando <nombre>_15min.nc en el mismo directorio.
"""
from pathlib import Path
import sys, xarray as xr, numpy as np

short_names = ("10u", "10v", "2t", "ssrd")  # amplía si añades más vars


def open_hres(path: Path) -> xr.Dataset:
    """
    • Si es NetCDF clásico → engine='netcdf4'.
    • Si no, abre cada shortName con cfgrib y fusiona.
    """
    try:                                       # intento NetCDF4
        return xr.open_dataset(path, engine="netcdf4")
    except Exception:
        def _load(short):
            return xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {"shortName": short},
                    "indexpath": "",
                },
            ).drop_vars("heightAboveGround", errors="ignore")

        ds = xr.merge([_load(s) for s in short_names], compat="override")

        # grib → pasar de 'time' + 'step' a eje horario 'valid_time'
        vtime = ds["time"] + ds["step"]
        ds = ds.assign_coords(valid_time=("step", vtime)).swap_dims(step="valid_time")
        ds = ds.drop_vars(["time", "step"])
        return ds


def main(fname: str) -> None:
    src = Path(fname)
    ds  = open_hres(src)

    # Resamplea todas las variables numéricas a 15 min (linear)
    ds15 = ds.resample(valid_time="15min").interpolate("linear")

    out = src.with_name(src.stem + "_15min.nc")
    ds15.to_netcdf(out, encoding={v: {"zlib": True, "complevel": 3} for v in ds15})
    print("✓ creado", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso:  python src/data/interp_time.py <archivo_hres>")
    main(sys.argv[1])
