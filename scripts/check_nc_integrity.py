# scripts/check_nc_integrity.py
import sys, pathlib, xarray as xr

root = pathlib.Path("data/raw/era5")   # cambia si tu ruta difiere
bad = []

for nc in root.rglob("*.nc"):
    try:
        xr.open_dataset(nc, engine="h5netcdf").close()
    except Exception as e:
        bad.append((nc, str(e)))

if bad:
    print("\nFicheros corruptos / incompletos:")
    for p, err in bad:
        print(f" ✗ {p}  ← {err.splitlines()[0]}")
    sys.exit(1)
else:
    print("✓ Todos los NetCDF se abrieron sin error.")
