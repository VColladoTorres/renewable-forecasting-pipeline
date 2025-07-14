"""Valida la presencia de todos los ficheros brutos requeridos (ESIOS + ERA5)."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
import yaml, logging, sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
RAW = Path("data/raw")

def _expected_dates(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)

def main(cfg_path: str = "configs/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    s = datetime.fromisoformat(cfg["data"]["start_date"].replace("Z", ""))
    e = datetime.fromisoformat(cfg["data"]["end_date"].replace("Z", ""))

    # --- ESIOS ----------------------------------------------------------------
    esios_root = RAW / "esios"
    all_json = {p.stem for p in esios_root.glob("**/*.json")}  # recursivo
    miss_esios = [d + ".json" for d in _expected_dates(s, e) if d not in all_json]

    logging.info("ESIOS faltantes: %d", len(miss_esios))
    for f in miss_esios[:5]:
        logging.info("  %s...", f)

    # --- ERA5 -----------------------------------------------------------------
    miss_era5 = []
    for spec in cfg["variables"]["era5"]:
        name = spec["name"]
        p = RAW / "era5" / name
        cur = s.replace(day=1)
        while cur <= e:
            fn = p / f"{cur:%Y-%m}.nc"
            if not fn.exists():
                miss_era5.append(fn.relative_to(RAW))
            # avanzar al mes siguiente
            nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
            cur = nxt
    logging.info("ERA5 faltantes: %d", len(miss_era5))
    for f in miss_era5[:5]:
        logging.info("  %s...", f)

    if miss_esios or miss_era5:
        sys.exit("❌  Faltan ficheros — completa la caché antes de continuar.")
    print("✅  Caché ESIOS + ERA5 completa.")

if __name__ == "__main__":
    main()
