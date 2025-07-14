#!/usr/bin/env python
"""
Descarga la última carrera disponible de ECMWF IFS-HRES (Open Data 0.25°)
para la Península Ibérica:

• Variables: 10u 10v 2t ssrd
• Pasos 0–144 h cada 3 h (06/18 UTC)   o  0–168 h (00/12 UTC) añadido 150–168 h cada 6 h
• Guarda GRIB2 y, si es compatible, NetCDF directamente
• Reintenta hasta 3 veces con back-off si el bucket aún no está listo
• Escritura atómica → primero *.tmp*, luego renombra
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from ecmwf.opendata import Client
from tenacity import retry, stop_after_attempt, wait_incrementing

# ───── configuración ──────────────────────────────────────────────────────
LOG = logging.getLogger("download_hres")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "hres"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS: list[str] = ["10u", "10v", "2t", "ssrd"]
RESOL = "0p25"
RUNS = (0, 6, 12, 18)


# ───── utilidades ──────────────────────────────────────────────────────────
def _latest_available_run(now: datetime) -> datetime:
    """
    Devuelve la última hora de inicio de pronóstico que debería estar
    ya publicada en el bucket Open-Data.
    • 00 / 12 UTC ⇒ lag ≈ 4 h
    • 06 / 18 UTC ⇒ lag ≈ 6 h
    """
    lag = 4 if now.hour in (0, 12) else 6
    cand = now - timedelta(hours=lag)
    run_hour = max(h for h in RUNS if h <= cand.hour)
    return cand.replace(hour=run_hour, minute=0, second=0, microsecond=0,
                        tzinfo=timezone.utc)


def _step_list(run_hour: int) -> List[int]:
    """Pasos permitidos según ECMWF Open-Data."""
    base = list(range(0, 145, 3))                     # 0–144 h / 3 h
    extra = list(range(150, 169, 6)) if run_hour in (0, 12) else []
    return base + extra


# ───── descarga con reintentos ─────────────────────────────────────────────
@retry(wait=wait_incrementing(start=30, increment=60, max=300),
       stop=stop_after_attempt(3), reraise=True)
def _fetch(run_time: datetime, target_tmp: Path, fmt: str) -> None:
    client = Client(source="ecmwf", model="ifs", resol=RESOL)
    client.retrieve(
        date=run_time.strftime("%Y-%m-%d"),
        time=run_time.hour,
        type="fc",
        param=PARAMS,
        step=_step_list(run_time.hour),
        target=str(target_tmp),
        format=fmt,                     # "grib" o "netcdf"
    )


# ───── nuevo cálculo de último run disponible ──────────────────────────────
def latest_run_candidates(now: datetime, attempts: int = 4) -> list[datetime]:
    """
    Devuelve una lista de candidatos (UTC) empezando por el run más reciente
    **posible** y retrocediendo de 6 h en 6 h `attempts` veces.
    """
    lag_hours = 5 if now.hour in (0, 12) else 8    # margen seguro
    first = now - timedelta(hours=lag_hours)
    # alinear a 00, 06, 12, 18
    aligned = first.replace(hour=max(h for h in RUNS if h <= first.hour),
                            minute=0, second=0, microsecond=0,
                            tzinfo=timezone.utc)
    return [aligned - timedelta(hours=6*i) for i in range(attempts)]

# ───── main con fallback + mensaje claro ───────────────────────────────────
def main() -> None:
    for run_time in latest_run_candidates(datetime.now(timezone.utc)):
        stem = f"ifs_hres_{run_time:%Y%m%d%H}"
        grib = OUT_DIR / f"{stem}.grib2"
        nc   = OUT_DIR / f"{stem}.nc"

        if nc.exists() or grib.exists():
            LOG.info("Ya existe %s (nc o grib2)", stem)
            return

        LOG.info("Intentando run %s UTC …", run_time.strftime("%Y-%m-%d %H"))
        try:
            # 1) NetCDF (si el servidor lo ofrece en futuro)
            tmp = nc.with_suffix(".nc.tmp")
            _fetch(run_time, tmp, fmt="netcdf")
            tmp.rename(nc)
            LOG.info("✓ NetCDF descargado → %s (%.1f MB)", nc.name, nc.stat().st_size/1e6)
            return
        except Exception as e_nc:                               # noqa: BLE001
            LOG.debug("NetCDF no disponible: %s", e_nc)

        try:
            # 2) GRIB2
            tmp = grib.with_suffix(".grib2.tmp")
            _fetch(run_time, tmp, fmt="grib")
            tmp.rename(grib)
            LOG.info("✓ GRIB2 descargado → %s (%.1f MB)", grib.name, grib.stat().st_size/1e6)
            return
        except Exception as e_grib:                             # noqa: BLE001
            LOG.warning("Run %s UTC no disponible aún (%s).", run_time.hour, e_grib)

    raise RuntimeError("Ninguno de los últimos 4 runs está publicado todavía.")



if __name__ == "__main__":
    main()
