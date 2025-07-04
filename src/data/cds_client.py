# src/data/cds_client.py
# MIT © 2025 MSc Candidate
"""
Cliente robusto para ERA5/ERA5-Land (Copernicus CDS).

• Sub-conjunto geográfico (area/grid)               • Descarga mensual con caché
• Endpoint 2025-06 (https://…/api)                  • Guarda en data/raw/cds/<var>/<YYYY-MM>.nc
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import cdsapi
import pandas as pd
from dotenv import load_dotenv            # carga .env (ESIOS_TOKEN, etc.)

load_dotenv()

_LOGGER = logging.getLogger(__name__)


class CDSClient:
    """Wrapper cdsapi con caché mensual en disco y soporte de área + grid."""

    def __init__(
        self,
        out_dir: Path | str = Path("data/raw/cds"),
        key: str | None = None,
        timeout: int = 3000,
    ) -> None:
        self._home = Path(out_dir).expanduser()
        self._home.mkdir(parents=True, exist_ok=True)

        # ───── credenciales ──────────────────────────────────────────────
        key = key or os.getenv("CDS_API_KEY")        # opcional; ~/.cdsapirc es preferente
        if key:                                      # si se pasa, sobre-escribe el entorno
            os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
            os.environ["CDSAPI_KEY"] = key
        # ─────────────────────────────────────────────────────────────────

        self._c = cdsapi.Client(timeout=timeout)

    # ------------------------------------------------------------------
    def download(
        self,
        cfg: Dict[str, str] | Sequence[Dict[str, str]],
        start: datetime,
        end: datetime,
    ) -> List[Path]:
        """Descarga una o varias variables ERA5/ERA5-Land según la lista cfg."""
        cfgs = cfg if isinstance(cfg, list) else [cfg]
        paths: list[Path] = []
        for spec in cfgs:
            paths.extend(self._download_one(spec, start, end))
        return paths

    # ------------------ internals ------------------
    def _download_one(
        self, spec: Dict[str, str], start: datetime, end: datetime
    ) -> List[Path]:
        months = _month_stubs(start, end)
        out_files: list[Path] = []

        for ym in months:
            year, month = ym.split("-")
            fn = self._home / spec["name"] / f"{ym}.nc"
            if fn.exists():
                _LOGGER.info("✓ %s existe – omitido", fn.name)
                out_files.append(fn)
                continue

            _LOGGER.info("ERA5 %s %s – descarga…", spec["name"], ym)
            fn.parent.mkdir(parents=True, exist_ok=True)

            # --------------------- request base -------------------------
            req = {
                "product_type": "reanalysis",
                "variable":     spec["short_name"],
                "year":         year,
                "month":        month,
                "day":          [f"{d:02d}" for d in range(1, 32)],
                "time":         [f"{h:02d}:00" for h in range(24)],
                "format":       spec.get("data_format", "netcdf"),  # default netcdf
            }
            # ------------------- área / grid opcionales ----------------
            for k in ("area", "grid"):
                if k in spec:
                    val = spec[k]
                    # API espera string "N/W/S/E" o "0.25/0.25"
                    if isinstance(val, (list, tuple)):
                        req[k] = "/".join(str(x) for x in val)
                    else:
                        req[k] = val
            # ------------------- extras arbitrarios --------------------
            req |= spec.get("extras", {})

            # LOG de depuración
            _LOGGER.debug("REQUEST JSON →\n%s", json.dumps(req, indent=2))

            # ------------------- descarga ------------------------------
            self._c.retrieve(spec["dataset"], req, str(fn))
            out_files.append(fn)

        return out_files


def _month_stubs(start: datetime, end: datetime) -> list[str]:
    """Devuelve ['YYYY-MM', …] para cada mes de start a end (incl.)."""
    start = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    months: list[str] = []
    cur = start
    while cur <= end:
        months.append(cur.strftime("%Y-%m"))
        cur = (cur.replace(day=28) + pd.Timedelta(days=4)).replace(day=1)  # +1 mes
    return months
