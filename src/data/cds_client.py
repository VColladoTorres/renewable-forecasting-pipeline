# src/data/cds_client.py
# MIT © 2025 MSc Candidate
"""
Cliente robusto para ERA5/ERA5-Land (Copernicus CDS).

• Descarga mensual con caché             • Acepta lista o dict de variables
• Endpoint actualizado 2025-06           • Guarda en data/raw/cds/<var>/<YYYY-MM>.nc
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import cdsapi
import pandas as pd

_LOGGER = logging.getLogger(__name__)


class CDSClient:
    """Wrapper cdsapi con caché mensual en disco."""

    def __init__(
        self,
        out_dir: Path | str = Path("data/raw/cds"),
        key: str | None = None,
    ) -> None:
        self._home = Path(out_dir).expanduser()
        self._home.mkdir(parents=True, exist_ok=True)

        key = key or os.getenv("CDS_API_KEY")
        if not key:
            raise ValueError("Falta CDS_API_KEY (.env o argumento)")

        # ─── endpoint CORRECTO (sin /api/v2) ───
        os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
        os.environ["CDSAPI_KEY"] = key

        self._c = cdsapi.Client(timeout=3000)

    # ------------------------------------------------------------------
    def download(
        self,
        cfg: Dict[str, str] | Sequence[Dict[str, str]],
        start: datetime,
        end: datetime,
    ) -> List[Path]:
        """Descarga una o varias variables ERA5."""
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

            req = {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": spec["short_name"],
                "year": year,
                "month": month,
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
            } | spec.get("extras", {})

            self._c.retrieve(spec["dataset"], req, str(fn))
            out_files.append(fn)

        return out_files


def _month_stubs(start: datetime, end: datetime) -> list[str]:
    """['YYYY-MM', …] para cada mes entre start y end (incl.)."""
    start = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    months: list[str] = []
    cur = start
    while cur <= end:
        months.append(cur.strftime("%Y-%m"))
        cur = (cur.replace(day=28) + pd.Timedelta(days=4)).replace(day=1)
    return months
