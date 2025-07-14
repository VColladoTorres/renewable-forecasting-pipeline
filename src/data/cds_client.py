# src/data/cds_client.py
# MIT © 2025 MSc Candidate
"""
Cliente robusto para ERA5 / ERA5-Land (Copernicus CDS).

• Sub-conjunto geográfico (area / grid)                • Caché mensual en disco
• Validación rápida de cada NetCDF                    • Salida → data/raw/cds/<var>/<YYYY-MM>.nc
"""

from __future__ import annotations

import calendar
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import cdsapi
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from requests import HTTPError

load_dotenv()
_LOGGER = logging.getLogger(__name__)


class CDSClient:
    """Wrapper cdsapi con caché, área y grid."""

    def __init__(
        self,
        out_dir: Path | str = Path("data/raw/cds"),
        key: str | None = None,
        timeout: int = 3000,
    ) -> None:
        self._home = Path(out_dir).expanduser()
        self._home.mkdir(parents=True, exist_ok=True)

        # ─── credenciales ────────────────────────────────────────────
        key = key or os.getenv("CDS_API_KEY")          # ~/.cdsapirc preferente
        if key:
            os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
            os.environ["CDSAPI_KEY"] = key
        # ─────────────────────────────────────────────────────────────

        self._c = cdsapi.Client(timeout=timeout)

    # =================================================================
    def download(
        self,
        cfg: Dict[str, str] | Sequence[Dict[str, str]],
        start: datetime,
        end: datetime,
    ) -> List[Path]:
        cfgs = cfg if isinstance(cfg, list) else [cfg]
        paths: list[Path] = []
        for spec in cfgs:
            paths.extend(self._download_one(spec, start, end))
        return paths

    # --------------------------- interno -----------------------------
    # CAMBIO: versión robusta con verificación de tamaño/validez y salto si ya existe
    def _download_one(
            self,
            spec: Dict[str, str],
            start: datetime,
            end: datetime,
    ) -> List[Path]:
        """Descarga un conjunto ERA5 mensual y devuelve las rutas resultantes."""
        months = _month_stubs(start, end)
        out_files: list[Path] = []

        for ym in months:
            year, month = ym.split("-")
            fn = self._home / spec["name"] / f"{ym}.nc"

            # ── 1. Omitir si el archivo ya existe y es válido (>0 B y NetCDF legible) ──
            if fn.exists():
                if fn.stat().st_size > 0 and _is_netcdf_ok(fn):
                    _LOGGER.info("✓ %s existe (%.1f MB) – omitido", fn.name, fn.stat().st_size / 1e6)
                    out_files.append(fn)
                    continue
                _LOGGER.warning("%s existente incompleto/corrupto – se vuelve a descargar", fn.name)
                fn.unlink(missing_ok=True)

            _LOGGER.info("ERA5 %s %s – descarga…", spec["name"], ym)
            fn.parent.mkdir(parents=True, exist_ok=True)

            # ── 2. Construcción de la petición ───────────────────────────────────────
            n_days = calendar.monthrange(int(year), int(month))[1]
            req = {
                # product_type puede venir en el YAML; si no, se fija a 'reanalysis'
                "product_type": spec.get("product_type", "reanalysis"),
                "variable": [spec["short_name"]],
                "year": year,
                "month": month,
                "day": [f"{d:02d}" for d in range(1, n_days + 1)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "format": "netcdf",
            }

            for k in ("area", "grid"):
                if k in spec:
                    val = spec[k]
                    req[k] = "/".join(map(str, val)) if isinstance(val, (list, tuple)) else val

            req |= spec.get("extras", {})
            _LOGGER.debug("REQUEST JSON →\n%s", json.dumps(req, indent=2))

            # ── 3. Descarga con manejo de errores ────────────────────────────────────
            try:
                self._c.retrieve(spec["dataset"], req, str(fn))
            except HTTPError as e:
                _LOGGER.error("CDS 400 Bad Request (%s):\n%s", spec["name"], e.response.text)
                raise

            # ── 4. Validación rápida ────────────────────────────────────────────────
            if not _is_netcdf_ok(fn):
                _LOGGER.warning("%s corrupto → reintento 1/1", fn.name)
                fn.unlink(missing_ok=True)
                self._c.retrieve(spec["dataset"], req, str(fn))
                if not _is_netcdf_ok(fn):
                    raise RuntimeError(f"NetCDF corrupto tras reintento: {fn}")

            out_files.append(fn)

        return out_files


# =====================================================================
def _is_netcdf_ok(path: Path) -> bool:
    try:
        xr.open_dataset(path, engine="h5netcdf").close()
        return True
    except Exception as exc:
        _LOGGER.error("NetCDF inválido (%s): %s", path.name, exc)
        return False


def _month_stubs(start: datetime, end: datetime) -> list[str]:
    start = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end   = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    months: list[str] = []
    cur = start
    while cur <= end:
        months.append(cur.strftime("%Y-%m"))
        cur = (cur.replace(day=28) + pd.Timedelta(days=4)).replace(day=1)  # +1 mes
    return months
