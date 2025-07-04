# src/data/esios_client.py
# MIT © 2025 MSc Candidate
"""
Cliente robusto para la API de ESIOS (y fallback a apidados.ree.es).

Características
───────────────
• Autenticación por x-api-key (si la hay)               • time_trunc=hour
• Back-off exponencial + reintentos (429 / 5xx)         • Fallback a API pública
• Troceo ≤ 24 h UTC                                     • Caché JSON por día
• Índice horario tz-aware sin duplicados                • Logs legibles
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────── Config ────────────────────────────
load_dotenv()
_LOGGER = logging.getLogger(__name__)
BASE_PRIV = "https://api.esios.ree.es"
BASE_PUB  = "https://apidatos.ree.es"        # fallback sin token
CACHE_DIR = Path("data/raw/esios")           # se crea al vuelo
HEADERS_BASE = {
    "Accept": "application/json; application/vnd.esios-api-v2+json",
    "Content-Type": "application/json",
}
IDS: dict[str, int] = {
    "wind": 551,
    "solar_pv": 1161,
    "demand": 1,
}
# ────────────────────────────────────────────────────────────────


def _chunks_day(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Ventanas de 24 h UTC exactas (requisito API)."""
    cur, out = start, []
    while cur <= end:
        nxt = min(cur + timedelta(hours=23), end)
        out.append((cur, nxt))
        cur = (nxt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return out


def _json_to_df(values: List[Dict[str, Any]]) -> pd.DataFrame:
    """Lista de valores → DataFrame UTC; tolera lista vacía."""
    if not values:
        return pd.DataFrame(columns=["mw"])
    df = pd.DataFrame(values)
    ts_col = next((c for c in ("datetime", "datetime_utc", "date", "timestamp") if c in df), None)
    if ts_col is None:
        raise ValueError(f"Timestamp no encontrado en {df.columns.tolist()}")
    df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df.set_index("ts", inplace=True)
    val_col = next((c for c in df.columns if c.endswith("value")), "value")
    return df[[val_col]].rename(columns={val_col: "mw"})


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados de índice manteniendo la 1.ª aparición."""
    return df[~df.index.duplicated(keep="first")]


class EsiosClient:
    """Cliente resiliente para series horarias de ESIOS."""

    def __init__(self, token: str | None = None) -> None:
        self.api_key = token or os.getenv("ESIOS_API_KEY") or os.getenv("ESIOS_TOKEN")
        self._session = self._build_session()

    # ───────────────────────── public ──────────────────────────
    def fetch_series(self, start: datetime, end: datetime, technology: str = "wind") -> pd.DataFrame:
        ind_id = IDS[technology]
        frames: list[pd.DataFrame] = []
        for s, e in _chunks_day(start, end):
            chunk = self._get_day(ind_id, s, e)
            if not chunk.empty:
                frames.append(chunk)
        if not frames:
            _LOGGER.warning("Sin datos %s entre %s y %s", technology, start, end)
            return pd.DataFrame(columns=[technology])
        df = (
            pd.concat(frames).sort_index()
            .tz_convert("Europe/Madrid")
            .pipe(_dedup)
            .rename(columns={"mw": technology})
        )
        return df

    # ───────────────────────── internal ────────────────────────
    def _build_session(self) -> requests.Session:
        """Sesión con retry exponencial."""
        retry_cfg = Retry(
            total=6,
            backoff_factor=0.7,                     # 0.7, 1.4, 2.8, …
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        s = requests.Session()
        s.mount("https://", HTTPAdapter(max_retries=retry_cfg))
        return s

    def _headers(self) -> Dict[str, str]:
        return HEADERS_BASE | ({"x-api-key": self.api_key} if self.api_key else {})

    # -------------- caché --------------
    def _cache_file(self, ind: int, day: datetime) -> Path:
        day_str = day.strftime("%Y-%m-%d")
        p = CACHE_DIR / str(ind) / f"{day_str}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load_cache(self, ind: int, day: datetime) -> list | None:
        p = self._cache_file(ind, day)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def _save_cache(self, ind: int, day: datetime, payload: list) -> None:
        p = self._cache_file(ind, day)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    # -------------- fetch --------------
    def _get_day(self, ind: int, s: datetime, e: datetime) -> pd.DataFrame:
        """Descarga un bloque de ≤24 h con caché y fallback."""
        cached = self._load_cache(ind, s)
        if cached is not None:
            return _json_to_df(cached)

        url_priv = self._build_url(BASE_PRIV, ind, s, e)
        resp = self._session.get(url_priv, headers=self._headers(), timeout=30)

        # fallback si token caducó o demasiados 5xx
        if resp.status_code in (401, 403) or resp.status_code >= 500:
            _LOGGER.warning("Falló %s → %s, probando API pública", url_priv, resp.status_code)
            url_pub = self._build_url(BASE_PUB, ind, s, e)
            resp = self._session.get(url_pub, headers=HEADERS_BASE, timeout=30)

        resp.raise_for_status()
        payload = resp.json()["indicator"]["values"]
        self._save_cache(ind, s, payload)
        return _json_to_df(payload)

    @staticmethod
    def _build_url(base: str, ind: int, s: datetime, e: datetime) -> str:
        return (
            f"{base}/indicators/{ind}"
            f"?start_date={s:%Y-%m-%dT%H:%M:%SZ}"
            f"&end_date={e:%Y-%m-%dT%H:%M:%SZ}"
            "&time_trunc=hour"
        )