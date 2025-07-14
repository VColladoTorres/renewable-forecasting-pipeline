#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genera una serie horaria de capacidad instalada (MW) interpolada
hacia delante, compatible con índices con o sin zona horaria.
"""

from __future__ import annotations
import pandas as pd


def _to_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Devuelve el mismo índice sin información de zona horaria."""
    return idx.tz_localize(None) if idx.tz is not None else idx


def build_capacity_series(timeline: dict[str, dict[str, float]],
                          index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Parameters
    ----------
    timeline : {tech: {iso-date: capacity_MW}}
    index    : DatetimeIndex (posiblemente tz-aware)

    Returns
    -------
    DataFrame con columnas = tecnologías, índice = `index`,
    valores ffill sobre la capacidad.
    """
    idx_naive = _to_naive(index)
    caps = {}
    for tech, pts in timeline.items():
        s = pd.Series({pd.to_datetime(k): v for k, v in pts.items()}).sort_index()
        caps[tech] = s.reindex(idx_naive, method="ffill").values
    out = pd.DataFrame(caps, index=index)          # conserva zona horaria original
    return out
