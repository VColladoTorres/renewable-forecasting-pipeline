# src/data/debug_esios.py
"""
Herramienta de depuración para la API de e·sios.

USO ──────────────────────────────────────────────────────────────────────
python -m src.data.debug_esios 551 2025-07-05          # ID y fecha única
python -m src.data.debug_esios 551 2025-07-05 2025-07-06
python -m src.data.debug_esios search "solar fotovoltaica"
python -m src.data.debug_esios scan 540 560
"""
from dotenv import load_dotenv, find_dotenv   # ← nuevo
load_dotenv(find_dotenv())                    # ← nuevo
import os, sys, json, requests, datetime as dt
from textwrap import indent

BASE = "https://api.esios.ree.es"
HDRS = {
    #  v2 sigue aceptando v1
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "x-api-key": os.getenv("ESIOS_TOKEN", ""),
}

def _req(path: str, params=None):
    r = requests.get(f"{BASE}{path}", headers=HDRS, params=params or {})
    if r.status_code == 403:
        raise SystemExit("403 Forbidden → token ausente, caducado o cabeceras incorrectas")
    r.raise_for_status()
    return r.json()

def show_values(ind: int, d0: str, d1: str | None):
    d1 = d1 or d0
    j = _req(f"/indicators/{ind}",
             {"start_date": f"{d0}T00:00:00Z",
              "end_date":   f"{d1}T23:00:00Z",
              "time_trunc": "hour"})
    vals = j["indicator"]["values"]
    print(f"{ind=}  rows={len(vals)}")
    if vals:
        print(indent(json.dumps(vals[:3], indent=2), "  "))

def search(term: str, limit=10):
    res = _req("/indicators", {"search": term, "page_size": limit})
    for x in res["indicators"]:
        print(f"{x['id']:>6}  {x['short_name'][:60]}")

def scan(lo: int, hi: int, date="2025-07-05"):
    for ind in range(lo, hi + 1):
        try:
            show_values(ind, date, date)
        except requests.HTTPError as e:
            print(f"{ind}: {e.response.status_code}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Argumentos insuficientes")
    if sys.argv[1] == "search":
        search(" ".join(sys.argv[2:]))
    elif sys.argv[1] == "scan":
        scan(int(sys.argv[2]), int(sys.argv[3]))
    else:
        ind = int(sys.argv[1])
        d0  = sys.argv[2]
        d1  = sys.argv[3] if len(sys.argv) > 3 else None
        show_values(ind, d0, d1)
