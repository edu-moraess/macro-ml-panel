"""World Bank WDI data access for global macro research.

Uses the official World Bank Indicators API v2. No authentication, synthetic
observations, or country-specific fallback data are used.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WB_BASE: Final[str] = "https://api.worldbank.org/v2"
CACHE_TTL_SECONDS: Final[int] = 900
HTTP_TIMEOUT_SECONDS: Final[int] = 20

# Brazil is intentionally excluded from the terminal's global comparison set.
COUNTRIES: Final[dict[str, str]] = {
    "USA": "United States",
    "CHN": "China",
    "DEU": "Germany",
    "IND": "India",
    "JPN": "Japan",
    "GBR": "United Kingdom",
    "MEX": "Mexico",
    "CAN": "Canada",
    "FRA": "France",
    "ITA": "Italy",
}

INDICATORS: Final[dict[str, dict[str, str]]] = {
    "gdp_growth": {"code": "NY.GDP.MKTP.KD.ZG", "name": "GDP growth", "unit": "%"},
    "inflation": {"code": "FP.CPI.TOTL.ZG", "name": "Inflation, consumer prices", "unit": "%"},
    "unemployment": {"code": "SL.UEM.TOTL.ZS", "name": "Unemployment", "unit": "%"},
    "current_account": {"code": "BN.CAB.XOKA.GD.ZS", "name": "Current account balance", "unit": "% GDP"},
    "trade": {"code": "NE.TRD.GNFS.ZS", "name": "Trade", "unit": "% GDP"},
    "exports": {"code": "NE.EXP.GNFS.ZS", "name": "Exports", "unit": "% GDP"},
    "fdi": {"code": "BX.KLT.DINV.WD.GD.ZS", "name": "FDI net inflows", "unit": "% GDP"},
    "reserves": {"code": "FI.RES.TOTL.CD", "name": "Total reserves", "unit": "current US$"},
    "life_expectancy": {"code": "SP.DYN.LE00.IN", "name": "Life expectancy", "unit": "years"},
    "population": {"code": "SP.POP.TOTL", "name": "Population", "unit": "people"},
}


def _session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Macro-Quant-Terminal/WorldBank-WDI",
        "Accept": "application/json",
    })
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Consultando World Bank WDI...")
def get_world_bank(indicator: str, countries: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Return annual WDI observations indexed by year and country columns."""
    if indicator not in INDICATORS:
        raise ValueError(f"Indicador World Bank desconhecido: {indicator}")
    selected = tuple(countries or tuple(COUNTRIES))
    selected = tuple(c for c in selected if c in COUNTRIES and c != "BRA")
    if not selected:
        raise ValueError("Nenhum país válido selecionado")

    code = INDICATORS[indicator]["code"]
    url = f"{WB_BASE}/country/{';'.join(selected)}/indicator/{code}"
    params = {"format": "json", "per_page": 20000, "date": "2000:2025"}
    try:
        response = _session().get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            raise ValueError("World Bank retornou estrutura sem observações")
        rows = []
        for obs in payload[1]:
            if obs.get("value") is None:
                continue
            rows.append({
                "year": pd.to_numeric(obs.get("date"), errors="coerce"),
                "country_code": obs.get("countryiso3code"),
                "country": COUNTRIES.get(obs.get("countryiso3code"), obs.get("country", "")),
                "value": pd.to_numeric(obs.get("value"), errors="coerce"),
            })
        df = pd.DataFrame(rows).dropna(subset=["year", "value"])
        if df.empty:
            raise ValueError(f"World Bank não retornou observações para {code}")
        df["year"] = df["year"].astype(int)
        return df.sort_values(["year", "country"])
    except requests.RequestException as exc:
        raise RuntimeError("World Bank WDI indisponível; sem fallback sintético.") from exc
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"World Bank WDI: {exc}; sem fallback sintético.") from exc


def latest_table(indicator: str, countries: tuple[str, ...] | None = None) -> pd.DataFrame:
    df = get_world_bank(indicator, countries)
    latest_year = int(df["year"].max())
    out = df[df["year"] == latest_year].copy()
    out["year"] = latest_year
    return out.sort_values("value", ascending=False)


def quality(indicator: str, countries: tuple[str, ...] | None = None) -> dict[str, object]:
    df = get_world_bank(indicator, countries)
    meta = INDICATORS[indicator]
    return {
        "source": "World Bank WDI",
        "indicator": meta["name"],
        "code": meta["code"],
        "unit": meta["unit"],
        "first_year": int(df["year"].min()),
        "last_year": int(df["year"].max()),
        "countries": int(df["country_code"].nunique()),
        "observations": int(len(df)),
        "queried_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
