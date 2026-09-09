"""Data access layer for the Macro-ML Panel.

Only public, real observations are accepted. The BCB/SGS client uses the official
JSON interface and a bounded retry strategy; it never fabricates observations.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Final
from io import StringIO

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BCB_BASE: Final[str] = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
FRED_URL: Final[str] = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
CACHE_TTL_SECONDS: Final[int] = 900
BCB_MAX_OBSERVATIONS: Final[int] = 10000

SGS: Final[dict[str, int]] = {
    "selic_meta": 432,
    "ipca_mensal": 433,
    "ibc_br": 24363,
    "cambio_ptax": 1,
    "desemprego_pnad": 24369,
}

FRED: Final[dict[str, str]] = {
    "t10y3m": "T10Y3M",
    "usrec": "USREC",
    "unrate": "UNRATE",
    "cpi_yoy": "CPIAUCSL",
    "fedfunds": "FEDFUNDS",
}


def _session() -> requests.Session:
    """Create an HTTP session with conservative retries for transient failures."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Macro-ML-Panel/2.0 (+research; public-data-client)",
        "Accept": "application/json, text/csv, */*",
    })
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _parse_bcb(payload: list[dict[str, object]], code: int) -> pd.Series:
    """Validate and normalize a BCData/SGS JSON payload."""
    if not isinstance(payload, list) or not payload:
        raise ValueError("BCB retornou uma lista vazia")
    df = pd.DataFrame(payload)
    if not {"data", "valor"}.issubset(df.columns):
        raise ValueError("resposta BCB sem os campos obrigatórios 'data' e 'valor'")
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["data", "valor"]).drop_duplicates("data")
    series = df.set_index("data")["valor"].sort_index()
    if series.empty:
        raise ValueError(f"BCB série {code} não contém observações numéricas")
    series.name = str(code)
    return series


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Consultando BCB/SGS...")
def get_bcb(code: int, start: str | None = None) -> pd.Series:
    """Fetch real observations from the official BCB/SGS public API.

    A 406 from hosted environments is retried through the official /ultimos/N
    endpoint. Both paths contain only observations published by BCB; no synthetic
    fallback is ever generated.
    """
    session = _session()
    base = f"{BCB_BASE}.{int(code)}/dados"
    params: dict[str, str] = {"formato": "json"}
    if start:
        params["dataInicial"] = pd.Timestamp(start).strftime("%d/%m/%Y")
    try:
        response = session.get(base, params=params, timeout=20)
        if response.status_code == 406:
            response = session.get(
                f"{base}/ultimos/{BCB_MAX_OBSERVATIONS}",
                params={"formato": "json"},
                timeout=20,
            )
        response.raise_for_status()
        series = _parse_bcb(response.json(), int(code))
        if start:
            series = series[series.index >= pd.Timestamp(start)]
        if series.empty:
            raise ValueError("nenhuma observação atende ao período solicitado")
        return series
    except Exception as exc:
        raise RuntimeError(
            f"BCB/SGS série {code}: {exc}. Fonte: BCData/SGS; sem fallback sintético."
        ) from exc


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Consultando FRED...")
def get_fred(code: str, start: str | None = None) -> pd.Series:
    """Fetch a real FRED series from its public CSV endpoint."""
    try:
        response = _session().get(FRED_URL.format(code=code), timeout=20)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        series = df.dropna().set_index("date")["value"].sort_index()
        if start:
            series = series[series.index >= pd.Timestamp(start)]
        if series.empty:
            raise ValueError("resposta FRED vazia para o período solicitado")
        series.name = code
        return series
    except Exception as exc:
        raise RuntimeError(
            f"FRED série {code}: {exc}. Fonte: Federal Reserve Bank of St. Louis; sem fallback sintético."
        ) from exc


def align(*series: pd.Series, freq: str = "MS") -> pd.DataFrame:
    """Resample and inner-align real series at a common frequency."""
    out = pd.concat([s.resample(freq).mean().rename(s.name) for s in series], axis=1).dropna()
    if out.empty:
        raise ValueError("as séries reais não possuem janela comum suficiente")
    return out


def data_status(source: str, reference: str, last_date: pd.Timestamp) -> str:
    """Return a compact provenance line for the research UI."""
    updated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return (
        f"Fonte: {source} · Série: {reference} · Última observação: {last_date:%Y-%m-%d} · "
        f"Consulta: {updated} · Cache TTL: {CACHE_TTL_SECONDS // 60} min"
    )
