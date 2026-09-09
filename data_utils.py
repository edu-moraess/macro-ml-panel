"""Data access layer for the Macro-ML Panel.

Only public, real observations are accepted. Network failures are surfaced as
controlled errors; no synthetic or placeholder observations are ever generated.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BCB_BASE: Final[str] = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
FRED_URL: Final[str] = "https://api.stlouisfed.org/fred/series/observations"
CACHE_TTL_SECONDS: Final[int] = 900
HTTP_TIMEOUT_SECONDS: Final[int] = 10
BCB_CHUNK_YEARS: Final[int] = 9
BCB_DEFAULT_START: Final[str] = "2000-01-01"

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
    "sp500": "SP500",
    "vix": "VIXCLS",
    "ust10y": "DGS10",
    "ust2y": "DGS2",
    "hy_spread": "BAMLH0A0HYM2",
    "wti": "DCOILWTICO",
    "gold": "GOLDAMGBD228NLBM",
    "brl_usd": "DEXBZUS",
}


def _session(retries: int = 1) -> requests.Session:
    retry = Retry(total=retries, connect=retries, read=retries, backoff_factor=0.4,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",),
                  raise_on_status=False)
    session = requests.Session()
    session.headers.update({"User-Agent": "Macro-ML-Panel/3.0 (+quant-research; public-data-client)",
                             "Accept": "application/json, text/csv, */*"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _parse_bcb(payload: list[dict[str, object]], code: int) -> pd.Series:
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


def _date_windows(start: pd.Timestamp, end: pd.Timestamp):
    cursor = start.normalize()
    while cursor <= end:
        window_end = min(cursor + pd.DateOffset(years=BCB_CHUNK_YEARS) - pd.Timedelta(days=1), end)
        yield cursor, window_end
        cursor = window_end + pd.Timedelta(days=1)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Consultando BCB/SGS...")
def get_bcb(code: int, start: str | None = None) -> pd.Series:
    session = _session(retries=2)
    base = f"{BCB_BASE}.{int(code)}/dados"
    requested_start = pd.Timestamp(start) if start else pd.Timestamp(BCB_DEFAULT_START)
    requested_end = pd.Timestamp.now().normalize()
    if requested_start > requested_end:
        raise ValueError("data inicial posterior à data atual")
    frames: list[pd.Series] = []
    try:
        for window_start, window_end in _date_windows(requested_start, requested_end):
            response = session.get(base, params={"formato": "json",
                "dataInicial": window_start.strftime("%d/%m/%Y"),
                "dataFinal": window_end.strftime("%d/%m/%Y")}, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            frames.append(_parse_bcb(response.json(), int(code)))
        if not frames:
            raise ValueError("nenhuma janela de consulta retornou dados")
        series = pd.concat(frames).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        series = series[series.index >= requested_start]
        if series.empty:
            raise ValueError("nenhuma observação atende ao período solicitado")
        series.name = str(code)
        return series
    except requests.RequestException as exc:
        raise RuntimeError(f"BCB/SGS série {code}: conexão/API indisponível após {HTTP_TIMEOUT_SECONDS}s. Fonte: BCData/SGS; sem fallback sintético.") from exc
    except Exception as exc:
        raise RuntimeError(f"BCB/SGS série {code}: {exc}. Fonte: BCData/SGS; sem fallback sintético.") from exc


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_fred(code: str, start: str | None = None) -> pd.Series:
    code = str(code).strip().upper()
    if not code or not code.replace("_", "").isalnum():
        raise ValueError(f"Código FRED inválido: {code!r}")
    try:
        api_key = st.secrets["FRED_API_KEY"]
    except Exception as exc:
        raise RuntimeError("FRED_API_KEY não configurada nos Streamlit Secrets. Fonte: FRED; sem fallback sintético.") from exc
    params = {"series_id": code, "api_key": api_key, "file_type": "json", "sort_order": "asc"}
    if start:
        params["observation_start"] = pd.Timestamp(start).strftime("%Y-%m-%d")
    try:
        response = _session(retries=0).get(FRED_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        observations = response.json().get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError("FRED retornou zero observações")
        df = pd.DataFrame(observations)
        if not {"date", "value"}.issubset(df.columns):
            raise ValueError("resposta FRED sem os campos obrigatórios 'date' e 'value'")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        series = df.dropna(subset=["date", "value"]).set_index("date")["value"].sort_index()
        if series.empty:
            raise ValueError("FRED retornou zero observações numéricas")
        series.name = code
        return series
    except requests.RequestException as exc:
        raise RuntimeError(f"FRED série {code}: conexão/API indisponível após {HTTP_TIMEOUT_SECONDS}s. Fonte: FRED; sem fallback sintético.") from exc
    except Exception as exc:
        raise RuntimeError(f"FRED série {code}: {exc}. Fonte: Federal Reserve Bank of St. Louis; sem fallback sintético.") from exc


def align(*series: pd.Series, freq: str = "MS") -> pd.DataFrame:
    out = pd.concat([s.resample(freq).mean().rename(s.name) for s in series], axis=1).dropna()
    if out.empty:
        raise ValueError("as séries reais não possuem janela comum suficiente")
    return out


def data_status(source: str, reference: str, last_date: pd.Timestamp) -> str:
    updated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return f"Fonte: {source} · Série: {reference} · Última observação: {last_date:%Y-%m-%d} · Consulta: {updated} · Cache TTL: {CACHE_TTL_SECONDS // 60} min"
