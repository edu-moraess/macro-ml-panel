"""Data access layer for the Macro Quant Research Terminal.

Only public, real observations are accepted (BCB/SGS and FRED). Network
failures are surfaced as controlled errors; no synthetic or placeholder
observations are ever generated, and no other data source is used.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Final, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Streamlit Community Cloud can attempt IPv6 to api.bcb.gov.br first. Force
# IPv4-only socket resolution so failed IPv6 routes do not stall requests.
urllib3.util.connection.HAS_IPV6 = False

BCB_BASE: Final[str] = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
FRED_URL: Final[str] = "https://api.stlouisfed.org/fred/series/observations"
CACHE_TTL_SECONDS: Final[int] = 900
HTTP_TIMEOUT_SECONDS: Final[int] = 35
BCB_DEFAULT_START: Final[str] = "2000-01-01"

SGS: Final[dict[str, int]] = {
    "selic_meta": 432,
    "ipca_mensal": 433,
    "ibc_br": 24363,
    "cambio_ptax": 1,
    "desemprego_pnad": 24369,
}

FRED: Final[dict[str, str]] = {
    "t10y3m": "T10Y3M", "usrec": "USREC", "unrate": "UNRATE", "cpi_yoy": "CPIAUCSL",
    "fedfunds": "FEDFUNDS", "sp500": "SP500", "vix": "VIXCLS", "ust10y": "DGS10",
    "ust2y": "DGS2", "hy_spread": "BAMLH0A0HYM2", "wti": "DCOILWTICO",
    "gold": "GOLDAMGBD228NLBM", "brl_usd": "DEXBZUS",
}

SERIES_META: Final[dict[str, dict[str, str]]] = {
    "432": {"name": "Selic Meta", "freq": "Daily→MS", "source": "BCB/SGS"},
    "433": {"name": "IPCA", "freq": "Monthly", "source": "BCB/SGS"},
    "24363": {"name": "IBC-Br", "freq": "Monthly", "source": "BCB/SGS"},
    "1": {"name": "USD/BRL PTAX", "freq": "Daily→MS", "source": "BCB/SGS"},
    "24369": {"name": "Desemprego PNAD", "freq": "Monthly", "source": "BCB/SGS"},
    "T10Y3M": {"name": "Term Spread 10Y-3M", "freq": "Daily→MS", "source": "FRED"},
    "USREC": {"name": "NBER Recession", "freq": "Monthly", "source": "FRED"},
    "UNRATE": {"name": "US Unemployment", "freq": "Monthly", "source": "FRED"},
    "CPIAUCSL": {"name": "US CPI", "freq": "Monthly", "source": "FRED"},
    "FEDFUNDS": {"name": "Fed Funds", "freq": "Monthly", "source": "FRED"},
    "SP500": {"name": "S&P 500", "freq": "Daily→MS", "source": "FRED"},
    "VIXCLS": {"name": "VIX", "freq": "Daily→MS", "source": "FRED"},
    "DGS10": {"name": "UST 10Y", "freq": "Daily→MS", "source": "FRED"},
    "DGS2": {"name": "UST 2Y", "freq": "Daily→MS", "source": "FRED"},
    "BAMLH0A0HYM2": {"name": "HY OAS", "freq": "Daily→MS", "source": "FRED"},
    "DCOILWTICO": {"name": "WTI", "freq": "Daily→MS", "source": "FRED"},
    "GOLDAMGBD228NLBM": {"name": "Gold", "freq": "Daily→MS", "source": "FRED"},
    "DEXBZUS": {"name": "BRL/USD", "freq": "Daily→MS", "source": "FRED"},
}


def _session(retries: int = 3) -> requests.Session:
    retry = Retry(total=retries, connect=retries, read=retries, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",),
                  raise_on_status=False)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Macro-Quant-Terminal/6.0 (+quant-research; public-data-client)",
        "Accept": "application/json, text/csv, */*",
    })
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _parse_bcb(payload: list[dict[str, Any]], code: int) -> pd.Series:
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
    return series.rename(str(code))


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Consultando BCB/SGS...")
def get_bcb(code: int, start: str | None = None) -> pd.Series:
    session = _session(retries=3)
    requested_start = pd.Timestamp(start) if start else pd.Timestamp(BCB_DEFAULT_START)
    requested_end = pd.Timestamp.now().normalize()
    if requested_start > requested_end:
        raise ValueError("data inicial posterior à data atual")
    url = f"{BCB_BASE}.{int(code)}/dados"
    params = {"formato": "json", "dataInicial": requested_start.strftime("%d/%m/%Y"),
              "dataFinal": requested_end.strftime("%d/%m/%Y")}
    try:
        response = session.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        series = _parse_bcb(response.json(), int(code))
        series = series[(series.index >= requested_start) & (series.index <= requested_end)]
        if series.empty:
            raise ValueError(f"nenhuma observação atende ao período ({requested_start:%Y-%m-%d} a {requested_end:%Y-%m-%d})")
        return series.rename(str(code))
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
        response = _session(retries=2).get(FRED_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
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
        return series.rename(code)
    except requests.RequestException as exc:
        raise RuntimeError(f"FRED série {code}: conexão/API indisponível após {HTTP_TIMEOUT_SECONDS}s. Fonte: FRED; sem fallback sintético.") from exc
    except Exception as exc:
        raise RuntimeError(f"FRED série {code}: {exc}. Fonte: Federal Reserve Bank of St. Louis; sem fallback sintético.") from exc


def align(*series: pd.Series, freq: str = "MS") -> pd.DataFrame:
    out = pd.concat([s.resample(freq).mean().rename(s.name) for s in series if s is not None and not s.empty], axis=1).dropna()
    if out.empty:
        raise ValueError("as séries reais não possuem janela comum suficiente")
    return out


def data_status(source: str, reference: str, last_date: pd.Timestamp) -> str:
    updated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return f"Fonte: {source} · Série: {reference} · Última observação: {last_date:%Y-%m-%d} · Consulta: {updated} · Cache TTL: {CACHE_TTL_SECONDS // 60} min"


def series_quality(series: pd.Series, source: str = "", series_id: str = "") -> dict[str, Any]:
    if series is None or series.empty:
        return {"source": source or "—", "series_id": series_id or "—", "name": "—", "freq": "—", "first_obs": None, "last_obs": None, "n_obs": 0, "missing_rate": 1.0, "freshness_days": None, "status": "DADOS INDISPONÍVEIS"}
    sid = series_id or str(series.name)
    meta = SERIES_META.get(sid, {})
    idx = pd.DatetimeIndex(series.index).dropna().sort_values().unique()
    n = len(idx)
    first = idx.min()
    last = idx.max()
    abnormal_gap_days = 0
    missing_rate = 0.0
    if n > 1:
        diffs = pd.Series(idx[1:] - idx[:-1]).dt.days.astype(float)
        freq = meta.get("freq", "")
        if "Daily" in freq:
            abnormal_gap_days = int((diffs > 7).sum())
            if abnormal_gap_days:
                missing_rate = min(1.0, abnormal_gap_days / max(n - 1, 1))
        elif "Monthly" in freq:
            periods = idx.to_period("M")
            expected = pd.period_range(periods.min(), periods.max(), freq="M")
            missing_rate = max(0.0, 1.0 - len(periods.unique()) / max(len(expected), 1))
        else:
            median_gap = diffs.median()
            if median_gap and median_gap > 0:
                expected = int((last - first).days / median_gap) + 1
                missing_rate = max(0.0, 1.0 - n / max(expected, 1))
    freshness = (pd.Timestamp.now().normalize() - pd.Timestamp(last).normalize()).days
    status = "OK"
    if freshness > 90: status = "STALE"
    elif freshness > 45: status = "LAG"
    if n < 24: status = "AMOSTRA CURTA"
    if abnormal_gap_days > 0 or missing_rate > 0.15: status = "GAPS"
    return {"source": meta.get("source", source) or "—", "series_id": sid, "name": meta.get("name", sid), "freq": meta.get("freq", "—"), "first_obs": first, "last_obs": last, "n_obs": n, "missing_rate": round(float(missing_rate), 4), "freshness_days": int(freshness), "status": status}


def format_quality_line(q: dict[str, Any]) -> str:
    last = q["last_obs"]
    last_str = last.strftime("%Y-%m") if last is not None else "—"
    return f"{q['source']} · {q['name']} · {q['series_id']} · {q['freq']} · Last: {last_str} · n={q['n_obs']} · miss={q['missing_rate']:.1%} · STATUS: {q['status']}"


def rolling_zscore(s: pd.Series, window: int = 60, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(24, window // 2)
    r = s.rolling(window, min_periods=min_periods)
    std = r.std().replace(0, np.nan)
    return (s - r.mean()) / std
