"""Data access layer for the Macro Quant Research Terminal.

Only public, real observations are accepted. Network failures are surfaced as
controlled errors; no synthetic or placeholder observations are ever generated.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Final, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BCB_BASE: Final[str] = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
FRED_URL: Final[str] = "https://api.stlouisfed.org/fred/series/observations"
IPEA_BASE: Final[str] = "https://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{}')"
YAHOO_CHART_URL: Final[str] = "https://query1.finance.yahoo.com/v8/finance/chart/{}"
CACHE_TTL_SECONDS: Final[int] = 900
HTTP_TIMEOUT_SECONDS: int = 20
BCB_DEFAULT_START: Final[str] = "2000-01-01"

SGS: Final[dict[str, int]] = {
    "selic_meta": 432,
    "ipca_mensal": 433,
    "ibc_br": 24363,
    "cambio_ptax": 1,
    "desemprego_pnad": 24369,
}

# The numeric keys are retained for compatibility with the existing modules.
# Transport for these Brazilian macro series is now provided by IPEADATA and,
# for PTAX/FX, Yahoo Finance. The application no longer depends on BCB/SGS.
BRAZIL_PROVIDER: Final[dict[int, str]] = {
    432: "BM12_TJOVER12",          # Selic monthly, IPEA
    433: "PRECOS12_IPCAG12",       # IPCA monthly variation, IPEA
    24363: "SGS12_IBCBRDESSAZ12",  # IBC-Br real seasonally adjusted, IPEA mirror
    24369: "PNADC12_TDESOC12",     # unemployment, IPEA
}

FRED: Final[dict[str, str]] = {
    "t10y3m": "T10Y3M", "usrec": "USREC", "unrate": "UNRATE", "cpi_yoy": "CPIAUCSL",
    "fedfunds": "FEDFUNDS", "sp500": "SP500", "vix": "VIXCLS", "ust10y": "DGS10",
    "ust2y": "DGS2", "hy_spread": "BAMLH0A0HYM2", "wti": "DCOILWTICO",
    "gold": "GOLDAMGBD228NLBM", "brl_usd": "DEXBZUS",
}

SERIES_META: Final[dict[str, dict[str, str]]] = {
    "432": {"name": "Selic Meta / Over", "freq": "Monthly", "source": "IPEA"},
    "433": {"name": "IPCA", "freq": "Monthly", "source": "IPEA"},
    "24363": {"name": "IBC-Br dessazonalizado", "freq": "Monthly", "source": "IPEA"},
    "1": {"name": "USD/BRL", "freq": "Daily", "source": "Yahoo Finance"},
    "24369": {"name": "Desemprego PNAD", "freq": "Monthly", "source": "IPEA"},
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


def _session(retries: int = 2) -> requests.Session:
    retry = Retry(total=retries, connect=retries, read=retries, backoff_factor=0.5,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",),
                  raise_on_status=False)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Macro-Quant-Terminal/5.0 (+quant-research; public-data-client)",
        "Accept": "application/json, text/csv, */*",
    })
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _parse_ipea(payload: dict[str, Any], code: str) -> pd.Series:
    values = payload.get("value") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values:
        raise ValueError(f"IPEA série {code} retornou zero observações")
    df = pd.DataFrame(values)
    if not {"VALDATA", "VALVALOR"}.issubset(df.columns):
        raise ValueError(f"IPEA série {code} sem VALDATA/VALVALOR")
    df["date"] = pd.to_datetime(df["VALDATA"], errors="coerce")
    df["value"] = pd.to_numeric(df["VALVALOR"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).drop_duplicates("date").sort_values("date")
    if df.empty:
        raise ValueError(f"IPEA série {code} não contém observações numéricas")
    return df.set_index("date")["value"].rename(code)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Consultando IPEA/Yahoo...")
def get_brazil_series(key: int, start: str | None = None) -> pd.Series:
    """Load a Brazilian macro series without using the BCB/SGS transport."""
    key = int(key)
    requested_start = pd.Timestamp(start) if start else pd.Timestamp(BCB_DEFAULT_START)
    requested_end = pd.Timestamp.now().normalize()

    if key == 1:
        # Yahoo Finance BRL=X is a public market-data endpoint and is used only
        # for the FX series; no synthetic observations or fallback values.
        try:
            period1 = int(requested_start.tz_localize("UTC").timestamp())
            period2 = int((requested_end + pd.Timedelta(days=1)).tz_localize("UTC").timestamp())
            response = _session(1).get(
                YAHOO_CHART_URL.format("BRL=X"),
                params={"period1": period1, "period2": period2, "interval": "1d", "events": "history"},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            result = response.json().get("chart", {}).get("result", [])
            if not result:
                raise ValueError("Yahoo Finance retornou zero resultados para BRL=X")
            chart = result[0]
            dates = pd.to_datetime(chart.get("timestamp", []), unit="s", utc=True).tz_convert(None)
            values = chart.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            series = pd.Series(pd.to_numeric(values, errors="coerce"), index=dates, name="1").dropna()
            series = series[(series.index >= requested_start) & (series.index <= requested_end)]
            if series.empty:
                raise ValueError("Yahoo Finance BRL=X não possui observações no período")
            return series
        except requests.RequestException as exc:
            raise RuntimeError("Yahoo Finance BRL=X indisponível; sem fallback sintético.") from exc

    ipea_code = BRAZIL_PROVIDER.get(key)
    if not ipea_code:
        raise ValueError(f"Nenhum provedor alternativo configurado para série brasileira {key}")
    try:
        response = _session(2).get(IPEA_BASE.format(ipea_code), timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        series = _parse_ipea(response.json(), ipea_code)
        series = series[(series.index >= requested_start) & (series.index <= requested_end)]
        if series.empty:
            raise ValueError(f"IPEA {ipea_code} não possui observações no período")
        return series.rename(str(key))
    except requests.RequestException as exc:
        raise RuntimeError(
            f"IPEA série {ipea_code}: conexão/API indisponível após {HTTP_TIMEOUT_SECONDS}s. "
            "Fonte: IPEA Data; sem fallback sintético."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"IPEA série {ipea_code}: {exc}. Sem fallback sintético.") from exc


# Compatibility surface: all existing modules keep calling get_bcb(code), but
# the function now routes Brazilian series through the alternative providers.
def get_bcb(code: int, start: str | None = None) -> pd.Series:
    return get_brazil_series(int(code), start=start)


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
        response = _session(1).get(FRED_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        observations = response.json().get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError("FRED retornou zero observações")
        df = pd.DataFrame(observations)
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
    n = len(series)
    span_days = max((series.index.max() - series.index.min()).days, 1)
    missing_rate = 0.0
    if n > 2:
        diffs = series.index.to_series().diff().dt.days.dropna()
        median_gap = diffs.median()
        if median_gap and median_gap > 0:
            expected = int(span_days / median_gap) + 1
            missing_rate = max(0.0, 1.0 - n / expected)
    last = series.index.max()
    freshness = (pd.Timestamp.now().normalize() - last).days
    status = "OK"
    if freshness > 90: status = "STALE"
    elif freshness > 45: status = "LAG"
    if n < 24: status = "AMOSTRA CURTA"
    if missing_rate > 0.15: status = "GAPS"
    return {"source": meta.get("source", source) or "—", "series_id": sid, "name": meta.get("name", sid), "freq": meta.get("freq", "—"), "first_obs": series.index.min(), "last_obs": last, "n_obs": n, "missing_rate": round(float(missing_rate), 4), "freshness_days": int(freshness), "status": status}


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
