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
FRED_API_URL: Final[str] = "https://api.stlouisfed.org/fred/series/observations"
CACHE_TTL_SECONDS: Final[int] = 900
BCB_MAX_OBSERVATIONS: Final[int] = 10000
HTTP_TIMEOUT_SECONDS: Final[int] = 10

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


def _session(retries: int = 1) -> requests.Session:
    """Create an HTTP session with a bounded retry policy for public data."""
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Macro-ML-Panel/2.2 (+research; public-data-client)",
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
    """Fetch real observations from the official BCB/SGS public API."""
    session = _session(retries=2)
    base = f"{BCB_BASE}.{int(code)}/dados"
    params: dict[str, str] = {"formato": "json"}
    if start:
        params["dataInicial"] = pd.Timestamp(start).strftime("%d/%m/%Y")
    try:
        response = session.get(base, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        if response.status_code == 406:
            response = session.get(
                f"{base}/ultimos/{BCB_MAX_OBSERVATIONS}",
                params={"formato": "json"},
                timeout=HTTP_TIMEOUT_SECONDS,
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


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_fred(code: str, start: str | None = None) -> pd.Series:
    """Fetch real FRED observations through the official API using Streamlit Secrets."""
    code = str(code).strip().upper()
    if not code or not code.replace("_", "").isalnum():
        raise ValueError(f"Código FRED inválido: {code!r}")

    api_key = str(st.secrets.get("FRED_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY não configurada nos Streamlit Secrets. "
            "Adicione a chave da FRED em Settings → Secrets. Fonte: FRED; sem fallback sintético."
        )

    params: dict[str, str] = {
        "series_id": code,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
    }
    if start:
        params["observation_start"] = pd.Timestamp(start).strftime("%Y-%m-%d")

    try:
        response = _session(retries=0).get(
            FRED_API_URL,
            params=params,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        observations = payload.get("observations")
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
        raise RuntimeError(
            f"FRED série {code}: conexão indisponível após {HTTP_TIMEOUT_SECONDS}s. "
            "Verifique FRED_API_KEY e o acesso de rede do Streamlit Cloud. "
            "Fonte: Federal Reserve Bank of St. Louis; sem fallback sintético."
        ) from exc
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
