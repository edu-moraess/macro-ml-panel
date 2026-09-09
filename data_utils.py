"""Data access layer for the Macro-ML Panel.

Only public, real observations are accepted. Network failures are surfaced as
controlled errors; no synthetic or placeholder observations are ever generated.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from typing import Final

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BCB_BASE: Final[str] = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
FRED_URL: Final[str] = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&download=1"
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
        "User-Agent": "Macro-ML-Panel/2.1 (+research; public-data-client)",
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
    """Fetch a real FRED series through the public CSV download endpoint.

    The request is deliberately bounded so a blocked/slow FRED connection cannot
    leave the Streamlit page spinning indefinitely. No fallback data are created.
    """
    code = str(code).strip().upper()
    if not code or not code.replace("_", "").isalnum():
        raise ValueError(f"Código FRED inválido: {code!r}")

    url = FRED_URL.format(code=code)
    try:
        response = _session(retries=0).get(url, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        if not response.text.strip():
            raise ValueError("FRED retornou uma resposta vazia")

        df = pd.read_csv(StringIO(response.text))
        if df.shape[1] < 2:
            raise ValueError(f"CSV FRED inválido: {df.shape[1]} coluna(s) recebida(s)")

        date_col = "DATE" if "DATE" in df.columns else df.columns[0]
        value_col = code if code in df.columns else df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        series = df.dropna(subset=[date_col, value_col]).set_index(date_col)[value_col].sort_index()

        if start:
            series = series[series.index >= pd.Timestamp(start)]
        if series.empty:
            raise ValueError("FRED retornou zero observações numéricas para o período solicitado")

        series.name = code
        return series
    except requests.RequestException as exc:
        raise RuntimeError(
            f"FRED série {code}: conexão indisponível após {HTTP_TIMEOUT_SECONDS}s. "
            "Verifique o acesso de rede do Streamlit Cloud. Fonte: FRED; sem fallback sintético."
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
