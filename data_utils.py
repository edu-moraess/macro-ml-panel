"""
data_utils.py — camada de dados do Macro-ML Panel.

Busca séries do BCB/SGS (API pública, sem chave) e do FRED (CSV público,
sem chave) usando apenas `requests`. Sem fallback sintético: se a API
falhar (rede fora do ar, série inexistente, resposta vazia), a função
estoura uma exceção clara — o app mostra o erro em vez de disfarçar com
dado fake.
"""
import pandas as pd
import requests
import streamlit as st

BCB_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"

# Séries BCB/SGS mais usadas no painel
SGS = {
    "selic_meta": 432,
    "ipca_mensal": 433,
    "ibc_br": 24363,
    "cambio_ptax": 1,
    "desemprego_pnad": 24369,
}

# Séries FRED mais usadas no painel
FRED = {
    "t10y3m": "T10Y3M",       # spread 10y-3m (proxy de curva)
    "usrec": "USREC",         # dummy de recessão do NBER
    "unrate": "UNRATE",       # desemprego EUA
    "cpi_yoy": "CPIAUCSL",    # CPI EUA (nível — calcular YoY depois)
    "fedfunds": "FEDFUNDS",
}


@st.cache_data(ttl=3600, show_spinner="Buscando dados reais...")
def get_bcb(code: int, start: str | None = None) -> pd.Series:
    try:
        r = requests.get(BCB_URL.format(code=code), timeout=8)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        s = df.dropna().set_index("data")["valor"].sort_index()
        if start:
            s = s[s.index >= start]
        if s.empty:
            raise ValueError("resposta da API veio vazia")
        return s
    except Exception as e:
        raise RuntimeError(f"BCB/SGS série {code}: {e}") from e


@st.cache_data(ttl=3600, show_spinner="Buscando dados reais...")
def get_fred(code: str, start: str | None = None) -> pd.Series:
    try:
        df = pd.read_csv(FRED_URL.format(code=code))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.dropna().set_index("date")["value"].sort_index()
        if start:
            s = s[s.index >= start]
        if s.empty:
            raise ValueError("resposta da API veio vazia")
        return s
    except Exception as e:
        raise RuntimeError(f"FRED série {code}: {e}") from e


def align(*series: pd.Series, freq="MS") -> pd.DataFrame:
    """Reamostra e alinha múltiplas séries num único DataFrame mensal."""
    out = pd.concat(
        [s.resample(freq).mean().rename(s.name) for s in series], axis=1
    )
    return out.dropna()
