"""DATA QUALITY page — centralized, objective, no duplication."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from data_utils import (
    FRED, SGS, get_bcb, get_fred, series_quality, format_quality_line, CACHE_TTL_SECONDS
)


def render() -> None:
    st.subheader("DATA QUALITY")
    st.caption("Provenance e status de todas as séries públicas usadas no terminal.")

    rows = []
    series_specs = [
        ("BCB", "432", "Selic", lambda: get_bcb(SGS["selic_meta"])),
        ("BCB", "433", "IPCA", lambda: get_bcb(SGS["ipca_mensal"])),
        ("BCB", "24363", "IBC-Br", lambda: get_bcb(SGS["ibc_br"])),
        ("BCB", "1", "USD/BRL", lambda: get_bcb(SGS["cambio_ptax"])),
        ("BCB", "24369", "Desemprego", lambda: get_bcb(SGS["desemprego_pnad"])),
        ("FRED", "T10Y3M", "Term Spread", lambda: get_fred(FRED["t10y3m"])),
        ("FRED", "VIXCLS", "VIX", lambda: get_fred(FRED["vix"])),
        ("FRED", "BAMLH0A0HYM2", "HY OAS", lambda: get_fred(FRED["hy_spread"])),
        ("FRED", "SP500", "S&P 500", lambda: get_fred(FRED["sp500"])),
        ("FRED", "DCOILWTICO", "WTI", lambda: get_fred(FRED["wti"])),
        ("FRED", "GOLDAMGBD228NLBM", "Gold", lambda: get_fred(FRED["gold"])),
    ]

    for source, sid, name, loader in series_specs:
        try:
            s = loader()
            q = series_quality(s, series_id=sid)
            rows.append({
                "Source": q["source"],
                "Series": q["name"],
                "ID": q["series_id"],
                "Freq": q["freq"],
                "Last Obs": q["last_obs"].strftime("%Y-%m-%d") if q["last_obs"] is not None else "—",
                "N": q["n_obs"],
                "Missing": f"{q['missing_rate']:.1%}",
                "Freshness (d)": q["freshness_days"],
                "Status": q["status"],
            })
            st.code(format_quality_line(q), language=None)
        except Exception as exc:
            rows.append({
                "Source": source, "Series": name, "ID": sid, "Freq": "—",
                "Last Obs": "—", "N": 0, "Missing": "—", "Freshness (d)": "—",
                "Status": "DADOS INDISPONÍVEIS",
            })
            st.error(f"{source} · {name} · {sid} → DADOS INDISPONÍVEIS — {exc}")

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(f"Cache TTL: {CACHE_TTL_SECONDS // 60} min · Sem dados sintéticos · Falhas expostas explicitamente")
