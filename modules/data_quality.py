"""DATA QUALITY page — objective provenance for every provider."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data_utils import FRED, SGS, get_bcb, get_fred, series_quality, format_quality_line, CACHE_TTL_SECONDS
from world_bank import COUNTRIES, INDICATORS, get_world_bank, quality


def render() -> None:
    st.subheader("DATA QUALITY")
    st.caption("Proveniência e frescor das séries públicas. Brasil usa IPEA/Yahoo; universo global usa World Bank/FRED.")

    rows = []

    # Brazil macro: no BCB/SGS transport. The compatibility keys map to IPEA
    # mirrors of the relevant macro series; FX uses Yahoo Finance BRL=X.
    brazil_specs = [
        ("IPEA", "432", "Selic", SGS["selic_meta"]),
        ("IPEA", "433", "IPCA", SGS["ipca_mensal"]),
        ("IPEA", "24363", "IBC-Br dessazonalizado", SGS["ibc_br"]),
        ("Yahoo Finance", "1", "USD/BRL", SGS["cambio_ptax"]),
        ("IPEA", "24369", "Desemprego PNAD", SGS["desemprego_pnad"]),
    ]
    for source, sid, name, key in brazil_specs:
        try:
            s = get_bcb(key)
            q = series_quality(s, series_id=sid)
            rows.append({
                "Source": q["source"], "Series": q["name"], "ID": q["series_id"],
                "Freq": q["freq"], "Last Obs": q["last_obs"].strftime("%Y-%m-%d") if q["last_obs"] is not None else "—",
                "N": q["n_obs"], "Countries": "Brazil", "Status": q["status"],
            })
            st.code(format_quality_line(q), language=None)
        except Exception as exc:
            rows.append({
                "Source": source, "Series": name, "ID": sid, "Freq": "—", "Last Obs": "—",
                "N": 0, "Countries": "Brazil", "Status": f"DADOS INDISPONÍVEIS: {exc}",
            })

    # Global structural macro indicators. Brazil remains explicitly excluded.
    for key in ["gdp_growth", "inflation", "unemployment", "current_account", "trade", "exports", "fdi", "reserves"]:
        meta = INDICATORS[key]
        try:
            df = get_world_bank(key, tuple(COUNTRIES))
            q = quality(key, tuple(COUNTRIES))
            rows.append({
                "Source": "World Bank WDI", "Series": meta["name"], "ID": meta["code"],
                "Freq": "Annual", "Last Obs": str(q["last_year"]), "N": len(df),
                "Countries": q["countries"], "Status": "OK",
            })
        except Exception as exc:
            rows.append({
                "Source": "World Bank WDI", "Series": meta["name"], "ID": meta["code"],
                "Freq": "Annual", "Last Obs": "—", "N": 0, "Countries": 0,
                "Status": f"DADOS INDISPONÍVEIS: {exc}",
            })

    for sid, name, loader in [
        ("T10Y3M", "Term Spread", lambda: get_fred(FRED["t10y3m"])),
        ("VIXCLS", "VIX", lambda: get_fred(FRED["vix"])),
        ("BAMLH0A0HYM2", "HY OAS", lambda: get_fred(FRED["hy_spread"])),
        ("SP500", "S&P 500", lambda: get_fred(FRED["sp500"])),
        ("DCOILWTICO", "WTI", lambda: get_fred(FRED["wti"])),
        ("GOLDAMGBD228NLBM", "Gold", lambda: get_fred(FRED["gold"])),
    ]:
        try:
            s = loader()
            q = series_quality(s, series_id=sid)
            rows.append({
                "Source": q["source"], "Series": q["name"], "ID": q["series_id"],
                "Freq": q["freq"], "Last Obs": q["last_obs"].strftime("%Y-%m-%d") if q["last_obs"] is not None else "—",
                "N": q["n_obs"], "Countries": "—", "Status": q["status"],
            })
            st.code(format_quality_line(q), language=None)
        except Exception as exc:
            rows.append({
                "Source": "FRED", "Series": name, "ID": sid, "Freq": "—", "Last Obs": "—",
                "N": 0, "Countries": "—", "Status": f"DADOS INDISPONÍVEIS: {exc}",
            })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Brasil: IPEA Data + Yahoo Finance · Global: World Bank WDI + FRED · Sem dados sintéticos · Cache TTL: {CACHE_TTL_SECONDS // 60} min")
