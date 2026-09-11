"""Global macro dashboard powered by World Bank WDI."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from world_bank import COUNTRIES, INDICATORS, get_world_bank, latest_table, quality


def render() -> None:
    st.subheader("WORLD BANK · GLOBAL MACRO")
    st.caption("Comparação internacional com dados oficiais do World Development Indicators. Brasil não faz parte do universo desta análise.")

    country_names = list(COUNTRIES.values())
    selected_names = st.multiselect(
        "Países",
        country_names,
        default=["United States", "China", "Germany", "India", "Japan", "United Kingdom", "Mexico"],
    )
    selected_codes = tuple(code for code, name in COUNTRIES.items() if name in selected_names)
    if not selected_codes:
        st.warning("Selecione pelo menos um país.")
        return

    indicator_options = list(INDICATORS)
    indicator = st.selectbox(
        "Indicador",
        indicator_options,
        format_func=lambda x: f"{INDICATORS[x]['name']} · {INDICATORS[x]['unit']}",
    )

    try:
        df = get_world_bank(indicator, selected_codes)
        meta = INDICATORS[indicator]
        latest = latest_table(indicator, selected_codes)
        q = quality(indicator, selected_codes)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Último ano", q["last_year"])
        c2.metric("Países", q["countries"])
        c3.metric("Observações", q["observations"])
        c4.metric("Unidade", meta["unit"])

        st.markdown("**Última observação disponível**")
        display = latest[["country", "year", "value"]].copy()
        display.columns = ["País", "Ano", "Valor"]
        st.dataframe(display, use_container_width=True, hide_index=True)

        pivot = df.pivot(index="year", columns="country", values="value").sort_index()
        st.line_chart(pivot, use_container_width=True)

        st.markdown("**Histórico**")
        st.dataframe(
            df.pivot(index="year", columns="country", values="value").sort_index().tail(15),
            use_container_width=True,
        )

        st.caption(
            f"Fonte: World Bank · WDI · {meta['code']} · {q['first_year']}–{q['last_year']} · "
            f"Consulta: {q['queried_at']} · Cache: 15 min · Sem dados sintéticos"
        )
    except (RuntimeError, ValueError) as exc:
        st.error("DADOS WORLD BANK INDISPONÍVEIS — o indicador não foi calculado.")
        st.caption(str(exc))
