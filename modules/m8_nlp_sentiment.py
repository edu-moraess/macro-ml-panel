"""Module 8 — transparent hawkish/dovish text scoring."""
from __future__ import annotations

import re
import pandas as pd
import streamlit as st

HAWKISH = ["aperto", "elevar", "elevação", "vigilância", "persistente", "acima da meta", "restritiva", "tighten", "hike", "elevated", "persistent", "vigilant", "restrictive"]
DOVISH = ["afrouxamento", "reduzir", "redução", "estímulo", "acomodatícia", "queda", "ease", "cut", "accommodative", "slack", "below target", "stimulus"]


def _score(texto: str) -> dict[str, float]:
    """Score supplied central-bank text using a transparent lexical dictionary."""
    palavras = re.findall(r"[a-zà-úçã-õ]+", texto.lower())
    n = max(len(palavras), 1)
    h = sum(texto.lower().count(t) for t in HAWKISH)
    d = sum(texto.lower().count(t) for t in DOVISH)
    return {"palavras": n, "hawkish": h, "dovish": d, "score (por 1000 palavras)": (h - d) / n * 1000}


def render() -> None:
    """Score real user-supplied statements; no illustrative text is injected."""
    st.header("8 · Sentimento hawkish/dovish em comunicados")
    st.caption("Cole comunicados reais no formato `data;texto`, um por linha. Nenhum texto sintético é carregado automaticamente.")

    raw = st.text_area("Comunicados reais", value="", height=190, placeholder="2026-08-01;[cole aqui o comunicado real]")
    rows: list[dict[str, float | str]] = []
    for line in raw.strip().splitlines():
        if ";" not in line:
            continue
        data, texto = line.split(";", 1)
        result = _score(texto)
        result["data"] = data.strip()
        rows.append(result)

    if not rows:
        st.info("Aguardando texto real. O módulo não gera exemplo sintético para preencher a tela.")
        return

    df = pd.DataFrame(rows).set_index("data")
    st.line_chart(df["score (por 1000 palavras)"])
    st.dataframe(df, use_container_width=True)
    st.info("Limitação: score lexical mede associação de termos, não identifica contexto, ironia ou causalidade. A fonte textual deve ser registrada pelo pesquisador.")
