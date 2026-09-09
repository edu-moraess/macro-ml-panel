"""
Módulo 8 — Sentimento hawkish/dovish (texto-como-dado)
Ideia: quantificar o tom de atas/comunicados de política monetária com um
score léxico simples, na linha da literatura de "text-as-data" em banca
central. Cole seus próprios trechos — os exemplos abaixo são ilustrativos,
escritos apenas para demonstrar o mecanismo (não são citações de atas reais).
"""
import re
import pandas as pd
import streamlit as st

HAWKISH = [
    "aperto", "elevar", "elevação", "vigilância", "persistente", "acima da meta",
    "restritiva", "tighten", "hike", "elevated", "persistent", "vigilant", "restrictive",
]
DOVISH = [
    "afrouxamento", "reduzir", "redução", "estímulo", "acomodatícia", "queda",
    "ease", "cut", "accommodative", "slack", "below target", "stimulus",
]

EXEMPLO = (
    "2024-06-01;O Comitê mantém vigilância e considera necessária uma postura mais "
    "restritiva diante da inflação persistente.\n"
    "2024-09-01;Diante da desaceleração da atividade, o Comitê vê espaço para reduzir "
    "o ritmo de aperto monetário.\n"
    "2025-01-01;O cenário atual indica espaço para afrouxamento gradual e uma postura "
    "mais acomodatícia adiante."
)


def _score(texto: str) -> dict:
    palavras = re.findall(r"[a-zà-úçã-õ]+", texto.lower())
    n = max(len(palavras), 1)
    h = sum(texto.lower().count(t) for t in HAWKISH)
    d = sum(texto.lower().count(t) for t in DOVISH)
    return {"palavras": n, "hawkish": h, "dovish": d, "score (por 1000 palavras)": (h - d) / n * 1000}


def render():
    st.header("8 · Sentimento hawkish/dovish em comunicados")
    st.caption(
        "Score léxico: conta termos hawkish vs. dovish por 1.000 palavras. "
        "Cole trechos no formato `data;texto`, um por linha."
    )

    raw = st.text_area("Comunicados", value=EXEMPLO, height=160)

    rows = []
    for line in raw.strip().splitlines():
        if ";" not in line:
            continue
        data, texto = line.split(";", 1)
        r = _score(texto)
        r["data"] = data.strip()
        rows.append(r)

    if not rows:
        st.warning("Adicione ao menos uma linha no formato `data;texto`.")
        return

    df = pd.DataFrame(rows).set_index("data")
    st.line_chart(df["score (por 1000 palavras)"])
    st.dataframe(df)
