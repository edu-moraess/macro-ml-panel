"""
Módulo 3 — Quebra estrutural (changepoint detection)
Ideia: em vez de assumir uma data de quebra (ex: mudança de regime cambial,
ancoragem de expectativas), o algoritmo de segmentação binária encontra os
pontos que mais reduzem a variância dentro de cada segmento.
"""
import numpy as np
import pandas as pd
import streamlit as st

from data_utils import get_bcb, get_fred, SGS, FRED

try:
    import ruptures as rpt
    RPT_OK = True
except ImportError:
    RPT_OK = False


def _l2_binseg(x: np.ndarray, n_bkps: int):
    """Fallback leve (custo L2) caso `ruptures` não esteja instalado."""
    def best_split(sub):
        n = len(sub)
        if n < 6:
            return None, np.inf
        best_i, best_cost = None, np.inf
        for i in range(3, n - 3):
            c = sub[:i].var() * i + sub[i:].var() * (n - i)
            if c < best_cost:
                best_cost, best_i = c, i
        return best_i, best_cost

    segments, bkps = [(0, len(x))], []
    for _ in range(n_bkps):
        candidates = []
        for a, b in segments:
            i, cost = best_split(x[a:b])
            if i is not None:
                candidates.append((cost, a, b, i))
        if not candidates:
            break
        candidates.sort(key=lambda c: c[0])
        _, a, b, i = candidates[0]
        bkps.append(a + i)
        segments.remove((a, b))
        segments += [(a, a + i), (a + i, b)]
    return sorted(bkps)


def render():
    st.header("3 · Quebra estrutural (changepoint detection)")
    st.caption(
        "Segmentação binária por custo L2 encontra automaticamente onde a média/"
        "variância da série muda de regime — sem você apontar a data a priori."
    )

    fonte = st.radio("Série", ["IPCA mensal (BR)", "CPI EUA (nível)"], horizontal=True)
    n_bkps = st.slider("Número de quebras a detectar", 1, 6, 2)

    s = get_bcb(SGS["ipca_mensal"]) if fonte.startswith("IPCA") else get_fred(FRED["cpi_yoy"])
    s = s.resample("MS").mean().dropna()
    x = s.to_numpy()

    if RPT_OK:
        bkps = rpt.Binseg(model="l2").fit(x).predict(n_bkps=n_bkps)[:-1]
    else:
        st.info("`ruptures` não instalado — usando segmentação binária própria (fallback).")
        bkps = _l2_binseg(x, n_bkps)

    df = pd.DataFrame({"valor": x}, index=s.index)
    st.line_chart(df)

    datas = [s.index[i].date() for i in bkps]
    st.write("**Datas de quebra detectadas:**")
    st.dataframe(pd.DataFrame({"data": datas}))

    bounds = [0] + bkps + [len(x)]
    resumo = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        resumo.append({
            "início": s.index[a].date(),
            "fim": s.index[b - 1].date(),
            "média": round(x[a:b].mean(), 3),
            "desvio padrão": round(x[a:b].std(), 3),
        })
    st.write("**Segmentos entre as quebras:**")
    st.dataframe(pd.DataFrame(resumo))
