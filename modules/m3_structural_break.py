"""Módulo 3 — Quebra estrutural (changepoint detection) com diagnóstico de magnitude."""
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
    def best_split(sub):
        n = len(sub)
        if n < 6:
            return None, np.inf
        best_i, best_cost = None, np.inf
        for i in range(3, n - 3):
            cost = sub[:i].var() * i + sub[i:].var() * (n - i)
            if cost < best_cost:
                best_cost, best_i = cost, i
        return best_i, best_cost
    segments, bkps = [(0, len(x))], []
    for _ in range(n_bkps):
        candidates = []
        for a, b in segments:
            i, cost = best_split(x[a:b])
            if i is not None: candidates.append((cost, a, b, i))
        if not candidates: break
        _, a, b, i = sorted(candidates)[0]
        bkps.append(a + i); segments.remove((a, b)); segments += [(a, a+i), (a+i, b)]
    return sorted(bkps)


def render():
    st.header("3 · Quebra estrutural (changepoint detection)")
    st.caption("Detecção de mudanças na média/variância com magnitude, persistência local e diagnóstico por segmento.")
    fonte = st.radio("Série", ["IPCA mensal (BR)", "CPI EUA (nível)"], horizontal=True)
    n_bkps = st.slider("Número de quebras a detectar", 1, 6, 2)
    s = get_bcb(SGS["ipca_mensal"]) if fonte.startswith("IPCA") else get_fred(FRED["cpi_yoy"])
    s = s.resample("MS").mean().dropna(); x = s.to_numpy()
    if RPT_OK:
        bkps = rpt.Binseg(model="l2").fit(x).predict(n_bkps=n_bkps)[:-1]
    else:
        st.info("`ruptures` não instalado — usando segmentação binária própria."); bkps = _l2_binseg(x, n_bkps)
    st.line_chart(pd.DataFrame({"valor": x}, index=s.index))
    bounds = [0] + bkps + [len(x)]
    rows = []
    for j, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        vals = x[a:b]
        rows.append({"segmento": j, "início": s.index[a].date(), "fim": s.index[b-1].date(), "n": len(vals), "média": vals.mean(), "desvio": vals.std()})
    seg = pd.DataFrame(rows)
    st.write("**Segmentos e magnitude das mudanças:**")
    st.dataframe(seg.round(4), use_container_width=True)
    changes = []
    for i, bp in enumerate(bkps):
        left = x[max(0, bp-12):bp]; right = x[bp:min(len(x), bp+12)]
        pooled = np.sqrt((left.var(ddof=1)+right.var(ddof=1))/2) if len(left)>1 and len(right)>1 else np.nan
        changes.append({"quebra": s.index[bp].date(), "Δ média (pós−pré)": right.mean()-left.mean(), "efeito padronizado": (right.mean()-left.mean())/pooled if pooled and pooled>0 else np.nan, "janela pré": len(left), "janela pós": len(right)})
    st.write("**Diagnóstico local (12 meses antes/depois):**")
    st.dataframe(pd.DataFrame(changes).round(3), use_container_width=True)
    st.caption("A data detectada é um changepoint estatístico, não prova causalidade nem significância econômica por si só.")
