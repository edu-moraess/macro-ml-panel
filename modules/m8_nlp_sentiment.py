"""Module 8 — transparent hawkish/dovish text scoring and policy surprise."""
from __future__ import annotations
import re
import pandas as pd
import streamlit as st
HAWKISH=["aperto","elevar","elevação","vigilância","persistente","acima da meta","restritiva","tighten","hike","elevated","persistent","vigilant","restrictive"]
DOVISH=["afrouxamento","reduzir","redução","estímulo","acomodatícia","queda","ease","cut","accommodative","slack","below target","stimulus"]
def _score(texto):
    low=texto.lower(); palavras=re.findall(r"[a-zà-úçã-õ]+",low); n=max(len(palavras),1); h=sum(low.count(t) for t in HAWKISH); d=sum(low.count(t) for t in DOVISH)
    return {"palavras":n,"hawkish":h,"dovish":d,"score":(h-d)/n*1000}
def render()->None:
    st.header("8 · Sentimento hawkish/dovish em comunicados")
    st.caption("Cole comunicados reais no formato `data;texto`; opcionalmente informe a fonte. A variação do score entre comunicados funciona como proxy transparente de surpresa de comunicação.")
    raw=st.text_area("Comunicados reais",value="",height=190,placeholder="2026-08-01;[cole aqui o comunicado real]")
    source=st.text_input("Fonte dos comunicados (URL ou identificação)",placeholder="Banco Central / Federal Reserve — referência da publicação")
    rows=[]
    for line in raw.strip().splitlines():
        if ";" not in line: continue
        data,texto=line.split(";",1); r=_score(texto); r["data"]=data.strip(); rows.append(r)
    if not rows: st.info("Aguardando texto real. O módulo não gera exemplo sintético para preencher a tela."); return
    df=pd.DataFrame(rows).set_index("data"); df["surpresa_vs_anterior"]=df["score"].diff()
    st.line_chart(df[["score","surpresa_vs_anterior"]]); st.dataframe(df,use_container_width=True)
    c=st.columns(2); c[0].metric("Último score",f"{df['score'].iloc[-1]:+.2f}"); c[1].metric("Surpresa vs anterior",f"{df['surpresa_vs_anterior'].iloc[-1]:+.2f}" if len(df)>1 else "—")
    if source: st.caption(f"Fonte registrada pelo pesquisador: {source}")
    st.info("Score lexical mede associação de termos e não entende contexto, ironia ou causalidade. A 'surpresa' é apenas a mudança no score; não é surpresa monetária medida pelo mercado.")
