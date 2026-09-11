"""Module 9 — macro anomaly detection via rolling Isolation Forest."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from data_utils import SGS, align, data_status, get_bcb

def render()->None:
    st.header("9 · Detecção de anomalia macro")
    st.caption("Isolation Forest em janela móvel para reduzir look-ahead: cada score histórico é estimado usando somente observações anteriores e contemporâneas.")
    contamination=st.slider("Fração esperada de meses anômalos",0.02,0.15,0.05,step=0.01); window=st.slider("Janela de estimação (meses)",36,120,60,12)
    selic=get_bcb(SGS["selic_meta"]).resample("MS").mean(); cambio=get_bcb(SGS["cambio_ptax"]).resample("MS").mean(); ipca=get_bcb(SGS["ipca_mensal"]).resample("MS").mean(); desemp=get_bcb(SGS["desemprego_pnad"]).resample("MS").mean()
    df=align(selic.rename("selic"),cambio.rename("cambio"),ipca.rename("ipca"),desemp.rename("desemprego")); df["cambio_var"]=df["cambio"].pct_change()*100; df=df.dropna(); feats=["selic","cambio_var","ipca","desemprego"]
    if len(df)<window+12: raise ValueError(f"Apenas {len(df)} observações; janela requer pelo menos {window+12}.")
    scores=[]; flags=[]
    for i in range(window,len(df)):
        tr=df.iloc[i-window:i][feats]; te=df.iloc[[i]][feats]; scaler=StandardScaler().fit(tr); model=IsolationForest(contamination=contamination,random_state=42).fit(scaler.transform(tr)); z=scaler.transform(te); scores.append(float(-model.score_samples(z)[0])); flags.append(bool(model.predict(z)[0]==-1))
    out=df.iloc[window:].copy(); out["score"]=scores; out["anomalia"]=flags
    c=st.columns(4); c[0].metric("OOS observações",f"{len(out):,}"); c[1].metric("Anomalias",f"{int(out.anomalia.sum())}"); c[2].metric("Janela",f"{window}m"); c[3].metric("Última leitura",f"{out.index.max():%b/%Y}")
    st.line_chart(out["score"],height=320); st.caption(data_status("Banco Central do Brasil · SGS","432 / 1 / 433 / 24369",out.index.max()))
    anomalies=out.loc[out.anomalia,feats+["score"]].sort_values("score",ascending=False); st.write("**Observações sinalizadas**"); st.dataframe(anomalies,use_container_width=True)
    st.write("**Persistência:**"); st.metric("Sequência máxima de anomalias",f"{max([len(g) for _,g in out[out.anomalia].groupby((~out.anomalia).cumsum())],default=0)} meses")
    st.info("Anomalia estatística não equivale a crise. A janela móvel evita treinar o detector com todo o futuro, mas o resultado continua sensível à janela e à contaminação.")
