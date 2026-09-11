"""Module 6 — K-Means clustering of observed FX behavior with stability diagnostics."""
from __future__ import annotations
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from data_utils import data_status, get_fred

FX={"Brasil":"EXBZUS","México":"EXMXUS","China":"EXCHUS","Japão":"EXJPUS","Índia":"EXINUS","África do Sul":"EXSFUS"}

def render()->None:
    st.header("6 · Clusterização de regimes cambiais (K-Means)")
    st.caption("Agrupamento por volatilidade e retornos observados, com comparação objetiva entre K e estabilidade por semente.")
    k=st.slider("Número de clusters",2,5,3)
    rows=[]; latest=None
    for country,code in FX.items():
        s=get_fred(code).resample("MS").mean().dropna()
        if len(s)<24: continue
        latest=s.index.max() if latest is None else min(latest,s.index.max()); ret=s.pct_change()
        rows.append({"pais":country,"vol_cambial":ret.rolling(12).std().iloc[-1]*100,"retorno_3m":(s.iloc[-1]/s.iloc[-4]-1)*100,"retorno_12m":(s.iloc[-1]/s.iloc[-13]-1)*100})
    panel=pd.DataFrame(rows).dropna(); feats=["vol_cambial","retorno_3m","retorno_12m"]
    if len(panel)<k+2: raise ValueError("Amostra real insuficiente para o número de clusters escolhido.")
    X=StandardScaler().fit_transform(panel[feats]); model=KMeans(n_clusters=k,n_init=20,random_state=42).fit(X); panel=panel.assign(cluster=model.labels_)
    c=st.columns(2); c[0].metric("Silhouette",f"{silhouette_score(X,model.labels_):.3f}"); c[1].metric("Países",f"{len(panel)}")
    comparison=[]
    for kk in range(2,min(5,len(panel)-1)+1):
        labels=KMeans(n_clusters=kk,n_init=20,random_state=42).fit_predict(X); comparison.append({"K":kk,"silhouette":silhouette_score(X,labels)})
    st.write("**Escolha de K:**"); st.dataframe(pd.DataFrame(comparison).round(3),use_container_width=True)
    stability=[]
    for seed in [0,7,42]:
        lab=KMeans(n_clusters=k,n_init=20,random_state=seed).fit_predict(X); stability.append({"seed":seed,"silhouette":silhouette_score(X,lab)})
    st.write("**Estabilidade por semente:**"); st.dataframe(pd.DataFrame(stability).round(3),use_container_width=True)
    fig,ax=plt.subplots(figsize=(9,4.5)); ax.scatter(panel["vol_cambial"],panel["retorno_12m"],c=panel["cluster"],cmap="tab10",s=90)
    for _,row in panel.iterrows(): ax.annotate(row["pais"],(row["vol_cambial"],row["retorno_12m"]),fontsize=8)
    ax.set_xlabel("Volatilidade cambial — rolling 12m (%)"); ax.set_ylabel("Retorno cambial — 12m (%)"); ax.grid(alpha=.15); st.pyplot(fig,clear_figure=True)
    st.dataframe(panel.sort_values("cluster"),use_container_width=True)
    if latest is not None: st.caption(data_status("Federal Reserve Bank of St. Louis · FRED",", ".join(FX.values()),latest))
    st.info("Clusters descrevem comportamento cambial observado; não são classificação oficial nem implicam causalidade.")
