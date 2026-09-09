"""
Módulo 5 — Nowcasting de atividade econômica (Gradient Boosting)
Ideia: o IBC-Br sai com defasagem. Indicadores de alta frequência que saem
antes (câmbio, juros, inflação/desemprego do mês anterior) alimentam um
modelo de ML para estimar o valor corrente antes da divulgação oficial —
o mesmo espírito do GDPNow, em poucas linhas.
"""
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from data_utils import get_bcb, align, SGS


def render():
    st.header("5 · Nowcasting de atividade (Gradient Boosting)")
    st.caption(
        "IBC-Br estimado a partir de câmbio e Selic do próprio mês + IPCA e "
        "desemprego do mês anterior (proxies que saem antes do IBC-Br)."
    )

    ibc = get_bcb(SGS["ibc_br"]).resample("MS").mean()
    cambio = get_bcb(SGS["cambio_ptax"]).resample("MS").mean()
    selic = get_bcb(SGS["selic_meta"]).resample("MS").mean()
    ipca = get_bcb(SGS["ipca_mensal"]).resample("MS").mean()
    desemprego = get_bcb(SGS["desemprego_pnad"]).resample("MS").mean()

    df = align(
        ibc.rename("ibc"), cambio.rename("cambio"), selic.rename("selic"),
        ipca.rename("ipca"), desemprego.rename("desemprego"),
    )
    df["ipca_lag1"] = df["ipca"].shift(1)
    df["desemprego_lag1"] = df["desemprego"].shift(1)
    df = df.dropna()

    feats = ["cambio", "selic", "ipca_lag1", "desemprego_lag1"]
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    model = GradientBoostingRegressor(random_state=0).fit(train[feats], train["ibc"])
    pred = pd.Series(model.predict(test[feats]), index=test.index, name="nowcast")
    naive = test["ibc"].shift(1).rename("ingênuo (t-1)")

    mae_model = mean_absolute_error(test["ibc"], pred)
    mae_naive = mean_absolute_error(test["ibc"].iloc[1:], naive.dropna())

    c1, c2 = st.columns(2)
    c1.metric("MAE — Gradient Boosting", f"{mae_model:.3f}")
    c2.metric("MAE — benchmark ingênuo", f"{mae_naive:.3f}")

    st.line_chart(pd.concat([test["ibc"].rename("observado"), pred], axis=1))

    imp = pd.Series(model.feature_importances_, index=feats, name="importância")
    st.write("**Importância das variáveis no nowcast:**")
    st.bar_chart(imp.sort_values(ascending=False))
