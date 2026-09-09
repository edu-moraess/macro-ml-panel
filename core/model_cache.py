"""Streamlit resource cache for fitted quantitative models."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.regimes import RegimeEngine


@st.cache_resource(show_spinner=False)
def fit_regime_engine(X: pd.DataFrame, n_regimes: int = 3, random_state: int = 42) -> RegimeEngine:
    """Fit once per unique factor matrix/configuration during a Streamlit session."""
    engine = RegimeEngine(n_regimes=n_regimes, random_state=random_state)
    engine.fit(X)
    return engine
