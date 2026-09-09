# Macro Quant Research Terminal (V2.1)

Institutional quantitative macro research terminal. Public data only (BCB/SGS + FRED).

```
PUBLIC DATA → DATA ENGINE → FEATURE ENGINE → FACTOR ENGINE
     → REGIME ENGINE → SIGNAL / RISK → BACKTEST / PORTFOLIO
     → YIELD CURVE INTELLIGENCE → QUANT TERMINAL
```

## Yield Curve Intelligence

- Real FRED constant-maturity Treasury vertices (3M…30Y)
- **Level / Slope (10Y−2Y, 10Y−3M, 30Y−10Y) / Curvature (2Y − 2×10Y + 30Y)**
- Curve regime: INVERTED · FLATTENING · NORMAL · STEEPENING · LONG-END PRESSURE
- Gaussian Process (RBF + WhiteKernel) for **curve construction/smoothing only**
  - length_scale = maturity correlation scale (years), not temporal memory
  - RMSE / MAE / residuals reported
  - **GP is not a causal macro forecast model**
- Historical spreads (2s10s, 3m10y, 10s30s)
- Curve factor optionally integrated into MACRO SCORE (documented weight)

## Navigation

QUANT STATE · QUANT LAB · YIELD CURVE · DATA QUALITY · legacy modules

Header: **QUANT RESEARCH / Macro Terminal** only. Theme (Light/Dark) is discrete in the sidebar.

## Install

```bash
pip install -r requirements.txt
# .streamlit/secrets.toml → FRED_API_KEY = "..."
streamlit run app.py
PYTHONPATH=. pytest -q
python -m compileall -q .
```

## Limitations

- FRED requires API key; BCB is public.
- HMM needs `hmmlearn` (GMM fallback).
- Full ARIMA/GARCH/VAR/VECM/Black-Litterman not forced without sufficient sample.
- Curve regimes are rule-based on observed spreads (documented thresholds).
