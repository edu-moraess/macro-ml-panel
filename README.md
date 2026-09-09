# Macro Quant Research Terminal

Quantitative macro research infrastructure — **not** investment advice.

```
PUBLIC DATA (BCB/SGS + FRED)
        ↓
   DATA ENGINE + Data Quality
        ↓
 FEATURE → FACTOR → CURVE → REGIME → SIGNAL → HISTORICAL VALIDATION
        ↓
   MACRO INTELLIGENCE DASHBOARD
```

## Macro Intelligence

- **Regime Engine**: RISK-ON / NEUTRAL / RISK-OFF with probability vector, duration, previous state, transition and stay probability. Method is **HMM** or **GMM** when the fallback is required.
- **Signal Engine**: MACRO SCORE ≈ 50·tanh(Σ wᵢ zᵢ) on [-100,+100]; factor contributions and confidence.
- **Historical Validation**: event studies and regime-transition studies; forward +1M/+3M/+6M outcomes **from t+1** to enforce anti-lookahead.
- **Yield Curve Intelligence**: rolling spread statistics, curve regime persistence and Gaussian-process construction.
- **Data Lineage**: reproducible metadata for inputs, factors, models, signals, curve state and validation without storing secrets.
- **Model caching**: fitted regime models are cached as Streamlit resources to avoid unnecessary refits on reruns.

## Anti-lookahead

`signal(t)` is always evaluated against outcomes starting at `t+1`. Normalization uses rolling windows only. Event counts require minimum sample.

## Navigation

MACRO INTELLIGENCE · QUANT STATE · QUANT LAB · YIELD CURVE · DATA QUALITY · legacy research modules

Header: **QUANT RESEARCH / Macro Terminal**. Theme Light/Dark in sidebar only.

## Install / Test

```bash
pip install -r requirements.txt
# secrets: FRED_API_KEY
streamlit run app.py
PYTHONPATH=. pytest -q
python -m compileall -q core
```

## Data policy

- Public data only: BCB/SGS and FRED.
- No synthetic, mock or fallback datasets.
- Data failures and insufficient samples are exposed explicitly.
- API credentials are never included in lineage output.

## Limitations

- FRED requires API key.
- HMM needs hmmlearn; GMM is used when the HMM path cannot be fitted.
- Historical validation depends on the overlapping sample of signals and asset returns.
- Curve regimes remain rule-based on observed spreads.
