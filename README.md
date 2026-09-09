# Macro Quant Research Terminal (V2.2)

Quantitative macro research infrastructure — **not** investment advice.

```
PUBLIC DATA (BCB/SGS + FRED)
        ↓
   DATA ENGINE + Data Quality
        ↓
 FEATURE → FACTOR → CURVE → REGIME 2.0 → SIGNAL → HISTORICAL VALIDATION
        ↓
   MACRO INTELLIGENCE DASHBOARD
```

## V2.2 Macro Intelligence

- **Regime Engine 2.0**: RISK-ON / NEUTRAL / RISK-OFF with full probability vector, duration, previous, transition, stay probability. Method field is **HMM** or **GMM** (explicit fallback).
- **Signal Engine**: MACRO SCORE ≈ 50·tanh(Σ wᵢ zᵢ) on [-100,+100]; factor contributions; confidence = 0.3·coverage + 0.3·agreement + 0.25·regime_p + 0.15·data_ok.
- **Historical Validation**: event study for score thresholds and regime transitions; forward +1M/+3M/+6M returns **from t+1** (anti-lookahead).
- **Yield Curve 2.0**: rolling mean/std/z/percentile on spreads; curve regime persistence; GP construction unchanged from V2.1.

## Anti-lookahead

`signal(t)` is always evaluated against outcomes starting at `t+1`. Normalization uses rolling windows only. Event counts require minimum sample.

## Navigation

MACRO INTELLIGENCE · QUANT STATE · QUANT LAB · YIELD CURVE · DATA QUALITY · legacy

Header: **QUANT RESEARCH / Macro Terminal**. Theme Light/Dark in sidebar only.

## Install / Test

```bash
pip install -r requirements.txt
# secrets: FRED_API_KEY
streamlit run app.py
PYTHONPATH=. pytest -q
python -m compileall -q core
```

## Limitations

- FRED requires API key.
- HMM needs hmmlearn (GMM fallback is labeled as GMM).
- Historical validation depends on overlapping sample of signal and equity returns.
- Curve regimes remain rule-based on observed spreads.
