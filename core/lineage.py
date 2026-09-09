"""Data lineage primitives for reproducible quantitative research."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable

import pandas as pd


LINEAGE_VERSION = "1.0"


def _date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def series_node(name: str, series: pd.Series, source: str, series_id: str) -> Dict[str, Any]:
    """Describe an input series without storing observations or secrets."""
    s = pd.to_numeric(series, errors="coerce").dropna() if series is not None else pd.Series(dtype=float)
    return {
        "name": name,
        "source": source,
        "series_id": str(series_id),
        "n_obs": int(len(s)),
        "first_obs": _date(s.index.min()) if len(s) else None,
        "last_obs": _date(s.index.max()) if len(s) else None,
        "status": "OK" if len(s) else "DADOS INDISPONÍVEIS",
    }


def build_lineage(
    inputs: Iterable[Dict[str, Any]],
    factors: pd.DataFrame,
    regime: Dict[str, Any],
    signal: Dict[str, Any],
    validation: Dict[str, Any],
    curve_persistence: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a compact, serializable lineage record for the current computation."""
    return {
        "version": LINEAGE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": ["public_data", "feature_engine", "factor_engine", "regime_engine", "signal_engine", "validation"],
        "inputs": list(inputs),
        "factors": {
            "columns": list(factors.columns),
            "n_obs": int(len(factors)),
            "first_obs": _date(factors.index.min()) if not factors.empty else None,
            "last_obs": _date(factors.index.max()) if not factors.empty else None,
        },
        "regime": {
            "method": regime.get("method"),
            "regime": regime.get("regime"),
            "probability": regime.get("probability"),
            "n_obs": regime.get("n_obs"),
        },
        "signal": {
            "macro_score": signal.get("macro_score"),
            "confidence": signal.get("confidence"),
            "n_factors": signal.get("n_factors"),
        },
        "curve": {
            "regime": curve_persistence.get("regime"),
            "duration": curve_persistence.get("duration"),
            "previous": curve_persistence.get("previous"),
        },
        "validation": {
            "status": validation.get("status"),
            "n_panel": validation.get("n_panel", 0),
        },
        "anti_lookahead": "signal(t) evaluated against forward outcomes beginning at t+1",
        "secrets_excluded": True,
    }
