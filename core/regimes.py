"""Regime Engine V2 — HMM / GMM based macro regimes on real factor data."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    from sklearn.mixture import GaussianMixture


class RegimeEngine:
    """Fit latent regimes. Requires sufficient sample. No synthetic labels."""

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.labels_: Optional[np.ndarray] = None
        self.method: str = "none"

    def fit(self, X: pd.DataFrame) -> "RegimeEngine":
        if X is None or len(X) < max(36, self.n_regimes * 12):
            raise ValueError(
                f"AMOSTRA INSUFICIENTE para regimes: n={0 if X is None else len(X)} "
                f"(mínimo ~{max(36, self.n_regimes * 12)})"
            )
        data = X.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 36:
            raise ValueError("AMOSTRA INSUFICIENTE após limpeza de NaN.")
        arr = data.values
        if HMM_AVAILABLE:
            self.model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="diag",
                n_iter=200,
                random_state=self.random_state,
            )
            self.model.fit(arr)
            self.labels_ = self.model.predict(arr)
            self.method = "HMM"
        else:
            self.model = GaussianMixture(
                n_components=self.n_regimes, random_state=self.random_state, covariance_type="diag"
            )
            self.model.fit(arr)
            self.labels_ = self.model.predict(arr)
            self.method = "GMM"
        self._index = data.index
        self._X = data
        return self

    def current(self) -> Dict:
        if self.labels_ is None or self.model is None:
            return {"regime": "N/A", "probability": 0.0, "method": self.method}
        last_label = int(self.labels_[-1])
        if self.method == "HMM" and hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(self._X.values)[-1]
            except Exception:
                proba = np.zeros(self.n_regimes)
                proba[last_label] = 1.0
        elif hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(self._X.values)[-1]
        else:
            proba = np.zeros(self.n_regimes)
            proba[last_label] = 1.0
        means = {}
        for k in range(self.n_regimes):
            mask = self.labels_ == k
            if mask.any():
                means[k] = float(self._X.iloc[mask].mean().mean())
        ordered = sorted(means.keys(), key=lambda k: means[k])
        name_map = {}
        if self.n_regimes == 2:
            name_map[ordered[0]] = "RISK-OFF"
            name_map[ordered[1]] = "RISK-ON"
        elif self.n_regimes == 3:
            name_map[ordered[0]] = "RISK-OFF"
            name_map[ordered[1]] = "NEUTRAL"
            name_map[ordered[2]] = "RISK-ON"
        else:
            for i, k in enumerate(ordered):
                name_map[k] = f"REGIME-{i}"
        duration = 1
        for i in range(len(self.labels_) - 2, -1, -1):
            if self.labels_[i] == last_label:
                duration += 1
            else:
                break
        prev = "—"
        if len(self.labels_) > duration:
            prev_label = int(self.labels_[-(duration + 1)])
            prev = name_map.get(prev_label, str(prev_label))
        return {
            "regime": name_map.get(last_label, f"R{last_label}"),
            "label": last_label,
            "probability": float(proba[last_label]) * 100,
            "proba_vector": {name_map.get(i, str(i)): float(proba[i]) * 100 for i in range(len(proba))},
            "duration_months": duration,
            "previous": prev,
            "method": self.method,
            "transition": getattr(self.model, "transmat_", None),
            "n_obs": len(self.labels_),
            "history": pd.Series(self.labels_, index=self._index, name="regime"),
            "name_map": name_map,
        }

    def history_labeled(self) -> pd.Series:
        st = self.current()
        if "history" not in st or not st.get("name_map"):
            return pd.Series(dtype=object)
        nm = st["name_map"]
        return st["history"].map(lambda x: nm.get(int(x), str(x)))
