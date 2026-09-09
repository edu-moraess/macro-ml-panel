"""Regime Engine 2.0 — HMM/GMM with probabilities, duration, transitions, persistence.

Fallback to GMM is EXPLICIT (method field never claims HMM when GMM is used).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    from sklearn.mixture import GaussianMixture


class RegimeEngine:
    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.labels_: Optional[np.ndarray] = None
        self.method: str = "none"
        self._index = None
        self._X = None
        self._name_map: Dict[int, str] = {}

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
        self._build_name_map()
        return self

    def _build_name_map(self) -> None:
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
        self._name_map = name_map

    def _proba_vector(self) -> np.ndarray:
        last_label = int(self.labels_[-1])
        if hasattr(self.model, "predict_proba"):
            try:
                return self.model.predict_proba(self._X.values)[-1]
            except Exception:
                pass
        proba = np.zeros(self.n_regimes)
        proba[last_label] = 1.0
        return proba

    def persistence_stats(self) -> Dict[str, Any]:
        labels = self.labels_
        n = len(labels)
        duration = 1
        for i in range(n - 2, -1, -1):
            if labels[i] == labels[-1]:
                duration += 1
            else:
                break
        spells = []
        cur, length = labels[0], 1
        for i in range(1, n):
            if labels[i] == cur:
                length += 1
            else:
                spells.append(length)
                cur, length = labels[i], 1
        spells.append(length)
        avg_duration = float(np.mean(spells)) if spells else np.nan
        n_trans = int(np.sum(labels[1:] != labels[:-1]))
        freq = {name: float(np.mean(labels == k)) for k, name in self._name_map.items()}
        cur_lab = labels[-1]
        mask_from = labels[:-1] == cur_lab
        stay = float(np.mean(labels[1:][mask_from] == cur_lab)) if mask_from.any() else np.nan
        prev = "—"
        if n > duration:
            prev = self._name_map.get(int(labels[-(duration + 1)]), str(labels[-(duration + 1)]))
        transition_label = f"{prev} → {self._name_map.get(int(labels[-1]), str(labels[-1]))}"
        return {
            "duration": duration,
            "avg_duration": avg_duration,
            "n_transitions": n_trans,
            "frequency": freq,
            "stay_probability": stay,
            "previous": prev,
            "transition": transition_label,
        }

    def current(self) -> Dict[str, Any]:
        if self.labels_ is None or self.model is None:
            return {"regime": "N/A", "probability": 0.0, "method": self.method, "probabilities": {}}
        last_label = int(self.labels_[-1])
        proba = self._proba_vector()
        if proba.sum() > 0:
            proba = proba / proba.sum()
        probs_named = {self._name_map.get(i, str(i)): float(proba[i]) * 100 for i in range(len(proba))}
        confidence = float(proba.max())
        pers = self.persistence_stats()
        return {
            "regime": self._name_map.get(last_label, f"R{last_label}"),
            "label": last_label,
            "probability": float(proba[last_label]) * 100,
            "probabilities": probs_named,
            "confidence": confidence,
            "duration_months": pers["duration"],
            "avg_duration": pers["avg_duration"],
            "n_transitions": pers["n_transitions"],
            "frequency": pers["frequency"],
            "stay_probability": pers["stay_probability"],
            "previous": pers["previous"],
            "transition": pers["transition"],
            "method": self.method,
            "transition_matrix": getattr(self.model, "transmat_", None),
            "n_obs": len(self.labels_),
            "history": pd.Series(self.labels_, index=self._index, name="regime"),
            "name_map": self._name_map,
        }

    def history_labeled(self) -> pd.Series:
        st = self.current()
        if "history" not in st or not st.get("name_map"):
            return pd.Series(dtype=object)
        nm = st["name_map"]
        return st["history"].map(lambda x: nm.get(int(x), str(x)))
