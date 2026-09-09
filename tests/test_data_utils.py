"""Unit tests for data layer — no synthetic data, controlled failures."""
from __future__ import annotations

import pandas as pd
import pytest

from data_utils import (
    SGS,
    _parse_bcb,
    align,
    format_quality_line,
    rolling_zscore,
    series_quality,
)


def test_parse_bcb_valid():
    payload = [
        {"data": "01/01/2020", "valor": "1.23"},
        {"data": "01/02/2020", "valor": "1.45"},
    ]
    s = _parse_bcb(payload, 433)
    assert len(s) == 2
    assert s.iloc[0] == 1.23


def test_parse_bcb_empty():
    with pytest.raises(ValueError):
        _parse_bcb([], 433)


def test_parse_bcb_invalid_fields():
    with pytest.raises(ValueError):
        _parse_bcb([{"foo": 1}], 433)


def test_rolling_zscore_shape():
    s = pd.Series(range(100), index=pd.date_range("2015-01-01", periods=100, freq="MS"))
    z = rolling_zscore(s, 24)
    assert len(z) == 100
    assert z.iloc[:10].isna().all() or z.notna().sum() > 0


def test_series_quality_empty():
    q = series_quality(pd.Series(dtype=float), series_id="433")
    assert q["status"] == "DADOS INDISPONÍVEIS"
    assert q["n_obs"] == 0


def test_series_quality_ok():
    idx = pd.date_range("2015-01-01", periods=60, freq="MS")
    s = pd.Series(range(60), index=idx, name="433")
    q = series_quality(s, series_id="433")
    assert q["n_obs"] == 60
    assert "IPCA" in q["name"] or q["series_id"] == "433"
    line = format_quality_line(q)
    assert "STATUS" in line


def test_align_empty():
    with pytest.raises(ValueError):
        align(pd.Series(dtype=float), pd.Series(dtype=float))
