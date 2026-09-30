import numpy as np
import pandas as pd
import pytest

from models import naive_baseline


@pytest.fixture
def hist():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.DataFrame(dict(date=dates, sales=np.arange(1, 31, dtype=float)))


def test_naive_baseline_is_same_day_last_week(hist):
    start, horizon = 10, 5
    out = naive_baseline(hist, start, horizon)
    expected = hist["sales"].shift(7).to_numpy(dtype=float)[start:start + horizon]
    assert np.allclose(out, expected)
    assert out[0] == hist["sales"].iloc[start - 7]


def test_naive_baseline_window_length(hist):
    assert len(naive_baseline(hist, 25, 5)) == 5


def test_naive_baseline_no_nan_after_first_week(hist):
    assert not np.isnan(naive_baseline(hist, 7, 10)).any()
