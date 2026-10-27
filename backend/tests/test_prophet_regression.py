"""Regression test for the Prophet short-history explosion.

Before the fix, `fit_prophet` always enabled yearly seasonality. On a series
shorter than ~2 years the yearly Fourier terms are unidentifiable and, with
multiplicative seasonality, blew the trend up (forecasts many times the demand
level). This test pins the corrected behaviour on a short, high-demand series.
"""
import numpy as np
import pandas as pd

from data import holiday_table
from models import fit_prophet, _metrics


def _short_series(days=140, start="2023-01-01"):
    dates = pd.date_range(start, periods=days)
    rng = np.random.default_rng(0)
    dow = dates.dayofweek
    sales = 500 + 70 * np.sin(2 * np.pi * dow / 7) + rng.normal(0, 20, days)
    frame = pd.DataFrame(dict(date=dates, product="A", sales=sales, discount=0.0,
                              price=1.0, temp=0.0, holiday_lift=0.0, stockout=0))
    hol = holiday_table([dates[0].year, dates[-1].year])
    return frame, hol


def test_prophet_short_history_does_not_explode():
    frame, hol = _short_series()
    split = len(frame) - 28
    train, test = frame.iloc[:split], frame.iloc[split:]

    pred, _ = fit_prophet(train, test, hol)
    pred = np.maximum(np.asarray(pred, dtype=float), 0)
    level = float(train["sales"].mean())
    metrics = _metrics(test["sales"].values, pred, np.full(len(test), level))

    # Forecasts stay in the same order of magnitude as demand (the bug produced
    # predictions ~10-20x the level and MAPE in the hundreds of percent).
    assert pred.max() <= 3 * level
    assert metrics["mape"] < 50
    assert metrics["mae"] < level
