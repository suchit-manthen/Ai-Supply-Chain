import numpy as np

from models import horizon_error_scale, prediction_interval


def _folds(residuals):
    """Build folds whose residuals equal the given arrays (actual - 0)."""
    return [dict(actual=np.asarray(r, dtype=float), pred=np.zeros(len(r))) for r in residuals]


def test_scale_has_horizon_length_and_is_non_negative():
    s = horizon_error_scale(_folds([np.sqrt(np.arange(1, 29))] * 3), 28)
    assert len(s) == 28
    assert np.all(s >= 0)


def test_scale_never_narrows_with_horizon():
    s = horizon_error_scale(_folds([np.sqrt(np.arange(1, 29)) + f for f in range(3)]), 28)
    assert np.all(np.diff(s) >= -1e-9)


def test_scale_grows_further_into_the_horizon():
    s = horizon_error_scale(_folds([np.sqrt(np.arange(1, 29)) + f for f in range(3)]), 28)
    assert s[27] > s[0]          # day 28 wider than day 1
    assert s[13] >= s[0]         # day 14 at least as wide as day 1


def test_scale_is_zero_for_perfect_forecasts():
    s = horizon_error_scale(_folds([np.zeros(28)] * 3), 28)
    assert np.allclose(s, 0.0)


def test_scale_is_flat_for_constant_errors():
    s = horizon_error_scale(_folds([np.full(28, 3.0)] * 3), 28)
    assert np.allclose(s, 3.0)


def test_prediction_interval_ordering():
    preds = np.array([0.0, 5.0, 100.0, 7.5])
    scale = np.array([1.0, 2.0, 3.0, 4.0])
    lo, hi = prediction_interval(preds, scale)
    assert np.all(lo <= preds)
    assert np.all(preds <= hi)
    assert np.all(lo >= 0)


def test_prediction_interval_widens_with_error_scale():
    lo, hi = prediction_interval(np.full(4, 50.0), np.array([1.0, 2.0, 3.0, 4.0]))
    widths = hi - lo
    assert np.all(np.diff(widths) > 0)


def test_prediction_interval_accepts_scalar_scale():
    lo, hi = prediction_interval([10.0, 10.0], 2.0, z=1.0)
    assert np.allclose(lo, [8.0, 8.0])
    assert np.allclose(hi, [12.0, 12.0])


def test_prediction_interval_zero_scale_is_a_point():
    lo, hi = prediction_interval([10.0], 0.0)
    assert lo[0] == hi[0] == 10.0


def test_prediction_interval_never_negative_lower_bound():
    lo, _ = prediction_interval([1.0], 100.0)
    assert lo[0] == 0.0
