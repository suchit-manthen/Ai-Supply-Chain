import numpy as np

from models import _metrics


def test_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    m = _metrics(y, y.copy(), y.copy())
    assert m["mae"] == 0
    assert m["rmse"] == 0
    assert m["mape"] == 0
    assert m["r2"] == 1.0
    assert m["naive_mape"] == 0


def test_mape_is_finite_with_zero_demand():
    y = np.array([0.0, 0.0, 10.0, 20.0])
    p = np.array([1.0, 1.0, 9.0, 18.0])
    naive = np.array([0.0, 0.0, 10.0, 20.0])
    m = _metrics(y, p, naive)
    assert np.isfinite(m["mape"])
    assert m["mape"] >= 0


def test_r2_guarded_for_constant_series():
    y = np.full(5, 7.0)
    m = _metrics(y, y + 1, y)
    assert m["r2"] == 0.0
    assert np.isfinite(m["rmse"])


def test_known_values():
    y = np.array([10.0, 20.0])
    p = np.array([12.0, 18.0])
    naive = np.array([10.0, 20.0])
    m = _metrics(y, p, naive)
    assert m["mae"] == 2.0
    assert m["rmse"] == 2.0
    assert m["mape"] == 15.0  # mean(2/10, 2/20) * 100
    assert m["r2"] == 0.84    # 1 - 8 / 50
