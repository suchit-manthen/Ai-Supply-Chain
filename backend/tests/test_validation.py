import numpy as np
import pandas as pd
import pytest

from validation import walk_forward_splits, evaluate_walk_forward, evaluate_baseline_walk_forward


def test_splits_count_and_bounds():
    n = 1000
    splits = walk_forward_splits(n, horizon=28, n_folds=3, min_train=365)
    assert len(splits) == 3
    for train_end, test_start, test_end in splits:
        assert train_end == test_start
        assert test_end - test_start == 28
        assert test_end <= n
        assert train_end >= 365
    assert splits[-1][2] == n  # newest fold ends at the end of the series


def test_splits_are_chronological_and_disjoint():
    splits = walk_forward_splits(1000, horizon=28, n_folds=3, min_train=365)
    for (_, _, prev_end), (_, nxt_start, _) in zip(splits, splits[1:]):
        assert prev_end <= nxt_start


def test_splits_respect_min_train():
    splits = walk_forward_splits(500, horizon=28, n_folds=5, min_train=400)
    assert splits
    assert all(train_end >= 400 for train_end, _, _ in splits)


def test_splits_raise_when_too_small():
    with pytest.raises(ValueError):
        walk_forward_splits(50, horizon=28, n_folds=1, min_train=365)
    with pytest.raises(ValueError):
        walk_forward_splits(500, horizon=0, n_folds=1, min_train=100)


def _small_history(n=140):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(dict(
        date=dates,
        sales=rng.integers(5, 20, n).astype(float),
        discount=np.zeros(n),
        price=np.full(n, 2.0),
        temp=np.full(n, 20.0),
        holiday_lift=np.zeros(n),
    ))


def test_evaluate_walk_forward_runs_and_uses_matching_windows():
    hist = _small_history(140)
    folds = evaluate_walk_forward("Random Forest", hist, None, horizon=28, n_folds=2, min_train=56)
    assert len(folds) == 2
    for f in folds:
        assert len(f["actual"]) == 28
        assert len(f["pred"]) == 28
        assert set(f["metrics"]) >= {"mae", "rmse", "mape", "naive_mape", "r2"}
        assert np.isfinite(f["metrics"]["mape"])
    assert folds[-1]["test_end"] == str(hist["date"].iloc[-1].date())


def test_baseline_uses_identical_windows_as_models():
    hist = _small_history(140)
    model_folds = evaluate_walk_forward("Random Forest", hist, None, horizon=28, n_folds=2, min_train=56)
    base_folds = evaluate_baseline_walk_forward(hist, horizon=28, n_folds=2, min_train=56)
    assert [f["test_start"] for f in base_folds] == [f["test_start"] for f in model_folds]
    assert [f["test_end"] for f in base_folds] == [f["test_end"] for f in model_folds]
    for f in base_folds:
        assert set(f["metrics"]) >= {"mae", "rmse", "mape", "naive_mape", "r2"}
        assert f["metrics"]["mape"] == f["metrics"]["naive_mape"]  # baseline predicted by itself
