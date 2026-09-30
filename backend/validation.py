"""Walk-forward (rolling-origin) validation framework.

Provides the reusable split + evaluation primitives used by selection.py to
compare the four candidate algorithms on identical windows, plus the lag-7
naive baseline for reference.
"""
import numpy as np

from config import HORIZON, N_FOLDS, MIN_TRAIN_DAYS
from models import _run, _metrics, naive_baseline


def walk_forward_splits(n_samples, horizon=HORIZON, n_folds=N_FOLDS, min_train=MIN_TRAIN_DAYS):
    """Return chronological (train_end, test_start, test_end) index splits.

    Rolling-origin with an expanding training window: each fold trains on
    [0, train_end) and tests on the following `horizon` days. The newest fold's
    test window ends exactly at `n_samples`, so the most recent data is always
    part of the evaluation. Folds are returned oldest-first.
    """
    if horizon <= 0 or n_folds <= 0:
        raise ValueError("horizon and n_folds must be positive")
    if n_samples < min_train + horizon:
        raise ValueError(f"need at least {min_train + horizon} samples, got {n_samples}")
    splits = []
    for k in range(n_folds):
        train_end = n_samples - horizon * (k + 1)
        if train_end < min_train:
            break
        splits.append((train_end, train_end, train_end + horizon))
    if not splits:
        raise ValueError("no valid folds for the given size/configuration")
    return list(reversed(splits))


def evaluate_walk_forward(algo, hist, hol, horizon=HORIZON, n_folds=N_FOLDS, min_train=MIN_TRAIN_DAYS):
    """Evaluate `algo` over every rolling-origin fold.

    Returns one record per fold with model metrics and the matching lag-7 naive
    baseline, both computed on identical test windows.
    """
    folds = []
    for fold_no, (train_end, test_start, test_end) in enumerate(
            walk_forward_splits(len(hist), horizon=horizon, n_folds=n_folds, min_train=min_train), start=1):
        train = hist.iloc[:train_end]
        test = hist.iloc[test_start:test_end]
        pred, _ = _run(algo, train, test, hol)
        pred = np.maximum(np.asarray(pred, dtype=float), 0)
        naive = naive_baseline(hist, test_start, horizon)
        folds.append(dict(
            fold=fold_no,
            train_end=str(hist["date"].iloc[train_end - 1].date()),
            test_start=str(hist["date"].iloc[test_start].date()),
            test_end=str(hist["date"].iloc[test_end - 1].date()),
            n_train=int(train_end),
            metrics=_metrics(test["sales"].values, pred, naive),
            actual=[float(v) for v in test["sales"].values],
            pred=[round(float(v), 1) for v in pred],
        ))
    return folds


def evaluate_baseline_walk_forward(hist, horizon=HORIZON, n_folds=N_FOLDS, min_train=MIN_TRAIN_DAYS):
    """Evaluate the lag-7 naive baseline over the same rolling-origin folds.

    Uses the identical windows as `evaluate_walk_forward`, so every model and
    the baseline are compared on exactly the same test periods.
    """
    folds = []
    for fold_no, (train_end, test_start, test_end) in enumerate(
            walk_forward_splits(len(hist), horizon=horizon, n_folds=n_folds, min_train=min_train), start=1):
        test = hist.iloc[test_start:test_end]
        naive = naive_baseline(hist, test_start, horizon)
        folds.append(dict(
            fold=fold_no,
            train_end=str(hist["date"].iloc[train_end - 1].date()),
            test_start=str(hist["date"].iloc[test_start].date()),
            test_end=str(hist["date"].iloc[test_end - 1].date()),
            n_train=int(train_end),
            metrics=_metrics(test["sales"].values, naive, naive),
        ))
    return folds
