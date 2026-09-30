"""Automatic, data-driven model selection.

Every SKU is evaluated against all four candidate algorithms with the
walk-forward framework in validation.py. The model with the lowest mean
validation error is selected, then refit on all history by forecast_product.
No model is assumed to win for any SKU.
"""
import numpy as np

from config import (HORIZON, N_FOLDS, MIN_TRAIN_DAYS, SELECTION_METRIC,
                    SELECTION_FALLBACK_METRIC, LOW_DEMAND_THRESHOLD)
from models import ALL_MODELS, forecast_product
from validation import evaluate_walk_forward, evaluate_baseline_walk_forward

METRIC_KEYS = ("mae", "rmse", "mape", "naive_mape", "r2")


def mean_metrics(folds):
    """Average a list of per-fold metric dicts into a single metric dict."""
    out = {}
    for k in METRIC_KEYS:
        out[k] = round(float(np.mean([f["metrics"][k] for f in folds])), 3 if k == "r2" else 2)
    return out


def choose_metric(mean_demand, primary=SELECTION_METRIC, fallback=SELECTION_FALLBACK_METRIC,
                  low_demand=LOW_DEMAND_THRESHOLD):
    """Choose the ranking metric.

    MAPE is the default. It is unreliable when demand is near zero (the
    max(actual, 1) guard makes the ratio explode), so below the low-demand
    threshold we fall back to RMSE.
    """
    if mean_demand < low_demand:
        return fallback, True
    return primary, False


def evaluate_all_models(hist, hol, horizon=HORIZON, n_folds=N_FOLDS, min_train=MIN_TRAIN_DAYS):
    """Walk-forward evaluate every candidate model and the naive baseline."""
    models = {}
    for algo in ALL_MODELS:
        folds = evaluate_walk_forward(algo, hist, hol, horizon=horizon, n_folds=n_folds, min_train=min_train)
        models[algo] = dict(folds=folds, mean=mean_metrics(folds))
    baseline_folds = evaluate_baseline_walk_forward(hist, horizon=horizon, n_folds=n_folds, min_train=min_train)
    return dict(models=models, baseline=dict(folds=baseline_folds, mean=mean_metrics(baseline_folds)))


def select_best(evaluation, hist):
    """Pick the candidate with the lowest mean validation error on the chosen metric."""
    mean_demand = float(hist["sales"].mean())
    metric, fallback_used = choose_metric(mean_demand)
    scores = {algo: evaluation["models"][algo]["mean"][metric] for algo in ALL_MODELS}
    best = min(scores, key=scores.get)
    baseline_score = evaluation["baseline"]["mean"][metric]
    n_folds = len(evaluation["models"][best]["folds"])

    if fallback_used:
        reason = (f"Lowest mean validation {metric.upper()} over {n_folds} rolling-origin folds; "
                  f"MAPE treated as unreliable (mean demand {mean_demand:.2f} < {LOW_DEMAND_THRESHOLD}/day).")
    else:
        reason = f"Lowest mean validation {metric.upper()} over {n_folds} rolling-origin folds"
        diff = round(float(baseline_score - scores[best]), 2)
        if baseline_score > 0:
            reason += f" ({diff:+.2f} pts vs naive baseline)"
        reason += "."

    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    return dict(
        model=best,
        metric=metric,
        score=round(float(scores[best]), 2),
        baseline_score=round(float(baseline_score), 2),
        improvement=round(float(baseline_score - scores[best]), 2),
        fallback_used=fallback_used,
        mean_demand=round(mean_demand, 2),
        ranking=[dict(model=a, score=round(float(s), 2)) for a, s in ordered],
        reason=reason,
    )


def select_and_forecast(hist, fut, hol, horizon=HORIZON, n_folds=N_FOLDS, min_train=MIN_TRAIN_DAYS):
    """Full per-SKU pipeline: evaluate all models, select the best, refit, forecast."""
    evaluation = evaluate_all_models(hist, hol, horizon=horizon, n_folds=n_folds, min_train=min_train)
    choice = select_best(evaluation, hist)
    folds = evaluation["models"][choice["model"]]["folds"]
    result = forecast_product(choice["model"], hist, fut, hol, folds=folds)
    result["selected_model"] = choice["model"]
    result["selection"] = choice
    result["evaluation"] = evaluation
    return result
