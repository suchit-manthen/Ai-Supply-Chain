"""Lightweight model/data monitoring.

Compares the latest completed run's validation error against the naive baseline
and against the previous run, and flags significant deterioration per SKU and
overall. Just SQLite reads plus two transparent, configurable thresholds - no
MLOps platform.
"""
from config import DRIFT_REL_THRESHOLD, DRIFT_ABS_POINTS
import db


def _avg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _delta(latest, prev):
    if latest is None or prev is None:
        return None
    return round(latest - prev, 2)


def _rel(latest, prev):
    if latest is None or not prev:
        return None
    return round((latest - prev) / prev, 3)


def _deteriorated(delta, rel):
    """Flag when error rose by at least the absolute OR the relative threshold."""
    if delta is None:
        return False
    return delta >= DRIFT_ABS_POINTS or (rel is not None and rel >= DRIFT_REL_THRESHOLD)


def compute(path=None):
    runs = db.list_runs(limit=2, path=path)
    if not runs:
        return dict(has_history=False, latest_run_id=None, previous_run_id=None,
                    thresholds=dict(rel=DRIFT_REL_THRESHOLD, abs_points=DRIFT_ABS_POINTS),
                    overall=None, skus=[], deteriorated=[])

    latest_id = runs[0]["run_id"]
    prev_id = runs[1]["run_id"] if len(runs) > 1 else None
    latest = db.sku_results(latest_id, path=path)
    prev = db.sku_results(prev_id, path=path) if prev_id else []
    prev_by_id = {r["sku_id"]: r for r in prev}

    skus, deteriorated = [], []
    for r in latest:
        p = prev_by_id.get(r["sku_id"])
        prev_mape = p["mape"] if p else None
        delta = _delta(r["mape"], prev_mape)
        rel = _rel(r["mape"], prev_mape)
        flagged = _deteriorated(delta, rel)
        if flagged:
            deteriorated.append(r["sku_id"])
        skus.append(dict(
            sku_id=r["sku_id"], model=r["selected_model"], mape=r["mape"], rmse=r["rmse"], mae=r["mae"],
            naive_mape=r["naive_mape"], prev_mape=prev_mape, delta_mape=delta, rel_change=rel,
            deteriorated=flagged, model_changed=bool(p and p["selected_model"] != r["selected_model"]),
        ))

    latest_avg, prev_avg, naive_avg = _avg(latest, "mape"), _avg(prev, "mape"), _avg(latest, "naive_mape")
    overall_delta = _delta(latest_avg, prev_avg)
    overall_rel = _rel(latest_avg, prev_avg)
    overall = dict(
        avg_mape=latest_avg, avg_naive=naive_avg,
        improvement=round(naive_avg - latest_avg, 2) if (naive_avg is not None and latest_avg is not None) else None,
        prev_avg_mape=prev_avg, delta_mape=overall_delta, rel_change=overall_rel,
        deteriorated=_deteriorated(overall_delta, overall_rel),
    )
    return dict(has_history=bool(prev_id), latest_run_id=latest_id, previous_run_id=prev_id,
                thresholds=dict(rel=DRIFT_REL_THRESHOLD, abs_points=DRIFT_ABS_POINTS),
                overall=overall, skus=skus, deteriorated=deteriorated)
