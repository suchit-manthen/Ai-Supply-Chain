"""Training pipeline (separate from API serving).

Runs the full per-SKU workflow - evaluate all models with walk-forward
validation, select the best per SKU, refit, forecast the next 28 days - builds
the owner and analyst payloads, and persists a completed run to SQLite.

Data comes from datasource.load_data (synthetic by default, or a CSV). The
`products` argument lets a caller supply its own SKU list (e.g. discovered from
an uploaded CSV); it defaults to the synthetic catalog.
"""
import time
from datetime import datetime, timezone

from data import PRODUCTS
from config import (HORIZON, HOLDOUT, N_FOLDS, MIN_TRAIN_DAYS, SEED, SERVICE_LEVEL,
                    SELECTION_METRIC, SELECTION_FALLBACK_METRIC, LOW_DEMAND_THRESHOLD, UPLOAD_MIN_TRAIN_DAYS)
from datasource import load_data, validate_sales, catalog_ids
from selection import select_and_forecast
from views import owner_view, analyst_view
import db


def _sku_rows(analyst):
    """Flatten the selected model's validation metrics into per-SKU rows."""
    rows = []
    for s in analyst["skus"]:
        comp = next(c for c in s["comparison"] if c["model"] == s["selected_model"])
        base = next(c for c in s["comparison"] if c.get("baseline"))
        rows.append(dict(
            sku_id=s["id"], selected_model=s["selected_model"], selection_metric=s["metric_used"],
            selection_score=s["selection_score"], baseline_score=s["baseline_score"],
            mape=comp["mape"], rmse=comp["rmse"], mae=comp["mae"], naive_mape=base["mape"],
        ))
    return rows


def train(progress_cb=None, path=None, source=None, seed=SEED, policy=None,
          products=None, data_source=None, min_train=None, n_folds=None):
    """Run one full training pass and persist it. Returns the saved run dict.

    source      : None (synthetic) or a .csv path
    products    : list of SKU specs (defaults to the synthetic catalog)
    data_source : label stored in the run ("synthetic" or "uploaded")
    min_train / n_folds : walk-forward settings (auto-computed when omitted)
    policy      : inventory overrides (service_level, z, ordering_cost, holding_rate)
    """
    strict_catalog = products is None
    products = products or PRODUCTS
    data_source = data_source or ("synthetic" if source is None else "uploaded")
    if min_train is None:
        min_train = MIN_TRAIN_DAYS if data_source == "synthetic" else UPLOAD_MIN_TRAIN_DAYS

    db.init_db(path)
    started = time.time()
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = db.new_run_id()

    df, end, hol = load_data(source, seed=seed)
    hist_df = df[df.date <= end].reset_index(drop=True)
    fut_df = df[df.date > end].reset_index(drop=True)

    present = set(hist_df["product"].astype(str))
    missing = sorted(set(p["id"] for p in products) - present)
    if missing:
        raise ValueError(f"Data is missing required SKUs: {missing}.")
    if strict_catalog and source is not None:
        catalog_missing = sorted(set(catalog_ids()) - present)
        if catalog_missing:
            raise ValueError(f"Data is missing catalog SKUs: {catalog_missing}. "
                             "Real CSVs mapped onto the catalog must include every catalog SKU, or use the upload flow.")

    quality = validate_sales(hist_df, product_ids=[p["id"] for p in products])
    if not quality["ok"]:
        bad = ", ".join(i["type"] for i in quality["issues"] if i["severity"] == "error")
        raise ValueError(f"Data validation failed ({bad}); fix the input data and retrain.")

    n_min = int(hist_df.groupby("product").size().min())
    need = min_train + HORIZON
    if n_min < need:
        raise ValueError(f"Not enough history: the shortest SKU has {n_min} daily rows; at least {need} are required.")
    if n_folds is None:
        n_folds = max(1, min(N_FOLDS, (n_min - min_train) // HORIZON))

    results = {}
    total = len(products)
    for i, p in enumerate(products):
        h = hist_df[hist_df["product"] == p["id"]].reset_index(drop=True)
        f = fut_df[fut_df["product"] == p["id"]].reset_index(drop=True)
        results[p["id"]] = select_and_forecast(h, f, hol, n_folds=n_folds, min_train=min_train)
        if progress_cb:
            progress_cb(round((i + 1) / total * 100), p["id"])

    owner = owner_view(results, hist_df, end, policy=policy, products=products, data_source=data_source)
    analyst = analyst_view(results, hist_df, end, hol, data_quality=quality, products=products,
                           data_source=data_source, n_folds=n_folds)

    run = dict(
        run_id=run_id, created_at=created_at, duration_s=round(time.time() - started, 1), status="completed",
        dataset=analyst["dataset"],
        config=dict(horizon=HORIZON, holdout=HOLDOUT, n_folds=n_folds, min_train=min_train, seed=seed,
                    selection_metric=SELECTION_METRIC, selection_fallback=SELECTION_FALLBACK_METRIC,
                    low_demand_threshold=LOW_DEMAND_THRESHOLD,
                    source=data_source, input=("synthetic" if source is None else str(source)),
                    service_level=float((policy or {}).get("service_level", SERVICE_LEVEL)),
                    n_skus=len(products), data_quality=quality),
        owner=owner, analyst=analyst, skus=_sku_rows(analyst),
    )
    db.save_run(run, path=path)
    return run
