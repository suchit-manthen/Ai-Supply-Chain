"""Streaming adapter for the Corporación Favorita ``train.csv`` format.

The raw file is scanned in chunks. Only the selected store, item ids and dates
are retained when preparing the small CSV consumed by the existing pipeline.
"""
from __future__ import annotations

import json
import csv
import os
import time
import uuid

import numpy as np
import pandas as pd

from config import UPLOAD_DIR, UPLOAD_MIN_TRAIN_DAYS, HORIZON, MAX_FORECAST_ROWS

REQUIRED = ("date", "store_nbr", "item_nbr", "unit_sales")
OPTIONAL = ("onpromotion",)
CHUNK_SIZE = 100_000
MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_SELECTED_ITEMS = 8


def is_favorita_header(line):
    """Identify Favorita files from their header without reading CSV rows."""
    try:
        columns = next(csv.reader([line.decode("utf-8-sig").strip()]))
    except (UnicodeDecodeError, csv.Error, StopIteration, AttributeError):
        return False
    normalized = {str(column).strip().lower() for column in columns}
    return set(REQUIRED).issubset(normalized)


def _paths(dataset_id):
    if not dataset_id or not str(dataset_id).replace("-", "").isalnum():
        raise ValueError("Invalid dataset id.")
    root = os.path.join(UPLOAD_DIR, f"{dataset_id}-favorita")
    return root + ".csv", root + ".json"


def raw_path(dataset_id):
    return _paths(dataset_id)[0]


def exists(dataset_id):
    try:
        return os.path.isfile(raw_path(dataset_id))
    except ValueError:
        return False


def discard_upload(dataset_id):
    """Remove an invalid staged source after its inspection failed."""
    source, metadata = _paths(dataset_id)
    for candidate in (source, metadata):
        if os.path.isfile(candidate):
            os.remove(candidate)


def save_metadata(dataset_id, metadata):
    _, meta_path = _paths(dataset_id)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)


def load_metadata(dataset_id):
    _, meta_path = _paths(dataset_id)
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def save_upload(stream):
    """Save an uploaded stream incrementally and return (id, path, byte_count)."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dataset_id = time.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    path, _ = _paths(dataset_id)
    total = 0
    try:
        with open(path, "wb") as out:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_FILE_BYTES:
                    raise ValueError("CSV exceeds the 2 GB upload limit.")
                out.write(block)
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        raise
    if not total:
        os.remove(path)
        raise ValueError("The uploaded CSV is empty.")
    return dataset_id, path, total


def _columns(path):
    cols = [str(c).strip().lower() for c in pd.read_csv(path, nrows=0).columns]
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ValueError("Missing required Favorita column(s): " + ", ".join(missing))
    return cols


def inspect(path):
    """Return basic file metadata and quality counts without loading all rows."""
    cols = _columns(path)
    read_cols = [c for c in (*REQUIRED, *OPTIONAL) if c in cols]
    stores, items = set(), set()
    row_count = invalid_dates = invalid_sales = missing_sales = missing_ids = missing_promotions = negative_sales = 0
    minimum, maximum = None, None
    wanted = set(read_cols)
    for chunk in pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in wanted, chunksize=CHUNK_SIZE):
        chunk.columns = [str(c).strip().lower() for c in chunk.columns]
        row_count += len(chunk)
        dates = pd.to_datetime(chunk["date"], errors="coerce")
        invalid_dates += int(dates.isna().sum())
        if dates.notna().any():
            lo, hi = dates.min(), dates.max()
            minimum = lo if minimum is None or lo < minimum else minimum
            maximum = hi if maximum is None or hi > maximum else maximum
        sales = pd.to_numeric(chunk["unit_sales"], errors="coerce")
        invalid_sales += int((chunk["unit_sales"].notna() & (sales.isna() | ~np.isfinite(sales))).sum())
        missing_sales += int(chunk["unit_sales"].isna().sum())
        negative_sales += int((sales < 0).sum())
        missing_ids += int(chunk["store_nbr"].isna().sum() + chunk["item_nbr"].isna().sum())
        stores.update(chunk["store_nbr"].dropna().astype(str).unique())
        items.update(chunk["item_nbr"].dropna().astype(str).unique())
        if "onpromotion" in chunk:
            missing_promotions += int(chunk["onpromotion"].isna().sum())
    quality = {
        "invalid_dates": invalid_dates, "invalid_sales": invalid_sales,
        "missing_sales": missing_sales, "missing_store_or_item": missing_ids,
        "missing_promotion": missing_promotions, "negative_unit_sales_rows": negative_sales,
    }
    warnings = []
    if invalid_dates: warnings.append(f"{invalid_dates} rows have invalid dates and will be excluded.")
    if invalid_sales: warnings.append(f"{invalid_sales} non-numeric sales values will be excluded.")
    if missing_sales: warnings.append(f"{missing_sales} rows have unknown/missing sales; they will be excluded, not treated as zero.")
    if missing_ids: warnings.append(f"{missing_ids} rows have a missing store or item id and will be excluded.")
    if negative_sales: warnings.append(f"{negative_sales} transaction rows have negative unit_sales (likely returns); selected negative daily net demand cannot be forecast by this workflow and will be reported during preparation.")
    if missing_promotions: warnings.append(f"{missing_promotions} promotion values are missing; they will be treated as no promotion and flagged in dataset quality.")
    if "onpromotion" not in cols:
        warnings.append("No onpromotion column is present; the promotion feature will be zero for all selected rows.")
    return dict(rows=row_count, date_min=str(minimum.date()) if minimum is not None else None,
                date_max=str(maximum.date()) if maximum is not None else None,
                stores=len(stores), skus=len(items), store_options=sorted(stores, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)),
                columns=cols, required_columns=list(REQUIRED), available_columns=cols,
                quality=quality, warnings=warnings)


def available_items(path, store_nbr):
    """List distinct item ids at a store, scanning only the id column pairs."""
    _columns(path)
    found = set()
    wanted = {"store_nbr", "item_nbr"}
    for chunk in pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in wanted, chunksize=CHUNK_SIZE):
        chunk.columns = [str(c).strip().lower() for c in chunk.columns]
        match = chunk[chunk["store_nbr"].astype(str) == str(store_nbr)]
        found.update(match["item_nbr"].dropna().astype(str).unique())
    return sorted(found, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))


def _promotion(values):
    """Normalize bool/numeric/text promotion flags to 0/1."""
    truthy = {"1", "true", "t", "yes", "y"}
    text = values.astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(values, errors="coerce")
    return (text.isin(truthy) | numeric.eq(1)).astype(int)


def prepare(path, store_nbr, item_nbrs, start_date=None, end_date=None):
    """Filter one store and <=8 items, aggregate duplicates, then write pipeline CSV.

    Missing sales rows are unknown observations and excluded. Date gaps are
    reported and block preparation: the existing forecasting code expects a
    daily sequence, so filling gaps with zero would fabricate demand.
    """
    cols = _columns(path)
    items = [str(x) for x in dict.fromkeys(item_nbrs or [])]
    if not items:
        raise ValueError("Select at least one item.")
    if len(items) > MAX_SELECTED_ITEMS:
        raise ValueError(f"Select at most {MAX_SELECTED_ITEMS} items for this first demo.")
    start = pd.Timestamp(start_date).normalize() if start_date else None
    end = pd.Timestamp(end_date).normalize() if end_date else None
    if start is not None and end is not None and start > end:
        raise ValueError("Start date must be on or before end date.")
    usecols = [c for c in (*REQUIRED, *OPTIONAL) if c in cols]
    # Keep only one aggregated record per daily store/item key in memory. This
    # lets a large transaction file stream through without retaining its rows.
    aggregated = {}
    valid_source_rows = 0
    quality = {"invalid_dates": 0, "invalid_sales": 0, "missing_sales": 0,
               "missing_ids": 0, "negative_sales_rows": 0, "missing_promotion": 0}
    wanted = set(usecols)
    for chunk in pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in wanted, chunksize=CHUNK_SIZE):
        chunk.columns = [str(c).strip().lower() for c in chunk.columns]
        chunk["store_nbr"] = chunk["store_nbr"].astype("string")
        chunk["item_nbr"] = chunk["item_nbr"].astype("string")
        subset = chunk[(chunk["store_nbr"] == str(store_nbr)) & chunk["item_nbr"].isin(items)].copy()
        if subset.empty:
            continue
        subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
        subset["date"] = subset["date"].dt.normalize()
        quality["invalid_dates"] += int(subset.date.isna().sum())
        if start is not None: subset = subset[subset.date.isna() | (subset.date >= start)]
        if end is not None: subset = subset[subset.date.isna() | (subset.date <= end)]
        subset["sales"] = pd.to_numeric(subset["unit_sales"], errors="coerce")
        quality["invalid_sales"] += int((subset.unit_sales.notna() & (subset.sales.isna() | ~np.isfinite(subset.sales))).sum())
        subset.loc[~np.isfinite(subset.sales), "sales"] = np.nan
        quality["missing_sales"] += int(subset.sales.isna().sum())
        quality["missing_ids"] += int((subset.store_nbr.isna() | subset.item_nbr.isna()).sum())
        quality["negative_sales_rows"] += int((subset.sales < 0).sum())
        if "onpromotion" in subset:
            quality["missing_promotion"] += int(subset.onpromotion.isna().sum())
            subset["promotion"] = _promotion(subset.onpromotion.fillna(False))
        else:
            subset["promotion"] = 0
        subset = subset.dropna(subset=["date", "sales", "store_nbr", "item_nbr"])
        if not subset.empty:
            valid_source_rows += len(subset)
            chunk_aggregated = subset.groupby(["date", "store_nbr", "item_nbr"], as_index=False).agg(
                sales=("sales", "sum"), promotion=("promotion", "max"))
            for day, store, item, sales, promotion in chunk_aggregated.itertuples(index=False, name=None):
                key = (day, store, item)
                if key in aggregated:
                    prior_sales, prior_promotion = aggregated[key]
                    aggregated[key] = (prior_sales + sales, max(prior_promotion, promotion))
                else:
                    aggregated[key] = (sales, promotion)
                    if len(aggregated) > MAX_FORECAST_ROWS:
                        raise ValueError(
                            f"The filtered forecasting dataset exceeds the {MAX_FORECAST_ROWS:,}-row limit. "
                            "Narrow the store, SKU selection, or date range and try again.")
    if not aggregated:
        raise ValueError("No valid rows remain for this store, item selection, and date range.")
    grouped = pd.DataFrame([
        {"date": key[0], "store_nbr": key[1], "item_nbr": key[2], "sales": value[0], "promotion": value[1]}
        for key, value in aggregated.items()
    ])
    grouped = grouped.sort_values(["item_nbr", "date"]).reset_index(drop=True)
    products = []
    gaps = []
    for item in items:
        series = grouped[grouped.item_nbr == item].sort_values("date")
        if series.empty:
            continue
        expected_start = start if start is not None else series.date.min()
        expected_end = end if end is not None else series.date.max()
        days = pd.date_range(expected_start, expected_end, freq="D")
        missing = len(days) - series.date.nunique()
        if missing:
            gaps.append({"item_nbr": item, "missing_days": int(missing)})
        if (series.sales < 0).any():
            raise ValueError(f"Item {item} has negative daily net sales after aggregation (returns). This forecasting workflow requires non-negative demand; resolve returns explicitly before training.")
        if len(series) < UPLOAD_MIN_TRAIN_DAYS + HORIZON:
            raise ValueError(f"Item {item} has {len(series)} observed days; at least {UPLOAD_MIN_TRAIN_DAYS + HORIZON} are required for validation and forecasting.")
        recent = float(series.sales.tail(14).mean())
        products.append(dict(id=item, name=f"Item {item}", category="Favorita", rows=int(len(series)),
                             suggested_stock=max(0, int(round(7 * recent))), store_nbr=str(store_nbr), item_nbr=item))
    if gaps:
        details = ", ".join(f"item {g['item_nbr']}: {g['missing_days']} day(s)" for g in gaps)
        raise ValueError("Selected items have missing calendar dates (" + details + "). Missing dates are not assumed to mean zero sales; choose a date range/items with daily observations for this initial adapter.")
    if len(products) != len(items):
        absent = sorted(set(items) - {p["id"] for p in products})
        raise ValueError("No usable rows for selected item(s): " + ", ".join(absent))

    # The existing pipeline uses `discount` as its promotion regressor. A binary
    # flag is carried through unchanged (not interpreted as a discount amount).
    output = grouped.rename(columns={"item_nbr": "product"})
    output["discount"] = output["promotion"].astype(float)
    output["price"] = 1.0  # neutral placeholder; Favorita train.csv has no price field
    output["temp"] = 0.0
    output["holiday_lift"] = 0.0
    output["stockout"] = 0
    output = output[["date", "product", "sales", "discount", "price", "temp", "holiday_lift", "stockout"]]
    output["date"] = output.date.dt.strftime("%Y-%m-%d")
    data = output.to_csv(index=False).encode("utf-8")
    return data, dict(rows=int(len(output)), stores=1, skus=len(products), store_nbr=str(store_nbr),
                      date_min=str(output.date.min()), date_max=str(output.date.max()),
                      products=products, columns=list(output.columns), source_columns=cols,
                      quality=quality, duplicate_rows_aggregated=int(valid_source_rows - len(grouped)),
                      warnings=["Favorita onpromotion is passed to the existing discount regressor as a binary promotion flag; it does not represent discount depth.",
                                "train.csv has no unit price, weather, or stockout fields. Neutral feature values are used where unavailable; inventory costs/stock must be configured separately."] + ([f"{quality['missing_sales']} rows with missing sales were excluded; they were not converted to zero."] if quality["missing_sales"] else []))

