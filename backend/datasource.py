"""Data input interface + validation.

Canonical schema - one row per product per day:

    date         datetime
    product      str      (catalog SKU id)
    sales        float    observed units sold (may be censored on stockout days)
    discount     float    promotion discount fraction, 0 = none
    price        float    shelf price
    temp         float    daily temperature
    holiday_lift float    holiday ramp factor (derived from the calendar)
    stockout     int      optional: 1 if the day was censored by a stockout

`load_data(None)` uses the existing synthetic generator. `load_data("file.csv")`
loads a real CSV and synthesizes the 28-day future window from the last known
exogenous values, so the forecasting pipeline runs unchanged. Real data must use
the catalog SKU ids (see data.PRODUCTS) because inventory metadata (lead time,
shelf life, costs) comes from that catalog.
"""
import os

import numpy as np
import pandas as pd

from config import HORIZON, SEED
from data import PRODUCTS, generate, holiday_table, _holiday_lift

REQUIRED = ["date", "product", "sales"]
OPTIONAL = ["discount", "price", "temp", "holiday_lift", "stockout"]
CATALOG = {p["id"]: p for p in PRODUCTS}


def catalog_ids():
    return [p["id"] for p in PRODUCTS]


def validate_sales(df, product_ids=None):
    """Check an observed-sales frame and return {ok, issues, summary}.

    Detects: missing dates, missing/unknown SKUs, duplicate records, invalid
    prices, negative sales and missing demand observations.
    """
    if not all(c in df.columns for c in REQUIRED):
        return dict(ok=False, issues=[dict(type="missing_columns", severity="error", count=1,
                                           detail=f"Required columns are {REQUIRED}")], summary={})
    ids = set(product_ids or catalog_ids())
    issues = []

    def add(kind, severity, count, detail):
        if count:
            issues.append(dict(type=kind, severity=severity, count=int(count), detail=detail))

    sales = pd.to_numeric(df["sales"], errors="coerce")
    prod = df["product"].astype("string")
    missing_sku = int((prod.isna() | (prod.str.strip() == "")).sum())
    unknown = sorted(set(prod.dropna().str.strip()) - ids)
    negative = int((sales < 0).sum())
    missing_sales = int(sales.isna().sum())
    dup = int(df.duplicated(subset=["product", "date"]).sum())

    add("missing_sku", "error", missing_sku, "Rows with a missing product id")
    add("unknown_sku", "warning", len(unknown), f"Product ids not in the catalog: {unknown}")
    add("negative_sales", "error", negative, "Rows with negative sales")
    add("missing_sales", "error", missing_sales, "Rows with no demand observation")
    add("duplicate_record", "error", dup, "Duplicate product + date rows")

    if "price" in df.columns:
        price = pd.to_numeric(df["price"], errors="coerce")
        add("invalid_price", "error", int(((price <= 0) | price.isna()).sum()), "Rows with missing or non-positive price")

    gaps = 0
    for _, g in df.groupby("product"):
        d = pd.to_datetime(g["date"]).dropna().sort_values()
        if len(d) >= 2:
            gaps += len(pd.date_range(d.min(), d.max(), freq="D")) - d.nunique()
    add("missing_dates", "warning", gaps, "Calendar days absent within each product's date range")

    dates = pd.to_datetime(df["date"])
    stockout_days = int(pd.to_numeric(df["stockout"], errors="coerce").fillna(0).sum()) if "stockout" in df.columns else 0
    summary = dict(
        rows=int(len(df)), skus=int(prod.nunique()), date_min=str(dates.min().date()), date_max=str(dates.max().date()),
        duplicates=dup, missing_sales=missing_sales, negative_sales=negative, missing_dates=int(gaps),
        stockout_days=stockout_days, unknown_skus=unknown, missing_skus=sorted(ids - set(prod.dropna().str.strip())),
    )
    ok = not any(i["severity"] == "error" for i in issues)
    return dict(ok=ok, issues=issues, summary=summary)


def load_synthetic(seed=SEED):
    """Existing generator; adds stockout and category columns."""
    df, end, hol = generate(seed)
    if "stockout" not in df.columns:
        df = df.assign(stockout=0)
    if "category" not in df.columns:
        categories = {p["id"]: p["category"] for p in PRODUCTS}
        df = df.assign(category=df["product"].map(categories).fillna("Unknown"))
    return df, end, hol


def _coerce_csv(raw):
    raw = raw.rename(columns={c: str(c).lower().strip() for c in raw.columns})
    for req in REQUIRED:
        if req not in raw.columns:
            raise ValueError(f"CSV is missing required column '{req}'")
    raw["date"] = pd.to_datetime(raw["date"])
    raw["product"] = raw["product"].astype(str).str.strip()
    raw["sales"] = pd.to_numeric(raw["sales"], errors="coerce")
    price_map = {k: v["price"] for k, v in CATALOG.items()}
    if "price" not in raw.columns:
        raw["price"] = raw["product"].map(price_map)
    # catalog price when known, else a neutral default (never 0, which would distort revenue/EOQ)
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce").fillna(raw["product"].map(price_map)).fillna(1.0)
    for col, default in [("discount", 0.0), ("temp", 0.0), ("holiday_lift", 0.0), ("stockout", 0)]:
        if col not in raw.columns:
            raw[col] = default
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["discount"] = raw["discount"].fillna(0.0)
    raw["temp"] = raw["temp"].fillna(0.0)
    raw["stockout"] = raw["stockout"].fillna(0).astype(int)
    return raw


def load_csv(path, seed=SEED):
    """Load a CSV of observed sales and append a synthesized 28-day future window."""
    raw = _coerce_csv(pd.read_csv(path))
    hist = raw[raw["sales"].notna() & raw["product"].notna()].sort_values(["product", "date"]).reset_index(drop=True)
    if hist.empty:
        raise ValueError("CSV contains no usable demand rows (need date, product, sales)")
    end = hist["date"].max()
    hol = holiday_table(range(hist["date"].min().year, (end + pd.Timedelta(days=HORIZON)).year + 1))
    hist["holiday_lift"] = _holiday_lift(pd.DatetimeIndex(hist["date"]), hol)

    has_category = "category" in hist.columns
    future_dates = pd.date_range(end + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    hl_future = _holiday_lift(pd.DatetimeIndex(future_dates), hol)
    frames = [hist]
    for pid in hist["product"].unique():
        last = hist[hist["product"] == pid].iloc[-1]
        future = dict(
            date=future_dates, product=pid, sales=np.nan, discount=0.0,
            price=float(last["price"]) if pd.notna(last["price"]) else 0.0,
            temp=float(last["temp"]) if pd.notna(last["temp"]) else 0.0,
            holiday_lift=hl_future, stockout=0,
        )
        if has_category:
            last_category = last.get("category", "Unknown")
            future["category"] = last_category if pd.notna(last_category) else "Unknown"
        frames.append(pd.DataFrame(future))
    combined = pd.concat(frames, ignore_index=True)
    columns = ["date", "product", "sales", "discount", "price", "temp", "holiday_lift", "stockout"]
    if has_category:
        combined["category"] = combined["category"].fillna("Unknown")
        columns = columns + ["category"]
    return combined[columns], end, hol


def load_data(source=None, seed=SEED):
    """Return (df, end, hol) for a data source.

    source=None -> synthetic generator; source="<path>.csv" -> real CSV.
    """
    if source is None or (isinstance(source, str) and source.lower() == "synthetic"):
        return load_synthetic(seed)
    if isinstance(source, (str, os.PathLike)) and str(source).lower().endswith(".csv"):
        return load_csv(str(source), seed=seed)
    raise ValueError(f"Unsupported data source {source!r}; use None (synthetic) or a .csv path")
