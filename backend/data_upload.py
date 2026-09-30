"""User-supplied supermarket datasets: parse, validate, normalize, persist.

Keeps all upload logic out of app.py. A CSV is validated with user-friendly
messages, normalized into the internal schema used by the forecasting pipeline,
and saved under backend/data/uploads. Inventory metadata (current stock, lead
time, service level) is supplied by the user after upload; nothing is silently
invented (missing values fall back to clearly-labelled demo assumptions).
"""
import io
import json
import os
import re
import time
import uuid

import numpy as np
import pandas as pd

from config import UPLOAD_DIR, UPLOAD_MIN_TRAIN_DAYS, MAX_SKUS, MAX_UPLOAD_ROWS, HORIZON

REQUIRED = ["date", "product", "sales"]
OPTIONAL = ["category", "price", "discount", "temp", "holiday_lift"]
CANONICAL = ["date", "product", "sales", "discount", "price", "temp", "holiday_lift"]
DEFAULT_LEAD = 3
DEFAULT_SERVICE_LEVEL = 0.95
TEMPLATE_HEADER = "date,product,category,sales,price,discount,temp,holiday_lift"


def sample_csv():
    """A small template showing the expected format (2 SKUs, 14 days)."""
    import datetime
    start = datetime.date(2025, 1, 1)
    rows = []
    for i in range(14):
        d = start + datetime.timedelta(days=i)
        disc = 0.10 if i % 7 == 3 else 0.0
        rows.append(f"{d},Fresh Milk 1L,Dairy,{280 + i * 3},{2.10:.2f},{disc:.2f},24.5,0")
        rows.append(f"{d},Sliced Bread,Bakery,{200 + i * 2},{1.80:.2f},0.00,22.1,0")
    return "\n".join([TEMPLATE_HEADER] + rows) + "\n"


def _lower_columns(df):
    return df.rename(columns=lambda c: str(c).lower().strip())


def validate(df, min_train=UPLOAD_MIN_TRAIN_DAYS, max_skus=MAX_SKUS):
    """Return (errors, warnings, detected) with user-friendly messages."""
    errors, warnings = [], []
    df = _lower_columns(df)

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        return [f"Missing required column(s): {', '.join(missing_cols)}. Expected columns: {TEMPLATE_HEADER}."], warnings, {}

    if len(df) == 0:
        return ["The file contains no data rows."], warnings, {}

    for col in REQUIRED:
        miss = int(df[col].isna().sum())
        if miss:
            errors.append(f"{miss} row(s) have a missing value in the required column '{col}'.")

    dates = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(dates.isna().sum())
    if bad_dates:
        errors.append(f"{bad_dates} row(s) have a date that could not be parsed (use YYYY-MM-DD).")

    sales = pd.to_numeric(df["sales"], errors="coerce")
    non_numeric = int((sales.isna() & df["sales"].notna()).sum())
    if non_numeric:
        errors.append(f"{non_numeric} row(s) have non-numeric sales values.")
    negative = int((sales < 0).sum())
    if negative:
        errors.append(f"{negative} row(s) have negative sales.")

    prod = df["product"].astype("string").str.strip()
    missing_prod = int((prod.isna() | (prod == "")).sum())
    if missing_prod:
        errors.append(f"{missing_prod} row(s) have a missing product id.")

    if errors:
        return errors, warnings, {}

    clean = df.assign(date=dates, product=prod, sales=sales)
    for col in ["price", "discount", "temp", "holiday_lift"]:
        if col in clean.columns:
            vals = pd.to_numeric(clean[col], errors="coerce")
            bad = int((clean[col].notna() & vals.isna()).sum())
            if bad:
                warnings.append(f"{bad} row(s) have non-numeric {col} values; defaults will be used for those cells.")
            if col in ("price", "discount") and int((vals < 0).sum()):
                warnings.append(f"Negative {col} values found; review whether these are valid for your dataset.")
    if "discount" in clean.columns and int((pd.to_numeric(clean["discount"], errors="coerce") > 1).sum()):
        warnings.append("Discount values above 1 were found. The model expects discount as a fraction from 0 to 1 (for example, 0.15 for 15%).")
    dup = int(clean.duplicated(subset=["product", "date"]).sum())
    if dup:
        errors.append(f"{dup} duplicate product + date row(s).")

    skus = sorted(clean["product"].unique())
    if len(skus) > max_skus:
        errors.append(f"Found {len(skus)} SKUs, which exceeds the demo limit of {max_skus}.")

    need = min_train + HORIZON
    for pid, g in clean.groupby("product"):
        if len(g) < need:
            errors.append(f"SKU '{pid}' has only {len(g)} daily row(s); at least {need} are required.")
        if g["sales"].nunique() <= 1:
            errors.append(f"SKU '{pid}' has a constant sales series and cannot be forecast.")

    gaps = 0
    for _, g in clean.groupby("product"):
        d = g["date"].dropna().sort_values()
        if len(d) >= 2:
            gaps += len(pd.date_range(d.min(), d.max(), freq="D")) - d.nunique()
    if gaps:
        warnings.append(f"{gaps} calendar day(s) are missing within SKU date ranges (the models handle gaps, but they reduce accuracy).")
    warnings.append("Future driver assumptions: the next 28 days use 0% discount, the last observed price and temperature per SKU, and a calendar-derived holiday lift. No future promotion or weather schedule is requested, so treat forecasts as conditional on these assumptions.")

    detected = dict(rows=int(len(clean)), skus=len(skus),
                    date_min=str(clean["date"].min().date()), date_max=str(clean["date"].max().date()),
                    products=[dict(id=str(p), rows=int((clean["product"] == p).sum())) for p in skus])
    return errors, warnings, detected


def normalize(df):
    """Coerce to the internal schema, filling optional columns with safe defaults."""
    df = _lower_columns(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product"] = df["product"].astype(str).str.strip()
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    if "category" not in df.columns:
        df["category"] = "Unknown"
    df["category"] = df["category"].fillna("Unknown").astype(str)
    if "price" not in df.columns:
        df["price"] = np.nan
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if df["price"].isna().any():
        # neutral default: per-SKU median if available, otherwise a flat 1.0
        median = df.groupby("product")["price"].transform("median")
        df["price"] = df["price"].fillna(median).fillna(1.0)
    for col in ["discount", "temp", "holiday_lift"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date", "product", "sales"]).sort_values(["product", "date"]).reset_index(drop=True)
    return df


def process(data_bytes):
    """Full parse -> validate -> normalize. Returns a JSON-friendly report."""
    try:
        raw = pd.read_csv(io.BytesIO(data_bytes))
    except Exception as e:  # unreadable file
        return dict(ok=False, errors=[f"Could not read the CSV file: {e}"], warnings=[], detected={}, optional={}, preview=[])

    if len(raw) > MAX_UPLOAD_ROWS:
        return dict(ok=False, errors=[f"The file has {len(raw)} rows, exceeding the limit of {MAX_UPLOAD_ROWS}."],
                    warnings=[], detected={}, optional={}, preview=[])

    raw = _lower_columns(raw)
    optional = {c: bool(c in raw.columns) for c in OPTIONAL}
    errors, warnings, detected = validate(raw)
    if errors:
        return dict(ok=False, errors=errors, warnings=warnings, detected=detected, optional=optional, preview=[])

    norm = normalize(raw)
    products = []
    for pid, g in norm.groupby("product"):
        g = g.sort_values("date")
        price = float(g["price"].median()) if g["price"].notna().any() else None
        cat = g["category"].iloc[-1] if g["category"].notna().any() else "Unknown"
        recent = float(g["sales"].iloc[-14:].mean())
        products.append(dict(id=str(pid), category=str(cat),
                             price=round(price, 2) if (price and price > 0) else None,
                             rows=int(len(g)), suggested_stock=int(max(0, round(7 * recent)))))

    preview = norm.head(5).assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
    out = norm.copy(); out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    normalized = out[CANONICAL].to_csv(index=False).encode()

    return dict(ok=True, errors=[], warnings=warnings,
                detected=dict(rows=int(len(norm)), skus=int(norm["product"].nunique()),
                              date_min=str(norm["date"].min().date()), date_max=str(norm["date"].max().date()),
                              products=products),
                optional=optional, preview=preview, normalized=normalized)


def build_products(detected, inventory=None):
    """Turn detected SKUs + user inventory inputs into the internal product spec."""
    inventory = inventory or {}
    products = []
    for mp in detected.get("products", []):
        pid = str(mp["id"])
        inv = inventory.get(pid) or {}
        price = mp.get("price") or 1.0
        lead = int(inv.get("lead", DEFAULT_LEAD) or DEFAULT_LEAD)
        stock = inv.get("stock")
        stock = mp.get("suggested_stock", 0) if stock in (None, "") else float(stock)
        svc = inv.get("service_level")
        products.append(dict(
            id=pid, name=pid, category=mp.get("category") or "Unknown", legacy_model=None,
            price=round(float(price), 2), cost=round(float(inv.get("unit_cost", float(price) * 0.7) or float(price) * 0.7), 2),
            lead=max(1, int(lead)), shelf=30, stock=float(max(0.0, stock)), stock_days=None,
            on_order=max(0, int(inv.get("on_order", 0) or 0)),
            max_cover=max(7, int(lead) + 14),
            service_level=float(svc) if svc not in (None, "") else None,
        ))
    return products


# --- persistence of uploaded datasets ---------------------------------------
_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def new_dataset_id():
    return time.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def _safe(dataset_id):
    return bool(dataset_id) and bool(_ID_RE.match(str(dataset_id)))


def dataset_path(dataset_id):
    return os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")


def meta_path(dataset_id):
    return os.path.join(UPLOAD_DIR, f"{dataset_id}.json")


def save_dataset(dataset_id, csv_bytes):
    if not _safe(dataset_id):
        raise ValueError("invalid dataset id")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(dataset_path(dataset_id), "wb") as f:
        f.write(csv_bytes)
    return dataset_path(dataset_id)


def save_meta(dataset_id, meta):
    if not _safe(dataset_id):
        raise ValueError("invalid dataset id")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(meta_path(dataset_id), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta_path(dataset_id)


def load_meta(dataset_id):
    p = meta_path(dataset_id)
    if not _safe(dataset_id) or not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def exists(dataset_id):
    return _safe(dataset_id) and os.path.exists(dataset_path(dataset_id))
