"""Generic CSV understanding: infer a schema then normalize to canonical.

Given any delimited sales file, this module:

1. profiles each column and scores how well it fits each semantic role
   (date, sales/demand, product/entity, store and known optional drivers),
   returning a ranked mapping with confidence scores;
2. detects the table *layout* - long (one observation per row) or wide (one
   column per product/series) - so a matrix export can be melted too;
3. converts a user-confirmed mapping into the canonical
   ``date, product, sales, discount, price, temp, holiday_lift, stockout``
   shape, handling dates, duplicates/aggregation and missing values.

Minimum requirement is ``date + product/entity + sales`` (for a wide file,
``date + value columns``); every other role is optional.
"""
from __future__ import annotations

import csv
import re

import numpy as np
import pandas as pd

from config import MAX_FORECAST_ROWS
import data_upload
from data_adapters import storage

SUFFIX = "generic"
SAMPLE_ROWS = 5000
SCAN_CHUNK_SIZE = 100_000  # raw rows read per chunk during inspect/prepare
CANONICAL = ["date", "product", "sales", "discount", "price", "temp", "holiday_lift", "stockout", "category"]

ROLES = ["date", "sales", "product", "store", "category", "price",
         "discount", "promo", "temp", "holiday_lift", "stockout"]
REQUIRED_ROLES = ["date", "product", "sales"]
OPTIONAL_ROLES = [r for r in ROLES if r not in REQUIRED_ROLES]

# Confidence needed before a role is auto-assigned. Required roles are a little
# stricter so an ambiguous column does not silently become the target.
THRESHOLD = {"date": 0.55, "sales": 0.55, "product": 0.45, "store": 0.5,
             "category": 0.5, "price": 0.5, "discount": 0.5, "promo": 0.5,
             "temp": 0.5, "holiday_lift": 0.5, "stockout": 0.5}

# Order matters: earlier roles claim an ambiguous column first.
ASSIGN_ORDER = ["date", "sales", "product", "store", "category", "price",
                "discount", "promo", "temp", "holiday_lift", "stockout"]

NAME_SYNONYMS = {
    "date": ["date", "ds", "datetime", "timestamp", "day", "period", "week", "month", "time", "datum", "fecha", "dt"],
    "sales": ["sales", "sale", "unitsales", "units", "qty", "quantity", "demand", "orders", "order",
              "volume", "sold", "unitssold", "salesqty", "consumption", "issues", "movement"],
    "product": ["product", "productid", "productname", "pid", "item", "itemid", "itemnbr", "itemno",
                "itemcode", "itemnumber", "sku", "skuid", "skucode", "upc", "ean", "gtin", "article",
                "articleno", "productcode", "material", "matnr", "barcode", "entity", "entityid", "series"],
    "store": ["store", "storeid", "storenbr", "storeno", "storenumber", "storecode", "branch", "branchid",
              "shop", "shopid", "location", "locationid", "site", "siteid", "outlet", "outletid", "region", "channel"],
    "category": ["category", "categoryid", "cat", "family", "dept", "department", "class", "group",
                 "segment", "subcategory", "productfamily", "commodity", "section"],
    "price": ["price", "unitprice", "sellprice", "sellingprice", "retail", "retailprice", "mrp",
              "listprice", "saleprice", "cost", "unitcost"],
    "discount": ["discount", "disc", "discountpct", "discountpercent", "discountrate", "markdown",
                 "rebate", "promodiscount", "discountamount", "promo", "promotion", "promorate",
                 "promopct", "promopercent", "promodepth"],
    "promo": ["onpromotion", "onpromo", "promo", "promotion", "promoted", "ispromo", "ispromotion",
              "promoflag", "promotionflag", "ad", "adflag", "feature", "featured"],
    "temp": ["temp", "temperature", "tempc", "tempf", "weathertemp", "maxtemp", "mintemp", "airtemp"],
    "holiday_lift": ["holiday", "holidaylift", "isholiday", "holidayflag", "hol", "holidayindicator", "event"],
    "stockout": ["stockout", "stockoutflag", "isstockout", "outofstock", "oos", "censored",
                 "stockoutindicator", "stockoutday"],
}

FALSY_TRUTHY = {"0", "1", "true", "false", "t", "f", "yes", "no", "y", "n", "on", "off"}


# --- header helpers ---------------------------------------------------------

def canonical_required_present(header_line):
    """True if a raw header already carries the canonical date/product/sales trio."""
    try:
        columns = next(csv.reader([header_line.decode("utf-8-sig").strip()]))
    except (UnicodeDecodeError, csv.Error, StopIteration, AttributeError):
        return False
    normalized = {str(column).strip().lower() for column in columns}
    return {"date", "product", "sales"}.issubset(normalized)


# --- persistence (thin wrappers over the shared storage helper) -------------

def save_upload(stream):
    return storage.save_upload(stream, SUFFIX)


def save_metadata(dataset_id, metadata):
    storage.save_metadata(dataset_id, SUFFIX, metadata)


def load_metadata(dataset_id):
    return storage.load_metadata(dataset_id, SUFFIX)


def raw_path(dataset_id):
    return storage.raw_path(dataset_id, SUFFIX)


def exists(dataset_id):
    return storage.exists(dataset_id, SUFFIX)


def discard(dataset_id):
    storage.discard(dataset_id, SUFFIX)


# --- date parsing -----------------------------------------------------------

def _day_month_hint(text):
    """Count ambiguous d/m vs m/d evidence so we can pick the right order."""
    day_first = month_first = 0
    for value in list(text.dropna().astype(str))[:300]:
        match = re.match(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.]", value)
        if match:
            first, second = int(match.group(1)), int(match.group(2))
            if first > 12:
                day_first += 1
            if second > 12:
                month_first += 1
    return day_first, month_first


def _parse_dates(series):
    """Parse a date column in several formats. Returns (series, format_note)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, "datetime"
    if pd.api.types.is_numeric_dtype(series):
        num = pd.to_numeric(series, errors="coerce")
        valid = num.dropna()
        if len(valid) and bool((valid % 1 == 0).all()) and valid.between(19000101, 29991231).mean() > 0.9:
            return pd.to_datetime(num.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce"), "%Y%m%d"
        return None, None

    text = series.astype("string").str.strip()
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"]:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return parsed, fmt

    day_first, month_first = _day_month_hint(text)
    if day_first >= month_first:
        ordered = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%m/%d/%Y", "%m-%d-%Y"]
    else:
        ordered = ["%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]
    ordered += ["%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y"]
    for fmt in ordered:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return parsed, fmt
    try:
        parsed = pd.to_datetime(text, errors="coerce")
    except Exception:
        return None, None
    return (parsed, "inferred") if parsed is not None else (None, None)


# --- profiling --------------------------------------------------------------

def _norm(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _tokens(name):
    return {t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t}


def _name_weight(column, role):
    """0..1 weight for how strongly a column name suggests a semantic role."""
    compact = _norm(column)
    if not compact:
        return 0.0
    synonyms = NAME_SYNONYMS[role]
    if compact in synonyms:
        return 1.0
    if _tokens(column) & set(synonyms):
        return 0.85
    if any(syn in compact for syn in synonyms if len(syn) >= 4):
        return 0.55
    return 0.0


def _boolean_like(series):
    values = series.dropna().unique()
    if len(values) == 0 or len(values) > 3:
        return False
    return all(str(v).strip().lower() in FALSY_TRUTHY for v in values)


def _features(series):
    n = int(len(series))
    if n == 0:
        return dict(rows=0, missing_frac=1.0, nunique=0, uniq_ratio=0.0, is_numeric=False,
                    numeric_frac=0.0, nonneg=0.0, integer_like=False, looks_date=False,
                    date_frac=0.0, plausible_date=False, boolean_like=False, date_format=None)
    num = pd.to_numeric(series, errors="coerce")
    numeric_frac = float(num.notna().mean())
    is_numeric = numeric_frac >= 0.9
    nonneg = float((num.dropna() >= 0).mean()) if num.notna().any() else 0.0
    integer_like = bool(is_numeric and ((num.dropna() % 1 == 0).mean() >= 0.99))
    dt, date_format = _parse_dates(series)
    if dt is not None:
        parsed = dt.notna()
        date_frac = float(parsed.mean())
        years = dt.dt.year.dropna()
        plausible_date = bool(len(years) and years.between(1990, 2100).all())
    else:
        date_frac, plausible_date, date_format = 0.0, False, None
    nunique = int(series.nunique(dropna=True))
    return dict(
        rows=n, missing_frac=float(series.isna().mean()), nunique=nunique,
        uniq_ratio=nunique / max(1, n), is_numeric=is_numeric, numeric_frac=numeric_frac,
        nonneg=nonneg, integer_like=integer_like,
        looks_date=bool(date_frac >= 0.9 and plausible_date and nunique > 1),
        date_frac=date_frac, plausible_date=plausible_date, boolean_like=_boolean_like(series),
        date_format=date_format,
    )


def _is_id_like(f):
    return f["uniq_ratio"] > 0.98 and f["integer_like"]


def _score(role, f, name_weight):
    score = 0.0
    reasons = []
    if name_weight > 0:
        score += 0.5 * name_weight
        reasons.append(f"name suggests {role}")

    if role == "date":
        if f["looks_date"]:
            score += 0.6
            reasons.append(f"{f['date_frac']:.0%} of sampled values parse as dates")
        elif f["date_frac"] > 0.5 and f["plausible_date"]:
            score += 0.2
            reasons.append(f"{f['date_frac']:.0%} of values parse as dates")
        if f["is_numeric"]:
            score -= 0.25
    elif role == "sales":
        if f["is_numeric"]:
            score += 0.25
            if f["nonneg"] >= 0.95:
                score += 0.15
                reasons.append("numeric and non-negative")
            if f["nunique"] > 3:
                score += 0.1
                reasons.append("varies across rows")
        if _is_id_like(f):
            score -= 0.3
    elif role == "product":
        if 2 <= f["nunique"] and f["uniq_ratio"] < 0.9:
            score += 0.3
            reasons.append(f"{f['nunique']} distinct values repeated across rows")
        if f["looks_date"]:
            score -= 0.4
    elif role == "store":
        if 2 <= f["nunique"] <= max(2, 0.3 * f["rows"]):
            score += 0.25
            reasons.append(f"{f['nunique']} grouping values (store-like)")
        if f["looks_date"]:
            score -= 0.4
    elif role == "category":
        if 2 <= f["nunique"] <= 200 and f["uniq_ratio"] < 0.5:
            score += 0.2
            reasons.append("low-cardinality label column")
    elif role == "price":
        if f["is_numeric"] and f["nonneg"] > 0.9:
            score += 0.3
            reasons.append("positive numeric values")
            if not f["integer_like"]:
                score += 0.1
    elif role == "discount":
        if f["is_numeric"] and f["nonneg"] > 0.9:
            score += 0.3
            reasons.append("numeric discount-like values")
    elif role == "promo":
        if f["boolean_like"]:
            score += 0.3
            reasons.append("binary/flag values")
    elif role == "temp":
        if f["is_numeric"] and f["nonneg"]:
            score += 0.25
            reasons.append("numeric weather-like column")
    elif role == "holiday_lift":
        if f["boolean_like"] or (f["is_numeric"] and f["nunique"] <= 5):
            score += 0.3
            reasons.append("small set of holiday/flag values")
    elif role == "stockout":
        if f["boolean_like"]:
            score += 0.35
            reasons.append("binary flag values")
    return max(0.0, min(0.99, score)), reasons


def _score_frame(df):
    """Return {role: [(column, confidence, reasons), ...]} sorted best-first."""
    features = {col: _features(df[col]) for col in df.columns}
    scored = {}
    for role in ROLES:
        candidates = []
        for col in df.columns:
            conf, reasons = _score(role, features[col], _name_weight(col, role))
            if conf > 0.05:
                candidates.append((str(col), round(conf, 3), reasons))
        candidates.sort(key=lambda item: item[1], reverse=True)
        scored[role] = candidates
    return scored, features


def _assign_core(scored):
    """Greedy exclusive assignment of columns to roles (no entity fallback)."""
    used = set()
    mapping = {}
    for role in ASSIGN_ORDER:
        for column, confidence, _ in scored[role]:
            if column in used:
                continue
            if confidence >= THRESHOLD[role]:
                mapping[role] = column
                used.add(column)
            break
    return mapping


def _finalize_long(mapping):
    """Let a grouping column act as the entity when no product column exists."""
    if "product" not in mapping:
        for fallback in ("category", "store"):
            if fallback in mapping:
                mapping["product"] = mapping[fallback]
                if fallback == "store":
                    mapping.pop("store", None)  # avoid keying as "store|store"
                break
    if "category" not in mapping and "product" in mapping:
        if _name_weight(mapping["product"], "category") > 0:
            mapping["category"] = mapping["product"]
    return mapping


def _detect_layout(df, mapping, features):
    """Return (layout, value_columns, id_columns). Long unless a matrix is found.

    A wide file has no dedicated sales column: its entities are the column
    headers. A numeric "product" role therefore does not prove a long layout
    (it is itself one of the sibling value columns); only a non-numeric entity
    column or an explicit sales column confirms long.
    """
    id_columns = [mapping["store"]] if mapping.get("store") else []
    if mapping.get("sales"):
        return "long", [], id_columns
    product_col = mapping.get("product")
    if product_col and not features[product_col]["is_numeric"]:
        return "long", [], id_columns
    date_col = mapping.get("date")
    if not date_col:
        return "long", [], id_columns
    reserved = {mapping.get(r) for r in ("date", "price", "discount", "promo", "temp",
                                         "holiday_lift", "stockout", "store", "category")}
    value_columns = []
    for col in df.columns:
        f = features[col]
        if col == date_col or col in reserved or f["looks_date"]:
            continue
        if _is_id_like(f) or not f["is_numeric"]:
            continue
        value_columns.append(str(col))
    if len(value_columns) >= 2:
        return "wide", value_columns, id_columns
    return "long", [], id_columns


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    return value


def _preview(df, limit=5):
    rows = []
    for _, row in df.head(limit).iterrows():
        rows.append({str(k): _jsonable(v) for k, v in row.items()})
    return rows


def profile(path, sample_rows=SAMPLE_ROWS):
    """Profile a CSV file and return guidance for mapping it to the schema."""
    df = pd.read_csv(path, nrows=sample_rows)
    df.columns = [str(c).strip() for c in df.columns]
    scored, features = _score_frame(df)

    core = _assign_core(scored)
    layout, value_columns, id_columns = _detect_layout(df, core, features)
    if layout == "wide":
        mapping = {"layout": "wide", "date": core.get("date")}
        if core.get("store"):
            mapping["store"] = core["store"]
        mapping["value_columns"] = value_columns
        mapping["id_columns"] = id_columns
        missing = [] if mapping.get("date") and value_columns else (
            ["date"] if not mapping.get("date") else ["value_columns"])
    else:
        mapping = _finalize_long(core)
        missing = [role for role in REQUIRED_ROLES if role not in mapping]

    columns = []
    for col in df.columns:
        f = features[col]
        columns.append(dict(
            name=str(col), dtype=str(df[col].dtype), missing_fraction=round(f["missing_frac"], 3),
            unique=f["nunique"], numeric=bool(f["is_numeric"]), date_like=bool(f["looks_date"]),
            sample_values=[_jsonable(v) for v in df[col].dropna().unique()[:5]],
        ))

    roles = {}
    for role in ROLES:
        roles[role] = [dict(column=column, confidence=confidence, reasons=reasons)
                       for column, confidence, reasons in scored[role][:6]]

    confidence = {role: (scored[role][0][1] if scored[role] else 0.0) for role in ROLES}

    warnings = []
    if layout == "wide":
        warnings.append(f"Detected a WIDE layout: {len(value_columns)} value columns treated as "
                        "separate products/series. Confirm the value columns before preparing.")
    elif missing:
        warnings.append("Could not confidently detect required role(s): " + ", ".join(missing)
                        + ". Choose the correct column for each before preparing.")
    if layout == "long" and mapping.get("store") and mapping.get("product"):
        warnings.append("Both a store and an entity column were detected; values are keyed as "
                        "'store|entity' unless you filter to a single store.")
    if layout == "long" and mapping.get("sales") and mapping.get("price") and mapping["sales"] == mapping["price"]:
        warnings.append("The sales and price roles point at the same column; override one of them.")
    if layout == "long" and not any(mapping.get(r) for r in ("discount", "promo")):
        warnings.append("No promotion/discount column detected; promotion features will be zero.")
    warnings.append("Only the first few thousand rows are profiled; the full file is used when you prepare the dataset.")

    stores = []
    store_col = mapping.get("store") if layout == "wide" else mapping.get("store")
    if store_col:
        stores = sorted([str(v) for v in df[store_col].dropna().unique()])[:100]

    entities = []
    if layout == "long" and mapping.get("product"):
        entities = sorted([str(v) for v in df[mapping["product"]].dropna().unique()])[:200]

    date_min = date_max = None
    date_format = None
    if "date" in mapping:
        parsed, date_format = _parse_dates(df[mapping["date"]])
        if parsed is not None:
            clean = parsed.dropna()
            if len(clean):
                date_min, date_max = clean.min().strftime("%Y-%m-%d"), clean.max().strftime("%Y-%m-%d")

    return dict(
        rows_sampled=int(len(df)), layout=layout, columns=columns, roles=roles, mapping=mapping,
        wide_columns=value_columns, id_columns=id_columns,
        confidence={k: round(v, 3) for k, v in confidence.items()},
        required=dict(roles=REQUIRED_ROLES, satisfied=not missing, missing=missing),
        optional=dict(roles=OPTIONAL_ROLES, present=[r for r in OPTIONAL_ROLES if r in mapping]),
        stores=stores, entities=entities, date_min=date_min, date_max=date_max, date_format=date_format,
        preview=_preview(df), warnings=warnings,
    )


# --- normalization ----------------------------------------------------------

def _clean_mapping(mapping, columns):
    cleaned = {}
    for role, column in (mapping or {}).items():
        if role not in ROLES or column in (None, ""):
            continue
        col = str(column).lower().strip()
        if col in columns:
            cleaned[role] = col
    return cleaned


def _promotion(values):
    truthy = {"1", "true", "t", "yes", "y", "on"}
    text = values.astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(values, errors="coerce")
    return (text.isin(truthy) | numeric.eq(1)).astype(float)


def _map_long(df, mapping):
    """Project a frame onto the canonical columns (no aggregation, no drops)."""
    out = pd.DataFrame(index=df.index)
    parsed, _ = _parse_dates(df[mapping["date"]])
    out["date"] = parsed if parsed is not None else pd.to_datetime(df[mapping["date"]], errors="coerce")
    entity = df[mapping["product"]].astype("string").str.strip()
    if mapping.get("store"):
        entity = df[mapping["store"]].astype("string").str.strip() + "|" + entity
    out["product"] = entity.replace("", pd.NA)
    out["sales"] = pd.to_numeric(df[mapping["sales"]], errors="coerce")
    out["category"] = (df[mapping["category"]].astype("string").fillna("Unknown")
                       if mapping.get("category") else "Unknown")
    out["price"] = pd.to_numeric(df[mapping["price"]], errors="coerce") if mapping.get("price") else np.nan
    if mapping.get("discount"):
        discount = pd.to_numeric(df[mapping["discount"]], errors="coerce")
        if discount.notna().any() and (discount.dropna() > 1).mean() > 0.5:
            discount = discount / 100.0  # percent-style discount
        out["discount"] = discount
    elif mapping.get("promo"):
        out["discount"] = _promotion(df[mapping["promo"]])
    else:
        out["discount"] = 0.0
    out["temp"] = pd.to_numeric(df[mapping["temp"]], errors="coerce") if mapping.get("temp") else 0.0
    out["holiday_lift"] = (pd.to_numeric(df[mapping["holiday_lift"]], errors="coerce")
                           if mapping.get("holiday_lift") else 0.0)
    out["stockout"] = pd.to_numeric(df[mapping["stockout"]], errors="coerce") if mapping.get("stockout") else 0
    return out


def _finalize_mapped(mapped):
    """Drop rows missing required values and apply neutral defaults."""
    before = len(mapped)
    clean = mapped.dropna(subset=["date", "product", "sales"]).copy()
    clean["discount"] = clean["discount"].fillna(0.0).clip(lower=0.0)
    clean["temp"] = clean["temp"].fillna(0.0)
    clean["holiday_lift"] = clean["holiday_lift"].fillna(0.0)
    clean["stockout"] = clean["stockout"].fillna(0)
    return clean, int(before - len(clean))


def _normalize_long(df, mapping, report=None):
    """Map a long frame to the canonical schema and aggregate duplicates."""
    report = report if report is not None else {}
    df = df.rename(columns=lambda c: str(c).lower().strip())
    mapping = _clean_mapping(mapping, set(df.columns))
    for role in REQUIRED_ROLES:
        if role not in mapping:
            raise ValueError(f"Mapping must set '{role}' to a column present in the file.")

    report["rows_in"] = int(len(df))
    _, date_format = _parse_dates(df[mapping["date"]])
    report.setdefault("date_format", date_format)

    out, dropped = _finalize_mapped(_map_long(df, mapping))
    report["dropped_missing_required"] = dropped
    aggregated = out.groupby(["product", "date"], as_index=False).agg(
        sales=("sales", "sum"), discount=("discount", "max"), price=("price", "mean"),
        temp=("temp", "mean"), holiday_lift=("holiday_lift", "max"),
        stockout=("stockout", "max"), category=("category", "first"))
    aggregated["stockout"] = aggregated["stockout"].astype(int)
    report["duplicates_aggregated"] = int(len(out) - len(aggregated))
    report["rows_out"] = int(len(aggregated))
    return aggregated.sort_values(["product", "date"]).reset_index(drop=True)


def normalize(df, mapping, report=None):
    """Map an arbitrary long frame to the canonical schema and aggregate duplicates."""
    return _normalize_long(df, mapping, report)


def _headers(path):
    original = [str(c).strip() for c in pd.read_csv(path, nrows=0).columns]
    return original, {c.lower(): c for c in original}


class _Aggregator:
    """Accumulate canonical rows across chunks, one row per product+date."""

    def __init__(self):
        self._rows = {}

    def add(self, frame):
        if frame.empty:
            return
        for (product, day), group in frame.groupby(["product", "date"], sort=False):
            key = (product, day)
            sales = float(group["sales"].sum())
            discount = float(group["discount"].max())
            holiday = float(group["holiday_lift"].max())
            stockout = float(group["stockout"].max())
            category = group["category"].iloc[0]
            price_sum = float(group["price"].sum()) if group["price"].notna().any() else None
            price_n = int(group["price"].notna().sum())
            temp_sum = float(group["temp"].sum()) if group["temp"].notna().any() else None
            temp_n = int(group["temp"].notna().sum())
            current = self._rows.get(key)
            if current is None:
                self._rows[key] = dict(sales=sales, discount=discount, holiday=holiday, stockout=stockout,
                                       category=category, price_sum=price_sum, price_n=price_n,
                                       temp_sum=temp_sum, temp_n=temp_n)
            else:
                current["sales"] += sales
                current["discount"] = max(current["discount"], discount)
                current["holiday"] = max(current["holiday"], holiday)
                current["stockout"] = max(current["stockout"], stockout)
                if price_sum is not None:
                    current["price_sum"] = (current["price_sum"] or 0.0) + price_sum
                    current["price_n"] += price_n
                if temp_sum is not None:
                    current["temp_sum"] = (current["temp_sum"] or 0.0) + temp_sum
                    current["temp_n"] += temp_n

    def __len__(self):
        return len(self._rows)

    def frame(self):
        if not self._rows:
            return pd.DataFrame(columns=["date", "product", "sales", "discount", "price",
                                         "temp", "holiday_lift", "stockout", "category"])
        rows = []
        for (product, day), a in self._rows.items():
            rows.append(dict(
                date=day, product=product, sales=a["sales"], discount=a["discount"],
                price=(a["price_sum"] / a["price_n"]) if a["price_n"] else np.nan,
                temp=(a["temp_sum"] / a["temp_n"]) if a["temp_n"] else 0.0,
                holiday_lift=a["holiday"], stockout=int(a["stockout"]), category=a["category"]))
        frame = pd.DataFrame(rows)
        frame["stockout"] = frame["stockout"].astype(int)
        return frame.sort_values(["product", "date"]).reset_index(drop=True)


def _new_quality():
    return dict(rows_scanned=0, rows_selected=0, invalid_dates=0, invalid_sales=0,
                missing_sales=0, missing_product=0, negative_sales=0,
                dropped_missing_required=0, duplicates_aggregated=0, wide_cells_filled_zero=0)


def _scan_quality_long(chunk, mapping, quality):
    dates, _ = _parse_dates(chunk[mapping["date"]])
    quality["invalid_dates"] += int(dates.isna().sum()) if dates is not None else 0
    raw_sales = chunk[mapping["sales"]]
    numeric = pd.to_numeric(raw_sales, errors="coerce")
    quality["invalid_sales"] += int((raw_sales.notna() & ~np.isfinite(numeric)).sum())
    quality["missing_sales"] += int(numeric.isna().sum())
    quality["negative_sales"] += int((numeric < 0).sum())
    product = chunk[mapping["product"]].astype("string").str.strip()
    quality["missing_product"] += int((product.isna() | (product == "")).sum())
    return dates, product


def _long_chunk(chunk, mapping, store, products, start_date, end_date, quality):
    dates, product = _scan_quality_long(chunk, mapping, quality)
    mask = pd.Series(True, index=chunk.index)
    if dates is not None:
        mask &= dates.notna()
        if start_date:
            mask &= dates >= pd.Timestamp(start_date).normalize()
        if end_date:
            mask &= dates <= pd.Timestamp(end_date).normalize()
    if store is not None and mapping.get("store"):
        mask &= chunk[mapping["store"]].astype("string").str.strip() == str(store)
    if products:
        mask &= product.isin({str(p) for p in products})
    filtered = chunk[mask]
    quality["rows_selected"] += int(len(filtered))
    clean, dropped = _finalize_mapped(_map_long(filtered, mapping))
    quality["dropped_missing_required"] += dropped
    return clean


def _wide_chunk(chunk, date_col, value_columns, id_col, labels, store, start_date, end_date,
                wide_missing, quality):
    dates, _ = _parse_dates(chunk[date_col])
    quality["invalid_dates"] += int(dates.isna().sum()) if dates is not None else 0
    for column in value_columns:
        raw = chunk[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        quality["invalid_sales"] += int((raw.notna() & ~np.isfinite(numeric)).sum())
        quality["negative_sales"] += int((numeric < 0).sum())

    mask = pd.Series(True, index=chunk.index)
    if dates is not None:
        mask &= dates.notna()
        if start_date:
            mask &= dates >= pd.Timestamp(start_date).normalize()
        if end_date:
            mask &= dates <= pd.Timestamp(end_date).normalize()
    if store is not None and id_col:
        mask &= chunk[id_col].astype("string").str.strip() == str(store)
    frame = chunk[mask].copy()
    frame["__date"] = dates[mask] if dates is not None else pd.to_datetime(frame[date_col], errors="coerce")
    id_vars = ["__date"] + ([id_col] if id_col else [])
    long = frame.melt(id_vars=id_vars, value_vars=value_columns, var_name="product", value_name="sales")
    if labels:
        long["product"] = long["product"].map(lambda v: labels.get(v, v))  # restore original header case
    if id_col:
        long = long.rename(columns={id_col: "store"})
    long = long.rename(columns={"__date": "date"})

    long_mapping = {"date": "date", "product": "product", "sales": "sales"}
    if id_col:
        long_mapping["store"] = "store"
    mapped = _map_long(long, long_mapping)
    if wide_missing != "dropped":
        filled = int(mapped["sales"].isna().sum())
        if filled:
            mapped = mapped.copy()
            mapped["sales"] = mapped["sales"].fillna(0.0)
            quality["wide_cells_filled_zero"] += filled
    quality["rows_selected"] += int(len(mapped))
    clean, dropped = _finalize_mapped(mapped)
    quality["dropped_missing_required"] += dropped
    return clean


def inspect(path, mapping=None, chunk_size=SCAN_CHUNK_SIZE):
    """Chunked, mapping-aware quality scan of the whole file (no full load)."""
    _, labels = _headers(path)
    columns = set(labels)
    mapping = {k: (str(v).lower().strip() if isinstance(v, str) else v) for k, v in dict(mapping or {}).items()}
    is_wide = str(mapping.get("layout", "long")) == "wide" or bool(mapping.get("value_columns"))
    quality = _new_quality()
    skus, stores = set(), set()
    minimum = maximum = None

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        chunk.columns = [str(c).lower().strip() for c in chunk.columns]
        quality["rows_scanned"] += int(len(chunk))
        if is_wide:
            date_col = mapping.get("date")
            value_columns = [str(c).lower().strip() for c in mapping.get("value_columns", [])
                             if str(c).lower().strip() in columns]
            if date_col:
                dates, _ = _parse_dates(chunk[date_col])
                quality["invalid_dates"] += int(dates.isna().sum()) if dates is not None else 0
                if dates is not None and dates.notna().any():
                    lo, hi = dates.min(), dates.max()
                    minimum = lo if minimum is None or lo < minimum else minimum
                    maximum = hi if maximum is None or hi > maximum else maximum
            for column in value_columns:
                raw = chunk[column]
                numeric = pd.to_numeric(raw, errors="coerce")
                quality["invalid_sales"] += int((raw.notna() & ~np.isfinite(numeric)).sum())
                quality["missing_sales"] += int(numeric.isna().sum())
                quality["negative_sales"] += int((numeric < 0).sum())
            skus.update(labels.get(c, c) for c in value_columns)
            id_col = mapping.get("store")
            if id_col and id_col in columns:
                stores.update(chunk[id_col].dropna().astype(str).unique())
        else:
            if not all(mapping.get(r) in columns for r in REQUIRED_ROLES):
                raise ValueError("inspect needs date, product and sales mapped.")
            dates, product = _scan_quality_long(chunk, mapping, quality)
            if dates is not None and dates.notna().any():
                lo, hi = dates.min(), dates.max()
                minimum = lo if minimum is None or lo < minimum else minimum
                maximum = hi if maximum is None or hi > maximum else maximum
            skus.update(product.dropna().unique())
            if mapping.get("store"):
                stores.update(chunk[mapping["store"]].dropna().astype(str).unique())

    quality["stores"] = len(stores)
    quality["skus"] = len(skus)
    quality["date_min"] = minimum.strftime("%Y-%m-%d") if minimum is not None else None
    quality["date_max"] = maximum.strftime("%Y-%m-%d") if maximum is not None else None
    return quality


def prepare(path, mapping, store=None, products=None, start_date=None, end_date=None, wide_missing="zero"):
    """Stream a mapped CSV in chunks, filter, aggregate and normalize it.

    The raw file is never fully loaded and has no row cap; only the filtered,
    aggregated forecasting set is capped at ``MAX_FORECAST_ROWS``. Store, SKU and
    date filters are applied per chunk so a huge source is reduced *before* that
    cap. A quality report is computed during the same scan.
    """
    _, labels = _headers(path)
    columns = set(labels)
    mapping = dict(mapping or {})
    mapping = {k: (str(v).lower().strip() if isinstance(v, str) else v) for k, v in mapping.items()}
    report = {}
    quality = _new_quality()

    is_wide = str(mapping.get("layout", "long")) == "wide" or bool(mapping.get("value_columns"))
    if is_wide:
        date_col = mapping.get("date")
        if not date_col or date_col not in columns:
            raise ValueError("A wide layout needs a valid date column.")
        value_columns = [str(c).lower().strip() for c in mapping.get("value_columns", [])
                         if str(c).lower().strip() in columns]
        if products:
            wanted = {str(p) for p in products}
            value_columns = [c for c in value_columns if labels.get(c, c) in wanted or c in wanted]
        if not value_columns:
            raise ValueError("A wide layout needs at least one value column.")
        id_col = mapping.get("store")
        id_col = id_col if id_col in columns else None
    else:
        mapping = _clean_mapping(mapping, columns)
        for role in REQUIRED_ROLES:
            if role not in mapping:
                raise ValueError(f"Mapping must set '{role}' to a column present in the file.")
        id_col = value_columns = date_col = None

    sample = pd.read_csv(path, nrows=20, low_memory=False)
    sample.columns = [str(c).lower().strip() for c in sample.columns]
    _, report["date_format"] = _parse_dates(sample[date_col if is_wide else mapping["date"]])

    aggregator = _Aggregator()
    for chunk in pd.read_csv(path, chunksize=SCAN_CHUNK_SIZE, low_memory=False):
        chunk.columns = [str(c).lower().strip() for c in chunk.columns]
        quality["rows_scanned"] += int(len(chunk))
        if is_wide:
            clean = _wide_chunk(chunk, date_col, value_columns, id_col, labels, store,
                                start_date, end_date, wide_missing, quality)
        else:
            clean = _long_chunk(chunk, mapping, store, products, start_date, end_date, quality)
        aggregator.add(clean)
        if len(aggregator) > MAX_FORECAST_ROWS:
            raise ValueError(
                f"The filtered forecasting dataset exceeds the {MAX_FORECAST_ROWS:,}-row limit. "
                "Narrow the store, SKU or date filters and try again.")

    canonical = aggregator.frame()
    if canonical.empty:
        raise ValueError("No valid rows remain for the selected filters.")
    quality["duplicates_aggregated"] = max(
        0, quality["rows_selected"] - quality["dropped_missing_required"] - len(canonical))
    quality["skus"] = int(canonical["product"].nunique())
    keyed = mapping.get("store") or (is_wide and id_col)
    quality["stores"] = int(canonical["product"].str.split("|", n=1).str[0].nunique()) if keyed else 1
    quality["date_min"] = str(canonical["date"].min().date())
    quality["date_max"] = str(canonical["date"].max().date())

    report["rows_scanned"] = quality["rows_scanned"]
    report["rows_selected"] = quality["rows_selected"]
    report["duplicates_aggregated"] = quality["duplicates_aggregated"]
    report["dropped_missing_required"] = quality["dropped_missing_required"]
    if quality["wide_cells_filled_zero"]:
        report["wide_cells_filled_zero"] = quality["wide_cells_filled_zero"]

    errors, warnings, _ = data_upload.validate(canonical)
    if errors:
        raise ValueError(" ".join(errors))
    norm = data_upload.normalize(canonical)

    product_meta = []
    for pid, group in norm.groupby("product"):
        group = group.sort_values("date")
        price = float(group["price"].median()) if group["price"].notna().any() else None
        category = group["category"].iloc[-1] if group["category"].notna().any() else "Unknown"
        recent = float(group["sales"].iloc[-14:].mean())
        product_meta.append(dict(id=str(pid), category=str(category),
                                 price=round(price, 2) if price and price > 0 else None,
                                 rows=int(len(group)), suggested_stock=int(max(0, round(7 * recent)))))

    detected = dict(
        rows=int(len(norm)), skus=int(norm["product"].nunique()),
        date_min=str(norm["date"].min().date()), date_max=str(norm["date"].max().date()),
        products=product_meta, columns=list(CANONICAL), warnings=warnings,
        layout="wide" if is_wide else "long", report=report, quality=quality,
        optional={role: (role in mapping or role in _clean_mapping(mapping, columns))
                  for role in OPTIONAL_ROLES},
    )
    preview = norm.head(5).assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
    detected["preview"] = preview
    normalized = norm[CANONICAL].to_csv(index=False).encode()
    return normalized, detected
