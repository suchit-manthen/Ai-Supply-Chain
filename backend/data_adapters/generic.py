"""Generic CSV understanding: infer a schema for an arbitrary sales export.

Given any delimited sales file, this module profiles each column, scores how
well it fits each semantic role (date, sales/demand, product/entity, store and
known optional drivers), and returns a ranked mapping with confidence scores.
Minimum requirement is ``date + product/entity + sales``; every other role is
optional and left out of the mapping when it cannot be found.

``profile`` never mutates the source; ``normalize`` turns a frame plus a mapping
into the canonical ``date, product, sales, discount, price, temp, holiday_lift,
stockout`` shape used by the rest of the pipeline.
"""
from __future__ import annotations

import csv
import re

import numpy as np
import pandas as pd

from config import MAX_UPLOAD_ROWS, HORIZON
import data_upload
from data_adapters import storage

SUFFIX = "generic"
SAMPLE_ROWS = 5000
CANONICAL = ["date", "product", "sales", "discount", "price", "temp", "holiday_lift", "stockout"]

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
                 "rebate", "promodiscount", "discountamount"],
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


def _try_datetime(series):
    if pd.api.types.is_numeric_dtype(series):
        return None
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return None


def _boolean_like(series):
    values = series.dropna().unique()
    if len(values) == 0 or len(values) > 3:
        return False
    return all(str(v).strip().lower() in FALSY_TRUTHY for v in values)


def _features(series):
    name = series.name
    n = int(len(series))
    if n == 0:
        return dict(rows=0, missing_frac=1.0, nunique=0, uniq_ratio=0.0, is_numeric=False,
                    numeric_frac=0.0, nonneg=0.0, integer_like=False, looks_date=False,
                    date_frac=0.0, plausible_date=False, boolean_like=False)
    num = pd.to_numeric(series, errors="coerce")
    numeric_frac = float(num.notna().mean())
    is_numeric = numeric_frac >= 0.9
    nonneg = float((num.dropna() >= 0).mean()) if num.notna().any() else 0.0
    integer_like = bool(is_numeric and ((num.dropna() % 1 == 0).mean() >= 0.99))
    dt = _try_datetime(series)
    if dt is not None:
        parsed = dt.notna()
        date_frac = float(parsed.mean())
        years = dt.dt.year.dropna()
        plausible_date = bool(len(years) and years.between(1990, 2100).all())
    else:
        date_frac, plausible_date = 0.0, False
    nunique = int(series.nunique(dropna=True))
    return dict(
        rows=n, missing_frac=float(series.isna().mean()), nunique=nunique,
        uniq_ratio=nunique / max(1, n), is_numeric=is_numeric, numeric_frac=numeric_frac,
        nonneg=nonneg, integer_like=integer_like,
        looks_date=bool(date_frac >= 0.9 and plausible_date and nunique > 1),
        date_frac=date_frac, plausible_date=plausible_date, boolean_like=_boolean_like(series),
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
            score += 0.5
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


def _assign(scored):
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
    # A grouping column can legitimately be the entity when no dedicated product
    # column exists (e.g. a file aggregated only by "family" or only by store).
    if "product" not in mapping:
        for fallback in ("category", "store"):
            if fallback in mapping:
                mapping["product"] = mapping[fallback]
                if fallback == "store":
                    mapping.pop("store", None)  # avoid keying as "store|store"
                break
    # The entity label doubles as the category when no separate category exists.
    if "category" not in mapping and "product" in mapping:
        if _name_weight(mapping["product"], "category") > 0:
            mapping["category"] = mapping["product"]
    return mapping


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
    mapping = _assign(scored)
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
    if missing:
        warnings.append("Could not confidently detect required role(s): " + ", ".join(missing)
                        + ". Choose the correct column for each before preparing.")
    if mapping.get("store") and mapping.get("product"):
        warnings.append("Both a store and an entity column were detected; values are keyed as "
                        "'store|entity' unless you filter to a single store.")
    if "sales" in mapping and "price" in mapping and mapping["sales"] == mapping["price"]:
        warnings.append("The sales and price roles point at the same column; override one of them.")
    if not any(mapping.get(r) for r in ("discount", "promo")):
        warnings.append("No promotion/discount column detected; promotion features will be zero.")
    warnings.append("Only the first few thousand rows are profiled; the full file is used when you prepare the dataset.")

    stores = []
    if mapping.get("store"):
        stores = sorted([str(v) for v in df[mapping["store"]].dropna().unique()])[:100]

    date_min = date_max = None
    if "date" in mapping:
        parsed = pd.to_datetime(df[mapping["date"]], errors="coerce").dropna()
        if len(parsed):
            date_min, date_max = parsed.min().strftime("%Y-%m-%d"), parsed.max().strftime("%Y-%m-%d")

    return dict(
        rows_sampled=int(len(df)), columns=columns, roles=roles, mapping=mapping,
        confidence={k: round(v, 3) for k, v in confidence.items()},
        required=dict(roles=REQUIRED_ROLES, satisfied=not missing, missing=missing),
        optional=dict(roles=OPTIONAL_ROLES, present=[r for r in OPTIONAL_ROLES if r in mapping]),
        stores=stores, date_min=date_min, date_max=date_max, preview=_preview(df), warnings=warnings,
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


def normalize(df, mapping):
    """Map an arbitrary frame to the canonical schema and aggregate duplicates."""
    df = df.rename(columns=lambda c: str(c).lower().strip())
    mapping = _clean_mapping(mapping, set(df.columns))
    for role in REQUIRED_ROLES:
        if role not in mapping:
            raise ValueError(f"Mapping must set '{role}' to a column present in the file.")

    out = pd.DataFrame(index=df.index)
    out["date"] = pd.to_datetime(df[mapping["date"]], errors="coerce")
    entity = df[mapping["product"]].astype("string").str.strip()
    if mapping.get("store"):
        store = df[mapping["store"]].astype("string").str.strip()
        entity = store + "|" + entity
    out["product"] = entity
    out["sales"] = pd.to_numeric(df[mapping["sales"]], errors="coerce")

    if mapping.get("category"):
        out["category"] = df[mapping["category"]].astype("string").fillna("Unknown")
    else:
        out["category"] = "Unknown"
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
    if mapping.get("stockout"):
        out["stockout"] = pd.to_numeric(df[mapping["stockout"]], errors="coerce")
    else:
        out["stockout"] = 0

    out["product"] = out["product"].replace("", pd.NA)
    out = out.dropna(subset=["date", "product", "sales"])
    out["discount"] = out["discount"].fillna(0.0).clip(lower=0.0)
    out["temp"] = out["temp"].fillna(0.0)
    out["holiday_lift"] = out["holiday_lift"].fillna(0.0)
    out["stockout"] = out["stockout"].fillna(0)

    aggregated = out.groupby(["product", "date"], as_index=False).agg(
        sales=("sales", "sum"), discount=("discount", "max"), price=("price", "mean"),
        temp=("temp", "mean"), holiday_lift=("holiday_lift", "max"),
        stockout=("stockout", "max"), category=("category", "first"))
    aggregated["stockout"] = aggregated["stockout"].astype(int)
    return aggregated.sort_values(["product", "date"]).reset_index(drop=True)


def prepare(path, mapping, store=None, products=None, start_date=None, end_date=None):
    """Filter + normalize a mapped CSV into the pipeline format."""
    raw = pd.read_csv(path)
    if len(raw) > MAX_UPLOAD_ROWS:
        raise ValueError(f"The file has {len(raw)} rows, exceeding the limit of {MAX_UPLOAD_ROWS}.")
    raw = raw.rename(columns=lambda c: str(c).lower().strip())
    mapping = _clean_mapping(mapping, set(raw.columns))
    for role in REQUIRED_ROLES:
        if role not in mapping:
            raise ValueError(f"Mapping must set '{role}' to a column present in the file.")

    if start_date or end_date:
        dates = pd.to_datetime(raw[mapping["date"]], errors="coerce")
        mask = pd.Series(True, index=raw.index)
        if start_date:
            mask &= dates.isna() | (dates >= pd.Timestamp(start_date).normalize())
        if end_date:
            mask &= dates.isna() | (dates <= pd.Timestamp(end_date).normalize())
        raw = raw[mask]
    if store is not None and mapping.get("store"):
        raw = raw[raw[mapping["store"]].astype(str) == str(store)]
    if products:
        wanted = {str(p) for p in products}
        raw = raw[raw[mapping["product"]].astype(str).isin(wanted)]

    canonical = normalize(raw, mapping)
    errors, warnings, _ = data_upload.validate(canonical)
    if errors:
        raise ValueError(" ".join(errors))
    norm = data_upload.normalize(canonical)

    product_meta = []
    for pid, g in norm.groupby("product"):
        g = g.sort_values("date")
        price = float(g["price"].median()) if g["price"].notna().any() else None
        category = g["category"].iloc[-1] if g["category"].notna().any() else "Unknown"
        recent = float(g["sales"].iloc[-14:].mean())
        product_meta.append(dict(id=str(pid), category=str(category),
                                 price=round(price, 2) if price and price > 0 else None,
                                 rows=int(len(g)), suggested_stock=int(max(0, round(7 * recent)))))

    detected = dict(
        rows=int(len(norm)), skus=int(norm["product"].nunique()),
        date_min=str(norm["date"].min().date()), date_max=str(norm["date"].max().date()),
        products=product_meta, columns=list(CANONICAL), warnings=warnings,
        optional={role: role in mapping for role in OPTIONAL_ROLES},
    )
    preview = norm.head(5).assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
    detected["preview"] = preview
    normalized = norm[CANONICAL].to_csv(index=False).encode()
    return normalized, detected
