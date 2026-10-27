"""Build small, ready-to-upload test datasets for the StockSense upload wizard.

Run from the repo root:  python scripts/fetch_test_data.py

Outputs into data/samples/:
  store_family_sample.csv   real Favorita "Store Sales" file, trimmed to one
                            store and a few families (chunked-trimmed from a
                            partial HTTP Range download, so only a few MB).
  generic_long_sample.csv   the same data under arbitrary column names, to
                            exercise the generic mapping wizard.
  wide_sample.csv           a date x product matrix (wide layout), synthetic.
  favorita_item_sample.csv  item-level Favorita schema, synthetic.

The real file is optional: if the download fails, a synthetic store+family
sample is written instead so the script always succeeds offline.
"""
from __future__ import annotations

import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples")
MIN_DAYS = 150                      # comfortably above the 118-day minimum
DOWNLOAD_BYTES = 16_000_000         # ~first 200 days of the 121 MB Favorita file
HF_STORE_FAMILY = ("https://huggingface.co/datasets/"
                   "mrcksggcfc/store-sales-time-series-forecasting/resolve/main/train.csv")


def _range_download(url, nbytes):
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    cut = raw.rfind(b"\n")  # drop any truncated final line
    return raw[:cut] if cut > 0 else raw


def _real_store_family():
    data = _range_download(HF_STORE_FAMILY, DOWNLOAD_BYTES)
    df = pd.read_csv(io.BytesIO(data))
    df = df[df["store_nbr"] == 1]
    counts = df.groupby("family")["date"].nunique()
    spread = df.groupby("family")["sales"].std()
    stable = [f for f in counts.index if counts[f] == counts.max() and spread.get(f, 0) > 0]
    # Prefer the higher-volume families so the sample is realistic to forecast.
    families = sorted(stable, key=lambda f: -df[df["family"] == f]["sales"].mean())[:6]
    df = df[df["family"].isin(families)]
    dates = sorted(df["date"].unique())[:MIN_DAYS]
    df = df[df["date"].isin(dates)].sort_values(["family", "date"])
    return df[["date", "store_nbr", "family", "sales", "onpromotion"]]


def _synthetic_store_family(days=MIN_DAYS):
    dates = pd.date_range("2024-01-01", periods=days)
    rows = []
    rng = np.random.default_rng(0)
    for family, base in (("GROCERY I", 900), ("BEVERAGES", 600), ("PRODUCE", 700),
                         ("DAIRY", 500), ("MEATS", 300), ("BREAD/BAKERY", 250)):
        for i, d in enumerate(dates):
            promo = 1 if i % 9 == 0 else 0
            rows.append(dict(date=d.strftime("%Y-%m-%d"), store_nbr=1, family=family,
                             sales=round(base * (1 + 0.15 * np.sin(2 * np.pi * i / 365)) * (1.3 if promo else 1)
                                         + rng.normal(0, base * 0.05), 1),
                             onpromotion=promo))
    return pd.DataFrame(rows)[["date", "store_nbr", "family", "sales", "onpromotion"]]


def _generic_long(store_family):
    out = store_family.rename(columns={"date": "TransactionDate", "store_nbr": "Branch",
                                       "family": "Department", "sales": "UnitsSold",
                                       "onpromotion": "PromoFlag"})
    return out


def _wide(days=200):
    dates = pd.date_range("2024-01-01", periods=days)
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Store": 7})
    for i, product in enumerate(["Amul Milk 1L", "Basmati Rice 5kg", "Ice Cream Tub",
                                 "Soft Drink 2L", "Potato Chips 150g", "Sliced Bread"]):
        base = 120 + i * 40
        frame[product] = np.round(base * (1 + 0.2 * np.sin(2 * np.pi * np.arange(days) / 365))
                                  + rng.normal(0, base * 0.06, days), 1)
    frame.loc[5, "Basmati Rice 5kg"] = None  # a missing cell to exercise handling
    return frame


def _favorita_item(days=MIN_DAYS, store=5, items=(100, 101, 102)):
    dates = pd.date_range("2024-01-01", periods=days)
    rows = []
    rng = np.random.default_rng(2)
    for item in items:
        for i, d in enumerate(dates):
            promo = (i % 8 == 0)
            rows.append(dict(date=d.strftime("%Y-%m-%d"), store_nbr=store, item_nbr=item,
                             unit_sales=max(0, round(30 + (item - 100) * 5 + (i % 7) * 2
                                                     + (12 if promo else 0) + rng.normal(0, 3), 1)),
                             onpromotion=promo))
    return pd.DataFrame(rows)[["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        store_family = _real_store_family()
        source = "real (Hugging Face: mrcksggcfc/store-sales-time-series-forecasting)"
    except Exception as exc:  # offline or hub unavailable
        print(f"[warn] download failed ({exc}); using synthetic store+family data", file=sys.stderr)
        store_family = _synthetic_store_family()
        source = "synthetic fallback"

    outputs = {
        "store_family_sample.csv": store_family,
        "generic_long_sample.csv": _generic_long(store_family),
        "wide_sample.csv": _wide(),
        "favorita_item_sample.csv": _favorita_item(),
    }
    for name, frame in outputs.items():
        path = os.path.join(OUT_DIR, name)
        frame.to_csv(path, index=False)
        print(f"wrote {path}  ({len(frame):,} rows, cols: {', '.join(map(str, frame.columns))})")
    print(f"\nstore_family source: {source}")
    print("Upload these from the wizard: Upload CSV -> (auto-detect) -> understand -> map -> quality -> select -> inventory -> forecast")


if __name__ == "__main__":
    main()
