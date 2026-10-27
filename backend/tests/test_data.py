import pandas as pd

from config import HORIZON, SEED
from data import PRODUCTS, generate

MODELS = {"ARIMA", "Prophet", "XGBoost", "Random Forest"}


def test_products_configuration():
    assert len(PRODUCTS) == 16
    assert len({p["id"] for p in PRODUCTS}) == 16
    # legacy_model is reference-only metadata
    assert {p["legacy_model"] for p in PRODUCTS} <= MODELS
    assert all("model" not in p for p in PRODUCTS)


def test_generate_is_deterministic_with_seed():
    a, end_a, _ = generate(SEED)
    b, end_b, _ = generate(SEED)
    assert end_a == end_b
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_data():
    a, _, _ = generate(SEED)
    b, _, _ = generate(SEED + 1)
    assert not a["sales"].equals(b["sales"])


def test_generate_shape_and_skus():
    df, _, _ = generate()
    total_days = 3 * 365 + HORIZON
    assert df["product"].nunique() == len(PRODUCTS)
    assert df.groupby("product").size().eq(total_days).all()
    assert len(df) == len(PRODUCTS) * total_days


def test_date_range_is_contiguous_daily():
    df, end, _ = generate()
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    assert dates.min() == end - pd.Timedelta(days=3 * 365 - 1)
    assert dates.max() == end + pd.Timedelta(days=HORIZON)
    assert dates.is_unique
    assert (dates.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_no_history_future_leakage():
    df, end, _ = generate()
    hist = df[df.date <= end]
    fut = df[df.date > end]
    assert hist.date.max() == end
    assert fut.date.min() == end + pd.Timedelta(days=1)
    assert hist.date.max() < fut.date.min()
    assert fut.groupby("product").size().eq(HORIZON).all()
    assert not set(hist.date).intersection(set(fut.date))
