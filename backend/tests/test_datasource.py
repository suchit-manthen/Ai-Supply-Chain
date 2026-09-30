import numpy as np
import pandas as pd
import pytest

from config import HORIZON
import datasource


def test_load_synthetic_shape_and_stockout_column():
    df, end, hol = datasource.load_synthetic(42)
    assert df["product"].nunique() == 8
    assert "stockout" in df.columns
    assert (df["stockout"] == 0).all()
    assert df[df.date > end].groupby("product").size().eq(HORIZON).all()


def test_load_data_dispatch():
    df, end, _ = datasource.load_data(None)
    assert df["product"].nunique() == 8
    with pytest.raises(ValueError):
        datasource.load_data("nonsense.txt")


def test_validate_clean_history_is_ok():
    df, end, _ = datasource.load_synthetic(42)
    report = datasource.validate_sales(df[df.date <= end])
    assert report["ok"] is True
    assert report["summary"]["rows"] == 8 * 3 * 365
    assert report["summary"]["negative_sales"] == 0
    assert report["summary"]["missing_skus"] == []


def test_validate_detects_each_issue():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame({
        "date": [dates[0], dates[1], dates[1], dates[2], dates[3]],
        "product": ["milk", "milk", "milk", "ghost", pd.NA],
        "sales": [10.0, -3.0, 10.0, 5.0, 4.0],
        "price": [1.2, 1.2, 0.0, 1.2, 1.2],
    })
    report = datasource.validate_sales(df)
    types = {i["type"] for i in report["issues"]}
    assert "duplicate_record" in types
    assert "negative_sales" in types
    assert "invalid_price" in types
    assert "missing_sku" in types
    assert "unknown_sku" in types
    assert report["ok"] is False


def test_validate_detects_missing_sales_and_date_gaps():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
        "product": ["milk", "milk"],
        "sales": [10.0, np.nan],
        "price": [1.2, 1.2],
    })
    report = datasource.validate_sales(df)
    types = {i["type"] for i in report["issues"]}
    assert "missing_sales" in types
    assert "missing_dates" in types
    assert report["summary"]["missing_dates"] == 1  # 2024-01-02 gap


def _write_csv(path, with_optional=True):
    rows = []
    for i in range(10):
        d = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        row = dict(date=d.strftime("%Y-%m-%d"), product="milk", sales=100 + i)
        if with_optional:
            row.update(discount=0.0, price=1.2, temp=20.0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_csv_appends_future_window(tmp_path):
    p = tmp_path / "sales.csv"
    _write_csv(p)
    df, end, hol = datasource.load_csv(str(p))
    assert end == pd.Timestamp("2024-01-10")
    assert len(df[df.date <= end]) == 10
    fut = df[df.date > end]
    assert len(fut) == HORIZON
    assert fut["sales"].isna().all()
    for c in ["date", "product", "sales", "discount", "price", "temp", "holiday_lift", "stockout"]:
        assert c in df.columns
    assert (fut["price"] == 1.2).all()


def test_load_csv_fills_missing_optional_columns(tmp_path):
    p = tmp_path / "sales.csv"
    _write_csv(p, with_optional=False)
    df, end, _ = datasource.load_csv(str(p))
    assert (df["discount"] == 0.0).all()
    assert (df["temp"] == 0.0).all()
    assert (df[df.date <= end]["price"] == datasource.CATALOG["milk"]["price"]).all()


def test_load_csv_missing_required_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-01"], "product": ["milk"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        datasource.load_csv(str(p))
