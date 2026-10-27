import pandas as pd

from models import _features, available_exog, CAL_FEATS
from data import holiday_table
import datasource


def test_available_exog_only_uses_varying_columns():
    d = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "discount": 0.0,                       # all zero -> dropped
        "price": 2.5,                          # constant -> dropped
        "temp": [10 + i for i in range(30)],   # varies -> kept
        "holiday_lift": [0.0] * 29 + [1.0],    # varies -> kept
    })
    assert available_exog(d) == ["temp", "holiday_lift"]


def test_features_are_calendar_plus_active_exog():
    d = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=20),
        "discount": [0.1 if i % 5 == 0 else 0.0 for i in range(20)],
        "price": 1.0, "temp": 0.0, "holiday_lift": 0.0,
    })
    x = _features(d)
    assert list(x.columns) == CAL_FEATS + ["discount"]
    assert x["sin_doy"].between(-1, 1).all() and x["cos_doy"].between(-1, 1).all()
    assert "price" not in x.columns and "temp" not in x.columns  # no dead features


def test_is_holiday_matches_holiday_lift():
    d = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5),
                      "holiday_lift": [0.0, 0.2, 0.0, 1.0, 0.0]})
    x = _features(d)
    assert list(x["is_holiday"]) == [0, 1, 0, 1, 0]


def test_holiday_table_is_enriched():
    hol = holiday_table([2024])
    names = set(hol["holiday"])
    assert {"Black Friday", "Cyber Monday", "Thanksgiving", "Independence Day"} <= names
    assert len(hol) >= 16
    assert hol["ds"].dt.year.eq(2024).all()


def test_load_csv_preserves_category(tmp_path):
    rows = []
    for i in range(10):
        d = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        rows.append(dict(date=d.strftime("%Y-%m-%d"), product="milk", category="Dairy", sales=100 + i))
    path = tmp_path / "sales.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    df, end, _ = datasource.load_csv(str(path))
    assert "category" in df.columns
    assert (df[df.date <= end]["category"] == "Dairy").all()
    assert (df[df.date > end]["category"] == "Dairy").all()  # future window keeps it too


def test_load_synthetic_has_categories():
    df, _, _ = datasource.load_synthetic(42)
    assert "category" in df.columns
    assert set(df[df["product"] == "milk"]["category"]) == {"Dairy"}
