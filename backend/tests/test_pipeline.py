import pandas as pd
import pytest

from datasource import catalog_ids
import pipeline


def test_pipeline_rejects_invalid_csv(tmp_path):
    # all catalog SKUs present (so the missing-SKU guard passes) + a duplicate row
    rows = [dict(date="2024-01-01", product=p, sales=10.0, price=1.2) for p in catalog_ids()]
    rows.append(dict(date="2024-01-01", product="milk", sales=10.0, price=1.2))
    csv = tmp_path / "bad.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    with pytest.raises(ValueError):
        pipeline.train(source=str(csv), path=str(tmp_path / "t.db"))


def test_pipeline_rejects_data_missing_skus(tmp_path):
    csv = tmp_path / "partial.csv"
    pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D").strftime("%Y-%m-%d"),
        "product": ["milk"] * 5,
        "sales": [10.0, 11, 12, 13, 14],
        "price": [1.2] * 5,
    }).to_csv(csv, index=False)

    with pytest.raises(ValueError):
        pipeline.train(source=str(csv), path=str(tmp_path / "t.db"))
