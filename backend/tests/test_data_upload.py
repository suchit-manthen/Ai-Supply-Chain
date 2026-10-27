import io

import pandas as pd
import pytest

import data_upload
import pipeline


def _frame(skus, days, start="2024-01-01"):
    rows = []
    base = pd.Timestamp(start)
    for pid in skus:
        for i in range(days):
            d = base + pd.Timedelta(days=i)
            sales = 100 + (i % 7) * 5 + (0 if pid == "a" else 40)
            rows.append(dict(date=d.strftime("%Y-%m-%d"), product=pid, category="Test",
                             sales=sales, price=2.0, discount=0.0, temp=20.0))
    return pd.DataFrame(rows)


def _csv_bytes(df):
    return df.to_csv(index=False).encode()


def test_sample_csv_format():
    s = data_upload.sample_csv()
    assert s.splitlines()[0] == "date,product,category,sales,price,discount,temp,holiday_lift"
    assert len(s.strip().splitlines()) > 5


def test_valid_csv_process():
    r = data_upload.process(_csv_bytes(_frame(["a", "b"], 120)))
    assert r["ok"] is True
    assert r["detected"]["rows"] == 240
    assert r["detected"]["skus"] == 2
    assert r["detected"]["date_min"] == "2024-01-01"
    assert len(r["preview"]) == 5
    assert r["optional"] == {"category": True, "price": True, "discount": True, "temp": True, "holiday_lift": False}
    assert r["normalized"] is not None


def test_missing_required_column():
    df = _frame(["a"], 120).drop(columns=["sales"])
    r = data_upload.process(_csv_bytes(df))
    assert r["ok"] is False
    assert any("sales" in e for e in r["errors"])


def test_invalid_dates():
    df = _frame(["a"], 120)
    df.loc[0, "date"] = "not-a-date"
    r = data_upload.process(_csv_bytes(df))
    assert r["ok"] is False
    assert any("date" in e.lower() for e in r["errors"])


def test_negative_sales():
    df = _frame(["a"], 120)
    df.loc[3, "sales"] = -5
    r = data_upload.process(_csv_bytes(df))
    assert r["ok"] is False
    assert any("negative" in e.lower() for e in r["errors"])


def test_duplicate_product_date():
    df = _frame(["a"], 120)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    r = data_upload.process(_csv_bytes(df))
    assert r["ok"] is False
    assert any("duplicate" in e.lower() for e in r["errors"])


def test_missing_values():
    df = _frame(["a"], 120)
    df.loc[5, "sales"] = None
    r = data_upload.process(_csv_bytes(df))
    assert r["ok"] is False
    assert any("missing" in e.lower() for e in r["errors"])


def test_multiple_skus_detected():
    r = data_upload.process(_csv_bytes(_frame([f"s{i}" for i in range(5)], 120)))
    assert r["ok"] is True
    assert r["detected"]["skus"] == 5


def test_missing_optional_columns_defaults():
    df = _frame(["a"], 120)[["date", "product", "sales"]]
    r = data_upload.process(_csv_bytes(df))
    assert r["ok"] is True
    assert r["optional"] == {"category": False, "price": False, "discount": False, "temp": False, "holiday_lift": False}
    norm = pd.read_csv(io.BytesIO(r["normalized"]))
    assert (norm["discount"] == 0).all()
    assert (norm["temp"] == 0).all()
    assert (norm["price"] > 0).all()  # neutral default, never 0


def test_insufficient_history():
    r = data_upload.process(_csv_bytes(_frame(["a"], 50)))
    assert r["ok"] is False
    assert any("at least" in e for e in r["errors"])


def test_constant_series():
    df = _frame(["a"], 120)
    df["sales"] = 7
    r = data_upload.process(_csv_bytes(df))
    assert r["ok"] is False
    assert any("constant" in e.lower() for e in r["errors"])


def test_too_many_skus():
    r = data_upload.process(_csv_bytes(_frame([f"s{i}" for i in range(30)], 120)))
    assert r["ok"] is False
    assert any("exceeds" in e for e in r["errors"])


def test_build_products_uses_inventory_inputs():
    r = data_upload.process(_csv_bytes(_frame(["a", "b"], 120)))
    prods = data_upload.build_products(r["detected"], {"a": {"stock": 500, "lead": 2, "service_level": 0.99}})
    a = next(p for p in prods if p["id"] == "a")
    b = next(p for p in prods if p["id"] == "b")
    meta_b = next(mp for mp in r["detected"]["products"] if mp["id"] == "b")
    assert a["stock"] == 500 and a["lead"] == 2 and a["service_level"] == 0.99
    assert b["stock"] == meta_b["suggested_stock"]  # falls back to the suggested demo assumption


def test_dataset_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(data_upload, "UPLOAD_DIR", str(tmp_path))
    did = data_upload.new_dataset_id()
    data_upload.save_dataset(did, b"a,b\n1,2\n")
    data_upload.save_meta(did, {"detected": {"rows": 1}})
    assert data_upload.exists(did)
    assert data_upload.load_meta(did)["detected"]["rows"] == 1
    assert not data_upload.exists("../../etc/passwd")


def test_invalid_dataset_id_rejected():
    with pytest.raises(ValueError):
        data_upload.save_dataset("../evil", b"x")


def test_end_to_end_uploaded_dataset(tmp_path):
    df = _frame(["a", "b"], 130)
    r = data_upload.process(_csv_bytes(df))
    assert r["ok"] is True
    src = tmp_path / "uploaded.csv"
    src.write_bytes(r["normalized"])
    products = data_upload.build_products(r["detected"], {})

    run = pipeline.train(source=str(src), products=products, data_source="uploaded",
                         min_train=90, n_folds=1, path=str(tmp_path / "t.db"))

    assert run["config"]["source"] == "uploaded"
    assert len(run["skus"]) == 2
    assert all(s["selected_model"] for s in run["skus"])
    assert run["owner"]["data_source"] == "uploaded"
    assert len(run["owner"]["items"]) == 2
    assert run["analyst"]["data_source"] == "uploaded"
    # price/temp are constant in this fixture, so only the holiday calendar is an
    # active exogenous driver - dead features must not be advertised or fed in.
    features = run["analyst"]["dataset"]["features"]
    assert "holiday lift" in features
    assert "temperature" not in features and "shelf price" not in features
    categories = run["analyst"]["dataset"]["categories"]
    assert categories and categories[0]["category"] == "Test"
    usage = run["analyst"]["skus"][0]["result"]["feature_usage"]
    assert usage["exog"] == ["holiday_lift"]  # only the varying driver is used
    for s in run["analyst"]["skus"]:
        assert len(s["result"]["forecast"]) == 28
        for pt in s["result"]["forecast"]:
            assert pt["lower_bound"] <= pt["pred"] <= pt["upper_bound"]
