import io

import pandas as pd
import pytest

from data_adapters import generic, storage


def _frame(entities=("A", "B"), days=140, start="2024-01-01", **extra):
    rows = []
    base = pd.Timestamp(start)
    for e in entities:
        for i in range(days):
            row = dict(date=base + pd.Timedelta(days=i), entity=e, sales=10 + (i % 5) + (0 if e == "A" else 3))
            row.update(extra)
            rows.append(row)
    return pd.DataFrame(rows)


def _write(path, df):
    df.to_csv(path, index=False)
    return path


def test_profile_detects_renamed_schema(tmp_path):
    df = _frame()
    df = df.rename(columns={"date": "ds", "entity": "sku", "sales": "qty"})
    df["store_id"] = 1
    df["unit_price"] = 2.5
    df["markdown"] = [0.1 if i % 7 == 0 else 0 for i in range(len(df))]
    df["onpromotion"] = df["markdown"] > 0
    df["temperature"] = 20.0
    prof = generic.profile(_write(tmp_path / "a.csv", df))
    m = prof["mapping"]
    assert m["date"] == "ds"
    assert m["sales"] == "qty"
    assert m["product"] == "sku"
    assert m["store"] == "store_id"
    assert m["price"] == "unit_price"
    assert m["discount"] == "markdown"
    assert m["promo"] == "onpromotion"
    assert m["temp"] == "temperature"
    assert prof["required"]["satisfied"] is True


def test_profile_minimal_required_only(tmp_path):
    prof = generic.profile(_write(tmp_path / "b.csv", _frame()))
    assert prof["required"]["satisfied"] is True
    assert prof["mapping"]["date"] == "date"
    assert prof["mapping"]["product"] == "entity"
    assert prof["mapping"]["sales"] == "sales"
    assert "price" not in prof["mapping"]
    assert "discount" not in prof["mapping"]


def test_profile_reports_missing_required_when_no_sales(tmp_path):
    df = _frame().drop(columns=["sales"])
    prof = generic.profile(_write(tmp_path / "c.csv", df))
    assert prof["required"]["satisfied"] is False
    assert prof["required"]["missing"] == ["sales"]


def test_family_column_falls_back_to_entity(tmp_path):
    rows = [dict(day=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), family="DAIRY", sales=5 + i % 3)
            for i in range(140)]
    prof = generic.profile(_write(tmp_path / "d.csv", pd.DataFrame(rows)))
    assert prof["required"]["satisfied"] is True
    assert prof["mapping"]["product"] == "family"
    assert prof["mapping"]["category"] == "family"


def test_date_not_confused_with_integer_id(tmp_path):
    df = _frame()
    df.insert(0, "id", range(len(df)))
    prof = generic.profile(_write(tmp_path / "e.csv", df))
    assert prof["mapping"]["date"] == "date"
    assert prof["mapping"]["sales"] == "sales"


def test_normalize_combines_store_and_aggregates_duplicates():
    df = pd.DataFrame([
        dict(date="2024-01-01", product="A", store="1", sales=2, promo=0),
        dict(date="2024-01-01", product="A", store="1", sales=3, promo=1),
        dict(date="2024-01-02", product="A", store="1", sales=4, promo=0),
    ])
    mapping = {"date": "date", "product": "product", "sales": "sales", "store": "store", "promo": "promo"}
    out = generic.normalize(df, mapping)
    assert set(out["product"]) == {"1|A"}
    first = out[out.date == pd.Timestamp("2024-01-01")].iloc[0]
    assert first.sales == 5        # duplicate rows summed
    assert first.discount == 1     # promotion combined with OR/max
    assert len(out) == 2


def test_normalize_percent_discount_scaled():
    df = pd.DataFrame([dict(date="2024-01-01", entity="A", sales=1, disc=20)])
    out = generic.normalize(df, {"date": "date", "product": "entity", "sales": "sales", "discount": "disc"})
    assert out.iloc[0].discount == pytest.approx(0.2)


def test_normalize_missing_required_raises():
    df = pd.DataFrame([dict(date="2024-01-01", entity="A", sales=1)])
    with pytest.raises(ValueError, match="sales"):
        generic.normalize(df, {"date": "date", "product": "entity"})


def test_prepare_end_to_end_canonical(tmp_path):
    df = _frame().rename(columns={"date": "ds", "entity": "sku", "sales": "qty"})
    path = _write(tmp_path / "f.csv", df)
    prof = generic.profile(path)
    normalized, detected = generic.prepare(path, prof["mapping"], start_date="2024-01-01", end_date="2024-05-19")
    out = pd.read_csv(io.BytesIO(normalized))
    assert list(out.columns) == ["date", "product", "sales", "discount", "price", "temp", "holiday_lift", "stockout", "category"]
    assert detected["skus"] == 2
    assert detected["rows"] == 280
    assert (out["price"] > 0).all()


def test_prepare_insufficient_history_raises(tmp_path):
    path = _write(tmp_path / "g.csv", _frame(days=50))
    prof = generic.profile(path)
    with pytest.raises(ValueError, match="at least"):
        generic.prepare(path, prof["mapping"])


def test_dates_day_first_and_yyyymmdd(tmp_path):
    dm = pd.DataFrame({"Ref Date": [f"{i % 28 + 1:02d}/{(i // 28) % 6 + 1:02d}/2024" for i in range(140)],
                       "SKU": "A", "Sold": [i % 4 + 1 for i in range(140)]})
    prof = generic.profile(_write(tmp_path / "dm.csv", dm))
    assert prof["date_format"] == "%d/%m/%Y"
    assert prof["mapping"]["date"] == "Ref Date"
    assert prof["mapping"]["sales"] == "Sold"
    _, detected = generic.prepare(_write(tmp_path / "dm.csv", dm), prof["mapping"])
    assert detected["date_min"] == "2024-01-01"

    dates = pd.date_range("2024-01-01", periods=140)
    ym = pd.DataFrame({"date": [int(d.strftime("%Y%m%d")) for d in dates], "sku": "B",
                       "qty": [i % 3 + 1 for i in range(140)]})
    prof2 = generic.profile(_write(tmp_path / "ym.csv", ym))
    assert prof2["date_format"] == "%Y%m%d"
    assert prof2["mapping"]["date"] == "date"
    assert prof2["required"]["satisfied"] is True


def test_wide_layout_detected_and_melted(tmp_path):
    dates = pd.date_range("2024-01-01", periods=140)
    wide = pd.DataFrame({"date": dates, "store": 7})
    for name in ["PRODUCT_A", "PRODUCT_B", "PRODUCT_C"]:
        wide[name] = [10 + i % 5 for i in range(140)]
    path = _write(tmp_path / "wide.csv", wide)
    prof = generic.profile(path)
    assert prof["layout"] == "wide"
    assert prof["mapping"]["date"] == "date"
    assert set(prof["wide_columns"]) == {"PRODUCT_A", "PRODUCT_B", "PRODUCT_C"}
    assert prof["required"]["satisfied"] is True

    normalized, detected = generic.prepare(path, prof["mapping"])
    out = pd.read_csv(io.BytesIO(normalized))
    assert detected["layout"] == "wide"
    assert detected["skus"] == 3
    assert detected["rows"] == 420
    assert set(out["product"]) == {"7|PRODUCT_A", "7|PRODUCT_B", "7|PRODUCT_C"}


def test_wide_missing_cells_zero_vs_dropped(tmp_path):
    dates = pd.date_range("2024-01-01", periods=140)
    wide = pd.DataFrame({"date": dates})
    for name in ["A", "B"]:
        wide[name] = [10 + i % 4 for i in range(140)]
    wide.loc[0, "B"] = None
    path = _write(tmp_path / "wide_missing.csv", wide)
    prof = generic.profile(path)

    _, filled = generic.prepare(path, prof["mapping"], wide_missing="zero")
    assert filled["report"]["wide_cells_filled_zero"] == 1
    assert filled["rows"] == 280

    _, dropped = generic.prepare(path, prof["mapping"], wide_missing="dropped")
    assert dropped["report"]["dropped_missing_required"] == 1
    assert dropped["rows"] == 279


def test_normalize_report_counts_duplicates_and_drops():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-01", "2024-01-02", None],
                       "product": ["A", "A", "A", "A"], "sales": [1, 2, 3, 9]})
    report = {}
    out = generic.normalize(df, {"date": "date", "product": "product", "sales": "sales"}, report=report)
    assert len(out) == 2
    assert out[out.date == pd.Timestamp("2024-01-01")].sales.iloc[0] == 3
    assert report["dropped_missing_required"] == 1
    assert report["duplicates_aggregated"] == 1
    assert report["rows_in"] == 4
    assert report["rows_out"] == 2


def test_large_file_streams_and_filters_before_cap(tmp_path):
    path = tmp_path / "large.csv"
    dates = pd.date_range("2024-01-01", periods=140)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("date,store,product,demand\n")
        for store in range(1, 501):
            for product in ("P1", "P2", "P3"):
                for d in dates:
                    f.write(f"{d:%Y-%m-%d},{store},{product},{10 + d.day % 5}\n")
    # 500 stores x 3 products x 140 days = 210,000 raw rows (> the 200k raw cap)
    prof = generic.profile(path)
    assert prof["mapping"]["sales"] == "demand"
    assert prof["mapping"]["store"] == "store"

    # Unfiltered, the forecasting set exceeds the cap and is rejected.
    with pytest.raises(ValueError, match="exceeds"):
        generic.prepare(path, prof["mapping"])

    # Filtering one store happens per chunk, before the cap, and succeeds.
    normalized, detected = generic.prepare(path, prof["mapping"], store="42")
    out = pd.read_csv(io.BytesIO(normalized))
    assert detected["rows"] == 420
    assert detected["skus"] == 3
    assert detected["report"]["rows_scanned"] == 210_000
    assert detected["quality"]["rows_scanned"] == 210_000
    assert (out["product"].str.startswith("42|")).all()


def test_inspect_quality_report(tmp_path):
    rows = []
    for d in pd.date_range("2024-01-01", periods=10):
        rows.append(dict(when=d.strftime("%Y-%m-%d"), sku="A", units=10))
    rows[0]["when"] = "not-a-date"   # invalid date
    rows[1]["units"] = None          # missing sales
    rows[2]["units"] = -3            # negative sales
    rows.append(dict(when="2024-01-03", sku="A", units=5))  # duplicate observation
    path = _write(tmp_path / "q.csv", pd.DataFrame(rows))
    prof = generic.profile(path)
    quality = generic.inspect(path, prof["mapping"])
    assert quality["rows_scanned"] == 11
    assert quality["invalid_dates"] == 1
    assert quality["missing_sales"] == 1
    assert quality["negative_sales"] == 1
    assert quality["skus"] == 1


def test_profile_exposes_entities(tmp_path):
    prof = generic.profile(_write(tmp_path / "e.csv", _frame()))
    assert set(prof["entities"]) == {"A", "B"}


def test_upload_route_canonical_file_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOAD_DIR", str(tmp_path / "uploads"))
    df = _frame().rename(columns={"entity": "product"})  # canonical date, product, sales
    path = _write(tmp_path / "canonical.csv", df)
    import app as api
    try:
        with path.open("rb") as stream:
            result = api.app.test_client().post(
                "/api/upload", data={"file": (stream, "canonical.csv")},
                content_type="multipart/form-data").get_json()
        assert result["adapter"] == "generic"
        assert result["profile"]["mapping"]["product"] == "product"
        assert result["profile"]["required"]["satisfied"] is True

        prepared = api.app.test_client().post(
            f"/api/generic/datasets/{result['dataset_id']}/prepare",
            json={"mapping": result["profile"]["mapping"]}).get_json()
        assert prepared["ok"] is True
        assert prepared["detected"]["skus"] == 2
    finally:
        for csv in (tmp_path / "uploads").glob("*-generic.csv"):
            generic.discard(csv.name.split("-generic.")[0])
        monkeypatch.undo()


def test_upload_route_prepares_wide_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOAD_DIR", str(tmp_path / "uploads"))
    dates = pd.date_range("2024-01-01", periods=140)
    wide = pd.DataFrame({"date": dates, "store": 3})
    for name in ["ITEM_X", "ITEM_Y"]:
        wide[name] = [8 + i % 4 for i in range(140)]
    path = _write(tmp_path / "wide.csv", wide)
    import app as api
    try:
        with path.open("rb") as stream:
            result = api.app.test_client().post(
                "/api/upload", data={"file": (stream, "wide.csv")},
                content_type="multipart/form-data").get_json()
        assert result["adapter"] == "generic"
        assert result["profile"]["layout"] == "wide"

        prepared = api.app.test_client().post(
            f"/api/generic/datasets/{result['dataset_id']}/prepare",
            json={"mapping": result["profile"]["mapping"], "wide_missing": "zero"}).get_json()
        assert prepared["ok"] is True
        assert prepared["detected"]["layout"] == "wide"
        assert prepared["detected"]["skus"] == 2
    finally:
        for csv in (tmp_path / "uploads").glob("*-generic.csv"):
            generic.discard(csv.name.split("-generic.")[0])
        monkeypatch.undo()


def test_upload_route_routes_to_generic(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOAD_DIR", str(tmp_path / "uploads"))
    df = _frame().rename(columns={"date": "ds", "entity": "sku", "sales": "qty"})
    path = _write(tmp_path / "h.csv", df)
    import app as api
    try:
        with path.open("rb") as stream:
            response = api.app.test_client().post(
                "/api/upload", data={"file": (stream, "sales.csv")},
                content_type="multipart/form-data")
        result = response.get_json()
        assert response.status_code == 200
        assert result["adapter"] == "generic"
        assert result["ok"] is True
        assert result["profile"]["required"]["satisfied"] is True

        prepared = api.app.test_client().post(
            f"/api/generic/datasets/{result['dataset_id']}/prepare",
            json={"mapping": result["profile"]["mapping"]})
        pdata = prepared.get_json()
        assert pdata["ok"] is True
        assert pdata["detected"]["skus"] == 2
    finally:
        for csv in (tmp_path / "uploads").glob("*-generic.csv"):
            generic.discard(csv.name.split("-generic.")[0])
        monkeypatch.undo()
