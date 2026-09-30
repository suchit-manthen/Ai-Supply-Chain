import io
import pandas as pd
import pytest

from data_adapters import store_family


def _store_family_file(path, *, stores=(1,), families=("GROCERY I",), days=120, start="2013-01-01"):
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for store in stores:
        for family in families:
            for index, day in enumerate(dates):
                rows.append(dict(date=day, store_nbr=store, family=family,
                                 sales=index % 7 + 1, onpromotion=(index % 2 == 0)))
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def test_is_store_family_header_distinguishes_schemas():
    assert store_family.is_store_family_header(b"date,store_nbr,family,sales,onpromotion\n")
    assert not store_family.is_store_family_header(b"date,store_nbr,item_nbr,unit_sales,onpromotion\n")


def test_inspect_valid_store_family_csv(tmp_path):
    path = tmp_path / "train.csv"
    _store_family_file(path, stores=(1, 2), families=("GROCERY I", "BEVERAGES"))
    info = store_family.inspect(path)
    assert info["rows"] == 480
    assert info["stores"] == 2
    assert info["skus"] == 2
    assert info["date_min"] == "2013-01-01"
    assert "onpromotion" in info["available_columns"]


def test_missing_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"date": "2013-01-01", "store_nbr": 1, "sales": 1}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="family"):
        store_family.inspect(path)


def test_invalid_dates_and_sales_are_reported_not_zero_filled(tmp_path):
    path = tmp_path / "train.csv"
    frame = _store_family_file(path)
    frame["date"] = frame["date"].astype(str)
    frame["sales"] = frame["sales"].astype(object)
    frame.loc[0, "date"] = "not-a-date"
    frame.loc[1, "sales"] = "not-a-number"
    frame.to_csv(path, index=False)
    info = store_family.inspect(path)
    assert info["quality"]["invalid_dates"] == 1
    assert info["quality"]["invalid_sales"] == 1
    normalized, prepared = store_family.prepare(path, 1, ["GROCERY I"])
    result = pd.read_csv(io.BytesIO(normalized))
    assert prepared["quality"]["invalid_dates"] == 1
    assert prepared["quality"]["invalid_sales"] == 1
    assert len(result) == 118  # bad rows are excluded, not synthesized as zero-sales observations


def test_duplicate_rows_aggregate_sales_and_promotion(tmp_path):
    path = tmp_path / "train.csv"
    frame = _store_family_file(path)
    duplicate = frame.iloc[[10]].copy()
    duplicate["sales"] = 5
    duplicate["onpromotion"] = True
    pd.concat([frame, duplicate], ignore_index=True).to_csv(path, index=False)
    normalized, info = store_family.prepare(path, 1, ["GROCERY I"])
    result = pd.read_csv(io.BytesIO(normalized))
    row = result[result.date == "2013-01-11"].iloc[0]
    assert row.sales == 9  # original daily value (4) + duplicate (5)
    assert row.discount == 1
    assert info["duplicate_rows_aggregated"] == 1


def test_promotion_conversion_and_chronological_order(tmp_path):
    path = tmp_path / "train.csv"
    frame = _store_family_file(path)
    frame["onpromotion"] = ["True" if i % 2 else "False" for i in range(len(frame))]
    frame = frame.sample(frac=1, random_state=5)
    frame.to_csv(path, index=False)
    normalized, _ = store_family.prepare(path, 1, ["GROCERY I"])
    result = pd.read_csv(io.BytesIO(normalized))
    assert result.date.is_monotonic_increasing
    assert set(result.discount.unique()) == {0, 1}


def test_store_and_family_filtering(tmp_path):
    path = tmp_path / "train.csv"
    _store_family_file(path, stores=(1, 2), families=("GROCERY I", "BEVERAGES"))
    normalized, info = store_family.prepare(path, 2, ["BEVERAGES"])
    result = pd.read_csv(io.BytesIO(normalized))
    assert set(result["product"].astype(str)) == {"BEVERAGES"}
    assert len(result) == 120
    assert info["store_nbr"] == "2"


def test_empty_after_filtering(tmp_path):
    path = tmp_path / "train.csv"
    _store_family_file(path)
    with pytest.raises(ValueError, match="No valid rows remain"):
        store_family.prepare(path, 99, ["GROCERY I"])


def test_missing_calendar_dates_are_detected_and_not_filled(tmp_path):
    path = tmp_path / "train.csv"
    frame = _store_family_file(path)
    frame = frame[frame.date != pd.Timestamp("2013-01-10")]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing calendar dates"):
        store_family.prepare(path, 1, ["GROCERY I"])


def test_upload_route_detects_store_family_adapter(tmp_path):
    path = tmp_path / "train.csv"
    _store_family_file(path, stores=(1, 2), families=("GROCERY I", "BEVERAGES"))
    import app as api
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store_family, "UPLOAD_DIR", str(tmp_path / "uploads"))
    try:
        with path.open("rb") as stream:
            response = api.app.test_client().post(
                "/api/upload", data={"file": (stream, "train.csv")},
                content_type="multipart/form-data")
        result = response.get_json()
        assert response.status_code == 200
        assert result["ok"] is True
        assert result["adapter"] == "store_family"
        assert result["dataset"]["stores"] == 2
        assert result["dataset"]["skus"] == 2
    finally:
        for dataset_id in [p.name.split("-storefamily.")[0] for p in (tmp_path / "uploads").glob("*-storefamily.csv")]:
            store_family.discard_upload(dataset_id)
        monkeypatch.undo()
