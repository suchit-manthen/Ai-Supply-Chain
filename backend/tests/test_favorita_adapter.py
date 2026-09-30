import io
import pandas as pd
import pytest

from data_adapters import favorita


def _favorita_file(path, *, stores=(1,), items=(100,), days=120, start="2017-01-01"):
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for store in stores:
        for item in items:
            for index, day in enumerate(dates):
                rows.append(dict(date=day, store_nbr=store, item_nbr=item,
                                 unit_sales=index % 7 + 1, onpromotion=(index % 2 == 0)))
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def test_inspect_valid_favorita_csv(tmp_path):
    path = tmp_path / "train.csv"
    _favorita_file(path, stores=(1, 2), items=(100, 101))
    info = favorita.inspect(path)
    assert info["rows"] == 480
    assert info["stores"] == 2
    assert info["skus"] == 2
    assert info["date_min"] == "2017-01-01"
    assert "onpromotion" in info["available_columns"]


def test_missing_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"date": "2017-01-01", "item_nbr": 4, "unit_sales": 1}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="store_nbr"):
        favorita.inspect(path)


def test_invalid_dates_and_sales_are_reported_not_zero_filled(tmp_path):
    path = tmp_path / "train.csv"
    frame = _favorita_file(path)
    frame["date"] = frame["date"].astype(str)
    frame["unit_sales"] = frame["unit_sales"].astype(object)
    frame.loc[0, "date"] = "not-a-date"
    frame.loc[1, "unit_sales"] = "not-a-number"
    frame.to_csv(path, index=False)
    info = favorita.inspect(path)
    assert info["quality"]["invalid_dates"] == 1
    assert info["quality"]["invalid_sales"] == 1
    normalized, prepared = favorita.prepare(path, 1, [100])
    result = pd.read_csv(io.BytesIO(normalized))
    assert prepared["quality"]["invalid_dates"] == 1
    assert prepared["quality"]["invalid_sales"] == 1
    assert len(result) == 118  # bad rows are excluded, not synthesized as zero-sales observations


def test_duplicate_rows_aggregate_sales_and_promotion(tmp_path):
    path = tmp_path / "train.csv"
    frame = _favorita_file(path)
    duplicate = frame.iloc[[10]].copy()
    duplicate["unit_sales"] = 5
    duplicate["onpromotion"] = True
    pd.concat([frame, duplicate], ignore_index=True).to_csv(path, index=False)
    normalized, info = favorita.prepare(path, 1, [100])
    result = pd.read_csv(io.BytesIO(normalized))
    row = result[result.date == "2017-01-11"].iloc[0]
    assert row.sales == 9  # original daily value (4) + duplicate (5)
    assert row.discount == 1
    assert info["duplicate_rows_aggregated"] == 1


def test_promotion_conversion_and_chronological_order(tmp_path):
    path = tmp_path / "train.csv"
    frame = _favorita_file(path)
    frame["onpromotion"] = ["True" if i % 2 else "False" for i in range(len(frame))]
    frame = frame.sample(frac=1, random_state=5)
    frame.to_csv(path, index=False)
    normalized, _ = favorita.prepare(path, 1, [100])
    result = pd.read_csv(io.BytesIO(normalized))
    assert result.date.is_monotonic_increasing
    assert set(result.discount.unique()) == {0, 1}


def test_store_and_item_filtering(tmp_path):
    path = tmp_path / "train.csv"
    _favorita_file(path, stores=(1, 2), items=(100, 200))
    normalized, info = favorita.prepare(path, 2, [200])
    result = pd.read_csv(io.BytesIO(normalized))
    assert set(result["product"].astype(str)) == {"200"}
    assert len(result) == 120
    assert info["store_nbr"] == "2"


def test_empty_after_filtering(tmp_path):
    path = tmp_path / "train.csv"
    _favorita_file(path)
    with pytest.raises(ValueError, match="No valid rows remain"):
        favorita.prepare(path, 99, [100])


def test_missing_calendar_dates_are_detected_and_not_filled(tmp_path):
    path = tmp_path / "train.csv"
    frame = _favorita_file(path)
    frame = frame[frame.date != pd.Timestamp("2017-01-10")]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing calendar dates"):
        favorita.prepare(path, 1, [100])


def test_raw_csv_above_200k_rows_can_be_uploaded_and_inspected(tmp_path):
    source = tmp_path / "favorita-large-raw.csv"
    with source.open("w", encoding="utf-8", newline="") as f:
        f.write("date,store_nbr,item_nbr,unit_sales,onpromotion\n")
        for _ in range(200_001):
            f.write("2017-01-01,1,100,1,False\n")
    import app as api
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(favorita, "UPLOAD_DIR", str(tmp_path / "uploads"))
    try:
        with source.open("rb") as stream:
            response = api.app.test_client().post(
                "/api/upload", data={"file": (stream, "train.csv")},
                content_type="multipart/form-data")
        result = response.get_json()
        assert response.status_code == 200
        assert result["ok"] is True
        assert result["adapter"] == "favorita"
        assert result["size_bytes"] > 0
        assert result["dataset"]["rows"] == 200_001
        assert result["dataset"]["stores"] == 1
        assert result["dataset"]["skus"] == 1
    finally:
        for dataset_id in [p.name.split("-favorita.")[0] for p in (tmp_path / "uploads").glob("*-favorita.csv")]:
            favorita.discard_upload(dataset_id)
        monkeypatch.undo()


def test_filtered_normalized_dataset_above_200k_rows_is_rejected(tmp_path):
    source = tmp_path / "favorita-filtered-too-large.csv"
    dates = pd.date_range("2020-01-01", periods=25_001, freq="D")
    with source.open("w", encoding="utf-8", newline="") as f:
        f.write("date,store_nbr,item_nbr,unit_sales,onpromotion\n")
        for day_index, day in enumerate(dates):
            for item in range(100, 108):
                f.write(f"{day:%Y-%m-%d},1,{item},1,{day_index % 2 == 0}\n")
    with pytest.raises(ValueError, match="filtered forecasting dataset exceeds the 200,000-row limit"):
        favorita.prepare(source, 1, list(range(100, 108)))
