import db
import monitoring


def _save(path, run_id, created, skus):
    db.save_run(dict(run_id=run_id, created_at=created, duration_s=1.0, status="completed",
                     dataset={"rows": 1, "skus": len(skus)}, config={"horizon": 28, "n_folds": 3},
                     owner={}, analyst={}, skus=skus), path=path)


def _sku(sku_id, mape, model="Prophet", naive=10.0):
    return dict(sku_id=sku_id, selected_model=model, selection_metric="mape", selection_score=mape,
                baseline_score=naive, mape=mape, rmse=mape * 2, mae=mape * 1.5, naive_mape=naive)


def test_no_runs(tmp_path):
    m = monitoring.compute(path=str(tmp_path / "t.db"))
    assert m["has_history"] is False
    assert m["latest_run_id"] is None
    assert m["deteriorated"] == []


def test_single_run_has_no_history(tmp_path):
    p = str(tmp_path / "t.db")
    _save(p, "r1", "2026-01-01T00:00:00+00:00", [_sku("milk", 7.0), _sku("rice", 11.0)])
    m = monitoring.compute(path=p)
    assert m["has_history"] is False
    assert m["overall"]["avg_mape"] == 9.0
    assert m["overall"]["avg_naive"] == 10.0
    assert m["overall"]["improvement"] == 1.0
    assert all(s["prev_mape"] is None for s in m["skus"])
    assert m["deteriorated"] == []


def test_deterioration_detected_absolute(tmp_path):
    p = str(tmp_path / "t.db")
    _save(p, "r1", "2026-01-01T00:00:00+00:00", [_sku("milk", 5.0), _sku("rice", 10.0)])
    _save(p, "r2", "2026-01-02T00:00:00+00:00", [_sku("milk", 8.0), _sku("rice", 10.2)])
    m = monitoring.compute(path=p)
    assert m["has_history"] is True
    assert "milk" in m["deteriorated"]
    milk = next(s for s in m["skus"] if s["sku_id"] == "milk")
    assert milk["deteriorated"] is True
    assert milk["delta_mape"] == 3.0
    rice = next(s for s in m["skus"] if s["sku_id"] == "rice")
    assert rice["deteriorated"] is False


def test_deterioration_detected_relative(tmp_path):
    p = str(tmp_path / "t.db")
    _save(p, "r1", "2026-01-01T00:00:00+00:00", [_sku("milk", 10.0)])
    # +1.8 points is below the absolute threshold but +18% is above the relative threshold
    _save(p, "r2", "2026-01-02T00:00:00+00:00", [_sku("milk", 11.8)])
    m = monitoring.compute(path=p)
    assert m["deteriorated"] == ["milk"]


def test_improvement_is_not_deterioration(tmp_path):
    p = str(tmp_path / "t.db")
    _save(p, "r1", "2026-01-01T00:00:00+00:00", [_sku("milk", 9.0)])
    _save(p, "r2", "2026-01-02T00:00:00+00:00", [_sku("milk", 6.0)])
    m = monitoring.compute(path=p)
    assert m["deteriorated"] == []
    assert m["overall"]["delta_mape"] == -3.0


def test_model_change_is_flagged_but_not_deterioration(tmp_path):
    p = str(tmp_path / "t.db")
    _save(p, "r1", "2026-01-01T00:00:00+00:00", [_sku("milk", 7.0, model="ARIMA")])
    _save(p, "r2", "2026-01-02T00:00:00+00:00", [_sku("milk", 7.0, model="Prophet")])
    m = monitoring.compute(path=p)
    assert m["skus"][0]["model_changed"] is True
    assert m["deteriorated"] == []
