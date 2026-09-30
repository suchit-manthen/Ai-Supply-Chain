import db


def _run(run_id, created, **kw):
    base = dict(
        run_id=run_id, created_at=created, duration_s=1.0, status="completed",
        dataset={"rows": 8760, "skus": 8}, config={"horizon": 28, "n_folds": 3},
        owner={"as_of": "2026-01-01", "items": []}, analyst={"skus": []},
        skus=[dict(sku_id="milk", selected_model="Prophet", selection_metric="mape",
                   selection_score=7.0, baseline_score=10.5, mape=7.0, rmse=20.0, mae=18.0, naive_mape=10.5)],
    )
    base.update(kw)
    return base


def test_latest_is_empty_before_any_run(tmp_path):
    p = str(tmp_path / "t.db")
    db.init_db(p)
    assert db.latest_run(p) is None


def test_save_and_latest_returns_most_recent_completed(tmp_path):
    p = str(tmp_path / "t.db")
    db.save_run(_run("r1", "2026-01-01T00:00:00+00:00"), path=p)
    db.save_run(_run("r2", "2026-01-02T00:00:00+00:00"), path=p)
    latest = db.latest_run(p)
    assert latest["run_id"] == "r2"
    assert latest["owner"] == {"as_of": "2026-01-01", "items": []}
    assert latest["analyst"] == {"skus": []}
    assert latest["dataset"]["skus"] == 8


def test_get_run_and_list_runs(tmp_path):
    p = str(tmp_path / "t.db")
    db.save_run(_run("r1", "2026-01-01T00:00:00+00:00"), path=p)
    run = db.get_run("r1", path=p)
    assert run["config"]["n_folds"] == 3
    assert db.get_run("missing", path=p) is None
    listing = db.list_runs(path=p)
    assert listing[0]["run_id"] == "r1"
    assert listing[0]["status"] == "completed"


def test_sku_results_roundtrip(tmp_path):
    p = str(tmp_path / "t.db")
    db.save_run(_run("r1", "2026-01-01T00:00:00+00:00"), path=p)
    rows = db.sku_results("r1", path=p)
    assert len(rows) == 1
    assert rows[0]["sku_id"] == "milk"
    assert rows[0]["selected_model"] == "Prophet"
    assert rows[0]["mape"] == 7.0


def test_latest_ignores_non_completed_runs(tmp_path):
    p = str(tmp_path / "t.db")
    db.save_run(_run("done", "2026-01-01T00:00:00+00:00"), path=p)
    db.save_run(_run("failed", "2026-01-02T00:00:00+00:00", status="failed", owner=None, analyst=None), path=p)
    assert db.latest_run(p)["run_id"] == "done"


def test_list_runs_limit(tmp_path):
    p = str(tmp_path / "t.db")
    for i in range(5):
        db.save_run(_run(f"r{i}", f"2026-01-0{i + 1}T00:00:00+00:00"), path=p)
    assert len(db.list_runs(limit=3, path=p)) == 3


def test_save_run_is_idempotent_for_same_run_id(tmp_path):
    p = str(tmp_path / "t.db")
    db.save_run(_run("r1", "2026-01-01T00:00:00+00:00", duration_s=1.0), path=p)
    db.save_run(_run("r1", "2026-01-01T00:00:00+00:00", duration_s=2.0), path=p)
    assert db.get_run("r1", path=p)["duration_s"] == 2.0
    assert len(db.list_runs(path=p)) == 1


def test_run_summaries_aggregate_metrics(tmp_path):
    p = str(tmp_path / "t.db")
    db.save_run(_run("r1", "2026-01-01T00:00:00+00:00"), path=p)
    db.save_run(_run("r2", "2026-01-02T00:00:00+00:00"), path=p)
    summaries = db.run_summaries(path=p)
    assert [s["run_id"] for s in summaries] == ["r2", "r1"]
    assert summaries[0]["avg_mape"] == 7.0
    assert summaries[0]["avg_naive"] == 10.5
    assert summaries[0]["n_skus"] == 1
    assert summaries[0]["horizon"] == 28


def test_get_run_meta_excludes_large_payloads(tmp_path):
    p = str(tmp_path / "t.db")
    db.save_run(_run("r1", "2026-01-01T00:00:00+00:00"), path=p)
    meta = db.get_run_meta("r1", path=p)
    assert meta["dataset"]["rows"] == 8760
    assert meta["config"]["horizon"] == 28
    assert "owner" not in meta and "analyst" not in meta
    assert db.get_run_meta("missing", path=p) is None
