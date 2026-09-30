"""Lightweight SQLite persistence for training runs.

Each completed training run is stored as metadata plus the full owner/analyst
JSON payloads, together with a normalized per-SKU summary row. The API loads the
latest completed run; the training pipeline writes a new one. No ORM and no
external services - just the standard-library sqlite3 module.
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    duration_s   REAL,
    status       TEXT NOT NULL,
    dataset_json TEXT,
    config_json  TEXT,
    owner_json   TEXT,
    analyst_json TEXT
);
CREATE TABLE IF NOT EXISTS sku_results (
    run_id           TEXT NOT NULL,
    sku_id           TEXT NOT NULL,
    selected_model   TEXT,
    selection_metric TEXT,
    selection_score  REAL,
    baseline_score   REAL,
    mape             REAL,
    rmse             REAL,
    mae              REAL,
    naive_mape       REAL,
    PRIMARY KEY (run_id, sku_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
"""


def _prepare(path=None):
    """Open a connection and ensure the schema exists (idempotent)."""
    p = path or DB_PATH
    parent = os.path.dirname(p)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def init_db(path=None):
    """Create the schema if it does not exist yet."""
    conn = _prepare(path)
    conn.close()


def new_run_id(now=None):
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def save_run(run, path=None):
    """Persist a run dict: run_id, created_at, duration_s, status, dataset, config, owner, analyst, skus."""
    conn = _prepare(path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, created_at, duration_s, status, dataset_json,"
            " config_json, owner_json, analyst_json) VALUES (?,?,?,?,?,?,?,?)",
            (run["run_id"], run["created_at"], run.get("duration_s"), run.get("status", "completed"),
             json.dumps(run.get("dataset")), json.dumps(run.get("config")),
             json.dumps(run.get("owner")), json.dumps(run.get("analyst"))),
        )
        conn.execute("DELETE FROM sku_results WHERE run_id = ?", (run["run_id"],))
        conn.executemany(
            "INSERT INTO sku_results (run_id, sku_id, selected_model, selection_metric, selection_score,"
            " baseline_score, mape, rmse, mae, naive_mape) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(run["run_id"], s["sku_id"], s.get("selected_model"), s.get("selection_metric"),
              s.get("selection_score"), s.get("baseline_score"), s.get("mape"), s.get("rmse"),
              s.get("mae"), s.get("naive_mape")) for s in run.get("skus", [])],
        )
        conn.commit()
    finally:
        conn.close()


def _hydrate(row):
    return dict(
        run_id=row["run_id"], created_at=row["created_at"], duration_s=row["duration_s"], status=row["status"],
        dataset=json.loads(row["dataset_json"]) if row["dataset_json"] else None,
        config=json.loads(row["config_json"]) if row["config_json"] else None,
        owner=json.loads(row["owner_json"]) if row["owner_json"] else None,
        analyst=json.loads(row["analyst_json"]) if row["analyst_json"] else None,
    )


def latest_run(path=None):
    """Return the most recent completed run, or None if there is none."""
    conn = _prepare(path)
    try:
        row = conn.execute("SELECT * FROM runs WHERE status='completed' ORDER BY created_at DESC LIMIT 1").fetchone()
        return _hydrate(row) if row else None
    finally:
        conn.close()


def get_run(run_id, path=None):
    conn = _prepare(path)
    try:
        row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return _hydrate(row) if row else None
    finally:
        conn.close()


def list_runs(limit=20, path=None):
    """Most recent runs first (id, timestamp, duration, status)."""
    conn = _prepare(path)
    try:
        rows = conn.execute(
            "SELECT run_id, created_at, duration_s, status FROM runs ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def sku_results(run_id, path=None):
    """Normalized per-SKU summary rows for a run."""
    conn = _prepare(path)
    try:
        rows = conn.execute("SELECT * FROM sku_results WHERE run_id=? ORDER BY sku_id", (run_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_run_meta(run_id, path=None):
    """Run metadata without the large owner/analyst payloads."""
    conn = _prepare(path)
    try:
        row = conn.execute(
            "SELECT run_id, created_at, duration_s, status, dataset_json, config_json FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        return dict(
            run_id=row["run_id"], created_at=row["created_at"], duration_s=row["duration_s"], status=row["status"],
            dataset=json.loads(row["dataset_json"]) if row["dataset_json"] else None,
            config=json.loads(row["config_json"]) if row["config_json"] else None,
        )
    finally:
        conn.close()


def run_summaries(limit=20, path=None):
    """Per-run aggregate validation metrics (most recent first), for history views."""
    conn = _prepare(path)
    try:
        rows = conn.execute(
            "SELECT r.run_id, r.created_at, r.duration_s, r.status, r.dataset_json, r.config_json,"
            " AVG(s.mape) AS avg_mape, AVG(s.naive_mape) AS avg_naive, COUNT(s.sku_id) AS n_skus"
            " FROM runs r LEFT JOIN sku_results s ON s.run_id = r.run_id"
            " WHERE r.status='completed' GROUP BY r.run_id ORDER BY r.created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        out = []
        for row in rows:
            cfg = json.loads(row["config_json"]) if row["config_json"] else {}
            out.append(dict(
                run_id=row["run_id"], created_at=row["created_at"], duration_s=row["duration_s"], status=row["status"],
                dataset=json.loads(row["dataset_json"]) if row["dataset_json"] else None,
                horizon=cfg.get("horizon"), n_folds=cfg.get("n_folds"), n_skus=row["n_skus"],
                avg_mape=round(row["avg_mape"], 2) if row["avg_mape"] is not None else None,
                avg_naive=round(row["avg_naive"], 2) if row["avg_naive"] is not None else None,
            ))
        return out
    finally:
        conn.close()
