"""Flask API.

Serves the latest persisted training run from SQLite and accepts user-uploaded
supermarket CSVs. Training is a separate job (pipeline.py / train.py), started by
POST /api/train or POST /api/retrain; the API never trains on startup.
"""
import threading

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

import db
import data_upload
from data_adapters import favorita, generic, store_family
import monitoring
from config import SERVICE_LEVEL, UPLOAD_MIN_TRAIN_DAYS
from inventory import service_z
from pipeline import train
from views import owner_view, analyst_view  # noqa: F401  re-exported for backwards compatibility

app = Flask(__name__)
CORS(app)

STATE = dict(status="untrained", progress=0, data=None, error=None,
             run_id=None, trained_at=None, duration_s=None, data_source=None,
             last_train=dict(source=None, products=None, data_source="synthetic", policy=None))
LOCK = threading.Lock()
_LOADED = False


def _set_state(**kw):
    with LOCK:
        STATE.update(**kw)


def load_latest():
    """Load the most recent completed run from SQLite into STATE (once per process)."""
    global _LOADED
    with LOCK:
        if _LOADED:
            return STATE["status"]
        _LOADED = True
    db.init_db()
    run = db.latest_run()
    if run is None:
        _set_state(status="untrained")
        return "untrained"
    source = (run.get("config") or {}).get("source", "synthetic")
    _set_state(status="ready", data=dict(owner=run["owner"], analyst=run["analyst"]), progress=100,
               run_id=run["run_id"], trained_at=run["created_at"], duration_s=run["duration_s"],
               data_source=source, error=None)
    return "ready"


def _start_training(**opts):
    """Atomically move to 'training' and launch the pipeline in a worker thread."""
    global _LOADED
    with LOCK:
        if STATE["status"] == "training":
            return False
        STATE.update(status="training", progress=0, error=None, data=None,
                     data_source=opts.get("data_source"))
        STATE["last_train"] = {k: v for k, v in opts.items()}
        _LOADED = True

    def job():
        try:
            run = train(progress_cb=lambda pct, sku=None: _set_state(progress=pct), **opts)
            _set_state(status="ready", data=dict(owner=run["owner"], analyst=run["analyst"]), progress=100,
                       run_id=run["run_id"], trained_at=run["created_at"], duration_s=run["duration_s"],
                       data_source=run["config"].get("source"))
        except Exception as e:  # surfaced via /api/status
            _set_state(status="error", error=str(e))

    threading.Thread(target=job, daemon=True).start()
    return True


@app.get("/api/status")
def status():
    load_latest()
    with LOCK:
        return jsonify(status=STATE["status"], progress=STATE["progress"], error=STATE["error"],
                       run_id=STATE["run_id"], trained_at=STATE["trained_at"], duration_s=STATE["duration_s"],
                       data_source=STATE["data_source"])


def _payload(key):
    load_latest()
    with LOCK:
        if STATE["status"] == "ready" and STATE["data"] is not None:
            return STATE["data"][key], None
        return None, STATE["status"]


@app.get("/api/owner")
def owner():
    data, st = _payload("owner")
    return jsonify(data) if data is not None else (jsonify(status=st), 202)


@app.get("/api/analyst")
def analyst():
    data, st = _payload("analyst")
    return jsonify(data) if data is not None else (jsonify(status=st), 202)


@app.get("/api/sample-csv")
def sample_csv_route():
    """Downloadable template showing the expected CSV format."""
    return Response(data_upload.sample_csv(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=stocksense_sample.csv"})


@app.post("/api/upload")
def upload():
    """Validate + normalize an uploaded CSV; returns a preview and detected info."""
    f = request.files.get("file")
    if f is None:
        return jsonify(ok=False, errors=["No file was uploaded."], warnings=[], detected={}, optional={}, preview=[])
    # Favorita's raw transaction file may contain millions of rows. Route it
    # into the chunked adapter before the generic normalized-row validator.
    header = f.stream.readline(64 * 1024)
    f.stream.seek(0)
    if favorita.is_favorita_header(header):
        try:
            dataset_id, path, size = favorita.save_upload(f.stream)
            metadata = favorita.inspect(path)
            favorita.save_metadata(dataset_id, metadata)
        except Exception as e:
            if "dataset_id" in locals():
                favorita.discard_upload(dataset_id)
            return jsonify(ok=False, errors=[str(e)], warnings=[], detected={}, optional={}, preview=[])
        return jsonify(ok=True, adapter="favorita", dataset_id=dataset_id,
                       size_bytes=size, dataset=metadata)
    if store_family.is_store_family_header(header):
        try:
            dataset_id, path, size = store_family.save_upload(f.stream)
            metadata = store_family.inspect(path)
            store_family.save_metadata(dataset_id, metadata)
        except Exception as e:
            if "dataset_id" in locals():
                store_family.discard_upload(dataset_id)
            return jsonify(ok=False, errors=[str(e)], warnings=[], detected={}, optional={}, preview=[])
        return jsonify(ok=True, adapter="store_family", dataset_id=dataset_id,
                       size_bytes=size, dataset=metadata)
    # Anything that is not already canonical is profiled so the UI can map its
    # columns (date + entity + sales required; everything else optional).
    if not generic.canonical_required_present(header):
        try:
            dataset_id, path, size = generic.save_upload(f.stream)
            prof = generic.profile(path)
            generic.save_metadata(dataset_id, prof)
        except Exception as e:
            if "dataset_id" in locals():
                generic.discard(dataset_id)
            return jsonify(ok=False, errors=[str(e)], warnings=[], detected={}, optional={}, preview=[])
        return jsonify(ok=True, adapter="generic", dataset_id=dataset_id, size_bytes=size,
                       profile=prof, preview=prof.get("preview", []))
    result = data_upload.process(f.read())
    normalized = result.pop("normalized", None)
    if not result.get("ok") or normalized is None:
        return jsonify(result)
    dataset_id = data_upload.new_dataset_id()
    data_upload.save_dataset(dataset_id, normalized)
    data_upload.save_meta(dataset_id, dict(detected=result["detected"], optional=result["optional"], warnings=result["warnings"]))
    result["dataset_id"] = dataset_id
    return jsonify(result)


@app.post("/api/favorita/upload")
def favorita_upload():
    """Accept a large Favorita train.csv incrementally and return file metadata."""
    f = request.files.get("file")
    if f is None:
        return jsonify(ok=False, error="Choose a CSV file to upload."), 400
    try:
        dataset_id, path, size = favorita.save_upload(f.stream)
        metadata = favorita.inspect(path)
        favorita.save_metadata(dataset_id, metadata)
    except Exception as e:
        if "dataset_id" in locals():
            favorita.discard_upload(dataset_id)
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, dataset_id=dataset_id, size_bytes=size, dataset=metadata)


@app.get("/api/favorita/datasets/<dataset_id>")
def favorita_dataset(dataset_id):
    """Read saved file metadata, optionally returning item ids for one store."""
    if not favorita.exists(dataset_id):
        return jsonify(ok=False, error="Unknown dataset. Upload the CSV again."), 404
    metadata = favorita.load_metadata(dataset_id) or {}
    store = request.args.get("store_nbr")
    if store is not None:
        try:
            metadata["store_items"] = favorita.available_items(favorita.raw_path(dataset_id), store)
        except Exception as e:
            return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, dataset=metadata)


@app.post("/api/favorita/datasets/<dataset_id>/prepare")
def favorita_prepare(dataset_id):
    """Prepare a bounded, user-selected subset in the existing pipeline format."""
    if not favorita.exists(dataset_id):
        return jsonify(ok=False, error="Unknown dataset. Upload the CSV again."), 404
    body = request.get_json(silent=True) or {}
    try:
        normalized, detected = favorita.prepare(
            favorita.raw_path(dataset_id), body.get("store_nbr"), body.get("item_nbrs") or [],
            body.get("start_date"), body.get("end_date"))
        data_upload.save_dataset(dataset_id, normalized)
        data_upload.save_meta(dataset_id, dict(detected=detected, optional={}, warnings=detected["warnings"],
                                               adapter="favorita", store_nbr=detected["store_nbr"]))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, dataset_id=dataset_id, detected=detected)


@app.post("/api/store-family/upload")
def store_family_upload():
    """Accept a large store/family train.csv incrementally and return file metadata."""
    f = request.files.get("file")
    if f is None:
        return jsonify(ok=False, error="Choose a CSV file to upload."), 400
    try:
        dataset_id, path, size = store_family.save_upload(f.stream)
        metadata = store_family.inspect(path)
        store_family.save_metadata(dataset_id, metadata)
    except Exception as e:
        if "dataset_id" in locals():
            store_family.discard_upload(dataset_id)
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, dataset_id=dataset_id, size_bytes=size, dataset=metadata)


@app.get("/api/store-family/datasets/<dataset_id>")
def store_family_dataset(dataset_id):
    """Read saved file metadata, optionally returning families for one store."""
    if not store_family.exists(dataset_id):
        return jsonify(ok=False, error="Unknown dataset. Upload the CSV again."), 404
    metadata = store_family.load_metadata(dataset_id) or {}
    store = request.args.get("store_nbr")
    if store is not None:
        try:
            metadata["store_families"] = store_family.available_families(store_family.raw_path(dataset_id), store)
        except Exception as e:
            return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, dataset=metadata)


@app.post("/api/store-family/datasets/<dataset_id>/prepare")
def store_family_prepare(dataset_id):
    """Prepare a bounded, user-selected subset in the existing pipeline format."""
    if not store_family.exists(dataset_id):
        return jsonify(ok=False, error="Unknown dataset. Upload the CSV again."), 404
    body = request.get_json(silent=True) or {}
    try:
        normalized, detected = store_family.prepare(
            store_family.raw_path(dataset_id), body.get("store_nbr"), body.get("families") or [],
            body.get("start_date"), body.get("end_date"))
        data_upload.save_dataset(dataset_id, normalized)
        data_upload.save_meta(dataset_id, dict(detected=detected, optional={}, warnings=detected["warnings"],
                                               adapter="store_family", store_nbr=detected["store_nbr"]))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, dataset_id=dataset_id, detected=detected)


@app.get("/api/generic/datasets/<dataset_id>")
def generic_dataset(dataset_id):
    """Return the saved column profile for a generically uploaded CSV."""
    if not generic.exists(dataset_id):
        return jsonify(ok=False, error="Unknown dataset. Upload the CSV again."), 404
    profile = generic.load_metadata(dataset_id) or {}
    return jsonify(ok=True, profile=profile)


@app.post("/api/generic/datasets/<dataset_id>/prepare")
def generic_prepare(dataset_id):
    """Apply a user-confirmed column mapping and normalize the dataset."""
    if not generic.exists(dataset_id):
        return jsonify(ok=False, error="Unknown dataset. Upload the CSV again."), 404
    body = request.get_json(silent=True) or {}
    try:
        normalized, detected = generic.prepare(
            generic.raw_path(dataset_id), body.get("mapping") or {},
            store=body.get("store"), products=body.get("products") or None,
            start_date=body.get("start_date"), end_date=body.get("end_date"))
        data_upload.save_dataset(dataset_id, normalized)
        data_upload.save_meta(dataset_id, dict(detected=detected, optional=detected.get("optional", {}),
                                               warnings=detected["warnings"], adapter="generic"))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, dataset_id=dataset_id, detected=detected)


@app.post("/api/train")
def train_endpoint():
    """Start a training run for either the synthetic demo or an uploaded dataset."""
    body = request.get_json(silent=True) or {}
    src = str(body.get("source", "synthetic")).lower()
    if src == "uploaded":
        dataset_id = body.get("dataset_id")
        if not data_upload.exists(dataset_id):
            return jsonify(ok=False, error="Unknown or expired dataset. Please upload the CSV again."), 404
        meta = data_upload.load_meta(dataset_id) or {}
        if meta.get("adapter") == "favorita":
            inventory_input = body.get("inventory") or {}
            missing_cost = []
            for product in (meta.get("detected") or {}).get("products", []):
                raw_cost = (inventory_input.get(str(product.get("id"))) or {}).get("unit_cost")
                try:
                    valid_cost = float(raw_cost) > 0
                except (TypeError, ValueError):
                    valid_cost = False
                if not valid_cost:
                    missing_cost.append(product.get("id"))
            if missing_cost:
                return jsonify(ok=False, error="Enter a positive unit cost for each selected Favorita item before forecasting: "
                                               + ", ".join(missing_cost)), 400
            dataset_source = "favorita"
        elif meta.get("adapter") == "store_family":
            dataset_source = "store_family"
        elif meta.get("adapter") == "generic":
            dataset_source = "generic"
        else:
            dataset_source = "uploaded"
        products = data_upload.build_products(meta.get("detected", {}), body.get("inventory") or {})
        if not products:
            return jsonify(ok=False, error="No SKUs found for this dataset."), 400
        service_level = float(body.get("service_level", SERVICE_LEVEL))
        policy = dict(service_level=service_level, z=service_z(service_level))
        _start_training(source=data_upload.dataset_path(dataset_id), products=products,
                        data_source=dataset_source, policy=policy,
                        min_train=UPLOAD_MIN_TRAIN_DAYS, n_folds=None)
        return jsonify(ok=True, source="uploaded", skus=len(products))
    opts = dict(source=None, products=None, data_source="synthetic", policy=None)
    if body.get("seed") is not None:
        opts["seed"] = int(body["seed"])
    _start_training(**opts)
    return jsonify(ok=True, source="synthetic")


@app.post("/api/retrain")
def retrain():
    load_latest()
    with LOCK:
        if STATE["status"] == "training":
            return jsonify(ok=False, status="training")
        last = dict(STATE.get("last_train") or {})
    if not last:
        last = dict(source=None, products=None, data_source="synthetic", policy=None)
    _start_training(**last)
    return jsonify(ok=True)


@app.get("/api/runs")
def runs():
    """Training history (most recent first) with per-run aggregate metrics."""
    limit = request.args.get("limit", 20, type=int)
    with LOCK:
        latest = STATE["run_id"]
    return jsonify(latest_run_id=latest, runs=db.run_summaries(limit=limit))


@app.get("/api/runs/<run_id>")
def run_detail(run_id):
    meta = db.get_run_meta(run_id)
    if meta is None:
        return jsonify(error="run not found"), 404
    meta["skus"] = db.sku_results(run_id)
    return jsonify(meta)


@app.get("/api/monitoring")
def monitoring_view():
    """Current validation error vs naive baseline and drift vs the previous run."""
    return jsonify(monitoring.compute())


if __name__ == "__main__":
    load_latest()
    app.run(port=5000, debug=False)
