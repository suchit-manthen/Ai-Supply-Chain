"""Shared helpers for adapters that stage a raw upload on disk.

Adapters keep the original uploaded file (streamed) plus a small JSON metadata
sidecar, then normalize a bounded subset into the pipeline format. Kept separate
so new adapters do not re-implement the same path/save/load logic.
"""
from __future__ import annotations

import json
import os
import time
import uuid

from config import UPLOAD_DIR

MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def _valid(dataset_id):
    return bool(dataset_id) and bool(str(dataset_id).replace("-", "").isalnum())


def paths(dataset_id, suffix):
    if not _valid(dataset_id):
        raise ValueError("Invalid dataset id.")
    root = os.path.join(UPLOAD_DIR, f"{dataset_id}-{suffix}")
    return root + ".csv", root + ".json"


def raw_path(dataset_id, suffix):
    return paths(dataset_id, suffix)[0]


def exists(dataset_id, suffix):
    try:
        return os.path.isfile(raw_path(dataset_id, suffix))
    except ValueError:
        return False


def discard(dataset_id, suffix):
    for candidate in paths(dataset_id, suffix):
        if os.path.isfile(candidate):
            os.remove(candidate)


def save_upload(stream, suffix):
    """Save an uploaded stream incrementally and return (id, path, byte_count)."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dataset_id = time.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    path, _ = paths(dataset_id, suffix)
    total = 0
    try:
        with open(path, "wb") as out:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_FILE_BYTES:
                    raise ValueError("CSV exceeds the 2 GB upload limit.")
                out.write(block)
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        raise
    if not total:
        os.remove(path)
        raise ValueError("The uploaded CSV is empty.")
    return dataset_id, path, total


def save_metadata(dataset_id, suffix, metadata):
    _, meta_path = paths(dataset_id, suffix)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)


def load_metadata(dataset_id, suffix):
    _, meta_path = paths(dataset_id, suffix)
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)
