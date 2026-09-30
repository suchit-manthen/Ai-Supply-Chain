"""Command-line entrypoint for the training pipeline.

Usage:
    python train.py                                  # synthetic data
    python train.py --source my_data.csv             # real observed-sales CSV
    python train.py --seed 7 --service-level 0.98    # overrides
"""
import argparse

from config import SERVICE_LEVEL
from inventory import service_z
import db
import pipeline


def main():
    ap = argparse.ArgumentParser(description="Run the StockSense training pipeline.")
    ap.add_argument("--source", default=None, help="Observed-sales CSV path (omit for synthetic data)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for synthetic data")
    ap.add_argument("--service-level", type=float, default=SERVICE_LEVEL, dest="service_level",
                    help="Target cycle service level for safety stock (default 0.95)")
    args = ap.parse_args()

    policy = dict(service_level=args.service_level, z=service_z(args.service_level))
    extra = {} if args.seed is None else dict(seed=args.seed)

    def progress(pct, sku=None):
        print(f"[{pct:3d}%] {sku or ''}", flush=True)

    try:
        run = pipeline.train(progress_cb=progress, source=args.source, policy=policy, **extra)
    except ValueError as e:
        print(f"ERROR: {e}")
        raise SystemExit(2)
    print()
    print(f"Saved run {run['run_id']} at {run['created_at']} ({run['duration_s']}s)")
    print(f"Data source: {run['config']['source']}  |  service level: {args.service_level}")
    print(f"{'SKU':<10}{'selected':<15}{'MAPE':>8}{'naive':>8}")
    for s in run["skus"]:
        print(f"{s['sku_id']:<10}{s['selected_model']:<15}{s['mape']:>8}{s['naive_mape']:>8}")
    latest = db.latest_run()
    print("Latest run in DB:", latest["run_id"] if latest else None)


if __name__ == "__main__":
    main()
