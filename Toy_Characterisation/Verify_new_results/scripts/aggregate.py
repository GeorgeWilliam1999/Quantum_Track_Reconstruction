#!/usr/bin/env python3
"""
Aggregate verification sweep results into CSV files.

Reads per-job results.json files and produces:
  - aggregated/verify_scatt.csv   (one row per job = per sweep point)
  - aggregated/verify_res.csv
  - aggregated/verify_scatt_per_event.csv  (one row per event)
  - aggregated/verify_res_per_event.csv
  - aggregated/run_status.json

Usage
-----
    python aggregate.py --results-dir /path/to/results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)


def aggregate(results_dir):
    results_dir = Path(results_dir)
    agg_dir = results_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Discover all job directories
    job_dirs = sorted(results_dir.glob("job_*/"))
    print(f"Found {len(job_dirs)} job directories")

    # Categorise by task
    tasks = {}
    done_count = 0
    fail_count = 0
    missing_count = 0

    for jd in job_dirs:
        done_file = jd / "DONE"
        fail_file = jd / "FAILED"
        results_file = jd / "results.json"

        if fail_file.exists():
            fail_count += 1
            continue
        if not done_file.exists():
            missing_count += 1
            continue
        if not results_file.exists():
            fail_count += 1
            continue

        done_count += 1
        data = load_json(results_file)
        task = data["task"]
        tasks.setdefault(task, []).append((jd, data))

    status = {
        "total_jobs": len(job_dirs),
        "completed": done_count,
        "failed": fail_count,
        "missing": missing_count,
        "tasks_found": {k: len(v) for k, v in tasks.items()},
    }
    save_json(status, agg_dir / "run_status.json")
    print(f"Status: {done_count} done, {fail_count} failed, {missing_count} pending")
    for t, jobs in tasks.items():
        print(f"  {t:20s}: {len(jobs)} jobs")

    # ─── Write CSVs ───────────────────────────────────────────────
    for task_name in ["verify_scatt", "verify_res"]:
        if task_name not in tasks:
            continue

        jobs = tasks[task_name]

        # Aggregated CSV (one row per job)
        agg_rows = []
        per_event_rows = []

        for jd, data in sorted(jobs, key=lambda j: j[1]["job_id"]):
            agg = data["aggregated"]
            agg_rows.append(agg)

            for ev in data["per_event"]:
                per_event_rows.append(ev)

        # Write aggregated
        _write_csv(agg_rows, agg_dir / f"{task_name}.csv")
        print(f"  {task_name}.csv: {len(agg_rows)} rows")

        # Write per-event
        _write_csv(per_event_rows, agg_dir / f"{task_name}_per_event.csv")
        print(f"  {task_name}_per_event.csv: {len(per_event_rows)} rows")

    print(f"\nAggregation complete. Results in: {agg_dir}")
    return status


def _write_csv(rows, path):
    """Write list of dicts to CSV."""
    if not rows:
        return
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate verification results")
    parser.add_argument("--results-dir", required=True,
                        help="Path to results directory")
    args = parser.parse_args()
    aggregate(args.results_dir)


if __name__ == "__main__":
    main()
