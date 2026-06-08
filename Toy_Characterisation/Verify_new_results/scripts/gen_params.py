#!/usr/bin/env python3
"""
Generate parameter files for verification sweep Condor jobs.

Task types
----------
verify_scatt : Sweep sigma_scatt with sigma_res=0 for given (angle, n_tracks).
verify_res   : Sweep sigma_res with sigma_scatt=1e-4 for given (angle, n_tracks).

Each job runs N_EVENTS independent events for one (sweep_val, n_tracks, angle)
combination and writes per-event + aggregated metrics.

Usage
-----
    python gen_params.py --outdir /path/to/results [--dry-run]
"""

import argparse
import json
from pathlib import Path


def generate_all_params(outdir, dry_run=False):
    """Generate all parameter JSON files and batches.txt."""

    outdir = Path(outdir)
    params_dir = outdir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    batches = []
    job_id = 0

    # ─── Shared parameters ────────────────────────────────────────
    ANGLE_SETTINGS     = [0.02, 0.2]
    TRACK_COUNTS       = [10, 20, 50, 100]
    N_EVENTS           = 20

    # Scattering sweep
    SIGMA_SCATT_VALUES = [0.5e-4, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3]
    SIGMA_RES_FIXED    = 0.0

    # Resolution sweep
    SIGMA_RES_VALUES   = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    SIGMA_SCATT_FIXED  = 1e-4

    # ─── verify_scatt: one job per (angle, sigma_scatt, n_tracks) ─
    for angle in ANGLE_SETTINGS:
        for sigma_s in SIGMA_SCATT_VALUES:
            for n_trk in TRACK_COUNTS:
                params = {
                    "task": "verify_scatt",
                    "job_id": job_id,
                    "angle": angle,
                    "n_tracks": n_trk,
                    "sigma_scatt": sigma_s,
                    "sigma_res": SIGMA_RES_FIXED,
                    "n_events": N_EVENTS,
                }
                _write_params(params_dir, job_id, params, dry_run)
                batches.append(str(params_dir / f"job_{job_id:05d}.json"))
                job_id += 1

    # ─── verify_res: one job per (angle, sigma_res, n_tracks) ─────
    for angle in ANGLE_SETTINGS:
        for sigma_r in SIGMA_RES_VALUES:
            for n_trk in TRACK_COUNTS:
                params = {
                    "task": "verify_res",
                    "job_id": job_id,
                    "angle": angle,
                    "n_tracks": n_trk,
                    "sigma_scatt": SIGMA_SCATT_FIXED,
                    "sigma_res": sigma_r,
                    "n_events": N_EVENTS,
                }
                _write_params(params_dir, job_id, params, dry_run)
                batches.append(str(params_dir / f"job_{job_id:05d}.json"))
                job_id += 1

    # ─── Write batches.txt ────────────────────────────────────────
    batches_path = outdir / "batches.txt"
    if not dry_run:
        with open(batches_path, "w") as f:
            for b in batches:
                f.write(b + "\n")

    # ─── Write run summary ────────────────────────────────────────
    summary = {
        "total_jobs": job_id,
        "task_counts": _count_tasks(batches, dry_run),
        "parameters": {
            "angle_settings": ANGLE_SETTINGS,
            "track_counts": TRACK_COUNTS,
            "n_events": N_EVENTS,
            "sigma_scatt_values": SIGMA_SCATT_VALUES,
            "sigma_res_fixed": SIGMA_RES_FIXED,
            "sigma_res_values": SIGMA_RES_VALUES,
            "sigma_scatt_fixed": SIGMA_SCATT_FIXED,
        },
    }
    if not dry_run:
        with open(outdir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Generated {job_id} jobs in {params_dir}")
    for task, count in summary["task_counts"].items():
        print(f"  {task:20s}: {count:4d} jobs")
    print(f"Batches file: {batches_path}")

    return job_id


def _write_params(params_dir, job_id, params, dry_run):
    """Write a single parameter JSON file."""
    path = params_dir / f"job_{job_id:05d}.json"
    if not dry_run:
        with open(path, "w") as f:
            json.dump(params, f, indent=2)


def _count_tasks(batches, dry_run):
    """Count jobs per task type."""
    counts = {}
    if dry_run:
        return counts
    for b in batches:
        with open(b) as f:
            task = json.load(f)["task"]
        counts[task] = counts.get(task, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Generate verification sweep parameters")
    parser.add_argument("--outdir", required=True, help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing files")
    args = parser.parse_args()

    n_jobs = generate_all_params(args.outdir, dry_run=args.dry_run)
    print(f"\nTotal: {n_jobs} jobs ready for submission.")


if __name__ == "__main__":
    main()
