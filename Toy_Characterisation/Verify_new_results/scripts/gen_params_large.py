#!/usr/bin/env python3
"""
Generate parameter files for large-event reconstruction Condor jobs.

Produces jobs for the reco_large task: full reconstruction metrics
using AcceleratedHamiltonian for events up to 1000 tracks.

Task types
----------
reco_large : Sweep over track counts [100..1000] with fixed physics params.
             Each job handles one n_tracks value for N_EVENTS events.

Usage
-----
    python gen_params_large.py --outdir /path/to/results_large [--dry-run]
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

    # ─── Physics parameters (user-specified: gamma=3) ─────────────
    SIGMA_SCATT = 1e-4      # 0.1 mrad
    SIGMA_RES   = 0.005     # 5 µm
    GAMMA       = 3.0
    DELTA       = 1.0
    ANGLE       = 0.2       # non-dense angular cone

    # ─── Sweep parameters ─────────────────────────────────────────
    TRACK_COUNTS = [10, 50, 100, 200, 300, 500, 700, 1000]
    N_EVENTS     = 50       # high statistics

    # ─── reco_large: one job per n_tracks ─────────────────────────
    for n_trk in TRACK_COUNTS:
        params = {
            "task": "reco_large",
            "job_id": job_id,
            "n_tracks": n_trk,
            "angle": ANGLE,
            "sigma_scatt": SIGMA_SCATT,
            "sigma_res": SIGMA_RES,
            "gamma": GAMMA,
            "delta": DELTA,
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
            "sigma_scatt": SIGMA_SCATT,
            "sigma_res": SIGMA_RES,
            "gamma": GAMMA,
            "delta": DELTA,
            "angle": ANGLE,
            "track_counts": TRACK_COUNTS,
            "n_events": N_EVENTS,
        },
    }
    if not dry_run:
        with open(outdir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Generated {job_id} jobs in {params_dir}")
    for task, count in summary["task_counts"].items():
        print(f"  {task:20s}: {count:4d} jobs")
    print(f"Batches file: {batches_path}")
    print(f"Parameters: gamma={GAMMA}, delta={DELTA}, "
          f"sigma_scatt={SIGMA_SCATT:.1e}, sigma_res={SIGMA_RES}, "
          f"angle={ANGLE}")
    print(f"Track counts: {TRACK_COUNTS}")
    print(f"Events per point: {N_EVENTS}")

    return job_id


def _write_params(params_dir, job_id, params, dry_run):
    path = params_dir / f"job_{job_id:05d}.json"
    if not dry_run:
        with open(path, "w") as f:
            json.dump(params, f, indent=2)


def _count_tasks(batches, dry_run):
    counts = {}
    if dry_run:
        return counts
    for b in batches:
        with open(b) as f:
            task = json.load(f)["task"]
        counts[task] = counts.get(task, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Generate large-event reconstruction Condor parameters")
    parser.add_argument("--outdir", required=True, help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without writing files")
    args = parser.parse_args()

    n_jobs = generate_all_params(args.outdir, dry_run=args.dry_run)
    print(f"\nTotal: {n_jobs} jobs ready for submission.")


if __name__ == "__main__":
    main()
