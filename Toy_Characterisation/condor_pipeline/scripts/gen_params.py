#!/usr/bin/env python3
"""
Generate parameter files for Condor pipeline jobs.

Each job gets one JSON file describing:
  - task type (which experiment to run)
  - all parameters for that specific job
  - which repeats to run

Task types
----------
char_scan       : Characterisation parametric scan (segment metrics vs track count)
char_hist       : Characterisation acceptance histograms
char_bulk_angles: Characterisation 100×50 bulk angle distributions
hc_density      : Hit-competition Steps 1-4 (occupancy/activation/competition/reco)
hc_roc          : Hit-competition Step 5 (threshold sweep / ROC)
hc_scatt        : Hit-competition Step 6 (scattering scan)
hc_scatt_hist   : Hit-competition Step 6 (scattering histograms)

Usage
-----
    python gen_params.py --outdir /path/to/results [--dry-run]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def generate_all_params(outdir, dry_run=False):
    """Generate all parameter JSON files and batches.txt."""

    outdir = Path(outdir)
    params_dir = outdir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    batches = []
    job_id = 0

    # ─── Shared parameters ────────────────────────────────────────
    ANGLE_SETTINGS    = [0.2, 0.1, 0.04]
    SCATT_MULTIPLIERS = [1, 2, 4]
    TRACK_SIZES       = list(range(10, 110, 10))  # 10..100 step 10
    N_REPEATS_SCAN    = 5

    # hit-competition
    TRACK_DENSITIES   = [5, 10, 20, 30, 50, 75, 100, 150]
    N_REPEATS_HC      = 5
    ROC_DENSITIES     = [10, 30, 50, 100, 150]
    ROC_REPEATS       = 5
    THRESHOLDS        = np.linspace(0.2, 0.9, 15).tolist()

    # histograms
    N_TRACKS_HIST     = 50
    N_EVENTS_HIST     = 20

    # bulk angles
    N_EVENTS_BULK     = 100
    N_TRACKS_BULK     = 50
    EVENTS_PER_JOB_BULK = 50  # split 100 events into 2 jobs

    # ─── char_scan: one job per (angle, scatt_mult) ───────────────
    for angle in ANGLE_SETTINGS:
        for mult in SCATT_MULTIPLIERS:
            params = {
                "task": "char_scan",
                "job_id": job_id,
                "angle": angle,
                "scatt_mult": mult,
                "track_sizes": TRACK_SIZES,
                "n_repeats": N_REPEATS_SCAN,
            }
            _write_params(params_dir, job_id, params, dry_run)
            batches.append(str(params_dir / f"job_{job_id:05d}.json"))
            job_id += 1

    # ─── char_hist: one job per (angle, scatt_mult) ───────────────
    for angle in ANGLE_SETTINGS:
        for mult in SCATT_MULTIPLIERS:
            params = {
                "task": "char_hist",
                "job_id": job_id,
                "angle": angle,
                "scatt_mult": mult,
                "n_tracks": N_TRACKS_HIST,
                "n_events": N_EVENTS_HIST,
            }
            _write_params(params_dir, job_id, params, dry_run)
            batches.append(str(params_dir / f"job_{job_id:05d}.json"))
            job_id += 1

    # ─── char_bulk_angles: split into EVENTS_PER_JOB_BULK ────────
    n_bulk_jobs = (N_EVENTS_BULK + EVENTS_PER_JOB_BULK - 1) // EVENTS_PER_JOB_BULK
    for chunk_i in range(n_bulk_jobs):
        start = chunk_i * EVENTS_PER_JOB_BULK
        end = min(start + EVENTS_PER_JOB_BULK, N_EVENTS_BULK)
        params = {
            "task": "char_bulk_angles",
            "job_id": job_id,
            "n_tracks": N_TRACKS_BULK,
            "event_start": start,
            "event_end": end,
        }
        _write_params(params_dir, job_id, params, dry_run)
        batches.append(str(params_dir / f"job_{job_id:05d}.json"))
        job_id += 1

    # ─── hc_density: one job per (angle, n_tracks) ───────────────
    # Steps 1-4 share the same event grid
    for angle in ANGLE_SETTINGS:
        for n_trk in TRACK_DENSITIES:
            params = {
                "task": "hc_density",
                "job_id": job_id,
                "angle": angle,
                "n_tracks": n_trk,
                "n_repeats": N_REPEATS_HC,
            }
            _write_params(params_dir, job_id, params, dry_run)
            batches.append(str(params_dir / f"job_{job_id:05d}.json"))
            job_id += 1

    # ─── hc_roc: one job per (angle, n_tracks) ───────────────────
    for angle in ANGLE_SETTINGS:
        for n_trk in ROC_DENSITIES:
            params = {
                "task": "hc_roc",
                "job_id": job_id,
                "angle": angle,
                "n_tracks": n_trk,
                "n_repeats": ROC_REPEATS,
                "thresholds": THRESHOLDS,
            }
            _write_params(params_dir, job_id, params, dry_run)
            batches.append(str(params_dir / f"job_{job_id:05d}.json"))
            job_id += 1

    # ─── hc_scatt: one job per scatt_mult ─────────────────────────
    for mult in SCATT_MULTIPLIERS:
        params = {
            "task": "hc_scatt",
            "job_id": job_id,
            "scatt_mult": mult,
            "track_sizes": TRACK_SIZES,
            "n_repeats": N_REPEATS_SCAN,
        }
        _write_params(params_dir, job_id, params, dry_run)
        batches.append(str(params_dir / f"job_{job_id:05d}.json"))
        job_id += 1

    # ─── hc_scatt_hist: one job per scatt_mult ────────────────────
    for mult in SCATT_MULTIPLIERS:
        params = {
            "task": "hc_scatt_hist",
            "job_id": job_id,
            "scatt_mult": mult,
            "n_tracks": N_TRACKS_HIST,
            "n_events": N_EVENTS_HIST,
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
        "task_counts": _count_tasks(batches, params_dir, dry_run),
        "parameters": {
            "angle_settings": ANGLE_SETTINGS,
            "scatt_multipliers": SCATT_MULTIPLIERS,
            "track_sizes": TRACK_SIZES,
            "n_repeats_scan": N_REPEATS_SCAN,
            "track_densities": TRACK_DENSITIES,
            "n_repeats_hc": N_REPEATS_HC,
            "roc_densities": ROC_DENSITIES,
            "roc_repeats": ROC_REPEATS,
            "thresholds": THRESHOLDS,
            "n_tracks_hist": N_TRACKS_HIST,
            "n_events_hist": N_EVENTS_HIST,
            "n_events_bulk": N_EVENTS_BULK,
            "n_tracks_bulk": N_TRACKS_BULK,
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


def _count_tasks(batches, params_dir, dry_run):
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
    parser = argparse.ArgumentParser(description="Generate Condor pipeline parameters")
    parser.add_argument("--outdir", required=True, help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing files")
    args = parser.parse_args()

    n_jobs = generate_all_params(args.outdir, dry_run=args.dry_run)
    print(f"\nTotal: {n_jobs} jobs ready for submission.")


if __name__ == "__main__":
    main()
