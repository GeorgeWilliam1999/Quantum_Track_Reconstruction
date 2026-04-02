#!/usr/bin/env python3
"""
Generate parameter files for the segment-grass ghost/drop-rate sweep.

Sweeps:
  - ghost_rate : [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
  - drop_rate  : [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
  - angle      : [0.2, 0.1]   (±0.2 shows grass, ±0.1 is the control)
  - n_tracks   : [20, 50, 100]

Each (ghost_rate, drop_rate, angle, n_tracks) combination is one job.
Each job runs N_REPEATS independent events and collects per-module
occupancy, activation spectra, and reconstruction metrics.

Usage
-----
    python gen_params.py --outdir /path/to/results [--dry-run]
"""

import argparse
import json
from pathlib import Path


def generate_params(outdir: str, dry_run: bool = False):
    outdir = Path(outdir)
    params_dir = outdir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    # ── sweep grid ────────────────────────────────────────────────
    GHOST_RATES  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    DROP_RATES   = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    ANGLES       = [0.2, 0.1]
    TRACK_COUNTS = [20, 50, 100]
    N_REPEATS    = 10

    batches = []
    job_id = 0

    for angle in ANGLES:
        for n_tracks in TRACK_COUNTS:
            for ghost_rate in GHOST_RATES:
                for drop_rate in DROP_RATES:
                    params = {
                        "job_id": job_id,
                        "angle": angle,
                        "n_tracks": n_tracks,
                        "ghost_rate": ghost_rate,
                        "drop_rate": drop_rate,
                        "n_repeats": N_REPEATS,
                    }
                    fname = params_dir / f"job_{job_id:05d}.json"
                    if not dry_run:
                        with open(fname, "w") as f:
                            json.dump(params, f, indent=2)
                    batches.append(str(fname))
                    job_id += 1

    # Write batches file
    batches_file = outdir / "batches.txt"
    if not dry_run:
        with open(batches_file, "w") as f:
            f.write("\n".join(batches) + "\n")

    # Summary
    summary = {
        "total_jobs": job_id,
        "ghost_rates": GHOST_RATES,
        "drop_rates": DROP_RATES,
        "angles": ANGLES,
        "track_counts": TRACK_COUNTS,
        "n_repeats": N_REPEATS,
    }
    summary_file = outdir / "run_summary.json"
    if not dry_run:
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Generated {job_id} jobs")
    print(f"  Ghost rates : {GHOST_RATES}")
    print(f"  Drop rates  : {DROP_RATES}")
    print(f"  Angles      : {ANGLES}")
    print(f"  Track counts: {TRACK_COUNTS}")
    print(f"  Repeats     : {N_REPEATS}")
    if dry_run:
        print("  [DRY RUN — no files written]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    generate_params(args.outdir, args.dry_run)
