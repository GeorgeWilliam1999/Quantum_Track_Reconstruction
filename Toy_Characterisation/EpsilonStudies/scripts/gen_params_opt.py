#!/usr/bin/env python3
"""
Generate parameter files for the Hamiltonian parameter-optimisation sweep.

Sweeps over:
  - k (scale factor)           → controls epsilon via the paper formula
  - theta_d (ERF smoothing)    → width of the error-function transition
  - measurement_error (σ_res)  → hit position smearing (mm)
  - collision_noise (σ_scatt)  → multiple scattering angle (rad)

All Hamiltonians are built with convolution=True (ERF-smoothed acceptance).

Evaluated at each combination of:
  - angular cone (phi_max = theta_max)
  - track multiplicity (n_tracks)
  - n_repeats independent events

Usage
-----
    python gen_params_opt.py --outdir /path/to/results_opt [--dry-run]
"""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np


# ─── Sweep grid ───────────────────────────────────────────────────
SCALE_VALUES          = [1, 2, 3, 5, 10, 20]           # k
THETA_D_VALUES        = [1e-4, 1e-3, 1e-2, 1e-1]       # ERF smoothing width (rad)
MEASUREMENT_ERRORS    = [0.0, 0.01, 0.05]               # σ_res (mm)
COLLISION_NOISES      = [5e-5, 1e-4, 4e-4]              # σ_scatt (rad)

# ─── Evaluation conditions ───────────────────────────────────────
ANGLE_SETTINGS        = [0.2, 0.1, 0.04]                # ±rad
TRACK_COUNTS          = [10, 50, 100]                    # n_particles per event
N_REPEATS             = 3                                # independent events per config


def generate_all_params(outdir, dry_run=False):
    outdir = Path(outdir)
    params_dir = outdir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    batches = []
    job_id = 0

    for angle, n_tracks in product(ANGLE_SETTINGS, TRACK_COUNTS):
        for k, theta_d, meas_err, coll_noise in product(
            SCALE_VALUES, THETA_D_VALUES,
            MEASUREMENT_ERRORS, COLLISION_NOISES,
        ):
            params = {
                "task":              "param_opt",
                "job_id":            job_id,
                "angle":             angle,
                "n_tracks":          n_tracks,
                "scale":             k,
                "theta_d":           theta_d,
                "measurement_error": meas_err,
                "collision_noise":   coll_noise,
                "convolution":       True,
                "n_repeats":         N_REPEATS,
            }
            _write_params(params_dir, job_id, params, dry_run)
            batches.append(str(params_dir / f"job_{job_id:05d}.json"))
            job_id += 1

    # Write batches.txt
    batches_path = outdir / "batches.txt"
    if not dry_run:
        with open(batches_path, "w") as f:
            for b in batches:
                f.write(b + "\n")

    summary = {
        "total_jobs": job_id,
        "convolution": True,
        "parameters": {
            "scale_values":       SCALE_VALUES,
            "theta_d_values":     THETA_D_VALUES,
            "measurement_errors": MEASUREMENT_ERRORS,
            "collision_noises":   COLLISION_NOISES,
            "angle_settings":     ANGLE_SETTINGS,
            "track_counts":       TRACK_COUNTS,
            "n_repeats":          N_REPEATS,
        },
    }
    if not dry_run:
        with open(outdir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Generated {job_id} jobs (all ERF convolution)")
    print(f"  Scales:     {SCALE_VALUES}")
    print(f"  Theta_d:    {THETA_D_VALUES}")
    print(f"  Meas err:   {MEASUREMENT_ERRORS}")
    print(f"  Coll noise: {COLLISION_NOISES}")
    print(f"  Angles:     {ANGLE_SETTINGS}")
    print(f"  Tracks:     {TRACK_COUNTS}")
    print(f"  Repeats:    {N_REPEATS}")
    print(f"Batches file: {batches_path}")

    return job_id


def _write_params(params_dir, job_id, params, dry_run):
    path = params_dir / f"job_{job_id:05d}.json"
    if not dry_run:
        with open(path, "w") as f:
            json.dump(params, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter files for Hamiltonian optimisation sweep")
    parser.add_argument("--outdir", required=True, help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without writing files")
    args = parser.parse_args()

    n_jobs = generate_all_params(args.outdir, dry_run=args.dry_run)
    print(f"\nTotal: {n_jobs} jobs ready for submission.")


if __name__ == "__main__":
    main()
