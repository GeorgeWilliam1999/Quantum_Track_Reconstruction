#!/usr/bin/env python3
"""
Generate parameter files for the Hamiltonian parameter-optimisation sweep.

Sweeps over:
  - k (scale factor)           → controls epsilon via the paper formula
  - theta_d (ERF smoothing)    → width of the error-function transition
  - measurement_error (σ_res)  → hit position smearing (mm)
  - collision_noise (σ_scatt)  → multiple scattering angle (rad)
  - convolution mode           → step function (baseline) vs ERF

Evaluated at each combination of:
  - angular cone (phi_max = theta_max)
  - track multiplicity (n_tracks)
  - n_repeats independent events

For step-function jobs theta_d is irrelevant, so that axis is collapsed
to a single sentinel value (None → stored as null in JSON).

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
CONVOLUTION_MODES     = [False, True]                    # step vs ERF

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
        for conv in CONVOLUTION_MODES:
            # For step function, theta_d is irrelevant → single value
            theta_d_grid = THETA_D_VALUES if conv else [None]

            for k, theta_d, meas_err, coll_noise in product(
                SCALE_VALUES, theta_d_grid,
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
                    "convolution":       conv,
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

    # Compute expected counts
    n_step = (len(SCALE_VALUES) * 1 * len(MEASUREMENT_ERRORS)
              * len(COLLISION_NOISES) * len(ANGLE_SETTINGS) * len(TRACK_COUNTS))
    n_erf  = (len(SCALE_VALUES) * len(THETA_D_VALUES) * len(MEASUREMENT_ERRORS)
              * len(COLLISION_NOISES) * len(ANGLE_SETTINGS) * len(TRACK_COUNTS))

    summary = {
        "total_jobs": job_id,
        "step_jobs":  n_step,
        "erf_jobs":   n_erf,
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

    print(f"Generated {job_id} jobs ({n_step} step + {n_erf} ERF)")
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
