#!/usr/bin/env python3
"""
gen_params.py – Full Cartesian hyperparameter scan for the Velo Toy Model.

Generates params.csv, batches.txt, batch directories, and run_summary.json
under a specified output directory (default: ../results relative to this script).

Parameter grid (from supervisor instructions):
  σ_scatt       : [0, 0.1, 0.2, 0.3, 0.5, 1.0] mrad
  σ_res         : [0, 10, 20, 50] µm
  window scale  : [3, 4, 5]
  drop rate     : [0%, 1%, 2%]
  noise fraction: [0, 0.0001, 0.001, 0.01]
  n_tracks      : [5, 10, 20, 50, 100]
  repeats       : 50

Total = 6 × 4 × 3 × 3 × 4 × 5 × 50 = 216 000 events
Batches of 50 → 4 320 condor jobs.

Defaults (for reference / 1-D scans):
  σ_scatt = 0.1 mrad,  σ_res = 10 µm,  drop = 0%,  noise = 0, scale = 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from itertools import product
from pathlib import Path

# =====================================================================
# Parameter grid
# =====================================================================

# Multiple scattering (rad) – values in mrad from instructions, stored in rad
SIGMA_SCATT_VALUES = [0.0, 0.0001, 0.0002, 0.0003, 0.0005, 0.001]
#                      0     0.1     0.2     0.3     0.5     1.0   mrad

# Hit resolution (mm) – values in µm from instructions, stored in mm
SIGMA_RES_VALUES = [0.0, 0.010, 0.020, 0.050]
#                    0     10     20     50   µm

# Window scale factor
SCALE_VALUES = [3, 4, 5]

# Hit drop rate (fraction)
DROP_VALUES = [0.0, 0.01, 0.02]

# Noise / ghost fraction
NOISE_VALUES = [0.0, 0.0001, 0.001, 0.01]

# Number of tracks per event
N_TRACKS_VALUES = [5, 10, 20, 50, 100]

# Repeats per configuration
N_REPEATS = 50

# Events packed into one condor job
EVENTS_PER_BATCH = 50

# =====================================================================
# Fixed detector geometry (from instructions)
# =====================================================================

N_MODULES = 26
Z_SPACING_MM = 33.0
HALF_XY_MM = 40.0          # 80 × 80 mm² → half-width 40 mm
Z_FIRST_MM = 100.0          # first module z
PHI_MAX = 0.2               # rad (±0.2)
THETA_MAX = 0.2             # rad (±0.2)

# Defaults for reference
DEFAULTS = {
    "sigma_scatt": 0.0001,   # 0.1 mrad in rad
    "sigma_res": 0.010,      # 10 µm in mm
    "drop": 0.0,
    "noise": 0.0,
    "scale": 5,
}

# =====================================================================
# Helpers
# =====================================================================

CSV_FIELDS = [
    "batch", "meas", "coll", "ghost", "drop",
    "repeat", "scale", "n_tracks",
    "scan_type",
]


def generate_params() -> list[dict]:
    """Return one dict per event across the full Cartesian grid."""
    params = []
    for coll, meas, scale, drop, noise, n_tracks in product(
        SIGMA_SCATT_VALUES,
        SIGMA_RES_VALUES,
        SCALE_VALUES,
        DROP_VALUES,
        NOISE_VALUES,
        N_TRACKS_VALUES,
    ):
        for repeat in range(N_REPEATS):
            params.append({
                "meas": meas,
                "coll": coll,
                "ghost": noise,
                "drop": drop,
                "repeat": repeat,
                "scale": scale,
                "n_tracks": n_tracks,
                "scan_type": "full_grid",
            })
    return params


def assign_batches(params: list[dict]) -> list[dict]:
    """Add a 'batch' key to every param dict."""
    for i, p in enumerate(params):
        p["batch"] = i // EVENTS_PER_BATCH
    return params


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate hyperparameter scan grid.")
    ap.add_argument(
        "--outdir", type=Path, default=Path(__file__).resolve().parent.parent / "results",
        help="Root output directory (default: ../results)",
    )
    args = ap.parse_args()
    outdir: Path = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Build grid
    params = generate_params()
    params = assign_batches(params)
    n_batches = params[-1]["batch"] + 1

    # ── params.csv ──────────────────────────────────────────────────
    csv_path = outdir / "params.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(params)

    # ── batches.txt ─────────────────────────────────────────────────
    batches_path = outdir / "batches.txt"
    with open(batches_path, "w") as f:
        for bid in range(n_batches):
            f.write(f"{bid}\n")

    # ── batch directories ───────────────────────────────────────────
    for bid in range(n_batches):
        (outdir / f"batch_{bid}").mkdir(exist_ok=True)

    # ── run_summary.json ────────────────────────────────────────────
    summary = {
        "run_name": "velo_scan",
        "description": "Full Cartesian hyperparameter scan (supervisor instructions)",
        "grid": {
            "sigma_scatt_rad": SIGMA_SCATT_VALUES,
            "sigma_res_mm": SIGMA_RES_VALUES,
            "scale": SCALE_VALUES,
            "drop_rate": DROP_VALUES,
            "noise_fraction": NOISE_VALUES,
            "n_tracks": N_TRACKS_VALUES,
        },
        "defaults": DEFAULTS,
        "geometry": {
            "n_modules": N_MODULES,
            "z_spacing_mm": Z_SPACING_MM,
            "half_xy_mm": HALF_XY_MM,
            "z_first_mm": Z_FIRST_MM,
            "phi_max_rad": PHI_MAX,
            "theta_max_rad": THETA_MAX,
        },
        "repeats_per_config": N_REPEATS,
        "events_per_batch": EVENTS_PER_BATCH,
        "total_events": len(params),
        "total_batches": n_batches,
    }
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── summary ─────────────────────────────────────────────────────
    print(f"Velo Scan – Full Cartesian Grid")
    print(f"{'=' * 50}")
    print(f"  σ_scatt  : {[s*1e3 for s in SIGMA_SCATT_VALUES]} mrad")
    print(f"  σ_res    : {[r*1e3 for r in SIGMA_RES_VALUES]} µm")
    print(f"  scale    : {SCALE_VALUES}")
    print(f"  drop     : {[d*100 for d in DROP_VALUES]}%")
    print(f"  noise    : {NOISE_VALUES}")
    print(f"  n_tracks : {N_TRACKS_VALUES}")
    print(f"  repeats  : {N_REPEATS}")
    print(f"{'=' * 50}")
    print(f"  Total events : {len(params):,}")
    print(f"  Total batches: {n_batches:,}")
    print(f"  Events/batch : {EVENTS_PER_BATCH}")
    print(f"{'=' * 50}")
    print(f"  Output dir   : {outdir}")
    print(f"  params.csv   : {csv_path}")
    print(f"  batches.txt  : {batches_path}")


if __name__ == "__main__":
    main()
