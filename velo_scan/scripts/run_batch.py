#!/usr/bin/env python3
"""
run_batch.py – Condor worker: generate, reconstruct and validate one batch.

For every row in --params whose batch == --batch:
  1. Build detector geometry (26 layers, 33 mm spacing, 80×80 mm²)
  2. Generate a truth event via StateEventGenerator
  3. Apply noise (drop + ghost) via make_noisy_event()
  4. Build Hamiltonian (SimpleHamiltonianFast), solve classically
  5. Reconstruct tracks with get_tracks()
  6. Validate with EventValidator.match_tracks(purity_min=0.7)
  7. Save:
     - truth event  → batch_<B>/truth_<tag>.json
     - noisy event  → batch_<B>/noisy_<tag>.json
     - reco event   → batch_<B>/reco_<tag>.json
     - metrics row  → batch_<B>/metrics.csv   (append)

Uses the lhcb_velo_toy package (installed in Velo_Toy_Tester conda env).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ── lhcb_velo_toy imports ──────────────────────────────────────────
from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers import SimpleHamiltonianFast, get_tracks
from lhcb_velo_toy.solvers.reconstruction import construct_event
from lhcb_velo_toy.analysis import EventValidator

# =====================================================================
# Constants
# =====================================================================

# Detector geometry (from supervisor instructions)
N_MODULES = 26
Z_FIRST_MM = 100.0
Z_SPACING_MM = 33.0
HALF_XY_MM = 40.0      # 80 × 80 mm²

# Angular acceptance
PHI_MAX = 0.2           # rad
THETA_MAX = 0.2         # rad

# Hamiltonian parameters (from baseline notebook)
GAMMA = 1.5
DELTA = 1.0

# Threshold: midpoint between uncoupled baseline and 1
BASELINE = DELTA / (DELTA + GAMMA)          # ≈ 0.4
THRESHOLD = (1.0 + BASELINE) / 2.0         # ≈ 0.7

# Minimum angle for epsilon window
THETA_MIN = 0.000015


# =====================================================================
# Helpers
# =====================================================================

def make_geometry() -> PlaneGeometry:
    """Build the standard 26-module VeLo-like geometry."""
    z_positions = [Z_FIRST_MM + i * Z_SPACING_MM for i in range(N_MODULES)]
    return PlaneGeometry(
        module_id=list(range(N_MODULES)),
        lx=[HALF_XY_MM] * N_MODULES,
        ly=[HALF_XY_MM] * N_MODULES,
        z=z_positions,
    )


def epsilon_window(meas_err: float, coll_noise: float, dz: float,
                   scale: int, theta_min: float = THETA_MIN) -> float:
    """
    Compute the angular tolerance (epsilon) for the Hamiltonian.

    epsilon = sqrt( (scale * σ_scatt)² + arctan(scale * σ_res / dz)² + θ_min² )
    """
    theta_s = scale * coll_noise
    theta_r = np.arctan((scale * meas_err) / dz) if dz != 0 else 0.0
    theta_m = theta_min
    return float(np.sqrt(theta_s**2 + theta_r**2 + theta_m**2))


def run_one(
    meas: float,
    coll: float,
    ghost: float,
    drop: float,
    repeat: int,
    scale: int,
    n_tracks: int,
    outdir: Path,
    geo: PlaneGeometry,
) -> dict:
    """
    Run a single event: generate → noise → reconstruct → validate.

    Returns a flat dict of parameters + metrics suitable for CSV.
    """
    # Reproducible randomness (seeded per param combo)
    seed = hash((meas, coll, ghost, drop, repeat, scale, n_tracks)) % (2**31)
    np.random.seed(seed)

    # ── Generate truth event ────────────────────────────────────
    gen = StateEventGenerator(
        detector_geometry=geo,
        events=1,
        n_particles=[n_tracks],
        phi_min=-PHI_MAX,
        phi_max=PHI_MAX,
        theta_min=-THETA_MAX,
        theta_max=THETA_MAX,
        measurement_error=meas,
        collision_noise=coll,
    )
    gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 50.0})
    particles = [[{"type": "pion", "mass": 139.6, "q": 1}] * n_tracks]
    gen.generate_particles(particles)
    truth_event = gen.generate_complete_events()

    # ── Apply noise / drops ─────────────────────────────────────
    noisy_event = gen.make_noisy_event(drop_rate=drop, ghost_rate=ghost)

    # ── Compute epsilon & build Hamiltonian ─────────────────────
    eps = epsilon_window(meas, coll, Z_SPACING_MM, scale)
    # Ensure epsilon is not too small (fallback to a minimum)
    eps = max(eps, 1e-6)

    ham = SimpleHamiltonianFast(epsilon=eps, gamma=GAMMA, delta=DELTA)
    ham.construct_hamiltonian(noisy_event)

    # ── Solve classically ───────────────────────────────────────
    x = ham.solve_classicaly()

    # ── Reconstruct tracks ──────────────────────────────────────
    reco_tracks = get_tracks(ham, x, noisy_event, threshold=THRESHOLD)

    # ── Build reco event for serialisation ──────────────────────
    reco_hit_ids = {hid for trk in reco_tracks for hid in trk.hit_ids}
    reco_hits = [h for h in noisy_event.hits if h.hit_id in reco_hit_ids]
    reco_event = construct_event(
        detector_geometry=geo,
        tracks=reco_tracks,
        hits=reco_hits,
    )

    # ── Validate ────────────────────────────────────────────────
    validator = EventValidator(truth_event, reco_tracks)
    matches, metrics = validator.match_tracks(purity_min=0.7)

    # ── File tag ────────────────────────────────────────────────
    tag = (
        f"m{meas}_c{coll}_g{ghost}_d{drop}"
        f"_r{repeat}_s{scale}_nt{n_tracks}"
    )

    # ── Save events as JSON ─────────────────────────────────────
    truth_event.to_json(str(outdir / f"truth_{tag}.json"))
    noisy_event.to_json(str(outdir / f"noisy_{tag}.json"))
    reco_event.to_json(str(outdir / f"reco_{tag}.json"))

    # ── Build flat metrics row ──────────────────────────────────
    row = {
        "hit_res": meas,
        "multi_scatter": coll,
        "ghost_rate": ghost,
        "drop_rate": drop,
        "repeat": repeat,
        "scale": scale,
        "n_tracks": n_tracks,
        "epsilon": eps,
        "n_segments": ham.n_segments,
        "n_reco_tracks": len(reco_tracks),
        "n_truth_tracks": len(truth_event.tracks),
        "n_hits_truth": len(truth_event.hits),
        "n_hits_noisy": len(noisy_event.hits),
    }
    # Add all metric keys returned by match_tracks
    for k, v in metrics.items():
        row[f"m_{k}"] = v

    return row


# =====================================================================
# CSV helpers
# =====================================================================

METRIC_FIELDS: list[str] | None = None   # set on first write


def append_metrics_row(csv_path: Path, row: dict) -> None:
    """Append one row to the batch metrics CSV."""
    global METRIC_FIELDS
    new_file = not csv_path.exists()

    if METRIC_FIELDS is None or new_file:
        METRIC_FIELDS = list(row.keys())

    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


# =====================================================================
# CLI
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Run one batch of the Velo scan.")
    ap.add_argument("--params", type=Path, required=True, help="Path to params.csv")
    ap.add_argument("--batch", type=int, required=True, help="Batch ID to process")
    ap.add_argument(
        "--runs-dir", type=Path, required=True,
        help="Root results directory containing batch_* folders",
    )
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.params)
    if "batch" not in df.columns:
        print("[ERROR] params.csv must have a 'batch' column", file=sys.stderr)
        sys.exit(1)

    sub = df[df["batch"] == args.batch]
    if sub.empty:
        print(f"[WARN] No rows for batch {args.batch}")
        return

    outdir = args.runs_dir / f"batch_{args.batch}"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "metrics.csv"

    geo = make_geometry()

    print(f"[INFO] Batch {args.batch}: {len(sub)} events → {outdir}")
    t0 = time.time()
    failures = 0

    for idx, row in sub.iterrows():
        try:
            result = run_one(
                meas=float(row["meas"]),
                coll=float(row["coll"]),
                ghost=float(row["ghost"]),
                drop=float(row["drop"]),
                repeat=int(row["repeat"]),
                scale=int(row["scale"]),
                n_tracks=int(row["n_tracks"]),
                outdir=outdir,
                geo=geo,
            )
            append_metrics_row(csv_path, result)
        except Exception as e:
            failures += 1
            print(f"[WARN] Row {idx} failed: {e}")
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"[INFO] Batch {args.batch} done in {elapsed:.1f}s  "
          f"({len(sub) - failures}/{len(sub)} OK)")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
