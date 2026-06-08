#!/usr/bin/env python3
"""
Condor worker for verification sweep jobs.

Dispatches verify_scatt and verify_res tasks.  Each task generates
N_EVENTS independent events for a single (sweep_val, n_tracks, angle)
point and writes per-event metrics + aggregated statistics.

Usage (called by run_worker.sh):
    python run_worker.py --params-json /path/to/job_XXXXX.json --outdir /path/to/output
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers import SimpleHamiltonianFast, get_tracks
from lhcb_velo_toy.analysis import EventValidator


# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════
DZ_MM     = 33.0
N_MODULES = 5
Z_FIRST   = 100.0
HALF_X    = 50.0
HALF_Y    = 50.0

GAMMA     = 1.5
DELTA     = 1.0
SCALE     = 3.0
THETA_MIN = 1.5e-5
THRESHOLD = (1 + DELTA / (DELTA + GAMMA)) / 2   # = 0.7

PV_SIGMA  = {"x": 0.1, "y": 0.1, "z": 50.0}

METRIC_KEYS = ["efficiency", "ghost_rate", "clone_fraction",
               "mean_purity", "hit_efficiency"]


# ═══════════════════════════════════════════════════════════════════
#  Helpers (same logic as helpers.py in condor_pipeline)
# ═══════════════════════════════════════════════════════════════════
def make_geometry():
    z_positions = [Z_FIRST + i * DZ_MM for i in range(N_MODULES)]
    return PlaneGeometry(
        module_id=list(range(N_MODULES)),
        lx=[HALF_X] * N_MODULES,
        ly=[HALF_Y] * N_MODULES,
        z=z_positions,
    )


def compute_epsilon(sigma_res, sigma_scatt):
    theta_s = SCALE * sigma_scatt
    theta_r = np.arctan((SCALE * sigma_res) / DZ_MM) if DZ_MM != 0 else 0.0
    return float(np.sqrt(2 * theta_s**2 + 12 * theta_r**2 + 2 * THETA_MIN**2))


def safe_generate(geo, n_tracks, measurement_error=0.0, collision_noise=1e-8,
                  phi_max=0.2, theta_max=0.2, max_retries=20):
    for _ in range(max_retries):
        gen = StateEventGenerator(
            detector_geometry=geo,
            events=1,
            n_particles=[n_tracks],
            phi_min=-phi_max, phi_max=phi_max,
            theta_min=-theta_max, theta_max=theta_max,
            measurement_error=measurement_error,
            collision_noise=collision_noise,
        )
        gen.generate_random_primary_vertices(PV_SIGMA)
        gen.generate_particles(
            [[{"type": "pion", "mass": 139.6, "q": 1}] * n_tracks]
        )
        event = gen.generate_complete_events()
        if event.tracks and min(len(t.hit_ids) for t in event.tracks) >= 3:
            return event
    return event  # best effort


def run_single_event(geo, n_tracks, sigma_res, sigma_scatt, phi_max):
    """Full pipeline: generate → solve → validate.  Returns metrics dict."""
    epsilon = compute_epsilon(sigma_res, sigma_scatt)
    event = safe_generate(
        geo, n_tracks,
        measurement_error=sigma_res,
        collision_noise=sigma_scatt,
        phi_max=phi_max,
        theta_max=phi_max,
    )
    ham = SimpleHamiltonianFast(epsilon=epsilon, gamma=GAMMA, delta=DELTA)
    ham.construct_hamiltonian(event)
    x = ham.solve_classicaly()

    reco_tracks = get_tracks(ham, x, event, threshold=THRESHOLD)
    if len(reco_tracks) == 0:
        return {
            "efficiency": 0.0, "ghost_rate": 0.0,
            "clone_fraction": 0.0, "mean_purity": 0.0,
            "hit_efficiency": 0.0,
            "n_candidates": 0, "n_reconstructible": len(event.tracks),
        }
    val = EventValidator(event, reco_tracks)
    _, metrics = val.match_tracks(purity_min=0.7)
    return metrics


def se(arr):
    """Standard error of the mean."""
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


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


# ═══════════════════════════════════════════════════════════════════
#  Task handler (shared by both verify_scatt and verify_res)
# ═══════════════════════════════════════════════════════════════════
def run_verify(params, outdir):
    """Run N_EVENTS for one (sweep_val, n_tracks, angle) point."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    task       = params["task"]
    n_tracks   = params["n_tracks"]
    sigma_scatt = params["sigma_scatt"]
    sigma_res  = params["sigma_res"]
    angle      = params["angle"]
    n_events   = params["n_events"]

    epsilon = compute_epsilon(sigma_res, sigma_scatt)
    geo = make_geometry()

    print(f"  task={task}  n_tracks={n_tracks}  angle={angle}  "
          f"sigma_scatt={sigma_scatt:.2e}  sigma_res={sigma_res:.4f}  "
          f"epsilon={epsilon:.6f}  n_events={n_events}")

    per_event = []
    for i in range(n_events):
        t0 = time.time()
        metrics = run_single_event(geo, n_tracks, sigma_res, sigma_scatt, angle)
        dt = time.time() - t0
        row = {
            "event": i,
            "n_tracks": n_tracks,
            "sigma_scatt": sigma_scatt,
            "sigma_res": sigma_res,
            "phi_max": angle,
            "epsilon": epsilon,
            "wall_time_s": round(dt, 2),
        }
        for key in METRIC_KEYS:
            row[key] = metrics[key]
        row["n_candidates"] = metrics.get("n_candidates", 0)
        row["n_reconstructible"] = metrics.get("n_reconstructible", 0)
        per_event.append(row)

        if (i + 1) % 5 == 0 or i == n_events - 1:
            print(f"    event {i+1}/{n_events}  "
                  f"eff={metrics['efficiency']:.3f}  "
                  f"ghost={metrics['ghost_rate']:.3f}  "
                  f"({dt:.1f}s)")

    # Aggregate
    agg = {"task": task, "n_tracks": n_tracks, "angle": angle,
           "sigma_scatt": sigma_scatt, "sigma_res": sigma_res,
           "epsilon": epsilon, "n_events": n_events}
    for key in METRIC_KEYS:
        vals = [r[key] for r in per_event]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_se"] = se(vals)
        agg[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    agg["wall_time_total_s"] = sum(r["wall_time_s"] for r in per_event)

    results = {
        "task": task,
        "job_id": params["job_id"],
        "params": params,
        "aggregated": agg,
        "per_event": per_event,
    }

    save_json(results, outdir / "results.json")
    print(f"  Saved results.json ({n_events} events, "
          f"eff={agg['efficiency_mean']:.3f}±{agg['efficiency_se']:.3f})")


# ═══════════════════════════════════════════════════════════════════
#  Dispatcher
# ═══════════════════════════════════════════════════════════════════
TASK_DISPATCH = {
    "verify_scatt": run_verify,
    "verify_res":   run_verify,
}


def main():
    parser = argparse.ArgumentParser(description="Verification sweep worker")
    parser.add_argument("--params-json", required=True, help="Path to job parameter JSON")
    parser.add_argument("--outdir", required=True, help="Output directory for this job")
    args = parser.parse_args()

    params_path = Path(args.params_json)
    outdir = Path(args.outdir)

    with open(params_path) as f:
        params = json.load(f)

    task = params["task"]
    job_id = params["job_id"]

    print(f"[Worker] Job {job_id}: task={task}")
    print(f"[Worker] Params: {params_path}")
    print(f"[Worker] Output: {outdir}")

    if task not in TASK_DISPATCH:
        print(f"[Worker] ERROR: Unknown task type '{task}'")
        sys.exit(1)

    t0 = time.time()
    try:
        TASK_DISPATCH[task](params, outdir)
        elapsed = time.time() - t0
        print(f"[Worker] Job {job_id} completed in {elapsed:.1f}s")

        with open(outdir / "DONE", "w") as f:
            f.write(f"completed in {elapsed:.1f}s\n")

    except Exception as e:
        elapsed = time.time() - t0
        print(f"[Worker] Job {job_id} FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        with open(outdir / "FAILED", "w") as f:
            f.write(f"{e}\n")
            traceback.print_exc(file=f)
        sys.exit(1)


if __name__ == "__main__":
    main()
