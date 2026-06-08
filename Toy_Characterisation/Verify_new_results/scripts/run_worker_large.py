#!/usr/bin/env python3
"""
Condor worker for large-event (up to 1000 tracks) reconstruction study.

Uses AcceleratedHamiltonian with Numba-parallel kernels for events that
are too large for the original SimpleHamiltonianFast.

Task types
----------
reco_large : Full reconstruction metrics for one (n_tracks, angle) point
             over N_EVENTS independent events using gamma=3 and the
             accelerated Hamiltonian.

Usage (called by run_worker_large.sh):
    python run_worker_large.py --params-json /path/to/job_XXXXX.json --outdir /path/to/output
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Add the parent dir (Verify_new_results/) so accelerated_hamiltonian is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.analysis import EventValidator
from accelerated_hamiltonian import AcceleratedHamiltonian, get_tracks_accel


# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════
DZ_MM     = 33.0
N_MODULES = 5
Z_FIRST   = 100.0
HALF_X    = 50.0
HALF_Y    = 50.0

SCALE     = 3.0
THETA_MIN = 1.5e-5

PV_SIGMA  = {"x": 0.1, "y": 0.1, "z": 50.0}

METRIC_KEYS = ["efficiency", "ghost_rate", "clone_fraction",
               "mean_purity", "hit_efficiency"]


# ═══════════════════════════════════════════════════════════════════
#  Helpers
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


def compute_threshold(gamma, delta):
    """Corrected threshold for gamma >= 2.

    Standard formula (1 + baseline)/2 fails when gamma > 2 because
    the max activation delta/(delta+gamma-2) is below threshold.
    Use midpoint between isolated baseline and chain activation.
    """
    baseline = delta / (delta + gamma)
    if delta + gamma - 2 > 0:
        max_chain = delta / (delta + gamma - 2)
    else:
        max_chain = 1.0
    return (baseline + max_chain) / 2


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
#  Task: reco_large
# ═══════════════════════════════════════════════════════════════════
def run_reco_large(params, outdir):
    """Full pipeline for large events: generate → accel-Hamiltonian → solve → validate.

    For each event, saves all 5 metrics plus timing and solution statistics.
    Also performs a threshold sweep to characterise threshold sensitivity.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_tracks    = params["n_tracks"]
    angle       = params["angle"]
    sigma_scatt = params["sigma_scatt"]
    sigma_res   = params["sigma_res"]
    gamma       = params["gamma"]
    delta       = params["delta"]
    n_events    = params["n_events"]

    epsilon   = compute_epsilon(sigma_res, sigma_scatt)
    threshold = compute_threshold(gamma, delta)
    geo       = make_geometry()

    # Threshold sweep values for sensitivity analysis
    baseline = delta / (delta + gamma)
    thr_sweep = np.linspace(max(baseline + 0.01, 0.26), 0.50, 10).tolist()
    if threshold not in thr_sweep:
        thr_sweep.append(threshold)
        thr_sweep.sort()

    print(f"  n_tracks={n_tracks}  angle={angle}  gamma={gamma}  delta={delta}")
    print(f"  sigma_scatt={sigma_scatt:.2e}  sigma_res={sigma_res:.4f}")
    print(f"  epsilon={epsilon:.6f}  threshold={threshold:.4f}")
    print(f"  n_events={n_events}")
    print(f"  threshold sweep: {[f'{t:.3f}' for t in thr_sweep]}")

    per_event = []
    thr_sweep_results = []  # threshold sensitivity per event

    for i in range(n_events):
        t0 = time.time()

        # Generate event
        event = safe_generate(
            geo, n_tracks,
            measurement_error=sigma_res,
            collision_noise=sigma_scatt,
            phi_max=angle,
            theta_max=angle,
        )
        t_gen = time.time() - t0

        # Build and solve
        t1 = time.time()
        ham = AcceleratedHamiltonian(epsilon=epsilon, gamma=gamma, delta=delta)
        A, b = ham.construct_hamiltonian(event)
        t_build = time.time() - t1

        t2 = time.time()
        x = ham.solve_classicaly()
        t_solve = time.time() - t2

        # Reconstruct at primary threshold
        t3 = time.time()
        reco_tracks = get_tracks_accel(ham, x, event, threshold=threshold)
        t_reco = time.time() - t3

        dt_total = time.time() - t0

        # Validate
        if len(reco_tracks) > 0:
            val = EventValidator(event, reco_tracks)
            _, metrics = val.match_tracks(purity_min=0.7)
            if "clone_fraction" not in metrics:
                metrics["clone_fraction"] = 0.0
        else:
            metrics = {
                "efficiency": 0.0, "ghost_rate": 0.0,
                "clone_fraction": 0.0, "mean_purity": 0.0,
                "hit_efficiency": 0.0,
            }

        row = {
            "event": i,
            "n_tracks": n_tracks,
            "angle": angle,
            "sigma_scatt": sigma_scatt,
            "sigma_res": sigma_res,
            "gamma": gamma,
            "delta": delta,
            "epsilon": epsilon,
            "threshold": threshold,
            "n_segments": ham.n_segments,
            "n_reco": len(reco_tracks),
            "n_truth": len(event.tracks),
            "n_hits": len(event.hits),
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "x_median": float(np.median(x)),
            "wall_time_s": round(dt_total, 2),
            "t_generate_s": round(t_gen, 2),
            "t_build_s": round(t_build, 2),
            "t_solve_s": round(t_solve, 2),
            "t_reco_s": round(t_reco, 3),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, 0.0)

        per_event.append(row)

        # Threshold sweep for this event
        for thr_val in thr_sweep:
            reco_thr = get_tracks_accel(ham, x, event, threshold=thr_val)
            if len(reco_thr) > 0:
                val_thr = EventValidator(event, reco_thr)
                _, m_thr = val_thr.match_tracks(purity_min=0.7)
                if "clone_fraction" not in m_thr:
                    m_thr["clone_fraction"] = 0.0
            else:
                m_thr = {"efficiency": 0.0, "ghost_rate": 0.0,
                         "clone_fraction": 0.0}
            thr_sweep_results.append({
                "event": i,
                "n_tracks": n_tracks,
                "threshold": thr_val,
                "n_reco": len(reco_thr),
                "efficiency": m_thr.get("efficiency", 0.0),
                "ghost_rate": m_thr.get("ghost_rate", 0.0),
                "clone_fraction": m_thr.get("clone_fraction", 0.0),
            })

        if (i + 1) % 5 == 0 or i == n_events - 1:
            print(f"    event {i+1}/{n_events}  "
                  f"eff={metrics.get('efficiency',0):.3f}  "
                  f"ghost={metrics.get('ghost_rate',0):.3f}  "
                  f"segs={ham.n_segments}  "
                  f"({dt_total:.1f}s)")

    # ─── Aggregate ────────────────────────────────────────────────
    agg = {
        "task": "reco_large",
        "n_tracks": n_tracks,
        "angle": angle,
        "sigma_scatt": sigma_scatt,
        "sigma_res": sigma_res,
        "gamma": gamma,
        "delta": delta,
        "epsilon": epsilon,
        "threshold": threshold,
        "n_events": n_events,
    }
    for key in METRIC_KEYS:
        vals = [r[key] for r in per_event]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_se"] = se(vals)
        agg[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    agg["wall_time_total_s"] = sum(r["wall_time_s"] for r in per_event)
    agg["n_segments_mean"] = float(np.mean([r["n_segments"] for r in per_event]))

    # Threshold sweep aggregation
    thr_agg = []
    for thr_val in thr_sweep:
        thr_events = [r for r in thr_sweep_results if r["threshold"] == thr_val]
        effs = [r["efficiency"] for r in thr_events]
        ghosts = [r["ghost_rate"] for r in thr_events]
        thr_agg.append({
            "threshold": thr_val,
            "efficiency_mean": float(np.mean(effs)),
            "efficiency_se": se(effs),
            "ghost_rate_mean": float(np.mean(ghosts)),
            "ghost_rate_se": se(ghosts),
        })

    results = {
        "task": "reco_large",
        "job_id": params["job_id"],
        "params": params,
        "aggregated": agg,
        "per_event": per_event,
        "threshold_sweep": thr_agg,
    }

    save_json(results, outdir / "results.json")
    print(f"  Saved results.json ({n_events} events, "
          f"eff={agg['efficiency_mean']:.3f}±{agg['efficiency_se']:.3f})")


# ═══════════════════════════════════════════════════════════════════
#  Dispatcher
# ═══════════════════════════════════════════════════════════════════
TASK_DISPATCH = {
    "reco_large": run_reco_large,
}


def main():
    parser = argparse.ArgumentParser(description="Large-event reconstruction worker")
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
