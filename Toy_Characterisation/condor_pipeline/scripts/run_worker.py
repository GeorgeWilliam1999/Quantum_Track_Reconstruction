#!/usr/bin/env python3
"""
Unified Condor worker for the Toy Characterisation pipeline.

Reads a parameter JSON file, dispatches to the appropriate task function,
and writes results to the output directory.

Usage
-----
    python run_worker.py --params-json /path/to/job_XXXXX.json --outdir /path/to/output
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Add the scripts directory to the path so helpers can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import (
    make_geometry, safe_generate, compute_epsilon,
    collect_segment_pair_angles, label_segments, build_hit_to_segments,
    build_and_solve, reconstruct_and_validate,
    SIGMA_RES, SIGMA_SCATT, DZ_MM, SCALE, GAMMA, DELTA,
    BASELINE, THRESHOLD, EPSILON, N_MODULES,
)


# ═══════════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════════
def se(a):
    """Standard error of the mean."""
    a = np.asarray(a, dtype=float)
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def save_json(obj, path):
    """Write JSON, converting numpy types to Python types."""
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
#  Task: char_scan
#  Characterisation parametric scan — segment metrics vs track count
#  for one (angle, scatt_mult) combination, all track sizes, n repeats
# ═══════════════════════════════════════════════════════════════════
def run_char_scan(params, outdir):
    angle     = params["angle"]
    mult      = params["scatt_mult"]
    track_sizes = params["track_sizes"]
    n_repeats = params["n_repeats"]

    sigma_s = SIGMA_SCATT * mult
    eps     = compute_epsilon(SIGMA_RES, sigma_s, DZ_MM, scale=SCALE)
    geo     = make_geometry()

    results = []

    for n_trk in track_sizes:
        per_repeat = []
        for rep in range(n_repeats):
            event = safe_generate(geo, n_trk, collision_noise=sigma_s,
                                  phi_max=angle, theta_max=angle)
            true_angles, false_angles = collect_segment_pair_angles(event)
            ta = np.array(true_angles)
            fa = np.array(false_angles)
            t_acc = int(np.sum(ta <= eps))
            f_acc = int(np.sum(fa <= eps))
            tot = t_acc + f_acc
            per_repeat.append({
                'n_true': len(ta), 'n_false': len(fa),
                'true_acc': t_acc,  'false_acc': f_acc,
                'eff': t_acc / len(ta) if len(ta) else 0,
                'fr':  f_acc / tot     if tot     else 0,
            })

        effs = np.array([r['eff'] for r in per_repeat])
        frs  = np.array([r['fr']  for r in per_repeat])
        results.append({
            'n_tracks':       n_trk,
            'eff_mean':       float(np.mean(effs) * 100),
            'eff_se':         float(se(effs) * 100),
            'fr_mean':        float(np.mean(frs) * 100),
            'fr_se':          float(se(frs) * 100),
            'n_true_mean':    float(np.mean([r['n_true']  for r in per_repeat])),
            'n_true_se':      float(se([r['n_true']  for r in per_repeat])),
            'n_false_mean':   float(np.mean([r['n_false'] for r in per_repeat])),
            'n_false_se':     float(se([r['n_false'] for r in per_repeat])),
            'true_acc_mean':  float(np.mean([r['true_acc']  for r in per_repeat])),
            'true_acc_se':    float(se([r['true_acc']  for r in per_repeat])),
            'false_acc_mean': float(np.mean([r['false_acc'] for r in per_repeat])),
            'false_acc_se':   float(se([r['false_acc'] for r in per_repeat])),
        })

    save_json({
        'task': 'char_scan',
        'angle': angle,
        'scatt_mult': mult,
        'epsilon': eps,
        'results': results,
    }, outdir / "results.json")


# ═══════════════════════════════════════════════════════════════════
#  Task: char_hist
#  Acceptance histograms — collect true/false segment-pair angles
#  for one (angle, scatt_mult) combination
# ═══════════════════════════════════════════════════════════════════
def run_char_hist(params, outdir):
    angle    = params["angle"]
    mult     = params["scatt_mult"]
    n_tracks = params["n_tracks"]
    n_events = params["n_events"]

    sigma_s = SIGMA_SCATT * mult
    geo     = make_geometry()

    all_true, all_false = [], []
    for ev_i in range(n_events):
        event = safe_generate(geo, n_tracks, collision_noise=sigma_s,
                              phi_max=angle, theta_max=angle)
        ta, fa = collect_segment_pair_angles(event)
        all_true.extend(ta)
        all_false.extend(fa)

    np.savez_compressed(
        str(outdir / "angles.npz"),
        true_angles=np.array(all_true, dtype=np.float64),
        false_angles=np.array(all_false, dtype=np.float64),
    )
    save_json({
        'task': 'char_hist',
        'angle': angle,
        'scatt_mult': mult,
        'n_tracks': n_tracks,
        'n_events': n_events,
        'n_true': len(all_true),
        'n_false': len(all_false),
    }, outdir / "results.json")


# ═══════════════════════════════════════════════════════════════════
#  Task: char_bulk_angles
#  100×50 bulk angle distributions (split into chunks)
# ═══════════════════════════════════════════════════════════════════
def run_char_bulk_angles(params, outdir):
    n_tracks    = params["n_tracks"]
    event_start = params["event_start"]
    event_end   = params["event_end"]

    geo = make_geometry()

    all_true, all_false = [], []
    for ev_i in range(event_start, event_end):
        event = safe_generate(geo, n_tracks)
        ta, fa = collect_segment_pair_angles(event)
        all_true.extend(ta)
        all_false.extend(fa)

    np.savez_compressed(
        str(outdir / "angles.npz"),
        true_angles=np.array(all_true, dtype=np.float64),
        false_angles=np.array(all_false, dtype=np.float64),
    )
    save_json({
        'task': 'char_bulk_angles',
        'n_tracks': n_tracks,
        'event_start': event_start,
        'event_end': event_end,
        'n_true': len(all_true),
        'n_false': len(all_false),
    }, outdir / "results.json")


# ═══════════════════════════════════════════════════════════════════
#  Task: hc_density
#  Hit-competition Steps 1-4 for one (angle, n_tracks) combo
#  Computes occupancy, activation, competition, and reco metrics
#  across n_repeats events
# ═══════════════════════════════════════════════════════════════════
def run_hc_density(params, outdir):
    angle     = params["angle"]
    n_tracks  = params["n_tracks"]
    n_repeats = params["n_repeats"]

    geo = make_geometry()

    # Accumulators
    occ_all = []          # Step 1: occupancy values
    true_x_all = []       # Step 2: activations of true segments
    false_x_all = []      # Step 2: activations of false segments
    comp_true_act = []    # Step 3: true segment activation at contested hits
    comp_false_sum = []   # Step 3: sum of false competitor activations
    comp_n_comp = []      # Step 3: number of false competitors
    reco_metrics = []     # Step 4: per-event reco metrics

    for rep in range(n_repeats):
        event = safe_generate(geo, n_tracks, phi_max=angle, theta_max=angle)
        ham, x, eps = build_and_solve(event)
        is_true = label_segments(ham, event)
        h2s = build_hit_to_segments(ham)

        # -- Step 1: Hit occupancy --
        for hit_id, seg_indices in h2s.items():
            occ_all.append(len(seg_indices))

        # -- Step 2: Activation spectrum --
        true_x_all.extend(x[is_true].tolist())
        false_x_all.extend(x[~is_true].tolist())

        # -- Step 3: Per-hit competition --
        for hit_id, seg_indices in h2s.items():
            if len(seg_indices) < 2:
                continue
            true_segs  = [s for s in seg_indices if is_true[s]]
            false_segs = [s for s in seg_indices if not is_true[s]]
            if len(true_segs) == 0 or len(false_segs) == 0:
                continue
            for ts in true_segs:
                comp_true_act.append(float(x[ts]))
                comp_false_sum.append(float(sum(x[fs] for fs in false_segs)))
                comp_n_comp.append(len(false_segs))

        # -- Step 4: Track-level reconstruction --
        reco_tracks, matches, metrics = reconstruct_and_validate(event, ham, x)
        reco_metrics.append({
            'angle':          angle,
            'n_tracks':       n_tracks,
            'repeat':         rep,
            'efficiency':     float(metrics.get('efficiency', 0.0)),
            'ghost_rate':     float(metrics.get('ghost_rate', 0.0)),
            'clone_fraction': float(metrics.get('clone_fraction', 0.0)),
            'n_reco':         len(reco_tracks),
            'n_truth':        len(event.tracks),
            'n_segments':     ham.n_segments,
        })

    # Save all data
    np.savez_compressed(
        str(outdir / "step1_occupancy.npz"),
        occupancy=np.array(occ_all, dtype=np.int32),
    )
    np.savez_compressed(
        str(outdir / "step2_activation.npz"),
        true_x=np.array(true_x_all, dtype=np.float64),
        false_x=np.array(false_x_all, dtype=np.float64),
    )
    np.savez_compressed(
        str(outdir / "step3_competition.npz"),
        true_act=np.array(comp_true_act, dtype=np.float64),
        false_sum=np.array(comp_false_sum, dtype=np.float64),
        n_competitors=np.array(comp_n_comp, dtype=np.int32),
    )
    save_json({
        'task': 'hc_density',
        'angle': angle,
        'n_tracks': n_tracks,
        'n_repeats': n_repeats,
        'reco_metrics': reco_metrics,
        'occ_mean': float(np.mean(occ_all)),
        'occ_median': float(np.median(occ_all)),
        'occ_max': int(np.max(occ_all)) if occ_all else 0,
        'occ_std': float(np.std(occ_all)),
    }, outdir / "results.json")


# ═══════════════════════════════════════════════════════════════════
#  Task: hc_roc
#  Step 5: Threshold sweep / ROC for one (angle, n_tracks) combo
#  Pre-solve events, then sweep thresholds
# ═══════════════════════════════════════════════════════════════════
def run_hc_roc(params, outdir):
    angle      = params["angle"]
    n_tracks   = params["n_tracks"]
    n_repeats  = params["n_repeats"]
    thresholds = np.array(params["thresholds"])

    geo = make_geometry()

    # Pre-solve events
    solved_events = []
    for rep in range(n_repeats):
        event = safe_generate(geo, n_tracks, phi_max=angle, theta_max=angle)
        ham, x, eps = build_and_solve(event)
        solved_events.append((event, ham, x))

    # Sweep thresholds
    eff_by_thr = []
    ghost_by_thr = []
    for thr in thresholds:
        effs, ghosts = [], []
        for event, ham, x in solved_events:
            reco_tracks, matches, metrics = reconstruct_and_validate(
                event, ham, x, threshold=float(thr))
            effs.append(metrics['efficiency'])
            ghosts.append(metrics['ghost_rate'])
        eff_by_thr.append(float(np.mean(effs)))
        ghost_by_thr.append(float(np.mean(ghosts)))

    save_json({
        'task': 'hc_roc',
        'angle': angle,
        'n_tracks': n_tracks,
        'n_repeats': n_repeats,
        'thresholds': thresholds.tolist(),
        'eff': eff_by_thr,
        'ghost': ghost_by_thr,
    }, outdir / "results.json")


# ═══════════════════════════════════════════════════════════════════
#  Task: hc_scatt
#  Step 6: Scattering scan for one scatt_mult
# ═══════════════════════════════════════════════════════════════════
def run_hc_scatt(params, outdir):
    mult        = params["scatt_mult"]
    track_sizes = params["track_sizes"]
    n_repeats   = params["n_repeats"]

    sigma_s = SIGMA_SCATT * mult
    eps     = compute_epsilon(SIGMA_RES, sigma_s, DZ_MM, scale=SCALE)
    geo     = make_geometry()

    results = []
    for n_trk in track_sizes:
        per_repeat = []
        for rep in range(n_repeats):
            event = safe_generate(geo, n_trk, collision_noise=sigma_s)
            true_angles, false_angles = collect_segment_pair_angles(event)
            ta = np.array(true_angles)
            fa = np.array(false_angles)
            t_acc = int(np.sum(ta <= eps))
            f_acc = int(np.sum(fa <= eps))
            tot = t_acc + f_acc
            per_repeat.append({
                'n_true': len(ta), 'n_false': len(fa),
                'true_acc': t_acc,  'false_acc': f_acc,
                'eff': t_acc / len(ta) if len(ta) else 0,
                'fr':  f_acc / tot     if tot     else 0,
            })

        effs = np.array([r['eff'] for r in per_repeat])
        frs  = np.array([r['fr']  for r in per_repeat])
        results.append({
            'n_tracks':       n_trk,
            'eff_mean':       float(np.mean(effs) * 100),
            'eff_se':         float(se(effs) * 100),
            'fr_mean':        float(np.mean(frs) * 100),
            'fr_se':          float(se(frs) * 100),
            'n_true_mean':    float(np.mean([r['n_true']  for r in per_repeat])),
            'n_true_se':      float(se([r['n_true']  for r in per_repeat])),
            'n_false_mean':   float(np.mean([r['n_false'] for r in per_repeat])),
            'n_false_se':     float(se([r['n_false'] for r in per_repeat])),
            'true_acc_mean':  float(np.mean([r['true_acc']  for r in per_repeat])),
            'true_acc_se':    float(se([r['true_acc']  for r in per_repeat])),
            'false_acc_mean': float(np.mean([r['false_acc'] for r in per_repeat])),
            'false_acc_se':   float(se([r['false_acc'] for r in per_repeat])),
        })

    save_json({
        'task': 'hc_scatt',
        'scatt_mult': mult,
        'epsilon': eps,
        'results': results,
    }, outdir / "results.json")


# ═══════════════════════════════════════════════════════════════════
#  Task: hc_scatt_hist
#  Step 6: Scattering histograms for one scatt_mult
# ═══════════════════════════════════════════════════════════════════
def run_hc_scatt_hist(params, outdir):
    mult     = params["scatt_mult"]
    n_tracks = params["n_tracks"]
    n_events = params["n_events"]

    sigma_s = SIGMA_SCATT * mult
    geo     = make_geometry()

    all_true, all_false = [], []
    for ev_i in range(n_events):
        event = safe_generate(geo, n_tracks, collision_noise=sigma_s)
        ta, fa = collect_segment_pair_angles(event)
        all_true.extend(ta)
        all_false.extend(fa)

    np.savez_compressed(
        str(outdir / "angles.npz"),
        true_angles=np.array(all_true, dtype=np.float64),
        false_angles=np.array(all_false, dtype=np.float64),
    )
    save_json({
        'task': 'hc_scatt_hist',
        'scatt_mult': mult,
        'n_tracks': n_tracks,
        'n_events': n_events,
        'n_true': len(all_true),
        'n_false': len(all_false),
    }, outdir / "results.json")


# ═══════════════════════════════════════════════════════════════════
#  Dispatcher
# ═══════════════════════════════════════════════════════════════════
TASK_DISPATCH = {
    'char_scan':        run_char_scan,
    'char_hist':        run_char_hist,
    'char_bulk_angles': run_char_bulk_angles,
    'hc_density':       run_hc_density,
    'hc_roc':           run_hc_roc,
    'hc_scatt':         run_hc_scatt,
    'hc_scatt_hist':    run_hc_scatt_hist,
}


def main():
    parser = argparse.ArgumentParser(description="Condor pipeline worker")
    parser.add_argument("--params-json", required=True,
                        help="Path to job parameter JSON file")
    parser.add_argument("--outdir", required=True,
                        help="Output directory for this job's results")
    args = parser.parse_args()

    params_path = Path(args.params_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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

        # Write a completion marker
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
