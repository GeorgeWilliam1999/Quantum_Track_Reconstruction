#!/usr/bin/env python3
"""
Worker script for the segment-grass ghost/drop-rate sweep.

Reads a single job parameter JSON and runs the occupancy / activation /
reconstruction analysis for one (ghost_rate, drop_rate, angle, n_tracks)
point, repeating over N events.

Output per job (saved as .npz + .json):
  - per-module hit counts and occupancy arrays
  - true/false activation spectra
  - reconstruction metrics (efficiency, ghost_rate, clone_fraction)
  - number of clipped tracks per event
  - PV z values
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np

from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers import SimpleHamiltonianFast, get_tracks
from lhcb_velo_toy.analysis import EventValidator

# ── Physics constants (match the main pipeline) ──────────────────
SIGMA_RES    = 0.0
SIGMA_SCATT  = 1e-4
DZ_MM        = 33.0
SCALE        = 3.0
GAMMA        = 1.5
DELTA        = 1.0
N_MODULES    = 5
Z_FIRST      = 100.0
HALF_XY      = 50.0
PV_SIGMA_Z   = 50.0

BASELINE  = DELTA / (DELTA + GAMMA)
THRESHOLD = (1 + BASELINE) / 2


def compute_epsilon():
    theta_s = SCALE * SIGMA_SCATT
    theta_r = np.arctan((SCALE * SIGMA_RES) / DZ_MM) if DZ_MM != 0 else 0.0
    return float(np.sqrt(2 * theta_s**2 + 12 * theta_r**2 + 2 * (1.5e-5)**2))

EPSILON = compute_epsilon()


def make_geometry():
    z_positions = [Z_FIRST + i * DZ_MM for i in range(N_MODULES)]
    return PlaneGeometry(
        module_id=list(range(N_MODULES)),
        lx=[HALF_XY] * N_MODULES,
        ly=[HALF_XY] * N_MODULES,
        z=z_positions,
    )


def label_segments(ham, event):
    true_seg_set = set()
    for trk in event.tracks:
        hids = trk.hit_ids
        for k in range(len(hids) - 1):
            true_seg_set.add((hids[k], hids[k + 1]))
    return np.array([
        ham._segment_to_hit_ids[i] in true_seg_set
        for i in range(ham.n_segments)
    ])


def build_hit_to_segments(ham):
    h2s = defaultdict(list)
    for idx, (fid, tid) in enumerate(ham._segment_to_hit_ids):
        h2s[fid].append(idx)
        h2s[tid].append(idx)
    return dict(h2s)


def run_job(params, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    angle    = params["angle"]
    n_tracks = params["n_tracks"]
    ghost_rate = params["ghost_rate"]
    drop_rate  = params["drop_rate"]
    n_repeats  = params["n_repeats"]

    geo = make_geometry()

    # Accumulators
    all_pv_z          = []
    all_n_clipped     = []
    all_occ           = []           # flat list of occupancy values (full corpus)
    all_occ_by_mod    = defaultdict(list)
    all_hits_per_mod  = defaultdict(list)
    all_true_x        = []
    all_false_x       = []
    reco_metrics_list = []
    all_n_segments    = []
    all_n_ghost_hits  = []
    all_n_dropped     = []
    # Selected-segment corpus (segments with x_i > THRESHOLD) ──
    all_occ_sel          = []
    all_occ_sel_by_mod   = defaultdict(list)
    all_hits_sel_per_mod = defaultdict(list)
    all_true_x_sel       = []
    all_false_x_sel      = []
    all_n_segments_sel   = []
    all_n_true_sel       = []
    all_n_false_sel      = []

    for rep in range(n_repeats):
        # Generate clean event
        gen = StateEventGenerator(
            detector_geometry=geo, events=1, n_particles=[n_tracks],
            phi_min=-angle, phi_max=angle,
            theta_min=-angle, theta_max=angle,
            measurement_error=0.0, collision_noise=1e-8,
        )
        gen.generate_random_primary_vertices(
            {"x": 0.1, "y": 0.1, "z": PV_SIGMA_Z})
        particles = [[{"type": "pion", "mass": 139.6, "q": 1}] * n_tracks]
        gen.generate_particles(particles)
        clean_event = gen.generate_complete_events()

        pv_z = gen.primary_vertices[0][2]
        all_pv_z.append(pv_z)

        # Count clipped tracks (truth-level, before noise)
        n_clipped = sum(1 for t in clean_event.tracks
                        if len(t.hit_ids) < N_MODULES)
        all_n_clipped.append(n_clipped)

        # Apply noise
        if ghost_rate > 0 or drop_rate > 0:
            event = gen.make_noisy_event(
                drop_rate=drop_rate, ghost_rate=ghost_rate)
        else:
            event = clean_event

        n_total_hits = len(event.hits)
        n_ghost_hits = sum(1 for h in event.hits if h.track_id == -1)
        n_dropped = len(clean_event.hits) - sum(
            1 for h in event.hits if h.track_id != -1)
        all_n_ghost_hits.append(n_ghost_hits)
        all_n_dropped.append(max(n_dropped, 0))

        # Hits per module
        hpm = defaultdict(int)
        for h in event.hits:
            hpm[h.module_id] += 1
        for m in range(N_MODULES):
            all_hits_per_mod[m].append(hpm[m])

        # Build Hamiltonian and solve
        ham = SimpleHamiltonianFast(epsilon=EPSILON, gamma=GAMMA, delta=DELTA)
        ham.construct_hamiltonian(event)
        all_n_segments.append(ham.n_segments)

        x = ham.solve_classicaly()

        # Occupancy per hit
        h2s = build_hit_to_segments(ham)
        hit_mod = {h.hit_id: h.module_id for h in event.hits}
        for hid, segs in h2s.items():
            occ = len(segs)
            all_occ.append(occ)
            all_occ_by_mod[hit_mod[hid]].append(occ)

        # Activation spectrum
        is_true = label_segments(ham, event)
        all_true_x.extend(x[is_true].tolist())
        all_false_x.extend(x[~is_true].tolist())

        # ── Selected-segment corpus (x > THRESHOLD) ───────────────
        is_selected = x > THRESHOLD
        n_sel = int(is_selected.sum())
        all_n_segments_sel.append(n_sel)
        all_n_true_sel.append(int((is_selected & is_true).sum()))
        all_n_false_sel.append(int((is_selected & ~is_true).sum()))
        all_true_x_sel.extend(x[is_selected & is_true].tolist())
        all_false_x_sel.extend(x[is_selected & ~is_true].tolist())

        # Per-hit occupancy restricted to selected segments
        # (hits with zero selected segments are excluded from the corpus)
        h2s_sel = defaultdict(int)
        for seg_idx in np.flatnonzero(is_selected):
            fid, tid = ham._segment_to_hit_ids[seg_idx]
            h2s_sel[fid] += 1
            h2s_sel[tid] += 1
        # Per-module hit count in selected corpus
        hits_sel_per_mod = defaultdict(int)
        for hid, occ_sel in h2s_sel.items():
            mod = hit_mod[hid]
            all_occ_sel.append(occ_sel)
            all_occ_sel_by_mod[mod].append(occ_sel)
            hits_sel_per_mod[mod] += 1
        for m in range(N_MODULES):
            all_hits_sel_per_mod[m].append(hits_sel_per_mod[m])

        # Reconstruction
        try:
            reco_tracks = get_tracks(ham, x, event, threshold=THRESHOLD)
            if len(reco_tracks) > 0:
                val = EventValidator(event, reco_tracks)
                _, metrics = val.match_tracks(purity_min=0.7)
                reco_metrics_list.append({
                    "efficiency": metrics.get("efficiency", 0.0),
                    "ghost_rate": metrics.get("ghost_rate", 0.0),
                    "clone_fraction": metrics.get("clone_fraction", 0.0),
                    "n_reco": len(reco_tracks),
                })
            else:
                reco_metrics_list.append({
                    "efficiency": 0.0, "ghost_rate": 0.0,
                    "clone_fraction": 0.0, "n_reco": 0,
                })
        except Exception:
            reco_metrics_list.append({
                "efficiency": 0.0, "ghost_rate": 0.0,
                "clone_fraction": 0.0, "n_reco": 0,
            })

    # ── Save results ──────────────────────────────────────────────
    # Arrays
    np.savez_compressed(
        outdir / "arrays.npz",
        occ_all=np.array(all_occ),
        **{f"occ_mod{m}": np.array(all_occ_by_mod[m])
           for m in range(N_MODULES)},
        **{f"hits_mod{m}": np.array(all_hits_per_mod[m])
           for m in range(N_MODULES)},
        true_x=np.array(all_true_x),
        false_x=np.array(all_false_x),
        pv_z=np.array(all_pv_z),
        n_clipped=np.array(all_n_clipped),
        n_segments=np.array(all_n_segments),
        n_ghost_hits=np.array(all_n_ghost_hits),
        n_dropped=np.array(all_n_dropped),
        # Selected-segment corpus
        occ_sel_all=np.array(all_occ_sel),
        **{f"occ_sel_mod{m}": np.array(all_occ_sel_by_mod[m])
           for m in range(N_MODULES)},
        **{f"hits_sel_mod{m}": np.array(all_hits_sel_per_mod[m])
           for m in range(N_MODULES)},
        true_x_sel=np.array(all_true_x_sel),
        false_x_sel=np.array(all_false_x_sel),
        n_segments_sel=np.array(all_n_segments_sel),
        n_true_sel=np.array(all_n_true_sel),
        n_false_sel=np.array(all_n_false_sel),
    )

    # Summary JSON
    mean_metrics = {}
    for key in ["efficiency", "ghost_rate", "clone_fraction", "n_reco"]:
        vals = [m[key] for m in reco_metrics_list]
        mean_metrics[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    occ_arr = np.array(all_occ)
    expected = {n_tracks, 2 * n_tracks}
    n_anomalous = int(np.sum(~np.isin(occ_arr, list(expected))))

    true_x_arr = np.array(all_true_x)
    false_x_arr = np.array(all_false_x)

    occ_sel_arr = np.array(all_occ_sel)
    expected_sel = {1, 2}
    n_anomalous_sel = int(np.sum(~np.isin(occ_sel_arr, list(expected_sel)))) \
        if len(occ_sel_arr) > 0 else 0

    summary = {
        "params": params,
        "n_events": n_repeats,
        "occ_mean": float(np.mean(occ_arr)),
        "occ_std": float(np.std(occ_arr)),
        "occ_max": int(np.max(occ_arr)) if len(occ_arr) > 0 else 0,
        "n_anomalous_occ": n_anomalous,
        "frac_anomalous_occ": float(n_anomalous / len(occ_arr))
            if len(occ_arr) > 0 else 0.0,
        "mean_n_clipped": float(np.mean(all_n_clipped)),
        "mean_n_ghost_hits": float(np.mean(all_n_ghost_hits)),
        "mean_n_dropped": float(np.mean(all_n_dropped)),
        "mean_n_segments": float(np.mean(all_n_segments)),
        "mean_true_activation": float(np.mean(true_x_arr))
            if len(true_x_arr) > 0 else 0.0,
        "mean_false_activation": float(np.mean(false_x_arr))
            if len(false_x_arr) > 0 else 0.0,
        # Selected-segment corpus summary
        "occ_sel_mean": float(np.mean(occ_sel_arr))
            if len(occ_sel_arr) > 0 else 0.0,
        "occ_sel_std": float(np.std(occ_sel_arr))
            if len(occ_sel_arr) > 0 else 0.0,
        "occ_sel_max": int(np.max(occ_sel_arr))
            if len(occ_sel_arr) > 0 else 0,
        "n_anomalous_occ_sel": n_anomalous_sel,
        "frac_anomalous_occ_sel": float(n_anomalous_sel / len(occ_sel_arr))
            if len(occ_sel_arr) > 0 else 0.0,
        "mean_n_segments_sel": float(np.mean(all_n_segments_sel)),
        "mean_n_true_sel": float(np.mean(all_n_true_sel)),
        "mean_n_false_sel": float(np.mean(all_n_false_sel)),
        "mean_true_activation_sel": float(np.mean(all_true_x_sel))
            if len(all_true_x_sel) > 0 else 0.0,
        "mean_false_activation_sel": float(np.mean(all_false_x_sel))
            if len(all_false_x_sel) > 0 else 0.0,
        "reco_metrics": mean_metrics,
        "hits_per_mod_mean": {
            str(m): float(np.mean(all_hits_per_mod[m]))
            for m in range(N_MODULES)
        },
        "hits_sel_per_mod_mean": {
            str(m): float(np.mean(all_hits_sel_per_mod[m]))
            for m in range(N_MODULES)
        },
    }

    with open(outdir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Marker
    (outdir / "DONE").touch()
    print(f"Job {params['job_id']} complete: "
          f"angle=±{angle}, n_trk={n_tracks}, "
          f"ghost={ghost_rate}, drop={drop_rate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-json", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    with open(args.params_json) as f:
        params = json.load(f)

    try:
        run_job(params, args.outdir)
    except Exception as e:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "FAILED", "w") as f:
            f.write(f"{e}\n")
            traceback.print_exc(file=f)
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
