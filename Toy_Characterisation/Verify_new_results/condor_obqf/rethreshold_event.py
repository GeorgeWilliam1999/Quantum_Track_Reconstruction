#!/usr/bin/env python3
"""Rethreshold a saved §14e/§18 quantum event at a new τ (no quantum solve).

Loads the existing pickle for (n_trk, rep, readout), reuses the stored
`sol_C` and `sol_Q_scaled`, regenerates the event deterministically with
the same seed as `run_event.py`, rebuilds the Hamiltonian (only the
segment skeleton — no solve), then runs the connected-component and
layered trackers at τ_abs = τ_rel * max(sol).  Writes a new pickle to
--outdir.

Designed for CPU-only Condor: the only cost is event regen + segment
construction + tracker, which is small compared to the quantum solve.
"""
from __future__ import annotations
import argparse, os, pickle, time, sys
from pathlib import Path
import numpy as np


def _import_toy():
    from lhcb_velo_toy.solvers.hamiltonians.fast import SimpleHamiltonianFast
    from lhcb_velo_toy.solvers.reconstruction.track_finder import (
        get_tracks, get_tracks_layered,
    )
    from lhcb_velo_toy.analysis.validation.validator import EventValidator
    return SimpleHamiltonianFast, get_tracks, get_tracks_layered, EventValidator


def _safe_generate(n_trk, measurement_error, collision_noise, n_modules=5,
                   dz_mm=33.0, max_attempts=20,
                   phi_max=0.2, theta_max=0.2):
    """Mirrors run_event.py exactly (same seed path)."""
    from lhcb_velo_toy.generation.generators.state_event import StateEventGenerator
    from lhcb_velo_toy.generation.geometry.plane import PlaneGeometry
    HALF_X = 40.0; HALF_Y = 40.0; Z_FIRST = 33.0
    PV_SIGMA = {"x": 0, "y": 0, "z": 1}
    z_positions = [Z_FIRST + i * dz_mm for i in range(n_modules)]
    geo = PlaneGeometry(
        module_id=list(range(n_modules)),
        lx=[HALF_X] * n_modules, ly=[HALF_Y] * n_modules, z=z_positions,
    )
    last = None
    for _ in range(max_attempts):
        gen = StateEventGenerator(
            detector_geometry=geo, events=1, n_particles=[n_trk],
            phi_min=-phi_max, phi_max=phi_max,
            theta_min=-theta_max, theta_max=theta_max,
            measurement_error=measurement_error,
            collision_noise=collision_noise,
        )
        gen.generate_random_primary_vertices(PV_SIGMA)
        gen.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * n_trk])
        event = gen.generate_complete_events()
        last = event
        if event is not None and event.tracks and \
           min(len(t.hit_ids) for t in event.tracks) >= 3:
            return event
    if last is None:
        raise RuntimeError(f"Failed to generate event for n_trk={n_trk}")
    return last


def _apply_hit_noise(event, drop_rate=0.0, rng=None):
    import copy
    if rng is None: rng = np.random.default_rng()
    evt = copy.deepcopy(event)
    if drop_rate > 0.0:
        kept = set()
        new_hits = []
        for h in evt.hits:
            if rng.random() >= drop_rate:
                kept.add(h.hit_id); new_hits.append(h)
        evt.hits = new_hits
        for trk in evt.tracks:
            trk.hit_ids = [hid for hid in trk.hit_ids if hid in kept]
        for mod in evt.modules:
            mod.hit_ids = [hid for hid in mod.hit_ids if hid in kept]
    return evt


def run_job(n_trk, rep, readout, tau_rel, srcdir, outdir,
            epsilon=2e-3, gamma=3.0, delta=1.0,
            sigma_res=5e-6, sigma_scatt=1e-4, drop_rate=0.01,
            purity_min=0.7, min_rec_hits=3):
    SimpleHamiltonianFast, get_tracks, get_tracks_layered, EventValidator = _import_toy()

    # ── load source pickle (provides sol_C, sol_Q_scaled, and original metrics)
    src = Path(srcdir) / f'event_n{n_trk:04d}_rep{rep:03d}.pkl'
    if not src.exists():
        raise FileNotFoundError(src)
    with open(src, 'rb') as f:
        prev = pickle.load(f)
    sol_C = np.asarray(prev['sol_C'], dtype=float)
    sol_Q = np.asarray(prev['sol_Q_scaled'], dtype=float)

    # ── regenerate event (same deterministic seed as run_event.py)
    np.random.seed(rep + n_trk * 100 + 140000)
    ev_clean = _safe_generate(n_trk,
                              measurement_error=sigma_res,
                              collision_noise=sigma_scatt)
    if drop_rate > 0.0:
        rng = np.random.default_rng(rep + n_trk * 100 + 240000)
        ev = _apply_hit_noise(ev_clean, drop_rate=drop_rate, rng=rng)
    else:
        ev = ev_clean

    # ── build ham (need ham.segments / adjacency for trackers; no solve)
    ham = SimpleHamiltonianFast(epsilon=epsilon, gamma=gamma, delta=delta)
    A_sparse, b = ham.construct_hamiltonian(ev, convolution=False)

    # Sanity: shapes must match the stored solution vectors
    if A_sparse.shape[0] != len(sol_C) or A_sparse.shape[0] != len(sol_Q):
        raise RuntimeError(
            f"shape mismatch n_seg={A_sparse.shape[0]} vs sol_C={len(sol_C)} sol_Q={len(sol_Q)}")

    # ── per-T thresholds
    tau_C = tau_rel * float(np.max(sol_C)) if sol_C.size else 0.0
    tau_Q = tau_rel * float(np.max(sol_Q)) if sol_Q.size else 0.0

    def _trk(sol, fn, kwargs):
        rec = fn(ham, sol, ev, **kwargs)
        _, m = EventValidator(ev, rec).match_tracks(
            purity_min=purity_min, hit_efficiency_min=0.0,
            min_rec_hits=min_rec_hits)
        return dict(efficiency=float(m['efficiency']),
                    ghost_rate=float(m['ghost_rate']),
                    clone_fraction=float(m['clone_fraction']),
                    mean_purity=float(m['mean_purity']),
                    hit_efficiency=float(m['hit_efficiency']),
                    n_rec_tracks=int(len(rec)))

    t0 = time.time()
    metrics = {
        'trk_C_cc':  _trk(sol_C, get_tracks,         dict(threshold=tau_C)),
        'trk_C_lay': _trk(sol_C, get_tracks_layered, dict(threshold=tau_C, min_hits=min_rec_hits)),
        'trk_Q_cc':  _trk(sol_Q, get_tracks,         dict(threshold=tau_Q)),
        'trk_Q_lay': _trk(sol_Q, get_tracks_layered, dict(threshold=tau_Q, min_hits=min_rec_hits)),
    }
    t_tracker = time.time() - t0

    out = dict(
        n_trk=n_trk, rep=rep,
        readout=readout, tau_rel=float(tau_rel),
        tau_abs_C=float(tau_C), tau_abs_Q=float(tau_Q),
        qmax_C=float(np.max(sol_C)) if sol_C.size else 0.0,
        qmax_Q=float(np.max(sol_Q)) if sol_Q.size else 0.0,
        n_seg=int(A_sparse.shape[0]),
        n_hits=int(prev.get('n_hits', sum(len(m.hit_ids) for m in ev.modules))),
        drop_rate=float(drop_rate),
        t_tracker=t_tracker,
        # Pass through baseline metrics for direct comparison
        prev_trk_C_cc=prev.get('trk_C_cc'),
        prev_trk_C_lay=prev.get('trk_C_lay'),
        prev_trk_Q_cc=prev.get('trk_Q_cc'),
        prev_trk_Q_lay=prev.get('trk_Q_lay'),
        **metrics,
    )
    fname = Path(outdir) / f'event_n{n_trk:04d}_rep{rep:03d}.pkl'
    fname.parent.mkdir(parents=True, exist_ok=True)
    with open(fname, 'wb') as f:
        pickle.dump(out, f)
    print(f"[ok] {fname.name}  τ_rel={tau_rel:.3f}  "
          f"eff_Q_lay={100*metrics['trk_Q_lay']['efficiency']:.1f}%  "
          f"ghost_Q_lay={100*metrics['trk_Q_lay']['ghost_rate']:.1f}%  "
          f"({t_tracker:.1f}s tracker)")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-trk', type=int, required=True)
    p.add_argument('--rep',   type=int, required=True)
    p.add_argument('--readout', type=str, required=True,
                   choices=['sampling', 'statevector'])
    p.add_argument('--tau-rel', type=float, required=True)
    p.add_argument('--srcdir', type=str, required=True)
    p.add_argument('--outdir', type=str, required=True)
    p.add_argument('--drop-rate', type=float, default=0.01)
    p.add_argument('--gamma', type=float, default=3.0)
    args = p.parse_args()

    fname = Path(args.outdir) / f'event_n{args.n_trk:04d}_rep{args.rep:03d}.pkl'
    if fname.exists():
        print(f"[skip] {fname.name} already exists")
        return 0
    try:
        return run_job(args.n_trk, args.rep, args.readout, args.tau_rel,
                       args.srcdir, args.outdir,
                       gamma=args.gamma, drop_rate=args.drop_rate)
    except Exception as e:
        import traceback
        err = Path(args.outdir) / f'event_n{args.n_trk:04d}_rep{args.rep:03d}.err'
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(f"{e}\n\n{traceback.format_exc()}")
        print(f"[err] {fname.name}: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
