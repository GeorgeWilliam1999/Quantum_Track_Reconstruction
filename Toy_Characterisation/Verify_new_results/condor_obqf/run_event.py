#!/usr/bin/env python3
"""
§18b/c condor worker — 1BQF evaluation on a single (n_trk, rep) event.

Each condor job runs this script once and writes one pickle to --outdir.
Aggregation is done back in the notebook (§18b/c plotting cells).

Arguments
---------
--n-trk     : int, truth track count
--rep       : int, repeat index (drives seed  rep + n_trk*100 + 140000)
--shots     : int, shots for the 1BQF simulator (default 8192)
--device    : 'CPU' or 'GPU'
--outdir    : directory for result pickle (one file per job)
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _import_toy():
    """Late import so we can print a clean error if env is wrong."""
    from lhcb_velo_toy.solvers.hamiltonians.fast import SimpleHamiltonianFast
    from lhcb_velo_toy.solvers.quantum.one_bit_hhl import OneBitHHL
    from lhcb_velo_toy.solvers.reconstruction.track_finder import (
        get_tracks, get_tracks_layered,
    )
    from lhcb_velo_toy.analysis.validation.validator import EventValidator
    return (SimpleHamiltonianFast, OneBitHHL, get_tracks,
            get_tracks_layered, EventValidator)


def _safe_generate(n_trk, measurement_error, collision_noise, n_modules=5,
                   dz_mm=33.0, max_attempts=20,
                   phi_max=0.2, theta_max=0.2):
    """Replicates the §14 notebook safe_generate exactly."""
    from lhcb_velo_toy.generation.generators.state_event import StateEventGenerator
    from lhcb_velo_toy.generation.geometry.plane import PlaneGeometry
    HALF_X = 40.0
    HALF_Y = 40.0
    Z_FIRST = 33.0
    PV_SIGMA = {"x": 0, "y": 0, "z": 1}
    z_positions = [Z_FIRST + i * dz_mm for i in range(n_modules)]
    geo = PlaneGeometry(
        module_id=list(range(n_modules)),
        lx=[HALF_X] * n_modules,
        ly=[HALF_Y] * n_modules,
        z=z_positions,
    )
    last = None
    for _ in range(max_attempts):
        gen = StateEventGenerator(
            detector_geometry=geo,
            events=1,
            n_particles=[n_trk],
            phi_min=-phi_max, phi_max=phi_max,
            theta_min=-theta_max, theta_max=theta_max,
            measurement_error=measurement_error,
            collision_noise=collision_noise,
        )
        gen.generate_random_primary_vertices(PV_SIGMA)
        gen.generate_particles(
            [[{"type": "pion", "mass": 139.6, "q": 1}] * n_trk]
        )
        event = gen.generate_complete_events()
        last = event
        if event is not None and event.tracks and \
           min(len(t.hit_ids) for t in event.tracks) >= 3:
            return event
    if last is None:
        raise RuntimeError(f"Failed to generate event for n_trk={n_trk}")
    return last  # best effort


def _apply_hit_noise(event, drop_rate=0.0, rng=None):
    """Mirror of notebook cell 63 apply_hit_noise (drop only)."""
    import copy
    if rng is None:
        rng = np.random.default_rng()
    evt = copy.deepcopy(event)
    if drop_rate > 0.0:
        kept_ids = set()
        new_hits = []
        for h in evt.hits:
            if rng.random() >= drop_rate:
                kept_ids.add(h.hit_id)
                new_hits.append(h)
        evt.hits = new_hits
        for trk in evt.tracks:
            trk.hit_ids = [hid for hid in trk.hit_ids if hid in kept_ids]
        for mod in evt.modules:
            mod.hit_ids = [hid for hid in mod.hit_ids if hid in kept_ids]
    return evt


def _seg_truth(ham):
    # Fast path: use cached (n_seg, 2) int64 array of [start_tid, end_tid]
    # populated by the vectorised construct_segments.  Falls back to the old
    # per-Segment Python loop for older builds without that attribute.
    arr = getattr(ham, "_segment_track_ids", None)
    if arr is not None and len(arr) == ham.n_segments:
        return (arr[:, 0] == arr[:, 1]) & (arr[:, 0] >= 0)
    return np.array([s.hit_start.track_id == s.hit_end.track_id
                     and s.hit_start.track_id >= 0
                     for s in ham.segments])


def _seg_metrics(sol, ham, threshold, n_true_clean):
    """§14-style segment metrics for a given solution vector."""
    truth = _seg_truth(ham)
    active = np.asarray(sol) > threshold
    n_true_all = int(truth.sum())
    n_false_all = int(ham.n_segments - n_true_all)
    n_true_active = int((active & truth).sum())
    n_active = int(active.sum())
    n_false_active = int(n_active - n_true_active)
    return dict(
        n_true_all=n_true_all,
        n_true_clean=int(n_true_clean) if n_true_clean is not None else n_true_all,
        n_false_all=n_false_all,
        n_true_active=n_true_active,
        n_false_active=n_false_active,
        n_active=n_active,
        n_seg=int(ham.n_segments),
    )


def _count_true_segments(event, epsilon, gamma, delta):
    from lhcb_velo_toy.solvers.hamiltonians.fast import SimpleHamiltonianFast
    h = SimpleHamiltonianFast(epsilon=epsilon, gamma=gamma, delta=delta)
    # Lazy path: only the cached track-id array is needed for the count, so
    # skip the O(N) Python Segment object materialisation when supported.
    try:
        h.construct_segments(event, materialize_segments=False)
    except TypeError:
        h.construct_segments(event)
    return int(_seg_truth(h).sum())


def run_job(n_trk: int, rep: int, shots: int, device: str,
            epsilon: float = 2e-3, gamma: float = 3.0, delta: float = 1.0,
            sigma_res: float = 0.005, sigma_scatt: float = 0.0001,
            threshold: float = 0.35, purity_min: float = 0.7,
            min_rec_hits: int = 3, drop_rate: float = 0.0,
            readout: str = 'sampling'):
    (SimpleHamiltonianFast, OneBitHHL, get_tracks,
     get_tracks_layered, EventValidator) = _import_toy()

    np.random.seed(rep + n_trk * 100 + 140000)
    ev_clean = _safe_generate(n_trk,
                              measurement_error=sigma_res,
                              collision_noise=sigma_scatt)

    # Efficiency denominator: true segments on the *clean* event (§14b convention).
    n_true_clean = _count_true_segments(ev_clean, epsilon, gamma, delta)

    if drop_rate > 0.0:
        rng = np.random.default_rng(rep + n_trk * 100 + 240000)
        ev = _apply_hit_noise(ev_clean, drop_rate=drop_rate, rng=rng)
    else:
        ev = ev_clean
    n_hits = sum(len(m.hit_ids) for m in ev.modules)

    ham = SimpleHamiltonianFast(epsilon=epsilon, gamma=gamma, delta=delta)
    # Library builder: returns sparse CSC A and dense b in one shot.
    # Note: SimpleHamiltonianFast.construct_hamiltonian returns -A internally
    # so the resulting (A, b) is already in the correct sign convention
    # (delta+gamma on the diagonal, -1 on connected off-diagonals — i.e. the
    # SAME matrix produced by the previous _build_ham_vectorized helper).
    A_sparse, b = ham.construct_hamiltonian(ev, convolution=False)
    n_seg = A_sparse.shape[0]

    t_c0 = time.time()
    sol_C = ham.solve_classicaly()
    t_classical = time.time() - t_c0

    # Quantum solve — pass the sparse CSC A directly. OneBQF.__init__ has a
    # sparse-aware branch that avoids O(n_seg^2) host allocations (previously
    # A_padded and B = c*I - A both blew up host RAM for n_seg >~ 1e5).
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    t_q0 = time.time()
    hhl = OneBitHHL(A_sparse, b, num_time_qubits=1, shots=shots, debug=False,
                    readout=readout)
    hhl.build_circuit()
    if device.upper() == 'GPU':
        # Auto-detect the number of GPUs Condor assigned to this job (via
        # CUDA_VISIBLE_DEVICES).  With >=2 GPUs we enable Aer's multi-device
        # statevector path (cuStateVec + blocking) and drop to single
        # precision to roughly double the per-GPU reach.  For a single GPU
        # we keep the original config so the small-T sweep is unaffected.
        _cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        n_gpus = len([x for x in _cvd.split(',') if x.strip() != '']) if _cvd else 1
        if n_gpus > 1:
            sim = AerSimulator(method='statevector', device='GPU',
                               cuStateVec_enable=True,
                               blocking_enable=True,
                               blocking_qubits=30,
                               precision='single')
            print(f"[aer] multi-GPU statevector: n_gpus={n_gpus}, "
                  f"blocking_qubits=30, precision=single")
        else:
            sim = AerSimulator(method='statevector', device='GPU')
    else:
        sim = AerSimulator(method='statevector', device='CPU')
    tqc = transpile(hhl.circuit, sim, optimization_level=1)
    if readout == 'statevector':
        # Exact, shot-free readout: run once, pull the full statevector.
        job = sim.run(tqc, shots=1)
        result = job.result()
        sv = np.asarray(result.data(0)['statevector'])
        sol_Q_raw, p_success = hhl.get_solution_from_statevector(sv)
        # success_prob is the ancilla=1 branch probability mass (continuous).
        n_success = p_success
    else:
        job = sim.run(tqc, shots=shots)
        hhl.counts = job.result().get_counts()
        sol_Q_raw, n_success = hhl.get_solution()
    t_quantum = time.time() - t_q0

    nC = float(np.linalg.norm(sol_C))
    nQ = float(np.linalg.norm(sol_Q_raw))
    sol_Q_scaled = sol_Q_raw * (nC / nQ) if nQ > 0 else np.zeros_like(sol_Q_raw)

    # Fidelity
    denom = nC * nQ
    cos = float(np.dot(sol_C, sol_Q_raw) / denom) if denom > 0 else 0.0
    rel_l2 = float(np.linalg.norm(sol_Q_scaled - sol_C) / max(nC, 1e-12))
    tauC = sol_C > threshold
    tauQ = sol_Q_scaled > threshold
    inter = int(np.sum(tauC & tauQ))
    union = int(np.sum(tauC | tauQ))
    jaccard = inter / union if union > 0 else 1.0

    # Track-level metrics for both trackers × both solutions
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

    kw_cc  = dict(threshold=threshold)
    kw_lay = dict(threshold=threshold, min_hits=min_rec_hits)
    metrics = {
        'trk_C_cc':  _trk(sol_C,        get_tracks,         kw_cc),
        'trk_C_lay': _trk(sol_C,        get_tracks_layered, kw_lay),
        'trk_Q_cc':  _trk(sol_Q_scaled, get_tracks,         kw_cc),
        'trk_Q_lay': _trk(sol_Q_scaled, get_tracks_layered, kw_lay),
    }

    # §14-style segment-level metrics for both solutions, w/ clean-event denom
    seg_C = _seg_metrics(sol_C,        ham, threshold, n_true_clean)
    seg_Q = _seg_metrics(sol_Q_scaled, ham, threshold, n_true_clean)

    return dict(
        n_trk=n_trk, rep=rep,
        n_hits=n_hits, n_seg=n_seg,
        shots=shots, device=device,
        readout=readout,
        drop_rate=float(drop_rate),
        success_prob=(n_success / shots) if readout == 'sampling' else float(n_success),
        t_classical=t_classical, t_quantum=t_quantum,
        sol_C=sol_C, sol_Q_raw=sol_Q_raw, sol_Q_scaled=sol_Q_scaled,
        fidelity=dict(cos=cos, rel_l2=rel_l2, jaccard=jaccard,
                      n_active_C=int(tauC.sum()), n_active_Q=int(tauQ.sum())),
        seg_C=seg_C, seg_Q=seg_Q,
        **metrics,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-trk', type=int, required=True)
    p.add_argument('--rep',   type=int, required=True)
    p.add_argument('--shots', type=int, default=8192)
    p.add_argument('--device', type=str, default='CPU', choices=['CPU', 'GPU'])
    p.add_argument('--drop-rate', type=float, default=0.0)
    p.add_argument('--gamma', type=float, default=3.0)
    p.add_argument('--readout', type=str, default='sampling',
                   choices=['sampling', 'statevector'])
    p.add_argument('--outdir', type=str, required=True)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f'event_n{args.n_trk:04d}_rep{args.rep:03d}.pkl'

    if fname.exists():
        print(f"[skip] {fname.name} already exists")
        return 0

    t0 = time.time()
    try:
        result = run_job(args.n_trk, args.rep, args.shots, args.device,
                         gamma=args.gamma, drop_rate=args.drop_rate,
                         readout=args.readout)
        result['wall_total'] = time.time() - t0
        result['host'] = os.uname().nodename
        with open(fname, 'wb') as f:
            pickle.dump(result, f)
        print(f"[ok] {fname.name}  "
              f"cos={result['fidelity']['cos']:.3f}  "
              f"eff_Q_lay={100*result['trk_Q_lay']['efficiency']:.1f}%  "
              f"({result['wall_total']:.0f}s)")
        return 0
    except Exception as e:
        err_path = outdir / f'event_n{args.n_trk:04d}_rep{args.rep:03d}.err'
        with open(err_path, 'w') as f:
            f.write(f"n_trk={args.n_trk} rep={args.rep}\n")
            f.write(traceback.format_exc())
        print(f"[err] {fname.name}: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
