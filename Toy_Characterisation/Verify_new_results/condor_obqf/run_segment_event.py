#!/usr/bin/env python3
"""
Condor worker: single-event segment-level statevector OneBitHHL evaluation.

Mirrors the §6/§8 setup of Quantum_segment_level_analysis.ipynb. Each job
produces ONE pickle in --outdir named:

    seg_g{gamma}_n{n_trk}_r{rep}.pkl

containing the full diagnostic record:
    {n_trk, rep, gamma, delta, eps, tau, n_seg, n_pad, n_qubits,
     P_anc, cos, t_q, mC, mQ, sol_C, sol_Q, truth, status}

Aggregation is done back in the notebook.
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Library imports (deferred so we can print clean errors on env failure).
# ---------------------------------------------------------------------------
def _import_toy():
    from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
    from lhcb_velo_toy.solvers import SimpleHamiltonianFast
    from lhcb_velo_toy.solvers.quantum.one_bit_hhl import OneBitHHL
    return (PlaneGeometry, StateEventGenerator,
            SimpleHamiltonianFast, OneBitHHL)


# ---------------------------------------------------------------------------
# Geometry — single source of truth, must match the notebook.
# ---------------------------------------------------------------------------
DZ_MM, N_MODULES, Z_FIRST = 33.0, 5, 33
HALF_X = HALF_Y = 40.0
PV_SIGMA = {'x': 0.0, 'y': 0.0, 'z': 1.0}


def _make_geometry(PlaneGeometry):
    z_positions = [Z_FIRST + DZ_MM * i for i in range(N_MODULES)]
    return PlaneGeometry(
        module_id=list(range(N_MODULES)),
        lx=[HALF_X] * N_MODULES,
        ly=[HALF_Y] * N_MODULES,
        z=z_positions,
    )


# ---------------------------------------------------------------------------
# Event generation (matches safe_generate in the notebook).
# ---------------------------------------------------------------------------
def safe_generate(n_tracks, seed, geom, StateEventGenerator,
                  measurement_error=0.005, collision_noise=1e-4,
                  phi_max=0.2, theta_max=0.2, max_retries=20):
    np.random.seed(seed)
    last = None
    for _ in range(max_retries):
        gen = StateEventGenerator(
            detector_geometry=geom, events=1, n_particles=[n_tracks],
            phi_min=-phi_max, phi_max=phi_max,
            theta_min=-theta_max, theta_max=theta_max,
            measurement_error=measurement_error,
            collision_noise=collision_noise,
        )
        gen.generate_random_primary_vertices(PV_SIGMA)
        gen.generate_particles(
            [[{"type": "pion", "mass": 139.6, "q": 1}] * n_tracks]
        )
        ev = gen.generate_complete_events()
        last = ev
        if ev.tracks and min(len(t.hit_ids) for t in ev.tracks) >= 3:
            return ev
    return last


# ---------------------------------------------------------------------------
# Statevector OneBitHHL (matches solve_quantum_statevector in the notebook).
# ---------------------------------------------------------------------------
def solve_quantum_statevector(A, b, OneBitHHL):
    from qiskit.quantum_info import Statevector
    hhl = OneBitHHL(A, b, num_time_qubits=1, shots=1, debug=False)
    hhl.build_circuit()
    qc = hhl.circuit.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc)
    amps = np.asarray(sv.data)
    n_sys = hhl.num_system_qubits
    amps = amps.reshape(2, 2 ** n_sys, 2)
    probs_anc1 = (np.abs(amps[1]) ** 2).sum(axis=-1)
    success = float(probs_anc1.sum())
    if success > 0:
        prob_norm = probs_anc1 / success
    else:
        prob_norm = np.zeros_like(probs_anc1)
    sol_padded = np.sqrt(prob_norm)
    nrm = float(np.linalg.norm(sol_padded))
    if nrm > 0:
        sol_padded /= nrm
    return sol_padded[: hhl.original_dim], success, n_sys


def rescale(sol_Q, sol_C):
    nQ = float(np.linalg.norm(sol_Q))
    nC = float(np.linalg.norm(sol_C))
    return sol_Q * (nC / nQ) if nQ > 0 else np.zeros_like(sol_Q)


def segment_truth(ham):
    arr = getattr(ham, "_segment_track_ids", None)
    if arr is not None and len(arr) == ham.n_segments:
        return ((arr[:, 0] == arr[:, 1]) & (arr[:, 0] >= 0)).astype(bool)
    return np.asarray([
        (s.hit_start.track_id == s.hit_end.track_id)
        and (s.hit_start.track_id >= 0)
        for s in ham.segments
    ], dtype=bool)


def seg_metrics(truth, sol, threshold):
    active = sol > threshold
    n_true_all     = int(truth.sum())
    n_false_all    = int(truth.size - n_true_all)
    n_true_active  = int((active & truth).sum())
    n_active       = int(active.sum())
    n_false_active = n_active - n_true_active
    return dict(
        n_true_all=n_true_all, n_false_all=n_false_all,
        n_true_active=n_true_active, n_false_active=n_false_active,
        n_active=n_active,
        segment_efficiency=n_true_active / max(n_true_all, 1),
        segment_purity=n_true_active / max(n_active, 1),
        segment_false_rate=n_false_active / max(n_active, 1),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-trk', type=int, required=True)
    ap.add_argument('--rep', type=int, required=True)
    ap.add_argument('--gamma', type=float, required=True)
    ap.add_argument('--delta', type=float, default=1.0)
    ap.add_argument('--epsilon', type=float, default=0.002)
    ap.add_argument('--threshold', type=float, default=0.35)
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--save-vectors', action='store_true',
                    help='also store sol_C, sol_Q, truth (large at high n_trk)')
    args = ap.parse_args()

    n_trk, rep, gamma = args.n_trk, args.rep, args.gamma
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"seg_g{gamma:g}_n{n_trk:04d}_r{rep:02d}.pkl"

    # Idempotency
    if out_file.exists():
        print(f"[done] already exists: {out_file}")
        return 0

    print(f"[start] n_trk={n_trk} rep={rep} gamma={gamma} delta={args.delta} "
          f"eps={args.epsilon} tau={args.threshold}", flush=True)
    t_total = time.time()

    try:
        PlaneGeometry, StateEventGenerator, SimpleHamiltonianFast, OneBitHHL = _import_toy()
        geom = _make_geometry(PlaneGeometry)
        seed = rep + n_trk * 100 + 140_000

        t0 = time.time()
        ev = safe_generate(n_trk, seed, geom, StateEventGenerator)
        t_gen = time.time() - t0

        t0 = time.time()
        ham = SimpleHamiltonianFast(epsilon=args.epsilon,
                                    gamma=gamma, delta=args.delta)
        ham.construct_segments(ev)
        ham.construct_hamiltonian(ev, convolution=False)
        sol_C = np.asarray(ham.solve_classicaly()).ravel()
        truth = segment_truth(ham)
        t_classical = time.time() - t0

        A = ham.A.toarray() if hasattr(ham.A, 'toarray') else np.asarray(ham.A)
        b = np.asarray(ham.b).ravel()
        n_seg = ham.n_segments
        n_pad = 1 << int(math.ceil(math.log2(max(n_seg, 2))))

        t0 = time.time()
        sol_Q_raw, P_anc, n_sys = solve_quantum_statevector(A, b, OneBitHHL)
        t_q = time.time() - t0
        sol_Q = rescale(sol_Q_raw, sol_C)

        cos = float(np.dot(sol_Q, sol_C) /
                    (np.linalg.norm(sol_Q) * np.linalg.norm(sol_C) + 1e-30))
        rel = float(np.linalg.norm(sol_Q - sol_C) /
                    max(np.linalg.norm(sol_C), 1e-12))

        mC = seg_metrics(truth, sol_C, args.threshold)
        mQ = seg_metrics(truth, sol_Q, args.threshold)

        record = dict(
            status='ok',
            n_trk=n_trk, rep=rep, gamma=gamma, delta=args.delta,
            epsilon=args.epsilon, threshold=args.threshold, seed=seed,
            n_seg=n_seg, n_pad=n_pad, n_qubits=n_sys + 2,
            t_gen=t_gen, t_classical=t_classical, t_q=t_q,
            t_total=time.time() - t_total,
            P_anc=P_anc, cosine=cos, rel_l2=rel,
            mC=mC, mQ=mQ,
        )
        if args.save_vectors:
            record['sol_C'] = sol_C
            record['sol_Q'] = sol_Q
            record['truth'] = truth

        with open(out_file, 'wb') as f:
            pickle.dump(record, f)
        print(f"[ok] {out_file}  n_seg={n_seg} cos={cos:.3f} "
              f"effC={mC['segment_efficiency']:.3f} "
              f"effQ={mQ['segment_efficiency']:.3f} "
              f"P(anc)={P_anc:.4f} t={record['t_total']:.1f}s", flush=True)
        return 0

    except Exception as exc:
        traceback.print_exc()
        record = dict(
            status='error', error=f'{type(exc).__name__}: {exc}',
            n_trk=n_trk, rep=rep, gamma=gamma, delta=args.delta,
            epsilon=args.epsilon, threshold=args.threshold,
            t_total=time.time() - t_total,
        )
        with open(out_file, 'wb') as f:
            pickle.dump(record, f)
        return 1


if __name__ == '__main__':
    sys.exit(main())
