#!/usr/bin/env python3
"""
Universal single-event worker for the segment-level Toy_Characterisation
experiments (Epsilon_study_2, Larger_Scatter, Larger_Scatter_Density, ERF).

One job runs one (n_trk, rep) point. It generates an event, builds the
Hamiltonian (optionally with ERF convolution), runs both the classical
and statevector-OneBitHHL quantum solvers, and writes a pickle with:

- per-pair angles and truth labels (true_angles, false_angles), so
  notebooks can do histogram-based threshold sweeps without re-running;
- the full classical and quantum solution vectors and the per-segment
  truth mask, so segment-level metrics at any threshold can be recomputed
  in analysis;
- per-event derived metrics at a default operating point (segment
  acceptance @ calculated epsilon for the *angle* histogram, plus
  classical- and quantum-solver metrics at the ABSOLUTE threshold
  tau_default = 0.35, matching segment_level_analysis.ipynb SOLVER_THRESHOLD);
- A-matrix sparsity statistics and wall-times.

Selecting epsilon
-----------------
``--epsilon`` is the calculated epsilon
``sqrt(2*theta_s^2 + 12*theta_r^2 + 2*theta_min^2)``. Compute it upstream
in ``gen_params*.py`` and pass the value here.

CSV columns (parsed by submit file):
    n_trk, rep, sigma_res, sigma_scatt, phi_max, hit_ineff,
    convolution, erf_sigma, epsilon, gamma, delta, tau_default,
    run_quantum, outdir, save_vectors
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


def _import_toy():
    # Deferred so a bad env produces a clean error.
    from lhcb_velo_toy.solvers import SimpleHamiltonianFast
    return SimpleHamiltonianFast


def _add_shared_to_path():
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-trk', type=int, required=True)
    ap.add_argument('--rep', type=int, required=True)
    ap.add_argument('--sigma-res', type=float, required=True,
                    help='measurement_error (sigma_res) in mm')
    ap.add_argument('--sigma-scatt', type=float, required=True,
                    help='collision_noise (sigma_scatt) in rad')
    ap.add_argument('--phi-max', type=float, default=0.2,
                    help='phi cone half-angle (also used for theta cone)')
    ap.add_argument('--hit-ineff', type=float, default=0.0,
                    help='probability each hit is randomly dropped')
    ap.add_argument('--convolution', type=int, default=0,
                    help='0=step (False), 1=erf (True)')
    ap.add_argument('--erf-sigma', type=float, default=1e-4,
                    help='ERF smoothing width theta_d (only used when convolution=1)')
    ap.add_argument('--epsilon', type=float, required=True,
                    help='angular acceptance threshold; pass the calculated value')
    ap.add_argument('--gamma', type=float, default=3.0)
    ap.add_argument('--delta', type=float, default=1.0)
    ap.add_argument('--tau-default', type=float, default=0.35,
                    help='ABSOLUTE segment threshold for default-metric reporting. '
                         'Hopfield false/isolated attractor = delta/(delta+gamma) = 0.25; '
                         'true segments sit on the ~0.375 plateau, so 0.35 separates them. '
                         'Matches segment_level_analysis.ipynb SOLVER_THRESHOLD. '
                         'NB: applied to sol_C and to the rescaled sol_Q (||sQ||==||sC||).')
    ap.add_argument('--run-quantum', type=int, default=1,
                    help='0 to skip the quantum solver (classical-only run)')
    ap.add_argument('--shots', type=int, default=0,
                    help='reserved / unused — kept for CLI compatibility')
    ap.add_argument('--device', type=str, default='CPU',
                    help='CPU: Python Statevector (T≤200). '
                         'GPU: AerSimulator CUDA statevector (T≤1000, ~1.5 h). '
                         'Matches Verify_new_results submit_gpu.sub pipeline.')
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--save-vectors', type=int, default=1,
                    help='1 to store full sol_C/sol_Q/truth (recommended)')
    ap.add_argument('--tag', type=str, default='seg',
                    help='filename prefix; defaults to "seg"')
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / (f"{args.tag}_n{args.n_trk:04d}_r{args.rep:03d}.pkl")

    if out_file.exists():
        print(f"[done] {out_file}", flush=True)
        return 0

    _add_shared_to_path()
    from helpers import (  # type: ignore
        make_geometry, safe_generate, apply_hit_inefficiency,
        solve_quantum_statevector, solve_quantum_aer_sampling, rescale_quantum,
        segment_truth_mask, solver_segment_metrics,
        fast_segment_metrics, collect_segment_pair_angles,
        compute_segment_metrics,
    )

    t_total = time.time()
    print(f"[start] n_trk={args.n_trk} rep={args.rep} "
          f"sigma_res={args.sigma_res} sigma_scatt={args.sigma_scatt} "
          f"phi_max={args.phi_max} hit_ineff={args.hit_ineff} "
          f"convolution={args.convolution} erf_sigma={args.erf_sigma} "
          f"eps={args.epsilon} run_quantum={args.run_quantum} device={args.device}", flush=True)

    try:
        SimpleHamiltonianFast = _import_toy()
        geom = make_geometry()

        seed = int(args.rep) + int(args.n_trk) * 100 + 200_000
        rng_drop = np.random.default_rng(seed + 31)

        t0 = time.time()
        ev = safe_generate(
            args.n_trk, seed=seed, geom=geom,
            measurement_error=float(args.sigma_res),
            collision_noise=float(args.sigma_scatt),
            phi_max=float(args.phi_max),
            theta_max=float(args.phi_max),  # same cone width as phi
        )
        if args.hit_ineff > 0.0:
            ev = apply_hit_inefficiency(ev, float(args.hit_ineff), rng_drop)
        t_gen = time.time() - t0

        # Angle-level metrics (histogram-ready): collect ALL pair angles
        # labelled true/false. Numba fast kernel gives totals; we also
        # store the per-pair angle arrays for downstream histograms.
        t0 = time.time()
        m_angle = fast_segment_metrics(ev, float(args.epsilon))
        # collect_segment_pair_angles is O(T^3) in BOTH time and output size —
        # the raw angle arrays alone exceed 128 GB at T=1000 (~3e9 triplets) and
        # OOM the job. The angle EFFICIENCY summary is already in m_angle
        # (fast_segment_metrics, cheap); only the raw per-pair arrays — used for
        # low-T threshold histograms — are skipped above T=300. Sampling-mode
        # jobs (shots > 0) skip it too.
        ANGLE_ARRAY_TMAX = 300
        if args.shots == 0 and args.n_trk <= ANGLE_ARRAY_TMAX:
            true_angles, false_angles = collect_segment_pair_angles(ev)
            true_angles = np.asarray(true_angles, dtype=np.float64)
            false_angles = np.asarray(false_angles, dtype=np.float64)
        else:
            true_angles = np.array([], dtype=np.float64)
            false_angles = np.array([], dtype=np.float64)
        t_angles = time.time() - t0

        # Hamiltonian + classical solver
        t0 = time.time()
        convolution = bool(int(args.convolution))
        ham = SimpleHamiltonianFast(
            epsilon=float(args.epsilon),
            gamma=float(args.gamma),
            delta=float(args.delta),
            theta_d=float(args.erf_sigma),
        )
        ham.construct_segments(ev)
        ham.construct_hamiltonian(ev, convolution=convolution)
        sol_C = np.asarray(ham.solve_classicaly()).ravel()
        truth = segment_truth_mask(ham)
        t_classical = time.time() - t0

        A_nnz = int(ham.A.nnz) if hasattr(ham.A, 'nnz') else int(np.count_nonzero(ham.A))
        n_seg = int(ham.n_segments)
        n_pad = 1 << int(math.ceil(math.log2(max(n_seg, 2))))

        # Classical solver metrics at a default operating point.
        # ABSOLUTE threshold (tau_default = 0.35): false/isolated segments sit at
        # the 0.25 Hopfield attractor, true segments on the ~0.375 plateau, so 0.35
        # separates them. A relative tau*max collapses below both -> FAR explosion.
        mC_default = solver_segment_metrics(
            truth, sol_C, threshold=args.tau_default
        )

        # Quantum solver
        sol_Q = None
        P_anc = None
        n_sys = None
        mQ_default = None
        cos_QC = None
        t_q = 0.0
        if args.run_quantum:
            t0 = time.time()
            sol_Q_raw, P_anc, n_sys = solve_quantum_statevector(
                ham.A, ham.b, device=args.device)
            sol_Q = rescale_quantum(sol_Q_raw, sol_C)
            t_q = time.time() - t0
            # sol_Q is rescaled so ||sol_Q|| == ||sol_C||, so the SAME absolute
            # threshold (tau_default = 0.35) is directly comparable to classical.
            mQ_default = solver_segment_metrics(
                truth, sol_Q, threshold=args.tau_default
            )
            if sol_Q.size and sol_C.size:
                cos_QC = float(
                    np.dot(sol_Q, sol_C)
                    / (np.linalg.norm(sol_Q) * np.linalg.norm(sol_C) + 1e-30)
                )

        record = dict(
            status='ok',
            # job identity
            n_trk=int(args.n_trk), rep=int(args.rep), seed=int(seed),
            tag=str(args.tag),
            # physics params
            sigma_res=float(args.sigma_res),
            sigma_scatt=float(args.sigma_scatt),
            phi_max=float(args.phi_max),
            hit_ineff=float(args.hit_ineff),
            convolution=bool(convolution),
            erf_sigma=float(args.erf_sigma),
            epsilon=float(args.epsilon),
            gamma=float(args.gamma),
            delta=float(args.delta),
            tau_default=float(args.tau_default),
            # geometry / sizes
            n_seg=n_seg, n_pad=n_pad, A_nnz=A_nnz,
            n_qubits=(int(n_sys) + 2) if n_sys is not None else None,
            # angle-level metrics (epsilon cut on per-triplet angles)
            angle_metrics=m_angle,
            # solver outputs at default tau
            classical_metrics_default=mC_default,
            quantum_metrics_default=mQ_default,
            cos_QC=cos_QC,
            P_anc=P_anc,
            # solver config
            device=str(args.device),
            # timings
            t_gen=t_gen, t_angles=t_angles,
            t_classical=t_classical, t_q=t_q,
            t_total=time.time() - t_total,
        )
        if args.save_vectors:
            record['sol_C'] = sol_C
            record['sol_Q'] = sol_Q
            record['truth'] = truth
            record['true_angles'] = true_angles
            record['false_angles'] = false_angles

        with open(out_file, 'wb') as f:
            pickle.dump(record, f)

        eff_str = f"effC={mC_default['segment_efficiency']:.3f}"
        if mQ_default is not None:
            eff_str += f" effQ={mQ_default['segment_efficiency']:.3f}"
        print(f"[ok] {out_file} n_seg={n_seg} A_nnz={A_nnz} "
              f"{eff_str} t={record['t_total']:.1f}s", flush=True)
        return 0

    except Exception as exc:
        traceback.print_exc()
        record = dict(
            status='error', error=f"{type(exc).__name__}: {exc}",
            n_trk=int(args.n_trk), rep=int(args.rep),
            sigma_res=float(args.sigma_res),
            sigma_scatt=float(args.sigma_scatt),
            phi_max=float(args.phi_max),
            hit_ineff=float(args.hit_ineff),
            convolution=bool(int(args.convolution)),
            erf_sigma=float(args.erf_sigma),
            epsilon=float(args.epsilon),
            t_total=time.time() - t_total,
        )
        with open(out_file, 'wb') as f:
            pickle.dump(record, f)
        return 1


if __name__ == '__main__':
    sys.exit(main())
