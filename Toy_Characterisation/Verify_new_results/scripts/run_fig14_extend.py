#!/usr/bin/env python
"""Single-event Condor worker for fig14 extension to large n.

Generates one clean event (with hit measurement noise + scattering),
solves SimpleHamiltonianFast on it (clean + 1%-hit-drop variant), and
writes a small pickle with the per-event metric record:

    {n: int, rep: int, base: dict, drop: dict}

The two dicts have keys n_true_all, n_true_clean, n_false_all,
n_true_active, n_false_active, n_active — exactly matching the cache
records in solver_segment_efficiency_ext{,_drop1pct}.pkl.
"""
import argparse, copy, pickle, time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix as _coo

from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers import SimpleHamiltonianFast


# ── constants (match notebook §0 + §14) ─────────────────────────────────
DZ_MM, N_MODULES, Z_FIRST = 33.0, 5, 33
HALF_X = HALF_Y = 40.0
GAMMA, DELTA = 3.0, 1.0
PV_SIGMA = {"x": 0, "y": 0, "z": 1}
FIXED_EPSILON     = 0.002
FIXED_RESOLUTION  = 0.005
FIXED_SCATTERING  = 1e-4
SOLVER_THRESHOLD  = 0.35
DROP_RATE_SOLVER  = 0.01

z_positions = [Z_FIRST + i * DZ_MM for i in range(N_MODULES)]
geo = PlaneGeometry(module_id=list(range(N_MODULES)),
                    lx=[HALF_X]*N_MODULES, ly=[HALF_Y]*N_MODULES,
                    z=z_positions)


def safe_generate(n_tracks, measurement_error, collision_noise,
                  phi_max=0.2, theta_max=0.2, max_retries=20):
    for _ in range(max_retries):
        gen = StateEventGenerator(
            detector_geometry=geo, events=1, n_particles=[n_tracks],
            phi_min=-phi_max, phi_max=phi_max,
            theta_min=-theta_max, theta_max=theta_max,
            measurement_error=measurement_error,
            collision_noise=collision_noise)
        gen.generate_random_primary_vertices(PV_SIGMA)
        gen.generate_particles(
            [[{"type": "pion", "mass": 139.6, "q": 1}] * n_tracks])
        ev = gen.generate_complete_events()
        if ev.tracks and min(len(t.hit_ids) for t in ev.tracks) >= 3:
            return ev
    return ev


def apply_hit_noise(event, drop_rate, rng):
    evt = copy.deepcopy(event)
    if drop_rate <= 0.0:
        return evt
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


def _build_ham_vectorized(ham):
    n = ham.n_segments
    if n == 0:
        ham.A = _coo((n, n)).tocsc(); ham.b = np.zeros(n); return
    hit_ids = np.asarray(ham._segment_to_hit_ids, dtype=np.int64)
    start_ids = hit_ids[:, 0]; end_ids = hit_ids[:, 1]
    vecs = np.asarray(ham._segment_vectors, dtype=np.float64)
    cos_eps = float(np.cos(ham.epsilon))
    order_s = np.argsort(start_ids, kind='stable')
    order_e = np.argsort(end_ids,   kind='stable')
    sorted_s = start_ids[order_s]; sorted_e = end_ids[order_e]
    uniq_s, first_s = np.unique(sorted_s, return_index=True)
    uniq_e, first_e = np.unique(sorted_e, return_index=True)
    first_s = np.append(first_s, n); first_e = np.append(first_e, n)
    end_map = {int(uniq_e[i]): order_e[first_e[i]:first_e[i+1]]
               for i in range(len(uniq_e))}
    chunks_r, chunks_c, chunks_v = [], [], []
    for k in range(len(uniq_s)):
        mid = int(uniq_s[k])
        sid_i = end_map.get(mid)
        if sid_i is None: continue
        sid_j = order_s[first_s[k]:first_s[k+1]]
        if sid_i.size == 0 or sid_j.size == 0: continue
        cos_mat = vecs[sid_i] @ vecs[sid_j].T
        np.clip(cos_mat, -1.0, 1.0, out=cos_mat)
        ii, jj = np.where(cos_mat >= cos_eps)
        if ii.size == 0: continue
        ri = sid_i[ii]; cj = sid_j[jj]
        chunks_r.append(np.concatenate([ri, cj]))
        chunks_c.append(np.concatenate([cj, ri]))
        chunks_v.append(np.ones(2*ri.size, dtype=np.float64))
    diag_val = -(ham.delta + ham.gamma)
    diag_idx = np.arange(n, dtype=np.int64)
    if chunks_r:
        off_r = np.concatenate(chunks_r); off_c = np.concatenate(chunks_c)
        off_v = np.concatenate(chunks_v)
    else:
        off_r = np.empty(0, dtype=np.int64)
        off_c = np.empty(0, dtype=np.int64)
        off_v = np.empty(0, dtype=np.float64)
    rows = np.concatenate([diag_idx, off_r])
    cols = np.concatenate([diag_idx, off_c])
    vals = np.concatenate([np.full(n, diag_val, dtype=np.float64), off_v])
    A = _coo((vals, (rows, cols)), shape=(n, n)).tocsc()
    ham.A = -A; ham.b = np.ones(n) * ham.delta


def _segment_truth(ham):
    return np.array(
        [s.hit_start.track_id == s.hit_end.track_id and s.hit_start.track_id >= 0
         for s in ham.segments])


def _solver_metrics(event, n_true_clean=None):
    ham = SimpleHamiltonianFast(epsilon=FIXED_EPSILON, gamma=GAMMA, delta=DELTA)
    ham.construct_segments(event)
    _build_ham_vectorized(ham)
    sol = ham.solve_classicaly()
    active = sol > SOLVER_THRESHOLD
    truth = _segment_truth(ham)
    n_true_all     = int(truth.sum())
    n_false_all    = ham.n_segments - n_true_all
    n_true_active  = int((active & truth).sum())
    n_active       = int(active.sum())
    n_false_active = n_active - n_true_active
    return dict(n_true_all=n_true_all,
                n_true_clean=n_true_clean if n_true_clean is not None else n_true_all,
                n_false_all=n_false_all,
                n_true_active=n_true_active,
                n_false_active=n_false_active,
                n_active=n_active)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n',   type=int, required=True)
    p.add_argument('--rep', type=int, required=True)
    p.add_argument('--outdir', required=True)
    args = p.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    fp = out / f'fig14_n{args.n:05d}_rep{args.rep:02d}.pkl'
    if fp.exists():
        print(f"[worker] {fp.name} exists, skipping"); return

    # Same seed scheme as the notebook: rep + n*100 + 140000
    np.random.seed(args.rep + args.n * 100 + 140000)
    rng_drop = np.random.default_rng(42 + args.rep + args.n * 100)

    t0 = time.time()
    ev_clean = safe_generate(args.n,
                             measurement_error=FIXED_RESOLUTION,
                             collision_noise=FIXED_SCATTERING)
    t_g = time.time() - t0
    rec_base = _solver_metrics(ev_clean)
    t_b = time.time() - t0 - t_g
    ev_drop  = apply_hit_noise(ev_clean, drop_rate=DROP_RATE_SOLVER, rng=rng_drop)
    rec_drop = _solver_metrics(ev_drop, n_true_clean=rec_base['n_true_all'])
    t_d = time.time() - t0 - t_g - t_b

    print(f"[worker] n={args.n} rep={args.rep}  gen={t_g:.1f}s  "
          f"base={t_b:.1f}s  drop={t_d:.1f}s  "
          f"n_seg(clean)={rec_base['n_true_all']+rec_base['n_false_all']}  "
          f"true_act={rec_base['n_true_active']}  false_act={rec_base['n_false_active']}")

    with open(fp, 'wb') as f:
        pickle.dump({'n': args.n, 'rep': args.rep,
                     'base': rec_base, 'drop': rec_drop}, f)
    print(f"[worker] wrote {fp}")


if __name__ == '__main__':
    main()
