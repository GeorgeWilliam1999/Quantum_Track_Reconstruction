#!/usr/bin/env python3
"""Control for Experiment A: how much of the result needs a solver at all?

At the truth-free reading (I) the classical solver and the one-bit filter switch on almost
every coupled segment (Run 3: 0.90 of true, 0.18 false), so the solve could be doing little
beyond "is this segment coupled". This control skips the solve: active = every segment with
at least one coupling (degree >= 1 in A), read out with the coupled readout + greedy hit
uniqueness (get_tracks_coupled), with the slot winner chosen by a solver-free score:
    rand    uniform random score (seed fixed per event): the uniqueness picks blindly
    deg     the number of couplings of the segment (a local classical count)
Same events, same Hamiltonian (SimpleHamiltonianFast, step, gamma 3, delta 1, the epsilon
of run_readouts.py), same EventValidator scoring.

Usage:  control_no_solver.py real|toy [n_proc]
Output: outputs/<mode>_control_events.csv
"""
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_readouts as RR  # noqa: E402

from lhcb_velo_toy.solvers import SimpleHamiltonianFast  # noqa: E402
from lhcb_velo_toy.solvers.reconstruction.coupled_readout import coupled_hit_groups  # noqa: E402

F7, L = RR.F7, RR.L


def run(job):
    t0 = time.perf_counter()
    if job[0] == "real":
        import real_lib as R
        path = job[1]
        ev, _ = R.load_real(path, seed=zlib.crc32((Path(path).parent.name + Path(path).stem).encode()))
        eps = RR.EPS_REAL
        base = dict(event=f"{Path(path).parent.name}/{Path(path).stem}", n_pv=len(ev.pv_xyz))
    else:
        _, T, npv, rep = job
        ev = L.generate_multi_pv(npv, T, seed=30000 + 1000 * npv + 10 * rep + T)
        eps = F7.EPS
        base = dict(event=f"T{T}_pv{npv}_r{rep}", T=T, n_pv=npv, rep=rep)
    hid = np.array([hh.hit_id for hh in ev.pkg_event.hits])
    hitpos = -np.ones(hid.max() + 1, int)
    hitpos[hid] = np.arange(len(hid))
    h = SimpleHamiltonianFast(epsilon=eps, gamma=3.0, delta=1.0)
    h.construct_segments(ev.pkg_event, materialize_segments=False)
    h.construct_hamiltonian(ev.pkg_event)
    A = h.A.tocsr()
    seg_hits = np.asarray(h._segment_to_hit_ids)
    tid = np.asarray(h._segment_track_ids)
    truth = (tid[:, 0] == tid[:, 1]) & (tid[:, 0] >= 0)
    deg = np.diff(A.indptr) - 1
    act = deg >= 1
    rng = np.random.default_rng(zlib.crc32(base["event"].encode()))
    rows = []
    for score_name, score in (("rand", rng.random(len(deg))), ("deg", deg + 1e-3 * rng.random(len(deg)))):
        g = coupled_hit_groups(A, seg_hits, act, solution=score, uniqueness="greedy")
        met, _ = F7.validate(ev, F7.make_tracks([list(hitpos[gg]) for gg in g], ev))
        r = dict(base, score=score_name, seg_eff=float((act & truth).sum() / max(truth.sum(), 1)),
                 seg_far=float((act & ~truth).sum() / max(act.sum(), 1)))
        r.update({k: met[k] for k in RR.METRICS})
        rows.append(r)
    for r in rows:
        r["t_event"] = time.perf_counter() - t0
    return rows


if __name__ == "__main__":
    mode = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    J = RR.real_jobs() if mode == "real" else RR.toy_jobs()[::-1]
    print(len(J), "events", flush=True)
    t0 = time.time()
    rows = []
    with Pool(nproc, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(run, J, chunksize=1)):
            rows += r
            print(f"{i + 1}/{len(J)} {r[0]['event']} {r[0]['t_event']:.0f}s", flush=True)
    pd.DataFrame(rows).sort_values(["event", "score"]).to_csv(RR.OUT / f"{mode}_control_events.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
