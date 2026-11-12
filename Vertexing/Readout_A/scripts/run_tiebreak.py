#!/usr/bin/env python3
"""Experiment A rerun with the deterministic tie-break of the coupled readout.

Package: LHCb_VeLo_Toy_Model branch claude/coupled-readout-tiebreak (53ea00f): the greedy hit
uniqueness ranks |amplitude| rounded to 9 significant digits and breaks ties by (start hit id,
end hit id) (postselect.hit_uniqueness_filter, sig_digits; get_tracks_coupled default 9).
Before, exact amplitude ties were decided by round-off (Cheap_VELO/C_small_circuits README).

Same events and solves as run_readouts.py / control_no_solver.py (Run 3: 103 events; toy:
T = 1000 x 1, 5, 10, 20 vertices, 3 events each). Reading I (truth-free) with the coupled
readout + greedy uniqueness (A's headline "ccu") for classical, 1BQF and QSVT, and the
no-solver control (every coupled segment, score = coupling count + 1e-3 x fixed random).
For each solver the readout is done three times:
    new      the fixed readout (sig_digits = 9)
    noisy    the fixed readout on |x| (1 + 1e-14 g), g standard normal: must equal "new"
    old      sig_digits = None (A's original ranking): must equal A's published numbers
Usage:  run_tiebreak.py real|toy [n_proc]
Output: outputs/tiebreak_<mode>_events.csv, outputs/tiebreak_<mode>_tracks.csv
"""
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

PKG = "/data/bfys/gscriven/worktrees/velo-toy-tiebreak/src"
PKG_MAIN = "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"
sys.path.insert(0, PKG)
sys.path.append(PKG_MAIN)                 # present, so fs_lib does not push it in front

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import importlib.util  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
# the Focusing_scan scripts are untracked files of the main checkout (read-only), as in A
VTX = Path("/data/bfys/gscriven/Quantum_Track_Reconstruction/Vertexing")
for p in (str(VTX / "Focusing_scan" / "scripts"), str(VTX / "Real_events" / "scripts")):
    sys.path.insert(1, p)
import fs_lib as L  # noqa: E402
import lhcb_velo_toy  # noqa: E402
from lhcb_velo_toy.solvers.reconstruction.coupled_readout import coupled_hit_groups  # noqa: E402

assert lhcb_velo_toy.__file__.startswith(PKG), lhcb_velo_toy.__file__
_spec = importlib.util.spec_from_file_location("fs7", VTX / "Focusing_scan" / "scripts" / "07_full_sweep.py")
F7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F7)
EPS_REAL = 0.005          # rad, as run_real.py / run_readouts.py
KEYS = ("efficiency", "ghost_rate", "n_matched", "n_reconstructible", "n_candidates", "n_ghosts", "n_clones")


def run(job):
    t0 = time.perf_counter()
    if job[0] == "real":
        import real_lib as R
        path = job[1]
        ev, ex = R.load_real(path, seed=zlib.crc32((Path(path).parent.name + Path(path).stem).encode()))
        F7.EPS = EPS_REAL
        base = dict(event=f"{Path(path).parent.name}/{Path(path).stem}")
    else:
        _, T, npv, rep = job
        ev = L.generate_multi_pv(npv, T, seed=30000 + 1000 * npv + 10 * rep + T)
        ex = None
        base = dict(event=f"T{T}_pv{npv}_r{rep}", T=T, n_pv=npv, rep=rep)
    hid = np.array([hh.hit_id for hh in ev.pkg_event.hits])
    hitpos = -np.ones(hid.max() + 1, int)
    hitpos[hid] = np.arange(len(hid))
    F7.hitpos_inv = hid
    reco = ev.n_hits_of_track >= 3
    h, truth, sols, sres = F7.segment_solutions(ev, int(reco.sum()))
    A = h.A.tocsr()
    sh = np.asarray(h._segment_to_hit_ids)
    iso_seg = np.diff(A.indptr) == 1
    rng = np.random.default_rng(zlib.crc32(base["event"].encode()))
    noise = 1 + 1e-14 * rng.standard_normal(A.shape[0])
    deg = np.diff(A.indptr) - 1
    crng = np.random.default_rng(zlib.crc32(base["event"].encode()))
    crng.random(len(deg))                    # control_no_solver.py draws the "rand" score first
    ctrl_score = deg + 1e-3 * crng.random(len(deg))
    jobs = {}
    for name, x in sols.items():
        iso = float(x[iso_seg].max()) if iso_seg.any() else 0.0
        jobs[name] = (x >= iso * (1 + 1e-6) + 1e-12, x)
    jobs["control"] = (deg >= 1, ctrl_score)
    rows, found = [], {}
    for name, (act, x) in jobs.items():
        for variant, score, sig in (("new", x, 9), ("noisy", np.abs(x) * noise, 9), ("old", x, None)):
            g = coupled_hit_groups(A, sh, act, solution=score, uniqueness="greedy", sig_digits=sig)
            groups = [list(hitpos[q]) for q in g]
            met, mt = F7.validate(ev, F7.make_tracks(groups, ev))
            rows.append(dict(base, solver=name, variant=variant, n_groups=len(groups),
                             groups_key=hash(tuple(sorted(tuple(sorted(q)) for q in groups))),
                             **{k: met[k] for k in KEYS}))
            found[f"{name}_{variant}"] = mt
    trows = []
    for t in np.nonzero(reco)[0]:
        d = dict(base, n_hits=int(ev.n_hits_of_track[t]))
        if ex is not None:
            d.update(prompt=bool(ex.prompt[t]), electron=bool(ex.electron[t]), p=float(ex.p[t]))
        d.update({f"found_{k}": int(t) in v for k, v in found.items()})
        trows.append(d)
    for r in rows:
        r["t_event"] = time.perf_counter() - t0
    return rows, trows


if __name__ == "__main__":
    mode = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    if mode == "real":
        import real_lib as R
        J = [("real", str(f)) for f in R.dump_files()]
    else:
        J = [("toy", 1000, npv, rep) for npv in (1, 5, 10, 20) for rep in range(3)]
    print(len(J), "events", flush=True)
    t0 = time.time()
    R_, T_ = [], []
    with Pool(nproc, maxtasksperchild=2) as pool:
        for i, (r, t) in enumerate(pool.imap_unordered(run, J, chunksize=1)):
            R_ += r
            T_ += t
            print(f"{i + 1}/{len(J)} {r[0]['event']}: {r[0]['t_event']:.0f}s", flush=True)
    pd.DataFrame(R_).sort_values(["event", "solver", "variant"]).to_csv(OUT / f"tiebreak_{mode}_events.csv", index=False)
    pd.DataFrame(T_).to_csv(OUT / f"tiebreak_{mode}_tracks.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
