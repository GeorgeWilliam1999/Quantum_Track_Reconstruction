#!/usr/bin/env python3
"""Experiment A: does the readout, not the filter, merge tracks? Four readouts, same solves.

Solvers (unchanged, Vertexing/Focusing_scan/scripts/07_full_sweep.segment_solutions): the
package segment Hamiltonian (SimpleHamiltonianFast, step kernel, gamma 3, delta 1), solved
exactly per connected component as
    classical  x = A^-1 b
    1bqf       exact one-bit-filter readout amplitude, t = pi/4
    qsvt       production comb, degree 40, half-width 0.18, |p(A) b|
Readings (which segments are active):
    I      truth-free: every segment above the isolated-segment value (as run_real.py)
    F      lowest threshold with segment false rate <= 1 % (truth-tuned, paper 1 comparison)
    P      paper 1 rule: classical and 1bqf at 99 % true-segment efficiency, qsvt at <= 1 % false
Readouts of the same active set:
    gt     package get_tracks rule: active segments sharing a hit form one track
           (07_full_sweep.tracks_from_segments, checked equal to get_tracks)
    gtu    gt after hit uniqueness (package postselect.hit_uniqueness_filter, greedy)
    cc     NEW package get_tracks_coupled without uniqueness: link two active segments
           only where A couples them (shared hit AND angle < epsilon)
    ccu    NEW get_tracks_coupled with greedy hit uniqueness (the default)
    lay    package get_tracks_layered (greedy angle-consistent chain following, >= 3 hits);
           not run at reading P
All scored by the package EventValidator (purity >= 0.7, reconstructible = >= 3 hits).
The readout is classical post-processing; qubits and two-qubit gates (seg_* columns) are
those of the solve and do not depend on the readout.

Usage:  run_readouts.py real [n_proc]    103 Run 3 Gauss Sim10 events
        run_readouts.py toy  [n_proc]    toy sweep, T = 100, 400, 1000 x n_pv = 1, 2, 5, 10, 15, 20
Output: outputs/<mode>_readouts_events.csv (one row per event x solver x reading x readout),
        outputs/<mode>_readouts_tracks.csv (per truth track, found flags, reading I and F)
"""
import importlib.util
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)
VTX = HERE.parent.parent
FSD = VTX / "Focusing_scan" / "scripts"
RED = VTX / "Real_events" / "scripts"
for p in (str(FSD), str(RED)):
    if p not in sys.path:
        sys.path.insert(0, p)
import fs_lib as L  # noqa: E402

spec = importlib.util.spec_from_file_location("fs7", FSD / "07_full_sweep.py")
F7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F7)

from lhcb_velo_toy.solvers.postselect import hit_uniqueness_filter  # noqa: E402
from lhcb_velo_toy.solvers.reconstruction.coupled_readout import coupled_hit_groups  # noqa: E402
from lhcb_velo_toy.solvers.reconstruction.track_finder import get_tracks_layered  # noqa: E402

EPS_REAL = 0.005          # rad, as run_real.py
METRICS = ("efficiency", "ghost_rate", "clone_fraction", "mean_purity", "hit_efficiency",
           "n_candidates", "n_ghosts", "n_clones", "n_reconstructible", "n_matched")
READOUTS = ("gt", "gtu", "cc", "ccu", "lay")
TOY_T = (100, 400, 1000)
TOY_PV = (1, 2, 5, 10, 15, 20)


def toy_jobs():
    return [("toy", T, npv, rep) for T in TOY_T for npv in TOY_PV for rep in range(5 if T < 700 else 3)]


def real_jobs():
    import real_lib as R
    return [("real", str(f)) for f in R.dump_files()]


def _layered(h, ev, x, act):
    """get_tracks_layered on the array Hamiltonian: a light segment list with the event hits."""
    hits = {hh.hit_id: hh for hh in ev.pkg_event.hits}
    sh = np.asarray(h._segment_to_hit_ids)
    segs = [SimpleNamespace(hits=(hits[int(a)], hits[int(b)])) for a, b in sh]
    ham = SimpleNamespace(segments=segs, _segment_vectors=h._segment_vectors, epsilon=h.epsilon)
    # layered keeps solution > threshold; feed a 0/1 score so the active set is exactly act,
    # ties broken by the solver amplitude
    xs = np.where(act, 1.0 + np.abs(x) / (np.abs(x).max() + 1e-300), 0.0)
    return get_tracks_layered(ham, xs, ev.pkg_event, threshold=0.5, min_hits=3)


def run(job):
    t0 = time.perf_counter()
    if job[0] == "real":
        import real_lib as R
        path = job[1]
        ev, ex = R.load_real(path, seed=zlib.crc32((Path(path).parent.name + Path(path).stem).encode()))
        F7.EPS = EPS_REAL
        base = dict(event=f"{Path(path).parent.name}/{Path(path).stem}", n_pv=len(ev.pv_xyz))
        cats = dict(prompt=ex.prompt, electron=ex.electron, p=ex.p)
    else:
        _, T, npv, rep = job
        ev = L.generate_multi_pv(npv, T, seed=30000 + 1000 * npv + 10 * rep + T)
        base = dict(event=f"T{T}_pv{npv}_r{rep}", T=T, n_pv=npv, rep=rep)
        cats = None
    hid = np.array([hh.hit_id for hh in ev.pkg_event.hits])
    hitpos = -np.ones(hid.max() + 1, int)
    hitpos[hid] = np.arange(len(hid))
    F7.hitpos_inv = hid
    reco = ev.n_hits_of_track >= 3
    T_reco = int(reco.sum())
    base.update(n_hits=len(hid), n_reco=T_reco)

    h, truth, sols, sres = F7.segment_solutions(ev, T_reco)
    seg_hits = np.asarray(h._segment_to_hit_ids)
    A = h.A.tocsr()
    iso_seg = np.diff(A.indptr) == 1
    # Hamiltonian components, in hits: the largest group a coupling readout could build
    from scipy.sparse.csgraph import connected_components
    C = A.copy()
    C.setdiag(0)
    C.eliminate_zeros()
    nc, lab = connected_components(C, directed=False)
    pairs = np.unique(np.c_[np.r_[lab, lab], np.r_[seg_hits[:, 0], seg_hits[:, 1]]], axis=0)
    base.update(sres, ham_max_comp_hits=int(np.bincount(pairs[:, 0]).max()), seg_n_true=int(truth.sum()))

    rows, matched = [], {}
    for name, x in sols.items():
        iso = float(x[iso_seg].max()) if iso_seg.any() else 0.0
        for reading in ("I", "F", "P"):
            if reading == "I":
                tau = iso * (1 + 1e-6) + 1e-12
            else:
                rule = "far1" if (name == "qsvt" or reading == "F") else "eff99"
                tau = F7.threshold(x, truth, rule)
            act = x >= tau
            uact = hit_uniqueness_filter(x, seg_hits, act, mode="greedy")
            for ro in READOUTS:
                if ro == "lay" and reading == "P":
                    continue          # P switches on nearly every segment; the layered
                                      # python loop then takes minutes per event (230 s on
                                      # a 1336-hit Run 3 event) and P is not a working point
                tr = time.perf_counter()
                if ro == "gt":
                    tracks = F7.make_tracks(F7.tracks_from_segments(seg_hits, act, hitpos), ev)
                elif ro == "gtu":
                    tracks = F7.make_tracks(F7.tracks_from_segments(seg_hits, uact, hitpos), ev)
                elif ro in ("cc", "ccu"):
                    g = coupled_hit_groups(A, seg_hits, act, solution=x,
                                           uniqueness="greedy" if ro == "ccu" else None)
                    tracks = F7.make_tracks([list(hitpos[gg]) for gg in g], ev)
                else:
                    tracks = _layered(h, ev, x, act)
                t_ro = time.perf_counter() - tr
                met, mt = F7.validate(ev, tracks)
                sizes = [len(t.hit_ids) for t in tracks]
                r = dict(base, solver=name, reading=reading, readout=ro, t_readout=t_ro,
                         seg_eff=float((act & truth).sum() / max(truth.sum(), 1)),
                         seg_far=float((act & ~truth).sum() / max(act.sum(), 1)),
                         n_active=int(act.sum()), n_active_unique=int(uact.sum()),
                         max_group_hits=max(sizes) if sizes else 0,
                         n_groups_gt20=int(sum(s > 20 for s in sizes)))
                r.update({k: met[k] for k in METRICS})
                rows.append(r)
                if reading in ("I", "F"):
                    matched[f"{name}{reading}_{ro}"] = mt
    trows = []
    for t in np.nonzero(reco)[0]:
        d = dict(event=base["event"], n_pv=base["n_pv"], n_hits=int(ev.n_hits_of_track[t]),
                 t=float(np.hypot(*ev.t_of_track[t])))
        if cats is not None:
            d.update(prompt=bool(cats["prompt"][t]), electron=bool(cats["electron"][t]), p=float(cats["p"][t]))
        else:
            d.update(T=base["T"])
        d.update({f"found_{k}": int(t) in v for k, v in matched.items()})
        trows.append(d)
    for r in rows:
        r["t_event"] = time.perf_counter() - t0
    return rows, trows


if __name__ == "__main__":
    mode = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    J = real_jobs() if mode == "real" else toy_jobs()
    # largest jobs first so the pool drains evenly
    print(len(J), "events", flush=True)
    t0 = time.time()
    R_, TR = [], []
    with Pool(nproc, maxtasksperchild=4) as pool:
        for i, (r, t) in enumerate(pool.imap_unordered(run, J[::-1] if mode == "toy" else J, chunksize=1)):
            R_ += r
            TR += t
            print(f"{i + 1}/{len(J)} {r[0]['event']}: {r[0]['seg_n']} segments, {r[0]['t_event']:.0f}s", flush=True)
    pd.DataFrame(R_).sort_values(["event", "solver", "reading", "readout"]).to_csv(
        OUT / f"{mode}_readouts_events.csv", index=False)
    pd.DataFrame(TR).to_csv(OUT / f"{mode}_readouts_tracks.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
