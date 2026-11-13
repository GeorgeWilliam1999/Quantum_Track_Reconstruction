#!/usr/bin/env python3
"""Item I/J, step 1: per event, the vertex-sampler oracle tables and every PV baseline.

Events
    toy   T = 100, 400, 1000 total tracks x n_pv = 1 ... 20, 3 events per point, the
          section 8 generator and seeds (fs_lib.generate_multi_pv, seed 30000 + 1000 n_pv
          + 10 rep + T; same events as 07_full_sweep / 08_pv_baseline / Readout_A).
    tune  toy tuning events, seeds 70000 + ..., T = 100, 400, 1000 x n_pv = 1, 3, 5, 10, 15, 20.
    real  the 103 Run 3 Gauss Sim10 events (Vertexing/Real_events, same smearing seeds as
          Readout_A). Events of job_5001 ... job_5008 tune, job_501x are the hold-out.

Per event, stored in cache/<mode>/<event>.npz:
    oracle tables of the vertex sampler (package solvers.vertexing.pv_sampler), at loose
    cuts (dmax <= DMAX_LOOSE, |t| >= TMIN, POCA z in ZWIN); step 2 tightens them:
        d_z0, d_doca         doublet items (package segments), d_n_items, d_nq
        t_z0, t_doca         triplet items (forward coupled segment pairs), t_n_items, t_nq
    truth: pv_z, pv_xyz, pv_nreco (reconstructible tracks per PV, >= 3 hits)
    found PV z lists of the baselines:
        scan                 focusing scan as specified (fs_lib.find_vertices)
        scan_refit           scan peaks seed candidates, z0 clustering (run_real.py REFIT)
        trk_<solver>_median  readout-A tracks (reading I, coupled + uniqueness), z0 median
                             per 1 mm gap cluster (the 08_pv_baseline rule)
        trk_<solver>_fit     same tracks, 3D vertex fit (package track_vertices, item J)
        trk_truth_fit        truth tracks (>= 3 hits), 3D vertex fit (upper bound for J)
    for solver in classical, 1bqf, qsvt.

Usage: 01_events.py {toy|tune|real} [n_proc]
"""
import importlib.util
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

PKG = "/data/bfys/gscriven/worktrees/velo-toy-pv-readout/src"
sys.path.insert(0, PKG)
import lhcb_velo_toy  # noqa: E402,F401  (cache the worktree package before fs_lib adds its own path)

FSD = Path("/data/bfys/gscriven/Quantum_Track_Reconstruction/Vertexing/Focusing_scan/scripts")
RED = Path("/data/bfys/gscriven/Quantum_Track_Reconstruction/Vertexing/Real_events/scripts")
for p in (str(FSD), str(RED)):
    if p not in sys.path:
        sys.path.insert(1, p)
import numpy as np  # noqa: E402
import fs_lib as L  # noqa: E402

assert lhcb_velo_toy.__file__.startswith(PKG), lhcb_velo_toy.__file__


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, FSD / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F7 = _load("fs7", "07_full_sweep.py")
F8 = _load("fs8", "08_pv_baseline.py")

from lhcb_velo_toy.solvers.postselect import hit_uniqueness_filter  # noqa: E402,F401
from lhcb_velo_toy.solvers.reconstruction.coupled_readout import coupled_hit_groups  # noqa: E402
from lhcb_velo_toy.solvers.vertexing import (  # noqa: E402
    doublet_table, fit_track_lines, triplet_table, vertices_from_tracks)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EPS_TOY = F7.EPS
EPS_REAL = 0.005                     # rad, as run_real.py / Readout_A
CUTS = dict(toy=dict(dmax=0.5, tmin=0.02, zwin=(-150.0, 30.0)),
            real=dict(dmax=3.0, tmin=0.02, zwin=(-200.0, 200.0)))
REFIT = dict(gap=0.5, min_trk=4, tmin=0.05)          # run_real.py
COARSE_TOY = dict(L.COARSE)
COARSE_REAL = dict(L.COARSE, lo=-250.0, hi=250.0)    # run_real.py


def jobs(mode):
    if mode == "toy":
        return [("toy", T, npv, rep, 30000 + 1000 * npv + 10 * rep + T)
                for T in (1000, 400, 100) for npv in range(1, 21) for rep in range(3)]
    if mode == "tune":
        return [("tune", T, npv, 0, 70000 + 1000 * npv + T)
                for T in (1000, 400, 100) for npv in (1, 3, 5, 10, 15, 20)]
    import real_lib as R
    return [("real", str(f)) for f in R.dump_files()]


def run(job):
    t0 = time.perf_counter()
    mode = job[0]
    if mode == "real":
        import real_lib as R
        path = job[1]
        name = f"{Path(path).parent.name}_{Path(path).stem}"
        ev, ex = R.load_real(path, seed=zlib.crc32((Path(path).parent.name + Path(path).stem).encode()))
        F7.EPS = EPS_REAL
        L.COARSE.update(COARSE_REAL)
        cuts = CUTS["real"]
        meta = dict(event=name, T=-1, n_pv=len(ev.pv_xyz), rep=-1)
    else:
        _, T, npv, rep, seed = job
        name = f"T{T}_pv{npv}_r{rep}"
        ev = L.generate_multi_pv(npv, T, seed=seed)
        F7.EPS = EPS_TOY
        L.COARSE.update(COARSE_TOY)
        cuts = CUTS["toy"]
        meta = dict(event=name, T=T, n_pv=npv, rep=rep)
    out = ROOT / "cache" / mode / f"{name}.npz"
    if out.exists():
        return name, "cached"
    hid = np.array([hh.hit_id for hh in ev.pkg_event.hits])
    hitpos = -np.ones(hid.max() + 1, int)
    hitpos[hid] = np.arange(len(hid))
    F7.hitpos_inv = hid
    reco = ev.n_hits_of_track >= 3
    T_reco = int(reco.sum())
    pv_nreco = np.array([int(((ev.pv_of_track == i) & reco).sum()) for i in range(len(ev.pv_xyz))])

    # ── segments, Hamiltonian, the three solvers (the Readout_A solves) ──────────────
    ts = time.perf_counter()
    h, truth, sols, sres = F7.segment_solutions(ev, T_reco)
    t_seg = time.perf_counter() - ts
    seg_hits = np.asarray(h._segment_to_hit_ids)
    seg_xyz = np.asarray(h._segment_endpoints_xyz)
    A = h.A.tocsr()

    # ── oracle tables (loose cuts) ──────────────────────────────────────────────────
    to = time.perf_counter()
    dt = doublet_table(seg_xyz, seg_hits, **cuts)
    tt = triplet_table(seg_xyz, seg_hits, A, **cuts)
    t_tab = time.perf_counter() - to
    # truth of the flagged items: all hits on one track, and that track's PV
    trk_of_hit = ev.track[hitpos[np.arange(hitpos.size)].clip(0)]

    def item_truth(items):
        if len(items) == 0:
            return np.zeros(0, int)
        tk = trk_of_hit[items]
        same = (tk == tk[:, :1]).all(1) & (tk[:, 0] >= 0)
        return np.where(same, ev.pv_of_track[tk[:, 0].clip(0)], -1)

    found = {}
    # ── focusing scan and scan-refit ────────────────────────────────────────────────
    tv = time.perf_counter()
    vs, _ = L.find_vertices(ev)
    found["scan"] = [v[0] for v in vs]
    seeds = L.tracks_at_vertices(ev, found["scan"], offsets=L.OFFSETS4, ngrid=512)
    found["scan_refit"] = F8.vertices_from_tracks(ev, [g for _, g in seeds], **REFIT)
    t_scan = time.perf_counter() - tv

    # ── PVs from first-pass tracks (item J) ─────────────────────────────────────────
    iso_seg = np.diff(A.indptr) == 1
    tj = {}
    for sname, x in sols.items():
        tt0 = time.perf_counter()
        iso = float(x[iso_seg].max()) if iso_seg.any() else 0.0
        act = x >= iso * (1 + 1e-6) + 1e-12
        g = coupled_hit_groups(A, seg_hits, act, solution=x, uniqueness="greedy")
        groups = [hitpos[np.asarray(gg)] for gg in g]
        lines = fit_track_lines(ev.x, ev.y, ev.z, groups)
        found[f"trk_{sname}_median"] = [v[0] for v in vertices_from_tracks(lines, fit=False)]
        found[f"trk_{sname}_fit"] = [v[0] for v in vertices_from_tracks(lines, fit=True)]
        tj[sname] = time.perf_counter() - tt0
    tg = [np.nonzero(ev.track == t)[0] for t in np.nonzero(reco)[0]]
    found["trk_truth_fit"] = [v[0] for v in vertices_from_tracks(fit_track_lines(ev.x, ev.y, ev.z, tg), fit=True)]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, meta=json.dumps(dict(meta, n_hits=len(hid), n_reco=T_reco, n_seg=len(seg_hits),
                                  t_seg=t_seg, t_tables=t_tab, t_scan=t_scan, t_trk=tj,
                                  t_event=time.perf_counter() - t0, cuts=cuts, sres=sres)),
        d_z0=dt.z0, d_doca=dt.doca, d_pv=item_truth(dt.items), d_n_items=dt.n_items, d_nq=dt.n_index_qubits,
        t_z0=tt.z0, t_doca=tt.doca, t_pv=item_truth(tt.items), t_n_items=tt.n_items, t_nq=tt.n_index_qubits,
        t_dmax_row=tt.cuts["d_max"],
        pv_xyz=ev.pv_xyz, pv_nreco=pv_nreco, found=json.dumps(found))
    return name, f"{time.perf_counter() - t0:.0f}s, {len(seg_hits)} segments, {dt.n_flagged}/{tt.n_flagged} flagged"


if __name__ == "__main__":
    mode = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    J = jobs(mode)
    print(len(J), "events", flush=True)
    t0 = time.time()
    with Pool(nproc, maxtasksperchild=4) as pool:
        for i, (n, s) in enumerate(pool.imap_unordered(run, J, chunksize=1)):
            print(f"{i + 1}/{len(J)} {n}: {s}", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
