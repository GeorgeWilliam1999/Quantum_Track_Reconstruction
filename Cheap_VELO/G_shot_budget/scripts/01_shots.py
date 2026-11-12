#!/usr/bin/env python3
"""Experiment G: how many measurement shots does each quantum solver need?

Plan: Notion research plan "cheap and effective VELO track finding", item G (readout shot
budget); to-do 4165d544b9d981db9a30ffcd22dc2550.

Events: the experiment B / A events. Toy: T = 100, 400, 1000 x 1, 5, 10, 20 vertices, seeds
of 07_full_sweep (5 events per point, 3 at T = 1000). Run 3: the 103 Gauss Sim10 events
(real_lib.load_real, smearing seed as run_real.py), segment epsilon 5 mrad as run_real.py.

Solvers and their exact post-selected output (no circuit is simulated):
  hit register at the focusing-scan vertices, per-vertex hit operator C at 4w
  (Cheap_VELO/B_graded_clique_filter/scripts/b_lib.evaluate):
    onebit8   one-bit filter on 8I - C (2 calls; the hit-register default since 12 Nov)
    poly6s    step polynomial, degree 6, shifted domain (6 calls)
    step2     ideal projector on every integer line >= 2 (not a circuit: no gate count)
    output y = f(C) u, u uniform over the real hits.
  segment register, package SimpleHamiltonianFast (07_full_sweep.segment_solutions):
    1bqf      one-bit filter, t = pi/4: |amplitude| = |V e^{-i w t/2} cos(w t/2) V^T 1|
    qsvt      comb, degree 40, half-width 0.18: |p(A) 1|
    classical x = A^-1 1 (reference only: no shots)
  The no-solver control of experiment A (Vertexing/Readout_A/outputs/*_control_events.csv,
  score "deg") is a reference line read from file.

Shots: p_i = |psi_i|^2 / sum over the real register items (padding excluded after
post-selection; for the polynomial the padded rows carry f(0)^2, recorded as pad_leak).
Nested multinomial draws (package solvers.quantum.shots.ShotSampler) at N = 10 ... 1e8,
half-decade steps, 5 seeds. Item on:
  count rule     c_i >= k
  estimate rule  c_i >= k and c_i / N >= p_min, where p_min is the exact-output reading of
                 A / B written as a probability: hit register weight >= tau (B) <=> p_i >=
                 tau^2 / (n P); segments at the truth-free reading I (amplitude above the
                 isolated-segment value) <=> p_i >= tau_I^2 / sum x^2.
  k = 1 and 2.
Readout: hit register = connected components of C over the on hits, >= 3 hits
(clique_filters.tracks_from_filtered_hits); segments = coupled readout with greedy hit
uniqueness (package get_tracks_coupled rule, coupled_hit_groups), slot winner = largest count
(ties broken by a fixed random jitter). EventValidator, purity >= 0.7, reconstructible >= 3 hits.
A curve (solver, rule, k, seed) stops once it is within 0.005 of its N -> infinity value at
two consecutive N.

Reproduction: the estimate rule at p_hat = p (N -> infinity) must give the experiment B
(hit) and A (segment, reading I, readout ccu) efficiencies event by event.

Usage:  01_shots.py toy|real [n_proc]
Columns mass_pmin / mass_true: probability mass on the items the exact reading keeps / on the
items of truth tracks (hit register: hits of reconstructible tracks; segments: true segments).

Output: outputs/<mode>_shots_events.csv (one row per event x solver), outputs/<mode>_shots_curves.csv
        (one row per event x solver x rule x k x seed x N)
"""
import math
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
os.environ["VELO_TOY_SRC"] = "/data/bfys/gscriven/worktrees/velo-toy-cheap-velo-G/src"
sys.path.insert(0, str(ROOT.parent / "B_graded_clique_filter" / "scripts"))
import b_lib as B  # noqa: E402

from lhcb_velo_toy.solvers.quantum.shots import ShotSampler, on_items, output_distribution, shot_grid  # noqa: E402
from lhcb_velo_toy.solvers.reconstruction.coupled_readout import coupled_hit_groups  # noqa: E402

L, F7, CF = B.L, B.F7, B.CF
EPS_TOY = F7.EPS
EPS_REAL = 0.005
GRID = shot_grid(1e1, 1e8, 2)
SEEDS = range(5)
KS = (1, 2)
RULES = ("count", "est")
HIT_SOLVERS = ("onebit8", "poly6s", "step2")
SEG_SOLVERS = ("1bqf", "qsvt")
METRICS = ("efficiency", "ghost_rate", "clone_fraction", "n_candidates", "n_matched", "n_reconstructible")


def rounds(P):
    return max(1.0, math.pi / (4 * math.sqrt(P))) if P > 0 else np.nan


def toy_jobs():
    J = [("toy", T, npv, rep) for T in (100, 400, 1000) for npv in (1, 5, 10, 20)
         for rep in range(5 if T < 700 else 3)]
    return J[::-1]                       # largest first


def real_jobs():
    import real_lib as R
    return [("real", str(f)) for f in R.dump_files()]


def curves(key, p, p_min, readout, limits, base, seed0):
    """Shot curves for one solver. readout(on, counts) -> metrics dict."""
    rows = []
    jit = np.random.default_rng(seed0 + 99).random(p.size) * 1e-3
    for seed in SEEDS:
        s = ShotSampler(p, seed=seed0 + seed)
        live = {(r, k): 0 for r in RULES for k in KS}        # consecutive saturated points
        for N in GRID:
            if all(v >= 2 for v in live.values()):
                break
            c = s.counts_at(N)
            for r in RULES:
                for k in KS:
                    if live[(r, k)] >= 2:
                        continue
                    on = on_items(c, N, k=k, p_min=p_min if r == "est" else None)
                    met = readout(on, c + jit)
                    rows.append(dict(base, solver=key, rule=r, k=k, seed=seed, N=int(N), n_on=int(on.sum()),
                                     **{m: met[m] for m in METRICS}))
                    live[(r, k)] = live[(r, k)] + 1 if met["efficiency"] >= limits[r] - 0.005 else 0
    return rows


def run(job):
    t0 = time.perf_counter()
    if job[0] == "real":
        import real_lib as R
        path = job[1]
        ev, _ = R.load_real(path, seed=zlib.crc32((Path(path).parent.name + Path(path).stem).encode()))
        F7.EPS = EPS_REAL
        name = f"{Path(path).parent.name}/{Path(path).stem}"
        base = dict(event=name)
        seed0 = zlib.crc32(name.encode()) % 100000
    else:
        _, T, npv, rep = job
        ev = L.generate_multi_pv(npv, T, seed=30000 + 1000 * npv + 10 * rep + T)
        F7.EPS = EPS_TOY
        base = dict(event=f"T{T}_pv{npv}_r{rep}", T=T, n_pv=npv, rep=rep)
        seed0 = 30000 + 1000 * npv + 10 * rep + T
    hid = np.array([hh.hit_id for hh in ev.pkg_event.hits])
    hitpos = -np.ones(hid.max() + 1, int)
    hitpos[hid] = np.arange(len(hid))
    F7.hitpos_inv = hid
    reco = ev.n_hits_of_track >= 3
    base.update(n_hits=len(hid), n_reco=int(reco.sum()))
    ev_rows, cur_rows = [], []

    def score(tracks):
        met, _ = F7.validate(ev, tracks)
        return met

    # ── hit register at the scan vertices ────────────────────────────────────
    vs, _ = L.find_vertices(ev)
    fz = [v[0] for v in vs]
    keep = {}
    brow, _ = B.evaluate(ev, fz, "scan", keep)
    C = keep["C"]
    n = C.shape[0]

    def hit_readout(on, counts):
        g = CF.tracks_from_filtered_hits(C, on.astype(float), threshold=0.5, min_hits=3)
        return score(F7.make_tracks([list(x) for x in g], ev))

    for key in HIT_SOLVERS:
        r, tau = keep[key]
        y = r["y"]
        P = float(y @ y)
        p = output_distribution(y)
        p_min = tau ** 2 / (n * P) * (1 - 1e-9)
        lim = {"est": hit_readout(p >= p_min, p), "count": hit_readout(p > 0, p)}
        pre = f"scan_{key}"
        row = dict(base, register="hit", solver=key, n_items=n, P=P, P_padded=brow[f"{pre}_P_padded"],
                   pad_leak=r["P_pad_leak"], p_min=p_min, mass_pmin=float(p[p >= p_min].sum()),
                   mass_true=float(p[reco[np.maximum(ev.track, 0)] & (ev.track >= 0)].sum()), n_pmin=int((p >= p_min).sum()), n_pos=int((p > 0).sum()),
                   qubits=brow.get(f"{pre}_qubits", np.nan), gates_per_solved=brow.get(f"{pre}_gates", np.nan),
                   rounds=brow.get(f"{pre}_rounds", np.nan), ref_efficiency=brow[f"{pre}_efficiency"],
                   ref_ghost_rate=brow[f"{pre}_ghost_rate"])
        for rl in RULES:
            row.update({f"inf_{rl}_{m}": lim[rl][m] for m in METRICS})
        ev_rows.append(row)
        cur_rows += curves(key, p, p_min, hit_readout, {rl: lim[rl]["efficiency"] for rl in RULES}, base, seed0)

    # ── segment register ─────────────────────────────────────────────────────
    h, truth, sols, sres = F7.segment_solutions(ev, int(reco.sum()))
    A = h.A.tocsr()
    seg_hits = np.asarray(h._segment_to_hit_ids)
    iso_seg = np.diff(A.indptr) == 1
    pad = sres["seg_n"] / 2.0 ** sres["seg_ns"]
    price = {"1bqf": (2, 2, sres["seg_P_1b"]), "qsvt": (40, 4, sres["seg_P_q"])}

    def seg_readout(on, scores):
        g = coupled_hit_groups(A, seg_hits, on, solution=scores, uniqueness="greedy")
        return score(F7.make_tracks([list(hitpos[x]) for x in g], ev))

    for key in ("classical",) + SEG_SOLVERS:
        x = sols[key]
        iso = float(x[iso_seg].max()) if iso_seg.any() else 0.0
        tau = iso * (1 + 1e-6) + 1e-12
        exact = seg_readout(x >= tau, x)
        row = dict(base, register="segment", solver=key, n_items=sres["seg_n"],
                   ref_efficiency=exact["efficiency"], ref_ghost_rate=exact["ghost_rate"])
        if key in SEG_SOLVERS:
            calls, anc, Pr = price[key]
            p = output_distribution(x)
            p_min = (tau ** 2 / float(x @ x)) * (1 - 1e-9)
            lim = {"est": seg_readout(p >= p_min, p), "count": seg_readout(p > 0, p)}
            Pp = Pr * pad
            row.update(P=Pr, P_padded=Pp, p_min=p_min, mass_pmin=float(p[p >= p_min].sum()),
                       mass_true=float(p[truth].sum()), n_pmin=int((p >= p_min).sum()), n_pos=int((p > 0).sum()),
                       qubits=sres["seg_ns"] + anc, rounds=rounds(Pp),
                       gates_per_solved=calls * 2 * sres["seg_cx_givens"] * rounds(Pp))
            for rl in RULES:
                row.update({f"inf_{rl}_{m}": lim[rl][m] for m in METRICS})
            cur_rows += curves(key, p, p_min, seg_readout, {rl: lim[rl]["efficiency"] for rl in RULES}, base, seed0)
        ev_rows.append(row)
    t = time.perf_counter() - t0
    for r in ev_rows:
        r["t_event"] = t
    return ev_rows, cur_rows


if __name__ == "__main__":
    mode = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    if mode == "real":
        RR = B.load("run_real", B.RE / "run_real.py")      # widens the scan range to [-250, 250) mm
        assert L.COARSE["lo"] == -250.0
    J = real_jobs() if mode == "real" else toy_jobs()
    if len(sys.argv) > 3:                                   # smoke test: first n jobs of a list
        J = J[-int(sys.argv[3]):]
    print(len(J), "events", flush=True)
    t0 = time.time()
    E, Cr = [], []
    with Pool(nproc, maxtasksperchild=2) as pool:
        for i, (e, c) in enumerate(pool.imap_unordered(run, J, chunksize=1)):
            E += e
            Cr += c
            print(f"{i + 1}/{len(J)} {e[0]['event']}: {e[0]['t_event']:.0f}s", flush=True)
    tag = mode if len(sys.argv) <= 3 else f"{mode}_smoke"
    pd.DataFrame(E).sort_values(["event", "register", "solver"]).to_csv(OUT / f"{tag}_shots_events.csv", index=False)
    pd.DataFrame(Cr).to_csv(OUT / f"{tag}_shots_curves.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
