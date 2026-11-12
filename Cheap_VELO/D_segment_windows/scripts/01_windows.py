#!/usr/bin/env python3
"""Experiment D: physics windows on the segment graph, with per-component circuits and shots.

Plan: Notion research plan "cheap and effective VELO track finding", candidate D (windowed
segment graph). Events: the A / B / C / G events (toy T = 100, 400, 1000 x 1, 5, 10, 20
vertices, 52 events; 103 Run 3 events).

Configurations (one segment system each, all from the package SimpleHamiltonianFast, step
kernel, gamma 3, delta 1):
  L4, L2   coupling angle epsilon x 4 and x 2, no window (the LOOSER direction: larger
           components; does the problem grow?)
  W0       the baseline: epsilon of A / C (toy: formula 1.70 mrad; Run 3: 5 mrad), no window
  W1 - W5  baseline epsilon + windows (package solvers.hamiltonians.windows): max slope
           |tx|, |ty| and max transverse impact parameter d0 to the beamline
              W1 0.60, 10 mm   W2 0.45, 3 mm   W3 0.35, 1 mm   W4 0.30, 0.3 mm   W5 0.25, 0.1 mm
           and z of closest approach inside the luminous region (Run 3 |z0| <= 250 mm; toy
           -200 ... 50 mm). A window gives exactly A[keep][:, keep] (restrict_system), so it can
           only shrink components. Segments join consecutive modules only (no layer skips in the
           package construction), so there is no layer-gap window to scan.
Per configuration:
  segments kept, true-segment retention, components (count with >= 2 items, largest size,
  qubits ceil(log2 m) + 2 / + 4), classical eigendecomposition time of all components;
  exact outputs per component (C's per_component): classical A^-1 1, one-bit filter (1BQF),
  QSVT comb (degree 40, domain from the windowed spectrum); reading I (above the
  isolated-segment value) with the coupled readout + greedy uniqueness (package
  get_tracks_coupled rule with the deterministic tie-break, sig_digits 9); the no-solver
  control (every coupled segment, score = coupling count + 1e-3 x fixed random);
  EventValidator efficiency and ghosts, per-track found flags (categories in 02);
  gates per solved shot per component (C's account: Givens price on the component's own
  register, r = 2, calls 2 (1BQF) / 40 (QSVT), amplification capped at 79 rounds for
  components without kept items);
  shots for the 1BQF: C's per-component allocation ceil(beta m ln m), estimate rule, 3 seeds,
  every component with >= 2 items a circuit; the target is the exact output with ties broken
  at random (mean of 5 draws), as in C. QSVT: exact output and gates per solved shot only.
Usage:  01_windows.py toy|real [n_proc] [n_smoke]
Output: outputs/<mode>_win_events.csv (event x config x solver), outputs/<mode>_win_tracks.csv
        (truth track x config: found flags), outputs/<mode>_win_curves.csv (1BQF shots)
"""
import importlib.util
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
PKG = "/data/bfys/gscriven/worktrees/velo-toy-cheap-velo-D/src"
os.environ["VELO_TOY_SRC"] = PKG
sys.path.insert(0, str(ROOT.parent / "B_graded_clique_filter" / "scripts"))
import b_lib as B  # noqa: E402  (loads the package from PKG)

_spec = importlib.util.spec_from_file_location("c1", ROOT.parent / "C_small_circuits" / "scripts" / "01_components.py")
C1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C1)
import lhcb_velo_toy  # noqa: E402
from lhcb_velo_toy.solvers import SimpleHamiltonianFast  # noqa: E402
from lhcb_velo_toy.solvers.hamiltonians.windows import restrict_system, segment_window_mask  # noqa: E402
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse  # noqa: E402
from lhcb_velo_toy.solvers.quantum.components import (  # noqa: E402
    ComponentShotSampler, component_givens_price, shot_allocation, split_components)
from lhcb_velo_toy.solvers.reconstruction.coupled_readout import coupled_hit_groups  # noqa: E402

assert lhcb_velo_toy.__file__.startswith(PKG), lhcb_velo_toy.__file__
L, F7 = B.L, B.F7
EPS_TOY, EPS_REAL = F7.EPS, 0.005
S, T1B = 4.0, np.pi / 4.0
WINDOWS = {"W1": (0.60, 10.0), "W2": (0.45, 3.0), "W3": (0.35, 1.0), "W4": (0.30, 0.3), "W5": (0.25, 0.1)}
CONFIGS = ["L4", "L2", "W0"] + list(WINDOWS)
BETAS = C1.BETAS
SEEDS = range(3)
MAX_COMP = 4000          # larger components are recorded but not solved densely
METRICS = ("efficiency", "ghost_rate", "n_matched", "n_reconstructible", "n_candidates", "n_ghosts")


def toy_jobs():
    return [("toy", T, npv, rep) for T in (100, 400, 1000) for npv in (1, 5, 10, 20)
            for rep in range(5 if T < 700 else 3)][::-1]


def real_jobs():
    import real_lib as R
    return [("real", str(f)) for f in R.dump_files()]


def build(ev, eps):
    h = SimpleHamiltonianFast(epsilon=eps, gamma=3.0, delta=1.0)
    h.construct_segments(ev.pkg_event, materialize_segments=False)
    h.construct_hamiltonian(ev.pkg_event)
    tid = np.asarray(h._segment_track_ids)
    truth = (tid[:, 0] == tid[:, 1]) & (tid[:, 0] >= 0)
    return h, truth


def run(job):
    t0 = time.perf_counter()
    if job[0] == "real":
        import real_lib as R
        path = job[1]
        ev, ex = R.load_real(path, seed=zlib.crc32((Path(path).parent.name + Path(path).stem).encode()))
        eps0, zr = EPS_REAL, (-250.0, 250.0)
        name = f"{Path(path).parent.name}/{Path(path).stem}"
        base = dict(event=name)
    else:
        _, T, npv, rep = job
        ev = L.generate_multi_pv(npv, T, seed=30000 + 1000 * npv + 10 * rep + T)
        ex = None
        eps0, zr = EPS_TOY, (-200.0, 50.0)
        name = f"T{T}_pv{npv}_r{rep}"
        base = dict(event=name, T=T, n_pv=npv, rep=rep)
    seed0 = zlib.crc32(name.encode()) % 100000
    hid = np.array([hh.hit_id for hh in ev.pkg_event.hits])
    hitpos = -np.ones(hid.max() + 1, int)
    hitpos[hid] = np.arange(len(hid))
    F7.hitpos_inv = hid
    reco = ev.n_hits_of_track >= 3
    base.update(n_hits=len(hid), n_reco=int(reco.sum()))
    E, TR, CU = [], [], []
    found = {}
    systems = {}
    for cfg in CONFIGS:
        eps = eps0 * (4 if cfg == "L4" else 2 if cfg == "L2" else 1)
        if eps not in systems:
            systems[eps] = build(ev, eps)
        h, truth_all = systems[eps]
        if cfg in WINDOWS:
            ms, md = WINDOWS[cfg]
            keep = segment_window_mask(h._segment_endpoints_xyz, max_slope=ms, max_d0=md, z_range=zr)
        else:
            keep = np.ones(h.A.shape[0], bool)
        A, b, idx = restrict_system(h.A, h.b, keep)
        N = A.shape[0]
        sh = np.asarray(h._segment_to_hit_ids)[idx]
        truth = truth_all[idx]
        comps, lab = split_components(A)
        sizes = np.array([len(c) for c in comps])
        row0 = dict(base, config=cfg, epsilon=eps, n_seg_all=int(h.A.shape[0]), n_seg=N,
                    true_kept=float(truth.sum() / max(truth_all.sum(), 1)), n_true=int(truth_all.sum()),
                    n_comp2=int((sizes > 1).sum()), max_comp=int(sizes.max()) if N else 0,
                    max_qubits_sys=int(math.ceil(math.log2(max(sizes.max(), 2)))) if N else 0)
        if N == 0 or sizes.max() > MAX_COMP:
            E.append(dict(row0, solver="skipped"))
            continue

        def readout(on, score, sig=9):
            g = coupled_hit_groups(A, sh, on, solution=score, uniqueness="greedy", sig_digits=sig)
            met, mt = F7.validate(ev, F7.make_tracks([list(hitpos[q]) for q in g], ev))
            return met, mt

        price = component_givens_price(A, comps, np.arange(N), C1.mcrx)
        x1c, lam, t_eig = C1.per_component(A, comps, lambda w: np.exp(-1j * w * T1B / 2) * np.cos(w * T1B / 2), True)
        lo_, hi_ = float(min(lam.min() if lam.size else S, S)), float(max(lam.max() if lam.size else S, S))
        pad = 0.05 * (hi_ - lo_) + 0.2
        comb = design_line_comb_inverse(degree=40, s=S, hw=0.18, domain=(lo_ - pad, hi_ + pad))
        xqc, _, _ = C1.per_component(A, comps, lambda w: comb(w), True)
        xcc, _, _ = C1.per_component(A, comps, lambda w: 1.0 / w, True)
        single = sizes[lab] == 1
        outs = {"1bqf": np.abs(x1c), "qsvt": np.abs(xqc.real), "classical": np.abs(xcc.real)}
        outs["qsvt"][single] = abs(float(comb(S)))
        outs["classical"][single] = 1.0 / S
        row0["t_eigh"] = t_eig
        for key in ("classical", "1bqf", "qsvt"):
            x = outs[key]
            iso = float(x[single].max()) if single.any() else 0.0
            tau = iso * (1 + 1e-6) + 1e-12
            exact_on = x >= tau
            met, mt = readout(exact_on, x)
            found[f"{cfg}_{key}"] = mt
            row = dict(row0, solver=key, seg_eff=float((exact_on & truth).sum() / max(truth_all.sum(), 1)),
                       seg_far=float((exact_on & ~truth).sum() / max(exact_on.sum(), 1)),
                       **{m: met[m] for m in METRICS})
            if key in ("1bqf", "qsvt"):
                s_c = np.bincount(lab, weights=x ** 2, minlength=len(comps))
                empty = np.bincount(lab, weights=exact_on.astype(float), minlength=len(comps)) == 0
                acc = C1.account(key, comps, price, s_c / sizes, 0.0, empty)
                circ = sizes > 1
                row.update(max_qubits=int(acc["qubits"][circ].max()) if circ.any() else 0,
                           sum_gates_solved=float(np.nansum(acc["gates_solved"][circ])))
                if key == "1bqf":
                    rng_t = np.random.default_rng(seed0 + 7)
                    xr = np.round(x / x.max(), 9)
                    row["exact_rand_efficiency"] = float(np.mean(
                        [readout(exact_on, xr + 1e-12 * rng_t.random(N), sig=None)[0]["efficiency"] for _ in range(5)]))
                    ok = (s_c > 0) & circ
                    p = np.where(ok[lab], x ** 2 / np.where(ok, s_c, 1.0)[lab], 0.0)
                    pm = np.where(ok, tau ** 2 / np.where(ok, s_c, 1.0) * (1 - 1e-9), 0.0)
                    cidx = np.flatnonzero(circ)
                    sub = [comps[c] for c in cidx]
                    Nitem = np.zeros(N)
                    jit = np.random.default_rng(seed0 + 99).random(N) * 1e-3
                    for seed in SEEDS:
                        smp = ComponentShotSampler(p, sub, seed=seed0 + seed)
                        for beta in BETAS:
                            budget = shot_allocation(sizes[cidx], beta)
                            cnt = smp.counts_at(budget)
                            Nitem[np.concatenate(sub)] = np.repeat(budget, sizes[cidx])
                            on = (cnt >= 1) & (cnt >= pm[lab] * Nitem)
                            m2, _ = readout(on, cnt + jit)
                            CU.append(dict(base, config=cfg, solver=key, beta=beta, seed=seed,
                                           shots=int(budget.sum()),
                                           gates=float(np.nansum(budget * acc["gates_solved"][cidx])),
                                           **{m: m2[m] for m in METRICS}))
            E.append(row)
        # no-solver control
        deg = np.diff(A.indptr) - 1
        crng = np.random.default_rng(zlib.crc32(name.encode()))
        crng.random(N)
        met, mt = readout(deg >= 1, deg + 1e-3 * crng.random(N))
        found[f"{cfg}_control"] = mt
        E.append(dict(row0, solver="control", **{m: met[m] for m in METRICS}))
    for t in np.nonzero(reco)[0]:
        d = dict(base, n_hits=int(ev.n_hits_of_track[t]))
        if ex is not None:
            d.update(prompt=bool(ex.prompt[t]), electron=bool(ex.electron[t]), p=float(ex.p[t]))
        d.update({f"found_{k}": int(t) in v for k, v in found.items()})
        TR.append(d)
    tt = time.perf_counter() - t0
    for r in E:
        r["t_event"] = tt
    return E, TR, CU


if __name__ == "__main__":
    mode = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    if mode == "real":
        B.load("run_real", B.RE / "run_real.py")          # real_lib on the path (scan range unused here)
    J = real_jobs() if mode == "real" else toy_jobs()
    tag = mode
    if len(sys.argv) > 3:
        J = J[-int(sys.argv[3]):]
        tag = f"{mode}_smoke"
    print(len(J), "events", flush=True)
    t0 = time.time()
    E, TR, CU = [], [], []
    with Pool(nproc, maxtasksperchild=1) as pool:
        for i, (e, t, c) in enumerate(pool.imap_unordered(run, J, chunksize=1)):
            E += e
            TR += t
            CU += c
            print(f"{i + 1}/{len(J)} {e[0]['event']}: {e[0]['t_event']:.0f}s", flush=True)
    pd.DataFrame(E).to_csv(OUT / f"{tag}_win_events.csv", index=False)
    pd.DataFrame(TR).to_csv(OUT / f"{tag}_win_tracks.csv", index=False)
    pd.DataFrame(CU).to_csv(OUT / f"{tag}_win_curves.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
