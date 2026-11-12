#!/usr/bin/env python3
"""Experiment C: every connected component as its own small circuit.

Plan: Notion research plan "cheap and effective VELO track finding", candidate C (decomposed
small circuits). Events: the A / B / G events (toy T = 100, 400, 1000 x 1, 5, 10, 20 vertices,
52 events; 103 Run 3 events).

Solvers (as experiment G): hit register at the focusing-scan vertices with the one-bit filter on
8I - C (onebit8) and the degree-6 shifted step polynomial (poly6s); segment register with the
one-bit filter (1bqf) and the QSVT comb (qsvt), truth-free reading I, coupled readout.

Per event and solver:
  1. Split the operator (C for hits, A for segments) into connected components
     (package solvers.quantum.components.split_components).
  2. Exactness. Each component is filtered on its own register with its own uniform probe
     (dense eigh per component, timed: that is the classical cost of the same problem). The
     outputs, rescaled to the whole-register probe, are compared with
       (a) the whole-register output used by B / G (per-component eigh inside mode_system or
           07_full_sweep.segment_solutions), and
       (b) an independent whole-register evaluation on the full sparse matrix with no split:
           cos / exp via scipy expm_multiply for the one-bit filters, Clenshaw recursion for the
           polynomials (step polynomial, comb).
     The tracks read from the per-component outputs are scored and compared with the tracks of
     the whole-register output read the same way (segments: scores rounded to 9 relative digits
     before the greedy uniqueness, because exact amplitude ties are common and round-off would
     otherwise pick the winner; see the README), and with G's exact efficiencies.
  3. Resources per component: items m, register qubits n_s = ceil(log2 m) (>= 1), + ancillas
     (2 one-bit filter, 4 polynomial / comb), Givens price per product step on the component's
     own register (package component_givens_price; MCRX CX counts transpiled for 1-14
     controls, MCRX_CX below, the 05 fit above), gates per solved shot = calls x r x step price x
     max(1, pi / (4 sqrt P_c,padded)), r = 4 (hits) / 2 (segments), calls 2 / 6 / 2 / 40.
     P_c = success probability of the component circuit with the uniform probe over its m
     items; padded P_c = P_c m / 2^n_s (+ f(0)^2 on the padded rows of the hit polynomial).
  4. Shots. Every circuit component gets N_c = ceil(beta m_c ln m_c) shots (package
     shot_allocation; the m ln m law of G), beta on a log grid, 5 seeds, nested multinomial per
     component (package ComponentShotSampler). Items on by G's estimate rule, per component:
     count >= 1 and count / N_c >= p_min,c, with p_min,c the exact reading as a probability in
     the component register. Readout and scoring as G. beta90 = smallest beta with mean
     efficiency >= exact - 0.02 (02_summary.py).
  5. Hybrid: components with at most t items (t = 1, 2, 4, 8) are solved classically (exact
     output, no circuit, no shots); only larger ones are circuits. t = 1 is the plain split
     (an isolated item has no coupling: no circuit can change it, f(diag) is known).

Usage:  01_components.py toy|real [n_proc] [n_smoke]
Output: outputs/<mode>_comp_events.csv (event x solver), outputs/<mode>_comp_sizes.csv
        (event x solver x size bin), outputs/<mode>_comp_curves.csv (event x solver x t x beta x seed)
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
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
os.environ["VELO_TOY_SRC"] = "/data/bfys/gscriven/worktrees/velo-toy-cheap-velo-C/src"
sys.path.insert(0, str(ROOT.parent / "B_graded_clique_filter" / "scripts"))
import b_lib as B  # noqa: E402

from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse  # noqa: E402
from lhcb_velo_toy.solvers.quantum.components import (  # noqa: E402
    ComponentShotSampler, component_givens_price, shot_allocation, split_components)
from lhcb_velo_toy.solvers.reconstruction.coupled_readout import coupled_hit_groups  # noqa: E402

L, F7, CF, MPV = B.L, B.F7, B.CF, B.MPV
EPS_TOY = F7.EPS
EPS_REAL = 0.005
S, T1B = 4.0, np.pi / 4.0
# MCRX CX counts, transpiled (qiskit, basis cx/rz/sx/x, optimisation 1) for 1 ... 14 controls,
# 12 Nov 2026 (same recipe as 05_resources.mcrx_fit; that fit is negative below 3 controls)
MCRX_CX = {1: 2, 2: 8, 3: 20, 4: 24, 5: 40, 6: 56, 7: 80, 8: 104, 9: 120, 10: 136, 11: 152,
           12: 168, 13: 184, 14: 200}
BETAS = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
SEEDS = range(5)
THRESH = (1, 2, 4, 8)
PRICE = {"onebit8": (2, 2, 4), "poly6s": (6, 4, 4), "1bqf": (2, 2, 2), "qsvt": (40, 4, 2)}  # calls, ancillas, r
METRICS = ("efficiency", "ghost_rate", "n_matched", "n_reconstructible")
SIZE_BINS = [1, 2, 3, 5, 9, 17, 33, 65, 129, 10 ** 9]


def mcrx(m):
    return float(MCRX_CX[m]) if m in MCRX_CX else MPV.mcrx_cx(m)


# Amplification cap for EMPTY components. A component that holds no item of the exact reading
# can have P ~ 1e-28 (its output cancels), where pi / (4 sqrt P) would be astronomical. It is
# run with at most R_CAP rounds (P floor 1e-4); nothing is flagged and it is read as empty.
# A component that holds kept items always gets its full rounds, however small its P (the QSVT
# comb has many such components). Which components are empty is taken from the exact output:
# an accounting choice, stated in the README.
P_FLOOR = 1e-4
R_CAP = np.pi / (4 * np.sqrt(P_FLOOR))


def rounds(P, empty):
    r = np.maximum(1.0, np.pi / (4 * np.sqrt(np.maximum(P, 1e-300))))
    return np.where(empty, np.minimum(R_CAP, r), r)


def clenshaw(poly, M, v):
    """p(M) v for a numpy Chebyshev series with a domain, on the whole sparse matrix."""
    lo, hi = poly.domain
    X = (2.0 * M - (lo + hi) * sp.identity(M.shape[0], format="csr")) / (hi - lo)
    c = poly.coef
    b1 = np.zeros_like(v)
    b2 = np.zeros_like(v)
    for ck in c[:0:-1]:
        b1, b2 = 2 * (X @ b1) - b2 + ck * v, b1
    return X @ b1 - b2 + c[0] * v


def toy_jobs():
    return [("toy", T, npv, rep) for T in (100, 400, 1000) for npv in (1, 5, 10, 20)
            for rep in range(5 if T < 700 else 3)][::-1]


def real_jobs():
    import real_lib as R
    return [("real", str(f)) for f in R.dump_files()]


def per_component(M, comps, f_of_lam, probe_ones):
    """Filter every component of size > 1 on its own register (dense eigh). Returns the output
    for a probe of ones on the component (rescale outside), the eigenvalues, and the time."""
    t0 = time.perf_counter()
    y = np.zeros(M.shape[0], dtype=complex)
    lams = []
    Mc = M.tocsr()
    for idx in comps:
        if len(idx) == 1:
            continue
        w, V = eigh(Mc[idx][:, idx].toarray())
        lams.append(w)
        y[idx] = V @ (f_of_lam(w) * (V.T @ np.ones(len(idx))))
    return y, (np.concatenate(lams) if lams else np.zeros(0)), time.perf_counter() - t0


def account(key, comps, price, P_c, Pleak, empty):
    calls, anc, r = PRICE[key]
    m = price["m"]
    ns = price["n_s"]
    Pp = P_c * m / 2.0 ** ns + Pleak
    g = calls * r * price["cx_step"] * rounds(Pp, empty)
    return dict(m=m, ns=ns, qubits=ns + anc, cx_step=price["cx_step"], P=P_c, P_padded=Pp, gates_solved=g)


def shots(key, base, comps, lab, acc, p, p_min_c, exact_on, readout, seed0):
    """Per-component shot budgets beta m ln m; hybrids t: components with m <= t classical."""
    rows = []
    jit = np.random.default_rng(seed0 + 99).random(p.size) * 1e-3     # tie-break among equal counts, as G
    m = acc["m"]
    circ = np.flatnonzero(m > 1)                      # components that can be circuits
    sub = [comps[c] for c in circ]
    pm = np.zeros(p.size)
    for c in circ:
        pm[comps[c]] = p_min_c[c]
    item_comp_size = m[lab].astype(int)
    for seed in SEEDS:
        smp = ComponentShotSampler(p, sub, seed=seed0 + seed)
        for beta in BETAS:
            budget = shot_allocation(m[circ], beta)
            cnt = smp.counts_at(budget)
            Nitem = np.zeros(p.size)
            for b, c in zip(budget, circ):
                Nitem[comps[c]] = b
            on_shot = (cnt >= 1) & (cnt >= pm * Nitem)
            score = cnt + jit
            for t in THRESH:
                classical = item_comp_size <= t
                on = np.where(classical, exact_on, on_shot)
                met = readout(on, score)
                use = m[circ] > t
                rows.append(dict(base, solver=key, t=t, beta=beta, seed=seed,
                                 n_circuits=int(use.sum()), shots=int(budget[use].sum()),
                                 gates=float(np.nansum(budget[use] * acc["gates_solved"][circ][use])),
                                 **{k: met[k] for k in METRICS}))
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
    E, SZ, CU = [], [], []

    def score(tracks):
        return F7.validate(ev, tracks)[0]

    def size_rows(key, m):
        h, _ = np.histogram(m, bins=SIZE_BINS)
        return [dict(base, solver=key, size_lo=lo, size_hi=hi - 1, n_comp=int(c))
                for lo, hi, c in zip(SIZE_BINS[:-1], SIZE_BINS[1:], h)]

    # ── hit register ─────────────────────────────────────────────────────────
    vs, _ = L.find_vertices(ev)
    fz = [v[0] for v in vs]
    keep = {}
    brow, _ = B.evaluate(ev, fz, "scan", keep)
    C = keep["C"].tocsr().astype(float)
    n = C.shape[0]
    reg = B.register_order(ev)
    comps, lab = split_components(C)
    price = component_givens_price(C, comps, reg, mcrx)

    def hit_readout(on, _scores):
        g = CF.tracks_from_filtered_hits(C, on.astype(float), threshold=0.5, min_hits=3)
        return score(F7.make_tracks([list(x) for x in g], ev))

    u = np.full(n, 1.0 / math.sqrt(n))
    for key in ("onebit8", "poly6s"):
        r, tau = keep[key]
        f = B.responses(brow["scan_alpha"], brow["scan_lo"])[key]
        yc, lam, t_eig = per_component(C, comps, lambda w: np.asarray(f(w), float), True)
        yc = yc.real
        f0 = float(np.asarray(f(np.array([0.0])))[0])
        for idx in comps:                                   # isolated hits: f(0) x probe
            if len(idx) == 1:
                yc[idx] = f0
        # component register probe 1 / sqrt(m): output per item = yc / sqrt(m)
        msz = np.zeros(n)
        for idx in comps:
            msz[idx] = len(idx)
        y_comp_probe = yc / np.sqrt(msz)
        y_whole_from_comp = yc / math.sqrt(n)               # ones -> whole-register probe
        d_b = float(np.max(np.abs(y_whole_from_comp - r["y"])))
        if key == "onebit8":
            th = (np.pi / 16.0) * (8.0 * sp.identity(n, format="csr") - C)
            y_ind = expm_multiply(-1j * th, u.astype(complex)).real
        else:
            y_ind = clenshaw(f, C, u)
        d_ind = float(np.max(np.abs(y_whole_from_comp - y_ind)))
        P_c = np.array([float(y_comp_probe[idx] @ y_comp_probe[idx]) for idx in comps])
        leak = np.where(price["m"] > 0, (2.0 ** price["n_s"] - price["m"]) / 2.0 ** price["n_s"] * f0 ** 2, 0.0)
        weight = np.abs(y_whole_from_comp) * math.sqrt(n)
        exact_on = weight >= tau
        empty = np.bincount(lab, weights=exact_on.astype(float), minlength=len(comps)) == 0
        acc = account(key, comps, price, P_c, leak, empty)
        ex = hit_readout(exact_on, weight)
        # estimate rule per component: weight >= tau <=> p_i,c >= tau^2 / (m_c P_c)
        p = np.zeros(n)
        pmin = np.zeros(len(comps))
        for c, idx in enumerate(comps):
            s_ = float(y_comp_probe[idx] @ y_comp_probe[idx])
            if s_ > 0:
                p[idx] = y_comp_probe[idx] ** 2 / s_
                pmin[c] = tau ** 2 / (len(idx) * s_) * (1 - 1e-9)
        circ = acc["m"] > 1
        capped = circ & empty & (acc["P_padded"] < P_FLOOR)
        kept_in_capped = int(exact_on[capped[lab]].sum())
        E.append(dict(base, n_comp_capped=int(capped.sum()), kept_in_capped=kept_in_capped, register="hit", solver=key, n_items=n, n_comp=len(comps),
                      n_comp_gt1=int(circ.sum()), max_items=int(acc["m"].max()),
                      max_qubits_sys=int(acc["ns"][circ].max()) if circ.any() else 0,
                      max_qubits=int(acc["qubits"][circ].max()) if circ.any() else 0,
                      sum_gates_solved=float(np.nansum(acc["gates_solved"][circ])),
                      max_gates_solved=float(np.nanmax(acc["gates_solved"][circ])) if circ.any() else 0.0,
                      **{f"n_circ_t{t}": int((acc["m"] > t).sum()) for t in THRESH},
                      **{f"sum_gates_solved_t{t}": float(np.nansum(acc["gates_solved"][acc["m"] > t])) for t in THRESH},
                      max_diff_vs_whole=d_b, max_diff_vs_independent=d_ind, t_eigh=t_eig,
                      exact_efficiency=ex["efficiency"], exact_ghost_rate=ex["ghost_rate"],
                      ref_efficiency=brow[f"scan_{key}_efficiency"]))
        SZ += size_rows(key, acc["m"])
        CU += shots(key, base, comps, lab, acc, p, pmin, exact_on, hit_readout, seed0)

    # ── segment register ─────────────────────────────────────────────────────
    h, truth, sols, sres = F7.segment_solutions(ev, int(reco.sum()))
    A = h.A.tocsr()
    N = A.shape[0]
    seg_hits = np.asarray(h._segment_to_hit_ids)
    iso_seg = np.diff(A.indptr) == 1
    comps, lab = split_components(A)
    price = component_givens_price(A, comps, np.arange(N), mcrx)
    single = price["m"][lab] == 1

    def seg_readout(on, scores):
        g = coupled_hit_groups(A, seg_hits, on, solution=scores, uniqueness="greedy")
        return score(F7.make_tracks([list(hitpos[x]) for x in g], ev))

    ones = np.ones(N)
    x1c, lam, t_eig = per_component(A, comps, lambda w: np.exp(-1j * w * T1B / 2) * np.cos(w * T1B / 2), True)
    lo_, hi_ = float(min(lam.min(), S)), float(max(lam.max(), S))
    pad = 0.05 * (hi_ - lo_) + 0.2
    comb = design_line_comb_inverse(degree=40, s=S, hw=0.18, domain=(lo_ - pad, hi_ + pad))
    xqc, _, t_eig_q = per_component(A, comps, lambda w: comb(w), True)
    outs = {"1bqf": np.abs(x1c), "qsvt": np.abs(xqc.real)}
    outs["qsvt"][single] = abs(float(comb(S)))                # isolated: (1 + e^{-isT})/2 = 0, |comb(s)|
    ind = {"1bqf": np.abs((ones + expm_multiply(-1j * T1B * A, ones.astype(complex))) / 2),
           "qsvt": np.abs(clenshaw(comb, A, ones))}
    for key in ("1bqf", "qsvt"):
        x = outs[key]
        d_b = float(np.max(np.abs(x - sols[key])))
        d_ind = float(np.max(np.abs(x - ind[key])))
        iso = float(x[iso_seg].max()) if iso_seg.any() else 0.0
        tau = iso * (1 + 1e-6) + 1e-12
        exact_on = x >= tau
        # Greedy hit uniqueness keeps the larger amplitude; many segments tie exactly (symmetric
        # chains), so a 1e-15 difference between two exact evaluations changes the winner. Both
        # outputs are rounded to 9 relative digits before the readout, so ties are broken by the
        # same (index) order. The whole-register output read the same way must give the same tracks.
        ex = seg_readout(exact_on, np.round(x / x.max(), 9))
        xw = sols[key]
        exw = seg_readout(xw >= tau, np.round(xw / xw.max(), 9))
        exw_raw = seg_readout(xw >= tau, xw)
        # what a measurement sees: tied amplitudes give equal expected counts, so the shots break
        # ties at random. Exact output with ties broken at random, mean of 5 draws: the target of
        # the shot curves for segments (02_summary.py).
        rng_t = np.random.default_rng(seed0 + 7)
        xr = np.round(x / x.max(), 9)
        ex_rand = [seg_readout(exact_on, xr + 1e-12 * rng_t.random(N)) for _ in range(5)]
        s_c = np.bincount(lab, weights=x ** 2, minlength=len(comps))
        P_c = s_c / price["m"]
        empty = np.bincount(lab, weights=exact_on.astype(float), minlength=len(comps)) == 0
        acc = account(key, comps, price, P_c, 0.0, empty)
        ok = (s_c > 0) & (price["m"] > 1)
        p = np.where(ok[lab], x ** 2 / np.where(ok, s_c, 1.0)[lab], 0.0)
        pmin = np.where(ok, tau ** 2 / np.where(ok, s_c, 1.0) * (1 - 1e-9), 0.0)
        circ = acc["m"] > 1
        capped = circ & empty & (acc["P_padded"] < P_FLOOR)
        kept_in_capped = int(exact_on[capped[lab]].sum())
        E.append(dict(base, n_comp_capped=int(capped.sum()), kept_in_capped=kept_in_capped, register="segment", solver=key, n_items=N, n_comp=len(comps),
                      n_comp_gt1=int(circ.sum()), max_items=int(acc["m"].max()),
                      max_qubits_sys=int(acc["ns"][circ].max()) if circ.any() else 0,
                      max_qubits=int(acc["qubits"][circ].max()) if circ.any() else 0,
                      sum_gates_solved=float(np.nansum(acc["gates_solved"][circ])),
                      max_gates_solved=float(np.nanmax(acc["gates_solved"][circ])) if circ.any() else 0.0,
                      **{f"n_circ_t{t}": int((acc["m"] > t).sum()) for t in THRESH},
                      **{f"sum_gates_solved_t{t}": float(np.nansum(acc["gates_solved"][acc["m"] > t])) for t in THRESH},
                      max_diff_vs_whole=d_b, max_diff_vs_independent=d_ind,
                      t_eigh=t_eig if key == "1bqf" else t_eig_q,
                      exact_efficiency=ex["efficiency"], exact_ghost_rate=ex["ghost_rate"],
                      whole_rounded_efficiency=exw["efficiency"], whole_raw_efficiency=exw_raw["efficiency"],
                      n_on_diff=int(np.sum(exact_on != (xw >= tau))), ref_efficiency=np.nan,
                      exact_rand_efficiency=float(np.mean([q["efficiency"] for q in ex_rand])),
                      exact_rand_sd=float(np.std([q["efficiency"] for q in ex_rand]))))
        SZ += size_rows(key, acc["m"])
        CU += shots(key, base, comps, lab, acc, p, pmin, exact_on, seg_readout, seed0)
    t = time.perf_counter() - t0
    for r in E:
        r["t_event"] = t
    return E, SZ, CU


if __name__ == "__main__":
    mode = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    if mode == "real":
        B.load("run_real", B.RE / "run_real.py")              # widens the scan range to [-250, 250) mm
        assert L.COARSE["lo"] == -250.0
    J = real_jobs() if mode == "real" else toy_jobs()
    tag = mode
    if len(sys.argv) > 3:
        J = J[-int(sys.argv[3]):]
        tag = f"{mode}_smoke"
    print(len(J), "events", flush=True)
    t0 = time.time()
    E, SZ, CU = [], [], []
    with Pool(nproc, maxtasksperchild=2) as pool:
        for i, (e, s, c) in enumerate(pool.imap_unordered(run, J, chunksize=1)):
            E += e
            SZ += s
            CU += c
            print(f"{i + 1}/{len(J)} {e[0]['event']}: {e[0]['t_event']:.0f}s", flush=True)
    pd.DataFrame(E).sort_values(["event", "solver"]).to_csv(OUT / f"{tag}_comp_events.csv", index=False)
    pd.DataFrame(SZ).to_csv(OUT / f"{tag}_comp_sizes.csv", index=False)
    pd.DataFrame(CU).to_csv(OUT / f"{tag}_comp_curves.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
