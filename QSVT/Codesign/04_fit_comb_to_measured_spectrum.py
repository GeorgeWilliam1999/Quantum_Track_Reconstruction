#!/usr/bin/env python3
"""Fit the polynomial to the MEASURED false pile-up (George's review, 2026-07-17).

Review of QSVT XI: study 03 retuned the *fixed* P4 line comb (centre + domain)
but never fit the response to where the false weight actually accumulates on
the heavy-noise (fork) Hamiltonian.  The demanded flow:

  1. take the events on the Hamiltonian (existing qtrk_pipeline events),
  2. MEASURE where the false amplitude piles up in the spectrum,
  3. FIT the polynomial with roots that eliminate the bad modes,
  4. re-measure — 1BQF/QSVT false rate should sit below classical;
     if it still does not, quantify exactly why.

Enabled by the probe: at heavy T=200 the compatibility graph has NO giant
component (max size ~17; ~870 nontrivial components), so the spectrum is
available EXACTLY by per-component dense eigh.  90% of wp99 false actives
live in components holding only ~142 true segments (nearly pure-false motifs).

MEASURED pile-up (rep 0): false-active formation concentrates on the FALSE
MOTIF LINES lam = s-1 (pair) and s-sqrt2 (triple), bin-overlap cosine with
the true formation only 0.022 — the fit's roots land there and crush ~1250 of
the 1454 false actives by ~4 orders of magnitude.

THE FLOOR THIS EXPOSES (the 'why' at eff >= 0.99): a handful of TRUE
segments (8/784 = 1.02% on rep 0) live in mixed 2-3 segment components whose
spectral formation is IDENTICAL to the false pair/triple motifs — any
lambda-response that roots the false lines kills them too (the mixed-motif
generalisation of the F3 floor theorem).  Because 8 > the 1% budget, the
strict eff>=0.99 working point is forced below the leak tier and the
suppressed band floods back.  One efficiency point lower the fit wins big.
Hence this study reports the FULL efficiency-matched frontier, not one gate.

Outputs: outputs/fork_noisy/fit_comb.{csv,png,log}; no new store rows.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.polynomial import chebyshev as npcheb
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import minres

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp  # noqa: E402
from qtrk_pipeline import hit_uniqueness_filter  # noqa: E402
import bif  # noqa: E402
from dp_matrix_characterisation import NOISES  # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "fork_noisy"
OUT.mkdir(parents=True, exist_ok=True)
GAMMA_REF, DELTA = 3.0, 1.0
T = 200
BIN_W = 0.02
EFF_TARGETS = (0.99, 0.985, 0.98, 0.97, 0.95)
MU_SELECT_EFF = 0.985          # mu chosen below the mixed-motif floor
MUS = np.geomspace(3e-3, 30.0, 13)
CHEB_DEGS = (40, 80, 160, 240)

CONFIGS = [("beta0", 0.0, "ref"), ("b0.5-gstar", 0.5, "gstar"),
           ("b1-gstar", 1.0, "gstar")]
REPS = (0, 1, 2)

LOG = open(OUT / "fit_comb.log", "w")


def say(*a):
    print(*a, flush=True)
    print(*a, file=LOG, flush=True)


def far_at_eff(x, truth, targets=EFF_TARGETS):
    """far when tau sits just below the k-th largest TRUE amplitude, k=ceil(eff*n).

    The cleanest efficiency-matched comparison: monotone in the target, no
    quantile-grid artefacts.  Returns {target: (eff_achieved, far, tau)}.
    """
    x = np.abs(np.asarray(x).ravel())
    truth = np.asarray(truth, bool)
    ts = np.sort(x[truth])[::-1]
    nt = len(ts)
    out = {}
    for tgt in targets:
        k = min(int(np.ceil(tgt * nt)), nt)
        tau = ts[k - 1] * (1 - 1e-12) if ts[k - 1] > 0 else 0.0
        act = x > tau
        e = int((act & truth).sum()) / nt
        f = int((act & ~truth).sum()) / max(int(act.sum()), 1)
        out[tgt] = (e, f, float(tau))
    return out


def roc(x, truth, npt=250):
    x = np.abs(np.asarray(x).ravel())
    ts = np.sort(x[truth])[::-1]
    ks = np.unique(np.linspace(1, len(ts), npt).astype(int))
    eff, far = [], []
    nt = len(ts)
    for k in ks:
        tau = ts[k - 1] * (1 - 1e-12)
        act = x > tau
        eff.append(int((act & truth).sum()) / nt)
        far.append(int((act & ~truth).sum()) / max(int(act.sum()), 1))
    return np.asarray(eff), np.asarray(far)


def build_system(rep, beta, gmode, eps, nz):
    ev, _ = qp.ensure_event(n_trk=T, rep=rep, **nz)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                               gamma=GAMMA_REF, delta=DELTA)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    seg_hits = np.asarray(ham._segment_to_hit_ids)
    n = int(ham.n_segments)
    A = ham.A.tocsr()
    if beta:
        B = bif.fork_graph_eps(ham._segment_to_hit_ids,
                               ham._segment_vectors, eps)
        A = (A + beta * B).tocsr()
    Off = A.copy()
    Off.setdiag(0)
    Off.eliminate_zeros()
    ncomp, lab = connected_components(Off, directed=False)
    sizes = np.bincount(lab)
    comps = {}
    lam_min = lam_max = GAMMA_REF + DELTA
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(ncomp + 1))
    for c in np.flatnonzero(sizes > 1):
        idx = order[bounds[c]:bounds[c + 1]]
        lam, V = eigh(A[np.ix_(idx, idx)].toarray())
        comps[c] = (idx, lam, V)
        lam_min, lam_max = min(lam_min, lam[0]), max(lam_max, lam[-1])
    g_pd = GAMMA_REF - lam_min
    g_win = lam_max - GAMMA_REF - 2 * DELTA
    g_star = max(GAMMA_REF, g_pd + 0.10, g_win + 0.10)
    g_use = GAMMA_REF if gmode == "ref" else g_star
    shift = g_use - GAMMA_REF
    comps = {c: (idx, lam + shift, V) for c, (idx, lam, V) in comps.items()}
    return dict(truth=truth, seg_hits=seg_hits, n=n,
                A=(A + shift * sp.identity(n, format="csr")).tocsr()
                if shift else A,
                comps=comps, iso=sizes[lab] == 1, s_p=g_use + DELTA,
                g_use=g_use, lam_min=lam_min + shift, lam_max=lam_max + shift)


def design_matrix(sysd, edges):
    nb = len(edges) - 1
    seg_rows, Dr = [], []
    for c, (idx, lam, V) in sysd["comps"].items():
        wb = V.T @ np.full(len(idx), DELTA)
        bi = np.clip(np.searchsorted(edges, lam) - 1, 0, nb - 1)
        M = np.zeros((len(idx), nb))
        contrib = V * wb[None, :]
        for i in range(len(lam)):
            M[:, bi[i]] += contrib[:, i]
        Dr.append(M)
        seg_rows.extend(idx.tolist())
    D = np.vstack(Dr)
    seg_rows = np.asarray(seg_rows)
    return D, seg_rows, sysd["truth"][seg_rows]


def amplitudes(sysd, p_of_lam):
    x = np.zeros(sysd["n"])
    x[sysd["iso"]] = abs(float(p_of_lam(np.array([sysd["s_p"]]))[0])) * DELTA
    for c, (idx, lam, V) in sysd["comps"].items():
        wb = V.T @ np.full(len(idx), DELTA)
        x[idx] = np.abs(V @ (p_of_lam(lam) * wb))
    return x


def binned_response(p_bins, edges):
    def f(lam):
        bi = np.clip(np.searchsorted(edges, np.asarray(lam)) - 1,
                     0, len(p_bins) - 1)
        return p_bins[bi]
    return f


def mode_table(sysd):
    rows = []
    for c, (idx, lam, V) in sysd["comps"].items():
        wb = V.T @ np.full(len(idx), DELTA)
        tm = sysd["truth"][idx]
        for i in range(len(lam)):
            vt = float((V[tm, i] ** 2).sum())
            rows.append((lam[i], wb[i], vt, 1.0 - vt))
    return np.asarray(rows)


def gate_frontier(x, truth, seg_hits, npt=40):
    """eff-matched frontier of threshold -> uniqueness-greedy gate."""
    x = np.abs(np.asarray(x).ravel())
    ts = np.sort(x[truth])[::-1]
    ks = np.unique(np.linspace(max(1, int(0.90 * len(ts))), len(ts),
                               npt).astype(int))
    eff, far = [], []
    for k in ks:
        tau = ts[k - 1] * (1 - 1e-12)
        act = hit_uniqueness_filter(x, seg_hits, x > tau, mode="greedy")
        eff.append(int((act & truth).sum()) / len(ts))
        far.append(int((act & ~truth).sum()) / max(int(act.sum()), 1))
    return np.asarray(eff), np.asarray(far)


def main():
    t00 = time.time()
    nz = NOISES["heavy"]
    eps = float(compute_epsilon(nz["sigma_res"], nz["sigma_scatt"]))
    say(f"[setup] heavy formula eps={eps*1e3:.3f} mrad, T={T}, bin={BIN_W}, "
        f"eff targets {EFF_TARGETS}")

    results, fitted_p0, panels = [], {}, {}
    for label, beta, gmode in CONFIGS:
        for rep in REPS:
            tt = time.time()
            sysd = build_system(rep, beta, gmode, eps, nz)
            truth, n, s_p = sysd["truth"], sysd["n"], sysd["s_p"]
            seg_hits = sysd["seg_hits"]
            edges = np.arange(sysd["lam_min"] - BIN_W,
                              sysd["lam_max"] + 2 * BIN_W, BIN_W)
            nb = len(edges) - 1

            x_cls = amplitudes(sysd, lambda l: 1.0 / l)
            xm, _ = minres(sysd["A"].tocsc(), DELTA * np.ones(n),
                           rtol=1e-10, maxiter=8000)
            xm = np.abs(np.asarray(xm).ravel())
            assert float(x_cls @ xm / (np.linalg.norm(x_cls)
                                       * np.linalg.norm(xm))) > 0.999999

            x_1b = amplitudes(sysd, lambda l: np.abs(np.cos(np.pi * l
                                                            / (2 * s_p))))
            lo, hi = sysd["lam_min"], sysd["lam_max"]
            half = max(s_p - lo, hi - s_p) + 0.05 * (hi - lo) + 0.2
            comb = design_line_comb_inverse(degree=40, s=s_p,
                                            domain=(s_p - half, s_p + half))
            x_comb = amplitudes(sysd, lambda l: np.abs(comb(l)))

            # ---- fit -------------------------------------------------------
            D, seg_rows, tru_rows = design_matrix(sysd, edges)
            y_true = x_cls[seg_rows[tru_rows]]
            DT, DF = D[tru_rows], D[~tru_rows]
            iso_row = np.zeros(nb)
            iso_row[np.clip(np.searchsorted(edges, s_p) - 1, 0, nb - 1)] = DELTA
            wiso = np.sqrt(float(np.count_nonzero(sysd["iso"] & ~truth))) / 30.0
            DF_full = np.vstack([DF, wiso * iso_row])
            GF, GT = DF_full.T @ DF_full, DT.T @ DT
            bT = DT.T @ y_true
            reg = 1e-8 * np.eye(nb)
            best = None
            for mu in MUS:
                p_b = np.linalg.solve(GF + mu * GT + reg, mu * bT)
                p_b = p_b / max(np.abs(p_b).max(), 1e-12)
                x_o = amplitudes(sysd, binned_response(p_b, edges))
                fe = far_at_eff(x_o, truth, (MU_SELECT_EFF,))[MU_SELECT_EFF]
                if best is None or fe[1] < best[0]:
                    best = (fe[1], mu, p_b, x_o)
            _, mu_b, p_opt, x_opt = best

            # Chebyshev realizability
            lam_grid = np.linspace(edges[0], edges[-1], 4000)
            targ = binned_response(p_opt, edges)(lam_grid)
            cheb = {}
            for d in CHEB_DEGS:
                dom = (edges[0], edges[-1])
                u = (2 * lam_grid - (dom[0] + dom[1])) / (dom[1] - dom[0])
                cf = npcheb.chebfit(u, targ, d)
                def pc(l, cf=cf, dom=dom):
                    uu = (2 * np.asarray(l) - (dom[0] + dom[1])) / (dom[1] - dom[0])
                    return npcheb.chebval(uu, cf)
                sc = max(np.max(np.abs(pc(lam_grid))), 1e-12)
                cheb[d] = amplitudes(sysd, lambda l: np.abs(pc(l)) / sc)

            # generalization: rep-0 response verbatim
            if rep == 0:
                fitted_p0[label] = (p_opt.copy(), edges.copy())
                x_gen = None
            else:
                p0, ed0 = fitted_p0[label]
                x_gen = amplitudes(sysd, binned_response(p0, ed0))

            # ---- efficiency-matched table ---------------------------------
            row = dict(config=label, rep=rep, gamma=sysd["g_use"], mu=mu_b,
                       n_true=int(truth.sum()))
            solvers = {"cls": x_cls, "1b": x_1b, "comb": x_comb,
                       "opt": x_opt, "cheb160": cheb[160]}
            if x_gen is not None:
                solvers["gen"] = x_gen
            for tag, xv in solvers.items():
                fe = far_at_eff(xv, truth)
                for tgt, (e_, f_, _) in fe.items():
                    row[f"far_{tag}_e{int(tgt*1000)}"] = f_
            for d in CHEB_DEGS:
                row[f"far_chebd{d}_e985"] = far_at_eff(
                    cheb[d], truth, (0.985,))[0.985][1]
            # gate frontier on OPT and on classical
            ge, gf = gate_frontier(x_opt, truth, seg_hits)
            gec, gfc = gate_frontier(x_cls, truth, seg_hits)
            for tgt in EFF_TARGETS:
                ok = ge >= tgt - 1e-9
                row[f"far_optgate_e{int(tgt*1000)}"] = (float(gf[ok].min())
                                                        if ok.any() else np.nan)
                okc = gec >= tgt - 1e-9
                row[f"far_clsgate_e{int(tgt*1000)}"] = (float(gfc[okc].min())
                                                        if okc.any() else np.nan)

            # ---- floor census ---------------------------------------------
            dead = truth & (np.abs(x_opt) < 1e-12)
            leak_med = float(np.median(np.abs(x_opt)[~truth & (x_cls >
                             qp.threshold_for(sysd["g_use"], DELTA))]))
            leak_tier = truth & (np.abs(x_opt) >= 1e-12) & \
                (np.abs(x_opt) < 10 * max(leak_med, 1e-15))
            row.update(n_true_dead=int(dead.sum()),
                       n_true_leak=int(leak_tier.sum()),
                       floor_frac=float((dead | leak_tier).sum()
                                        / max(int(truth.sum()), 1)))
            comp_shapes = []
            for c, (idx, lam, V) in sysd["comps"].items():
                m = dead[idx] | leak_tier[idx]
                if m.any():
                    comp_shapes.append((len(idx), int(truth[idx].sum()),
                                        int((~truth[idx]).sum())))
            results.append(row)
            say(f"[{label} rep{rep}] g={sysd['g_use']:.2f} mu={mu_b:.2g} | "
                f"far@eff .99/.985/.98/.97: "
                f"cls {row['far_cls_e990']:.3f}/{row['far_cls_e985']:.3f}/"
                f"{row['far_cls_e980']:.3f}/{row['far_cls_e970']:.3f} · "
                f"1bqf {row['far_1b_e990']:.3f}/{row['far_1b_e985']:.3f}/"
                f"{row['far_1b_e980']:.3f}/{row['far_1b_e970']:.3f} · "
                f"OPT {row['far_opt_e990']:.3f}/{row['far_opt_e985']:.3f}/"
                f"{row['far_opt_e980']:.3f}/{row['far_opt_e970']:.3f} · "
                f"OPT+gate@.985 {row['far_optgate_e985']:.3f} | floor "
                f"{row['n_true_dead']}+{row['n_true_leak']} of {row['n_true']} "
                f"({100*row['floor_frac']:.2f}%) shapes {comp_shapes[:6]} "
                f"[{time.time()-tt:.0f}s]")
            if rep == 0:
                panels[label] = dict(mt=mode_table(sysd), p_opt=p_opt,
                                     edges=edges, s_p=s_p, comb=comb,
                                     x_cls=x_cls, x_opt=x_opt, x_1b=x_1b,
                                     truth=truth, seg_hits=seg_hits)

    df = pd.DataFrame(results)
    df.to_csv(OUT / "fit_comb.csv", index=False)

    # ------------------------------- figure --------------------------------
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.8))

    ax = axes[0, 0]
    P = panels["beta0"]
    mt, edges, s_p = P["mt"], P["edges"], P["s_p"]
    lam, wb, vt, vf = mt[:, 0], mt[:, 1], mt[:, 2], mt[:, 3]
    wsol = (wb / np.maximum(np.abs(lam), 1e-9)) ** 2
    ax.hist(lam, bins=edges, weights=wsol * vt, histtype="stepfilled",
            alpha=0.45, color="#79b465", label="true formation weight")
    ax.hist(lam, bins=edges, weights=wsol * vf, histtype="stepfilled",
            alpha=0.5, color="#e34948", label="FALSE formation weight")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)
    ax.set_ylabel("mode weight (v·b/λ)² × content  [log]")
    ax.set_xlabel("eigenvalue λ")
    for l4 in s_p - 2 * np.cos(np.arange(1, 5) * np.pi / 5):
        ax.axvline(l4, color="#33322e", lw=0.6, ls=":")
    for lf, nm in [(s_p - 1, "s−1 pair"), (s_p - np.sqrt(2), "s−√2")]:
        ax.axvline(lf, color="#e34948", lw=0.6, ls="--")
        ax.text(lf, ax.get_ylim()[1] * 0.5, nm, fontsize=6, color="#a83232",
                rotation=90, va="top")
    ax2 = ax.twinx()
    lg = np.linspace(edges[0], edges[-1], 1500)
    cb = np.abs(P["comb"](lg))
    ax2.plot(lg, cb / max(cb.max(), 1e-12), color="#9a9890", lw=1.1,
             label="production comb (fixed)")
    ax2.plot(lg, np.abs(binned_response(P["p_opt"], edges)(lg)),
             color="#2a78d6", lw=1.5, label="FITTED response (roots at pile-up)")
    ax2.set_ylim(0, 1.12)
    ax2.grid(False)
    ax2.set_ylabel("|response| (normalised)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="upper right")
    ax.set_title("(a) the measurement: false pile-up on the motif lines vs "
                 "the responses (beta0 rep0)", loc="left", fontsize=9)

    ax = axes[0, 1]
    for xv, cc, lab, lw in [(P["x_cls"], "#33322e", "classical", 1.6),
                            (P["x_1b"], "#e34948", "1BQF notch", 1.3),
                            (P["x_opt"], "#2a78d6", "FITTED response", 2.0)]:
        e_, f_ = roc(xv, P["truth"])
        ax.plot(np.maximum(f_, 2e-3), e_, "-", color=cc, lw=lw, label=lab)
    ge, gf = gate_frontier(P["x_opt"], P["truth"], P["seg_hits"], npt=60)
    ax.plot(np.maximum(gf, 2e-3), ge, "-", color="#1e9e6a", lw=2.0,
            label="FITTED + uniqueness gate")
    for tgt, ls in [(0.99, ":"), (0.985, "--")]:
        ax.axhline(tgt, color="#79776f", lw=0.6, ls=ls)
    ax.set_xscale("log")
    ax.set_ylim(0.90, 1.005)
    ax.set_xlabel("false rate")
    ax.set_ylabel("segment efficiency")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("(b) the frontier after fitting (beta0 rep0): the floor "
                 "bites only above eff≈0.99", loc="left", fontsize=9)

    ax = axes[1, 0]
    order = [c[0] for c in CONFIGS]
    agg = df.groupby("config").mean(numeric_only=True).reindex(order)
    width = 0.13
    series = [("far_cls_e985", "#33322e", "classical"),
              ("far_1b_e985", "#e34948", "1BQF"),
              ("far_comb_e985", "#9a9890", "comb (fixed)"),
              ("far_cheb160_e985", "#7fb0e0", "fitted cheb d=160"),
              ("far_opt_e985", "#2a78d6", "fitted (optimal)"),
              ("far_optgate_e985", "#1e9e6a", "fitted + gate")]
    for k, (col, cc, lab) in enumerate(series):
        ax.bar(np.arange(len(order)) + (k - 2.5) * width, agg[col], width - 0.01,
               color=cc, label=lab)
    ax.set_xticks(np.arange(len(order)), order, fontsize=8)
    ax.set_ylabel("false rate at matched eff = 0.985")
    ax.legend(fontsize=6.8)
    ax.set_title("(c) matched-efficiency verdict below the floor (rep-mean, "
                 "eff=0.985)", loc="left", fontsize=9)

    ax = axes[1, 1]
    d0 = df[df.config == "beta0"]
    degs = list(CHEB_DEGS)
    ax.semilogy(degs, [d0[f"far_chebd{d}_e985"].mean() for d in degs], "o-",
                color="#2a78d6", label="Chebyshev-fit far@0.985")
    ax.axhline(d0.far_opt_e985.mean(), color="#1e9e6a", ls="--",
               label="optimal binned response")
    ax.axhline(d0.far_cls_e985.mean(), color="#33322e", ls=":",
               label="classical")
    gen = d0[d0.rep > 0]
    if len(gen) and "far_gen_e985" in gen:
        ax.axhline(gen.far_gen_e985.mean(), color="#eb6834", ls="-.",
                   label="rep-0 fit on reps 1–2")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("false rate at eff=0.985 [log]")
    ax.legend(fontsize=7.5)
    ax.set_title("(d) realizability + generalization (beta0); floor = "
                 f"{df[df.config=='beta0'].n_true_dead.mean():.1f}+"
                 f"{df[df.config=='beta0'].n_true_leak.mean():.1f} mixed-motif "
                 "trues", loc="left", fontsize=9)

    fig.suptitle("Fitting the polynomial to the measured false pile-up — "
                 "exact spectra, heavy T=200 (response to review of QSVT XI)",
                 y=0.995, fontsize=11)
    foot = (r"heavy ($\sigma_{scatt}$ 1e-4, $\sigma_{res}$ 20 µm, drop 1%), "
            r"formula $\varepsilon$=6.313 mrad, T=200, reps 0–2 · exact "
            r"per-component eigh (max comp ~18; no giant component) · fit: "
            r"min $\Sigma_F x^2$ + $\mu\Sigma_T(x-x_{cls})^2$ over binned "
            r"response (bin 0.02), $\mu$ by far@eff0.985, |p|≤1 · "
            r"efficiency-matched far: $\tau$ just below the k-th true "
            r"amplitude, k=⌈eff·n_true⌉ · floor = trues with formation ≡ "
            r"false pair/triple motifs (dead <1e-12 or ≤10× leak tier) · "
            "gate = hit-uniqueness greedy · provenance: QSVT/Codesign/"
            "04_fit_comb_to_measured_spectrum.py")
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    fig.text(0.01, 0.004, foot, fontsize=5.6, color="#52514e", va="bottom")
    fig.savefig(OUT / "fit_comb.png", bbox_inches="tight")
    plt.close(fig)

    cols = (["config", "rep"] +
            [f"far_{t}_e985" for t in ("cls", "1b", "comb", "opt", "cheb160")] +
            ["far_optgate_e985", "far_clsgate_e985", "n_true_dead",
             "n_true_leak", "floor_frac"])
    say(df[[c for c in cols if c in df]].round(4).to_string(index=False))
    say(f"[done] {time.time()-t00:.0f}s -> outputs/fork_noisy/fit_comb.png")


if __name__ == "__main__":
    main()
