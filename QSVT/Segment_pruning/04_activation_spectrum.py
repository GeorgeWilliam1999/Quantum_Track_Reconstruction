#!/usr/bin/env python3
"""Activation histograms and the eigenvalues behind them (paper corpus).

For T = 400 and 1000, all ten repetitions pooled:

fig7_activations_T{T}
    One panel per solver (classical, one-bit circuit, QSVT): histograms of the
    segment amplitudes of the FULL solve, true segments against false
    segments, with the false segments that survive the decoupled prune and
    the components-under-3 prune drawn as outlines.  Pruning does not change
    any amplitude (the kept values are identical), so the pruned histogram is
    the full one minus the removed segments.  The median 99 % working-point
    thresholds of the full and the component-pruned readings are marked.

fig8_eigenvalues_T{T}
    Top: the three filter responses against the eigenvalue (classical
    delta/lambda, one-bit |cos(lambda t/2)|, the comb |p(lambda)|), each
    normalised to its maximum on the domain.
    Bottom left: the eigenvalue spectrum of A, each eigenpair split by its
    weight on true segments (sum of v_i^2 over true i) against false
    segments.  Bottom right: the same spectrum by component type.  Decoupled
    segments are the single bar at lambda = s; two-segment pieces are the
    bars at s -/+ 1; those are exactly what the two prunes remove.

Amplitudes come from outputs/amplitudes/ (stored by 01_run_study.py); the
eigenvalues are recomputed here, component by component, from the matrix
rebuilt on the stored events.  Also writes outputs/eigen_classes_T{T}.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components

from common import (COMB_DEGREE, FIG, OUT, S, T_ONEBIT, bounds_for, build_full, epsilon_for,
                    load_event, truth_full)
from lhcb_velo_toy.solvers.hamiltonians import coupling_pattern
from lhcb_velo_toy.solvers.quantum import QSVT

CFG = "paper"
TS = [400, 1000]
REPS = range(10)
SOLVERS = [("classical", "classical"), ("onebit_circuit", "one-bit filter (package circuit)"),
           ("qsvt", "QSVT, degree 40")]
C_TRUE, C_FALSE = "#0072B2", "#7F7F7F"
C_DEG1, C_COMP3 = "#D55E00", "#8B0000"
CLASSES = ["isolated false segment", "isolated true segment", "false pair", "true pair",
           "false triple or bridge", "hub", "complete track", "broken track", "contaminated track"]
C_CLASS = {"isolated false segment": "#BBBBBB", "isolated true segment": "#9ECAE1",
           "false pair": "#D55E00", "true pair": "#56B4E9",
           "false triple or bridge": "#E69F00", "hub": "#CC79A7",
           "complete track": "#0072B2", "broken track": "#009E73", "contaminated track": "#F0E442"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})


def classify(n, maxdeg, n_true):
    if n == 1:
        return "isolated true segment" if n_true else "isolated false segment"
    if n == 2:
        return "true pair" if n_true == 2 else ("contaminated track" if n_true else "false pair")
    if n_true == n:
        return "complete track" if (n == 4 and maxdeg <= 2) else "broken track"
    if n_true:
        return "contaminated track"
    return "false triple or bridge" if maxdeg <= 2 else "hub"


_CACHE = {}


def event_data(T, rep):
    """Build once per event: truth, coupled, comp3 masks, and the eigen census
    of the coupled part (sliced from the PRUNED matrix, which is small)."""
    key = (T, rep)
    if key in _CACHE:
        return _CACHE[key]
    from common import pruned_copy
    eps = epsilon_for(CFG)
    ev, _ = load_event(CFG, T, rep)
    ham = build_full(ev, eps)
    truth = truth_full(ham)
    poly = QSVT.comb(ham.A, ham.b, degree=COMB_DEGREE, spectral_bounds=bounds_for(ham)).poly
    hp = pruned_copy(ham)                       # decoupled segments dropped
    kept = hp.kept_index
    coupled = np.zeros(ham.n_segments, bool)
    coupled[kept] = True
    Ak = hp.A.tocsr()
    C = coupling_pattern(Ak)
    ncomp, lab = connected_components(C, directed=False)
    sizes = np.bincount(lab, minlength=ncomp)
    comp3 = np.zeros(ham.n_segments, bool)
    comp3[kept[sizes[lab] >= 3]] = True
    deg = np.asarray(C.getnnz(axis=1)).ravel()
    order = np.argsort(lab, kind="stable")
    starts = np.r_[0, np.cumsum(sizes)]
    tk = truth[kept]
    lam, wtrue, cls, groups = [], [], [], []
    for c in range(ncomp):
        idx = order[starts[c]:starts[c + 1]]
        w, V = np.linalg.eigh(Ak[idx][:, idx].toarray())
        t = tk[idx]
        lam.append(w)
        wtrue.append((V[t] ** 2).sum(axis=0))
        cls += [classify(len(idx), int(deg[idx].max()), int(t.sum()))] * len(idx)
        groups.append((idx, w, V, V.T @ np.ones(len(idx))))   # (kept-space idx, evals, evecs, <v|1>)
    n_iso_true = int((truth & ~coupled).sum())
    n_iso_false = int((~truth & ~coupled).sum())
    lam = np.r_[np.concatenate(lam), np.full(n_iso_false + n_iso_true, S)]
    wtrue = np.r_[np.concatenate(wtrue), np.zeros(n_iso_false), np.ones(n_iso_true)]
    cls = np.r_[np.asarray(cls), ["isolated false segment"] * n_iso_false,
                ["isolated true segment"] * n_iso_true]
    out = dict(truth=truth, coupled=coupled, comp3=comp3, lam=lam, wtrue=wtrue, cls=cls, poly=poly,
               groups=groups, kept=kept, n_kept=int(len(kept)), bounds=bounds_for(ham),
               comp3_kept=(sizes[lab] >= 3))
    _CACHE[key] = out
    return out


def eigen_census(T):
    """Pooled eigenvalues over the repetitions with true weight and class."""
    ds = [event_data(T, rep) for rep in REPS]
    return (np.concatenate([d["lam"] for d in ds]), np.concatenate([d["wtrue"] for d in ds]),
            np.concatenate([d["cls"] for d in ds]), ds[0]["poly"])


def fig_activations(T, so):
    files = [OUT / "amplitudes" / f"{CFG}_T{T}_rep{rep}.npz" for rep in REPS]
    files = [f for f in files if f.exists()]
    data = [np.load(f) for f in files]
    ds = [event_data(T, rep) for rep in range(len(files))]
    truth = np.concatenate([d["truth"] for d in ds])
    coupled = np.concatenate([d["coupled"] for d in ds])
    comp3 = np.concatenate([d["comp3"] for d in ds])
    assert all(np.array_equal(d["truth"], e["truth"]) for d, e in zip(data, ds))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, (sol, lab) in zip(axes, SOLVERS):
        x = np.concatenate([d[sol].astype(float) for d in data])
        x = np.abs(x)
        scale = float(np.median(x[truth & comp3]))      # metrics are scale-invariant
        x = x / scale
        xmax = 2.6
        bins = np.linspace(0, xmax, 130)
        ax.hist(x[~truth], bins, color=C_FALSE, alpha=0.5, label="false, all")
        ax.hist(x[~truth & coupled], bins, histtype="step", color=C_DEG1, lw=1.4,
                label="false, after decoupled prune")
        ax.hist(x[~truth & comp3], bins, histtype="step", color=C_COMP3, lw=1.2, ls="--",
                label="false, after components < 3 prune")
        ax.hist(x[truth], bins, histtype="stepfilled", color=C_TRUE, alpha=0.45, label="true, all")
        ax.hist(x[truth & comp3], bins, histtype="step", color=C_TRUE, lw=1.2,
                label="true, after components < 3 prune")
        d = so[(so.config == CFG) & (so["T"] == T) & (so.solver == sol)]
        for var, col, ls in (("full", "#000000", "-"), ("comp3", C_COMP3, "--")):
            tau = d[d.variant == var].tau_e990.median() / scale
            ax.axvline(tau, color=col, ls=ls, lw=0.9)
        ax.set_yscale("log"); ax.set_title(lab, fontsize=8)
        ax.set_xlabel("amplitude / median true amplitude")
        ax.set_xlim(0, xmax)
    axes[0].set_ylabel(f"segments (10 events, T = {T})")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([], [], color="#000000", lw=0.9),
                    plt.Line2D([], [], color=C_COMP3, lw=0.9, ls="--")],
               l + ["99 % threshold, full reading", "99 % threshold, components >= 3 reading"],
               fontsize=6.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(FIG / f"fig7_activations_T{T}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"fig7_activations_T{T}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def fig_eigenvalues(T):
    lam, wtrue, cls, poly = eigen_census(T)
    pd.DataFrame({"class": CLASSES,
                  "n_eigenvalues": [int((cls == c).sum()) for c in CLASSES],
                  "true_weight": [float(wtrue[cls == c].sum()) for c in CLASSES]}
                 ).to_csv(OUT / f"eigen_classes_T{T}.csv", index=False)
    lo, hi = 0.0, 8.0
    bins = np.linspace(lo, hi, 161)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.0), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 2.2]))
    dlo, dhi = float(poly.domain[0]), float(poly.domain[1])
    grid = np.linspace(max(lo + 0.05, dlo), min(hi, dhi), 2000)
    resp = {"classical  delta / lambda": np.abs(1.0 / grid),
            "one-bit  |cos(lambda t / 2)|": np.abs(np.cos(grid * T_ONEBIT / 2)),
            "QSVT comb  |p(lambda)|": np.abs(poly(grid))}
    # normalise every response to its value at the lowest true-track line
    # s - 2 cos(pi/5), so the three are comparable where the tracks live
    l1 = S - 2 * np.cos(np.pi / 5)
    for ax in axes[0]:
        for (lab, y), col in zip(resp.items(), ["#000000", "#E69F00", "#009E73"]):
            ref = float(np.interp(l1, grid, y))
            ax.plot(grid, y / ref, color=col, lw=1.4, label=lab)
        ax.set_ylabel("response / response at first track line"); ax.set_ylim(0, 1.6)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    ax = axes[1, 0]
    ax.hist(lam, bins, weights=1 - wtrue, color=C_FALSE, alpha=0.6, label="weight on false segments")
    ax.hist(lam, bins, weights=wtrue, histtype="step", color=C_TRUE, lw=1.5, label="weight on true segments")
    ax.set_yscale("log"); ax.set_xlabel("eigenvalue of A"); ax.set_ylabel(f"eigenpairs (10 events, T = {T})")
    ax.legend(fontsize=7, loc="upper right")
    ax = axes[1, 1]
    present = [c for c in CLASSES if (cls == c).any()]
    ax.hist([lam[cls == c] for c in present], bins, stacked=True, color=[C_CLASS[c] for c in present],
            label=present)
    ax.set_yscale("log"); ax.set_xlabel("eigenvalue of A"); ax.set_ylabel("eigenpairs by component type")
    ax.legend(fontsize=6.5, loc="upper right", ncol=2)
    for ax in axes[1]:
        ax.axvline(S, color="#555555", lw=0.6, ls=":")
        ax.text(S, ax.get_ylim()[1] * 0.5, " s (decoupled)", fontsize=6.5, color="#555555")
        for k in (1, 2, 3, 4):
            ax.axvline(S - 2 * np.cos(k * np.pi / 5), color=C_TRUE, lw=0.5, ls=":")
    fig.savefig(FIG / f"fig8_eigenvalues_T{T}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"fig8_eigenvalues_T{T}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def main():
    so = pd.read_csv(OUT / f"{CFG}_solves.csv")
    for T in TS:
        fig_activations(T, so)
        fig_eigenvalues(T)
        print("done", T, flush=True)


if __name__ == "__main__":
    main()
