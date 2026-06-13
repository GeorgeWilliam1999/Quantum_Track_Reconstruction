#!/usr/bin/env python3
"""
Sensitivity of the segment metrics to the Hamiltonian acceptance epsilon.

    sigma_p^2   = sigma_scatt^2 + 6 arctan^2(sigma_res/dz)     (per projection)
    p_kink(eps) = exp(-(eps^2 - 2 theta_min^2)/(2 sigma_p^2))  (Rayleigh miss)

Four measurements for the Notion epsilon-sensitivity report:

  1 direct-eps : segment efficiency / false rate vs eps on a shared log grid,
                 PAIRED events (same event reused across eps, only A changes).
                 Family A: sigma_res = 0.01 fixed, sigma_scatt varies.
                 Family B: sigma_scatt = 1e-4 fixed, sigma_res varies.
  2 sigma scan : same metrics vs sigma_scatt (sigma_res = 0.01 fixed) and vs
                 sigma_res (sigma_scatt = 1e-4 fixed) with eps from the FORMULA
                 at every point — the "reverse" view, dense in sigma.
  3 analytic   : closed-form overlays. Efficiency: chain-fragmentation
                 enumeration eff(p) = (1-p)^2 (1+p/2) with the exact Hopfield
                 motif ladder (P1=1/4, P2=1/3, P3=5/14, P4=4/11 vs tau=0.35).
                 False rate: accepted-false-coupling load proportional to eps^2.
  4 roc        : tau sweep on stored solution vectors, one representative cell,
                 three eps values — the efficiency-first working-point view.

Figures -> figures/epsilon_sensitivity/*.png
Numbers -> outputs/epsilon_sensitivity_scan.json (+ _roc.npz)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

import helpers  # noqa: E402
from helpers import solve_quantum_statevector, rescale_quantum  # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402
from lhcb_velo_toy.analysis.segment_metrics import (  # noqa: E402
    DEFAULT_DZ, DEFAULT_THETA_MIN, segment_truth_mask, solver_segment_metrics,
)
from lhcb_velo_toy.solvers import SimpleHamiltonianFast  # noqa: E402
from qtrk_pipeline.metrics import rescale_to_signal  # noqa: E402

FIGDIR = HERE / "figures" / "epsilon_sensitivity"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR = HERE / "outputs"

DZ = DEFAULT_DZ
TH_MIN = DEFAULT_THETA_MIN
GEOM = helpers.make_geometry()
TAU = 0.35
T, NREP = 30, 3

EPS_GRID = np.logspace(-4.0, np.log10(4e-2), 10)

SR_FIX, SS_FIX = 0.01, 1e-4
FAMILY_A = [(SR_FIX, ss) for ss in (1e-4, 5e-4, 2e-3)]      # sigma_res fixed
FAMILY_B = [(sr, SS_FIX) for sr in (0.0, 0.02, 0.05)]       # sigma_scatt fixed
CELLS = FAMILY_A + FAMILY_B

SCAN_SS = [1e-5, 5e-5, 1e-4, 3e-4, 7.5e-4, 1.3e-3, 2e-3]    # at sr = SR_FIX
SCAN_SR = [0.0, 0.002, 0.005, 0.01, 0.02, 0.035, 0.05]      # at ss = SS_FIX

ROC_CELL = (SR_FIX, SS_FIX)
ROC_EPS_IDX = (4, 5, 7)   # ~0.45x, ~0.88x, ~3.3x the formula eps of ROC_CELL


def sigma_p(sr: float, ss: float) -> float:
    return float(np.sqrt(ss**2 + 6.0 * np.arctan(sr / DZ) ** 2))


def p_kink(eps: float, sr: float, ss: float) -> float:
    x = eps**2 - 2.0 * TH_MIN**2
    if x <= 0.0:
        return 1.0
    return float(np.exp(-x / (2.0 * sigma_p(sr, ss) ** 2)))


def eff_analytic(p: float | np.ndarray) -> float | np.ndarray:
    # 5-hit track = P4 segment chain, 3 kinks each kept w.p. q = 1-p.
    # tau = 0.35 kills P1 (1/4) and P2 (1/3); P3 (>=5/14) and P4 survive whole.
    # all kept: 4 active; one END kink lost (2 ways): P3+P1 -> 3; everything
    # else -> 0. eff = [4 q^3 + 3*2 q^2 p]/4 = (1-p)^2 (1+p/2).
    q = 1.0 - np.asarray(p, dtype=float)
    return q**2 * (1.0 + 0.5 * (1.0 - q))


def chain_levels(n: int, extra_edges=(), n_path: int | None = None) -> np.ndarray:
    # Hopfield amplitudes of a path/tree motif: (4 I - Adj) x = 1 (gamma=3,
    # delta=1 -> diag gamma+delta=4, kept coupling -1, b=1; verified on the
    # real A of a clean event). Path runs over the first n_path nodes.
    A = 4.0 * np.eye(n)
    for i in range((n if n_path is None else n_path) - 1):
        A[i, i + 1] = A[i + 1, i] = -1.0
    for i, j in extra_edges:
        A[i, j] = A[j, i] = -1.0
    return np.linalg.solve(A, np.ones(n))


def false_coupling_count(ham, truth: np.ndarray) -> int:
    # couplings = off-diagonal nnz pairs of A; true-true pairs can only be
    # consecutive same-track segments (segments couple via a shared hit and
    # hits belong to exactly one track), so n_false = n_pairs - n_true_pairs.
    coo = ham.A.tocoo()
    up = coo.row < coo.col
    r, c = coo.row[up], coo.col[up]
    return int(len(r) - int(np.sum(truth[r] & truth[c])))


ARCHIVE: dict[str, np.ndarray] = {}   # raw vectors: metrics stay a VIEW


def solve_cell(ev, eps: float, tag: str = "", want_vectors: bool = False) -> dict:
    ham = SimpleHamiltonianFast(epsilon=eps, gamma=helpers.GAMMA,
                                delta=helpers.DELTA, theta_d=1e-4)
    ham.construct_segments(ev)
    ham.construct_hamiltonian(ev, convolution=False)
    truth = segment_truth_mask(ham)

    t0 = time.time()
    sol_C = np.asarray(ham.solve_classicaly()).ravel()
    mC = solver_segment_metrics(truth, sol_C, threshold=TAU)
    t_c = time.time() - t0

    t0 = time.time()
    sol_Q_raw, p_anc, n_sys = solve_quantum_statevector(ham.A, ham.b,
                                                        device='CPU')
    # PRIMARY: classical-signal-support rescale — the qtrk_pipeline
    # build_metrics convention, so these numbers are store-comparable.
    sol_Q = rescale_to_signal(sol_Q_raw, sol_C, threshold=TAU)
    mQ = solver_segment_metrics(truth, sol_Q, threshold=TAU)
    # secondary: legacy full-norm rescale, recorded for the convention note
    mQn = solver_segment_metrics(truth, rescale_quantum(sol_Q_raw, sol_C),
                                 threshold=TAU)
    t_q = time.time() - t0

    if tag:
        ARCHIVE[f"solC|{tag}"] = sol_C.astype(np.float32)
        ARCHIVE[f"solQraw|{tag}"] = np.asarray(sol_Q_raw, np.float32)
        ARCHIVE[f"truth|{tag}"] = truth.astype(bool)

    out = dict(
        eps=eps, A_nnz=int(ham.A.nnz), n_sys=int(n_sys),
        P_anc=float(p_anc),
        n_false_couplings=false_coupling_count(ham, truth),
        eff_C=mC["segment_efficiency"], far_C=mC["segment_false_rate"],
        eff_Q=mQ["segment_efficiency"], far_Q=mQ["segment_false_rate"],
        eff_Qn=mQn["segment_efficiency"], far_Qn=mQn["segment_false_rate"],
        n_active_C=mC["n_active"], n_active_Q=mQ["n_active"],
        n_true_all=mC["n_true_all"],
        t_classical=t_c, t_quantum=t_q,
    )
    if want_vectors:
        out["vectors"] = (truth, sol_C, sol_Q)
    return out


# ---------------------------------------------------------------------------
# Part 1 — direct-eps sweep on paired events
# ---------------------------------------------------------------------------

direct: dict[str, list] = {}
roc_store: list[dict] = []
for sr, ss in CELLS:
    key = f"sr{sr:g}_ss{ss:g}"
    rows = []
    for rep in range(NREP):
        ev = helpers.safe_generate(T, seed=60_000 + rep, geom=GEOM,
                                   measurement_error=sr, collision_noise=ss)
        for ie, eps in enumerate(EPS_GRID):
            want = (sr, ss) == ROC_CELL and ie in ROC_EPS_IDX
            r = solve_cell(ev, float(eps),
                           tag=f"direct|{key}|r{rep}|e{ie}",
                           want_vectors=want)
            if want:
                truth, sC, sQ = r.pop("vectors")
                roc_store.append(dict(eps_idx=ie, rep=rep, truth=truth,
                                      sol_C=sC, sol_Q=sQ))
            r.update(rep=rep, sr=sr, ss=ss,
                     p_kink=p_kink(float(eps), sr, ss))
            rows.append(r)
            print(f"[1] {key} rep{rep} eps={eps:.3e} "
                  f"effC={r['eff_C']:.3f} farC={r['far_C']:.3f} "
                  f"effQ={r['eff_Q']:.3f} farQ={r['far_Q']:.3f} "
                  f"nfc={r['n_false_couplings']} tq={r['t_quantum']:.0f}s",
                  flush=True)
    direct[key] = rows

np.savez_compressed(
    OUTDIR / "epsilon_sensitivity_roc.npz",
    eps_grid=EPS_GRID, roc_cell=np.array(ROC_CELL),
    **{f"truth_e{d['eps_idx']}_r{d['rep']}": d["truth"] for d in roc_store},
    **{f"solC_e{d['eps_idx']}_r{d['rep']}": d["sol_C"] for d in roc_store},
    **{f"solQ_e{d['eps_idx']}_r{d['rep']}": d["sol_Q"] for d in roc_store},
)
np.savez_compressed(OUTDIR / "epsilon_sensitivity_vectors.npz", **ARCHIVE)

# ---------------------------------------------------------------------------
# Part 2 — sigma scans at the formula eps
# ---------------------------------------------------------------------------

scans: dict[str, list] = {"ss": [], "sr": []}
for axis, points in (("ss", SCAN_SS), ("sr", SCAN_SR)):
    for ip, val in enumerate(points):
        sr, ss = (SR_FIX, val) if axis == "ss" else (val, SS_FIX)
        eps = float(compute_epsilon(sr, ss))
        for rep in range(NREP):
            ev = helpers.safe_generate(T, seed=70_000 + 100 * ip + rep,
                                       geom=GEOM, measurement_error=sr,
                                       collision_noise=ss)
            r = solve_cell(ev, eps, tag=f"scan|{axis}{val:g}|r{rep}")
            r.update(rep=rep, sr=sr, ss=ss, sigma_p=sigma_p(sr, ss))
            scans[axis].append(r)
            print(f"[2] {axis}={val:g} rep{rep} eps={eps:.3e} "
                  f"effC={r['eff_C']:.3f} farC={r['far_C']:.3f} "
                  f"effQ={r['eff_Q']:.3f} farQ={r['far_Q']:.3f}", flush=True)


# ---------------------------------------------------------------------------
# aggregation helpers
# ---------------------------------------------------------------------------

def agg(rows, xkey, ykey):
    xs = sorted(set(r[xkey] for r in rows))
    m, e = [], []
    for x in xs:
        v = [r[ykey] for r in rows if r[xkey] == x]
        m.append(np.mean(v))
        e.append(np.std(v) / np.sqrt(len(v)))
    return np.array(xs), np.array(m), np.array(e)


def fit_quadratic(eps, far):
    # least-squares c for far = c eps^2, on unsaturated points
    keep = (far > 0) & (far < 0.5)
    if keep.sum() < 2:
        return float("nan")
    return float(np.sum(far[keep] * eps[keep] ** 2)
                 / np.sum(eps[keep] ** 4))


CMAP = plt.get_cmap("viridis")
P_DEFAULT = float(np.exp(-9.0))

# ---------------------------------------------------------------------------
# Figs 1+2 — direct-eps sweeps
# ---------------------------------------------------------------------------

fits: dict[str, float] = {}
for figname, family, fixed_lab, var_lab in (
    ("eps_scan_fixed_sres", FAMILY_A,
     rf"$\sigma_{{\rm res}}={SR_FIX}$ mm fixed", "ss"),
    ("eps_scan_fixed_sscatt", [(SR_FIX, SS_FIX)] + FAMILY_B,
     rf"$\sigma_{{\rm scatt}}={SS_FIX:g}$ fixed", "sr"),
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    cols = [CMAP(x) for x in np.linspace(0.0, 0.85, len(family))]
    for (sr, ss), col in zip(family, cols):
        key = f"sr{sr:g}_ss{ss:g}"
        rows = direct[key]
        lab = (rf"$\sigma_s={ss:g}$" if var_lab == "ss"
               else rf"$\sigma_r={sr:g}$ mm")
        eps_f = float(compute_epsilon(sr, ss))
        for ax, met in zip(axes, ("eff", "far")):
            for solver, ls, mk in (("C", "-", "o"), ("Q", "--", "s")):
                x, m, e = agg(rows, "eps", f"{met}_{solver}")
                ax.errorbar(x, m, yerr=e, fmt=mk, ls=ls, ms=4.5, capsize=2.5,
                            color=col, mfc=(col if solver == "C" else "none"),
                            label=f"{lab} "
                                  f"({'classical' if solver=='C' else '1BQF'})")
            ax.axvline(eps_f, color=col, lw=1.2, alpha=0.55)
        # analytic overlays
        xx = np.logspace(np.log10(EPS_GRID[0]), np.log10(EPS_GRID[-1]), 300)
        axes[0].plot(xx, eff_analytic([p_kink(x, sr, ss) for x in xx]),
                     color=col, lw=1.0, alpha=0.8, ls=":")
        x, mC, _ = agg(rows, "eps", "far_C")
        c = fit_quadratic(x, mC)
        fits[key] = c
        if np.isfinite(c):
            axes[1].plot(xx, np.clip(c * xx**2, 0, 1.05), color=col,
                         lw=1.0, alpha=0.8, ls=":")
    axes[0].set_ylabel("segment efficiency")
    axes[1].set_ylabel("segment false rate")
    axes[0].set_ylim(-0.03, 1.06)
    axes[1].set_ylim(-0.03, 1.06)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\varepsilon$ [rad]")
        ax.grid(alpha=0.25)
    axes[0].set_title(r"dotted: analytic $(1-p)^2(1+p/2)$,"
                      r"  $p=e^{-\varepsilon^2/2\sigma_p^2}$", fontsize=9.5)
    axes[1].set_title(r"dotted: $c\,\varepsilon^2$ acceptance law"
                      r"  (vertical lines: formula $\varepsilon$)",
                      fontsize=9.5)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(rf"Segment metrics vs Hamiltonian $\varepsilon$ — "
                 rf"{fixed_lab}  (paired events, $T={T}$, {NREP} reps, "
                 rf"$\tau={TAU}$)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGDIR / f"{figname}.png", dpi=160)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3 — motif ladder + universal collapse eff(p)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                         gridspec_kw=dict(width_ratios=[1, 1.5]))
ax = axes[0]
motifs = [
    ("P1 isolated", chain_levels(1), "tab:red"),
    ("P2 fragment", chain_levels(2), "tab:red"),
    ("P3 fragment", chain_levels(3), "tab:green"),
    ("P4 full track", chain_levels(4), "tab:green"),
    ("false pendant\non P4 interior",
     chain_levels(5, [(1, 4)], n_path=4)[[4]], "tab:orange"),
]
for i, (lab, lv, col) in enumerate(motifs):
    for v in np.unique(np.round(lv, 6)):
        ax.hlines(v, i - 0.32, i + 0.32, color=col, lw=3)
        ax.text(i + 0.36, v, f"{v:.3f}", fontsize=8, va="center")
ax.axhline(TAU, color="k", ls="--", lw=1.5)
ax.text(-0.45, TAU + 0.008, r"$\tau=0.35$", fontsize=9)
ax.set_xticks(range(len(motifs)))
ax.set_xticklabels([m[0] for m in motifs], fontsize=8)
ax.set_ylabel("Hopfield amplitude")
ax.set_ylim(0.2, 0.62)
ax.set_title(r"motif ladder: $(4I-\mathrm{Adj})\,x=\mathbf{1}$"
             r"  ($\gamma=3,\ \delta=1$)", fontsize=10)

ax = axes[1]
pp = np.logspace(-6, 0, 400)
ax.plot(pp, eff_analytic(pp), "k-", lw=2,
        label=r"$(1-p)^2(1+p/2)$ enumeration")
cols6 = [CMAP(x) for x in np.linspace(0.0, 0.85, len(CELLS))]
for (sr, ss), col in zip(CELLS, cols6):
    rows = direct[f"sr{sr:g}_ss{ss:g}"]
    x, m, e = agg(rows, "eps", "eff_C")
    pk = np.array([p_kink(v, sr, ss) for v in x])
    ax.errorbar(np.clip(pk, 1e-6, 1), m, yerr=e, fmt="o", ms=4.5, capsize=2.5,
                color=col, label=rf"$\sigma_r={sr:g}$, $\sigma_s={ss:g}$")
ax.axvline(P_DEFAULT, color="tab:green", lw=1.5, alpha=0.7)
ax.text(P_DEFAULT, 0.15, r"formula default $p=e^{-9}$", rotation=90,
        fontsize=8, color="tab:green", va="bottom", ha="right")
ax.set_xscale("log")
ax.set_xlabel(r"computed per-kink miss probability "
              r"$p(\varepsilon)=e^{-\varepsilon^2/2\sigma_p^2}$")
ax.set_ylabel("measured classical segment efficiency")
ax.set_title("all cells collapse onto the enumeration curve", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIGDIR / "eff_universal_collapse.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 4 — sigma scans at formula eps, with the sigma_p^2 law
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharey="row")
for j, (axis, xlab, fixed) in enumerate((
    ("ss", r"$\sigma_{\rm scatt}$ [rad]",
     rf"$\sigma_{{\rm res}}={SR_FIX}$ mm fixed"),
    ("sr", r"$\sigma_{\rm res}$ [mm]",
     rf"$\sigma_{{\rm scatt}}={SS_FIX:g}$ fixed"),
)):
    rows = scans[axis]
    xkey = "ss" if axis == "ss" else "sr"
    for i, met in enumerate(("eff", "far")):
        ax = axes[i, j]
        for solver, ls, mk, col in (("C", "-", "o", "tab:blue"),
                                    ("Q", "--", "s", "tab:red")):
            x, m, e = agg(rows, xkey, f"{met}_{solver}")
            ax.errorbar(x, m, yerr=e, fmt=mk, ls=ls, ms=5, capsize=3,
                        color=col,
                        label="classical" if solver == "C" else "1BQF")
        if met == "eff":
            ax.axhline(eff_analytic(P_DEFAULT), color="k", ls=":", lw=1.5,
                       label=r"analytic $(1-p)^2(1+p/2)$ at $p=e^{-9}$")
            # keep the 1BQF ~0.75 ladder plateau in view
            ax.set_ylim(0.55, 1.04)
        else:
            xs = np.array(sorted(set(r[xkey] for r in rows)))
            eps_f = np.array([float(compute_epsilon(
                (SR_FIX if axis == "ss" else v),
                (v if axis == "ss" else SS_FIX))) for v in xs])
            _, mC, _ = agg(rows, xkey, "far_C")
            _, mQ, _ = agg(rows, xkey, "far_Q")
            c = fit_quadratic(eps_f, mC)
            fits[f"scan_{axis}"] = c
            ax.plot(xs, np.clip(c * eps_f**2, 0, 1.0), "k:", lw=1.5,
                    label=r"$c\,\varepsilon^2(\sigma)\ \propto\ \sigma_p^2$")
            ax.set_ylim(-0.02, max(0.45, 1.3 * mC.max(), 1.3 * mQ.max()))
        if axis == "ss":
            ax.set_xscale("log")
            xstar = np.sqrt(6.0) * np.arctan(SR_FIX / DZ)
            ax.axvline(xstar, color="gray", lw=1.2, alpha=0.7)
            if i == 0:
                ax.text(xstar * 1.07, 0.86,
                        r"$\sigma^*_{\rm scatt}=\sqrt{6}\,"
                        r"\arctan(\sigma_r/\Delta z)$",
                        fontsize=8, color="gray", rotation=90, va="bottom")
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_title(fixed, fontsize=10.5)
axes[0, 0].set_ylabel("segment efficiency")
axes[1, 0].set_ylabel("segment false rate")
axes[0, 0].legend(fontsize=8, loc="lower left")
axes[1, 0].legend(fontsize=8, loc="upper left")
fig.suptitle(rf"Noise scans at the formula $\varepsilon$ "
             rf"($p_{{\rm miss}}=e^{{-9}}$) — $T={T}$, {NREP} reps, "
             rf"$\tau={TAU}$", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGDIR / "sigma_scan_formula_eps.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 5 — ROC: tau sweep at three eps, representative cell
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 5.6))
taus = np.linspace(0.05, 0.95, 181)
cols3 = ["tab:purple", "tab:green", "tab:orange"]
eps_f_roc = float(compute_epsilon(*ROC_CELL))
for ie, col in zip(ROC_EPS_IDX, cols3):
    eps = EPS_GRID[ie]
    for skey, ls in (("C", "-"), ("Q", "--")):
        effs, fars = [], []
        for tau in taus:
            ev_, fv_ = [], []
            for d in roc_store:
                if d["eps_idx"] != ie:
                    continue
                m = solver_segment_metrics(
                    d["truth"], d[f"sol_{skey}"], threshold=float(tau))
                ev_.append(m["segment_efficiency"])
                fv_.append(m["segment_false_rate"])
            effs.append(np.mean(ev_))
            fars.append(np.mean(fv_))
        ax.plot(fars, effs, ls=ls, color=col, lw=1.8,
                label=rf"$\varepsilon={eps:.2e}$"
                      rf" ({eps/eps_f_roc:.2f}$\times$ formula), "
                      f"{'classical' if skey=='C' else '1BQF'}")
        it = int(np.argmin(np.abs(taus - TAU)))
        ax.plot(fars[it], effs[it], "o", ms=8, color=col,
                mfc=(col if skey == "C" else "none"), mew=1.8)
ax.set_xlabel(r"segment false rate ($\tau$ swept 0.05$\to$0.95)")
ax.set_ylabel("segment efficiency")
ax.set_title(rf"Working-point view at $\sigma_r={ROC_CELL[0]}$, "
             rf"$\sigma_s={ROC_CELL[1]:g}$, $T={T}$ "
             rf"(markers: $\tau={TAU}$)", fontsize=10.5)
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIGDIR / "roc_tau_sweep.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 6 — high-T cross-check from the qtrk store (formula eps, coarse grid)
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402

df = pd.read_csv("/data/bfys/gscriven/qtrk_store/manifest/metrics.csv")
df = df[df["study"] == "Epsilon_study_2"]
T_SHOW = [50, 100, 200, 400]
fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharey="row")
for j, (xcol, sel, xlab, fixed) in enumerate((
    ("sigma_scatt", df["sigma_res"] == SR_FIX, r"$\sigma_{\rm scatt}$ [rad]",
     rf"$\sigma_{{\rm res}}={SR_FIX}$ mm fixed"),
    ("sigma_res", df["sigma_scatt"] == SS_FIX, r"$\sigma_{\rm res}$ [mm]",
     rf"$\sigma_{{\rm scatt}}={SS_FIX:g}$ fixed"),
)):
    sub = df[sel]
    colsT = [CMAP(x) for x in np.linspace(0.0, 0.85, len(T_SHOW))]
    for i, ycol in enumerate(("segment_efficiency", "segment_false_rate")):
        ax = axes[i, j]
        for t, col in zip(T_SHOW, colsT):
            for solver, ls, mk in (("classical", "-", "o"),
                                   ("quantum", "--", "s")):
                g = (sub[(sub["n_trk"] == t) & (sub["solver"] == solver)]
                     .groupby(xcol)[ycol].agg(["mean", "sem", "count"]))
                g = g[g["count"] >= 2]
                if g.empty:
                    continue
                ax.errorbar(g.index, g["mean"], yerr=g["sem"], fmt=mk, ls=ls,
                            ms=5, capsize=3, color=col,
                            mfc=(col if solver == "classical" else "none"),
                            label=f"T={t} {solver}")
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_title(fixed, fontsize=10.5)
axes[0, 0].set_ylabel("segment efficiency")
axes[1, 0].set_ylabel("segment false rate")
axes[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle(r"qtrk-store cross-check: same scans at high $T$ "
             r"(formula $\varepsilon$, $\tau=0.35$, 20 reps classical / "
             r"3 quantum)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGDIR / "store_grid_highT.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

out = dict(
    T=T, nrep=NREP, tau=TAU, eps_grid=EPS_GRID.tolist(),
    quantum_convention=("eff_Q/far_Q: rescale_to_signal (qtrk_pipeline "
                        "build_metrics convention, store-comparable); "
                        "eff_Qn/far_Qn: legacy full-norm rescale_quantum"),
    family_A=FAMILY_A, family_B=FAMILY_B,
    scan_ss=SCAN_SS, scan_sr=SCAN_SR, roc_cell=ROC_CELL,
    roc_eps=[float(EPS_GRID[i]) for i in ROC_EPS_IDX],
    motif_levels=dict(
        P1=chain_levels(1).tolist(), P2=chain_levels(2).tolist(),
        P3=chain_levels(3).tolist(), P4=chain_levels(4).tolist(),
        P4_pendant_interior=chain_levels(5, [(1, 4)], n_path=4).tolist(),
        P5=chain_levels(5).tolist(),
    ),
    quadratic_fits=fits,
    direct=direct,
    scans=scans,
)
with (OUTDIR / "epsilon_sensitivity_scan.json").open("w") as f:
    json.dump(out, f, indent=2)
print("[done] figures + JSON written", flush=True)
