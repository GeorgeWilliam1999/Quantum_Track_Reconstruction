#!/usr/bin/env python
"""ERF Youden-J / EER threshold characterisation from pooled per-segment scores.

Pools the per-segment activation scores (true vs false, from the stored solution
vectors + recomputed truth) for every ERF cell (noise pair x theta_d x T x solver)
and characterises separability and threshold choice:

  - ROC / AUC per cell (histogram-based, 6000 bins over [-0.5, 2.5]);
  - Youden J* = max(TPR - FPR) and its threshold tau_J;
  - EER (equal error rate: FPR = FNR) and its threshold tau_EER;
  - the wp99 working point (tau at the 1% quantile of TRUE scores; the 1BQF
    headline convention) and the fixed gamma-aware absolute tau = 0.35;
  - segment efficiency / false rate (n_false_active / n_active) at each of the
    four thresholds.

Youden/EER are DIAGNOSTICS of separability (decision 2026-06-14: the reporting
headline stays wp99 for the 1BQF and fixed tau=0.35 for classical); this script
quantifies what each alternative would buy.

Quantum (1BQF) scores are rescaled onto the classical SIGNAL support of the
matched classical solution (same event_key + ham_key) via the canonical
``rescale_to_signal`` — NOT the full-norm ratio, which is false-bulk dominated,
grows ~sqrt(n_seg), and at T>=700 inflates the true band past any fixed
histogram range (the first run's AUC->0 artefact).

Inputs : $QTRK_STORE/manifest/metrics.csv (study == ERF) + stored solutions/events
Outputs: results/erf_youden_eer.csv + figures/erf_youden_*.png (6 figures)

Run (Q_env):
  export PYTHONPATH=".../Toy_Characterisation/_shared:.../LHCb_VeLo_Toy_Model/src"
  python youden_eer_store.py
"""
from __future__ import annotations

import os, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared")
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")
import qtrk_pipeline as qp  # noqa: E402
import helpers  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
RES = HERE / "results"
FIGS.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# ---------------------------------------------------------------- constants
GAMMA, DELTA = 3.0, 1.0
TAU_ABS = qp.threshold_for(GAMMA, DELTA)          # 0.35 at gamma=3
# Non-uniform score grid: tau-resolution 5e-4 where thresholds live ([-1, 3]),
# coarse log bins 3 -> 50 for the drifted-but-orderly tail (dense coupled
# clusters push the top activations past the pure-chain 1.5 level at high T).
BINS = np.concatenate([np.linspace(-1.0, 3.0, 8001), np.geomspace(3.0006, 50.0, 120)])
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])
# Per-solve validity gate: |x| beyond this is the catastrophic lambda_min -> 0
# explosion (measured landscape: healthy/drifted solves top out ~2-40; exploded
# ones sit at 1e2-1e24 — nothing in between at the gate).
EXPLODE_MAX = 50.0
PAIRS = [(1e-4, 0.0, "clean"), (3e-4, 0.01, "moderate"), (5e-4, 0.02, "heavy")]
PAIR_LABEL = {(ss, sr): lab for ss, sr, lab in PAIRS}
# 2026-07-04: + the kink-matched widths theta_d = eps/3 = sigma_kink (one per
# pair, crossed with all pairs). Non-round floats: pool keys are SNAPPED onto
# this grid (snap_td) so hard-coded lookups and df filters match exactly.
KINK = {"clean": 1.4159802e-4, "moderate": 1.1322446e-3, "heavy": 2.2153451e-3}
TDS = sorted([1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3] + list(KINK.values()))


def snap_td(x: float) -> float:
    """Snap a stored erf_sigma onto the canonical TDS grid (isclose, not ==)."""
    for t in TDS:
        if np.isclose(x, t, rtol=1e-3):
            return t
    return float(x)


# palette (validated reference set; one color-meaning per figure family)
C_TRUE, C_FALSE = "#2a78d6", "#eb6834"            # truth state (distributions)
C_CLASS, C_QUANT = "#2a78d6", "#e34948"           # solver identity (trends)
# theta_d = ordered magnitude -> single-hue sequential ramp (light -> dark blue)
TD_RAMP = ["#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5",
           "#08519c", "#08306b", "#041e42"]
TD_COLOR = dict(zip(TDS, TD_RAMP))
PAIR_RAMP = {"clean": "#9ecae1", "moderate": "#4292c6", "heavy": "#08306b"}

FOOT = (r"ERF store (qtrk_store, abs $\tau$=0.35 ref) · $\gamma$=3 $\delta$=1 · erf kernel, "
        r"$\theta_d$: $\{10^{-6},10^{-5},5{\times}10^{-5},10^{-4},5{\times}10^{-4},10^{-3}\}$ "
        r"+ kink-matched $\varepsilon/3$ per pair (0.142 · 1.132 · 2.215 mrad); $10^{-6}$ = step "
        r"· pairs ($\sigma_{scatt}$,$\sigma_{res}$): "
        "clean(1e-4, 0) · moderate(3e-4, 0.01) · heavy(5e-4, 0.02) · "
        r"$\phi_{max}$=0.2 · T$\in$[10,1000] · classical MINRES / 1BQF matrix-free "
        "statevector, 20 events/point (2026-07-04 refresh), quantum rescaled on the "
        "classical signal support")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def footer(fig):
    fig.text(0.01, 0.005, FOOT, fontsize=6, color="#52514e", va="bottom")


# ---------------------------------------------------------------- pooling
def pool_scores() -> dict:
    """One pass over the ERF solves -> {(pair, td, T, solver): [hist_true, hist_false]}."""
    m = pd.read_csv(os.path.join(qp.store_root(), "manifest", "metrics.csv")) \
        if hasattr(qp, "store_root") else pd.read_csv(
            os.path.join(os.environ.get("QTRK_STORE", "/data/bfys/gscriven/qtrk_store"),
                         "manifest", "metrics.csv"))
    # membership via `studies` (token match), NOT `study==`: shared cells are
    # attributed to one owner study and a study== filter silently drops them.
    is_erf = m.studies.fillna("").map(lambda s: "ERF" in re.split(r"[|,;\s]+", str(s)))
    # qsvt solves live on their own comb scale — pooling them raw next to the
    # 0.35-referenced classical/quantum scores would be meaningless; excluded.
    erf = m[is_erf & m.solver.isin(["classical", "quantum"])].copy()
    erf["pair"] = [PAIR_LABEL.get((ss, sr)) for ss, sr in zip(erf.sigma_scatt, erf.sigma_res)]
    erf = erf.sort_values(["event_key", "solver"])   # classical before quantum
    from qtrk_pipeline.metrics import rescale_to_signal
    pools: dict = {}
    solC_event: dict = {}                             # ham_key -> sol_C (current event only)
    truth_cache_key, truth = None, None
    t0, n_loaded = time.time(), 0
    for _, r in erf.iterrows():
        if r.event_key != truth_cache_key:
            ev = qp.load_event(qp.event_path(r.event_key))
            truth = qp.truth_from_event(ev).astype(bool)
            truth_cache_key = r.event_key
            solC_event = {}                           # reset: partners are per-event
        sol = qp.load_solution(r.sol_key)["sol"].astype(np.float64)
        if r.solver == "classical":
            solC_event[r.ham_key] = sol
        elif r.solver == "quantum":
            sol_C = solC_event.get(r.ham_key)
            if sol_C is None:   # no classical partner (should not happen)
                continue
            # canonical: rescale on the classical SIGNAL support (n-independent)
            sol = rescale_to_signal(sol, sol_C, TAU_ABS)
        key = (r.pair, snap_td(r.erf_sigma), int(r.n_trk), r.solver)
        if key not in pools:
            pools[key] = [np.zeros(len(BINS) - 1, np.int64), np.zeros(len(BINS) - 1, np.int64),
                          np.zeros(2, np.int64)]  # [n_excluded_solves, n_total_solves]
        pools[key][2][1] += 1
        # Per-solve VALIDITY GATE. Wide-eps high-T cells are in the INDEFINITENESS
        # regime (lambda_min -> 0, Eps2 §7.12): the classical solve explodes to
        # |x| ~ 1e2-1e24 (vs the healthy/drifted <= ~40) and the signal-rescaled
        # quantum follows. An exploded solve is a solver/regime failure, not a
        # threshold question — pooling it floods the score axis and corrupts the
        # absolute-tau pool. Exclude the whole solve; report the excluded fraction
        # as the measured indefiniteness-onset map.
        if float(np.max(np.abs(sol))) > EXPLODE_MAX:
            pools[key][2][0] += 1
            continue
        pools[key][0] += np.histogram(sol[truth], bins=BINS)[0]
        pools[key][1] += np.histogram(sol[~truth], bins=BINS)[0]
        n_loaded += 1
        if n_loaded % 400 == 0:
            print(f"  … {n_loaded} solves pooled ({time.time()-t0:.0f}s)", flush=True)
    print(f"[pool] {n_loaded} solves -> {len(pools)} pools in {time.time()-t0:.0f}s")
    return pools


# ---------------------------------------------------------------- per-pool metrics
def roc_from_hists(ht: np.ndarray, hf: np.ndarray):
    """Descending-threshold ROC. Returns (fpr, tpr, taus) with tau = bin edge."""
    # above-threshold counts for tau = BINS[i]: sum of bins >= i
    ct = np.concatenate([[0], np.cumsum(ht[::-1])])[::-1].astype(np.float64)  # len 6001
    cf = np.concatenate([[0], np.cumsum(hf[::-1])])[::-1].astype(np.float64)
    tpr = ct / max(ct[0], 1)
    fpr = cf / max(cf[0], 1)
    return fpr, tpr, BINS


def analyse_pool(ht, hf) -> dict:
    n_t, n_f = int(ht.sum()), int(hf.sum())
    if n_t == 0 or n_f == 0:      # pool fully excluded by the validity gate
        nan = float("nan")
        out = dict(n_true=n_t, n_false=n_f, auc=nan, youden_J=nan, tau_J=nan,
                   eer=nan, tau_EER=nan, tau_wp99=nan)
        for tag in ("abs", "J", "EER", "wp99"):
            out[f"eff_{tag}"] = nan
            out[f"far_{tag}"] = nan
        return out
    fpr, tpr, taus = roc_from_hists(ht, hf)
    # AUC: integrate TPR over FPR (both monotone decreasing in tau)
    auc = float(np.trapezoid(tpr[::-1], fpr[::-1]))
    j = tpr - fpr
    i_j = int(np.argmax(j))
    # EER: FPR = 1 - TPR
    i_e = int(np.argmin(np.abs(fpr - (1 - tpr))))
    # wp99: tau just below the 1% quantile of true scores
    cdf_t = np.cumsum(ht) / max(n_t, 1)
    i_w = int(np.searchsorted(cdf_t, 0.01))
    tau_wp = BINS[max(i_w, 0)]
    out = dict(n_true=n_t, n_false=n_f, auc=auc,
               youden_J=float(j[i_j]), tau_J=float(taus[i_j]),
               eer=float(0.5 * (fpr[i_e] + 1 - tpr[i_e])), tau_EER=float(taus[i_e]),
               tau_wp99=float(tau_wp))
    for tag, tau in (("abs", TAU_ABS), ("J", taus[i_j]), ("EER", taus[i_e]), ("wp99", tau_wp)):
        i = int(np.searchsorted(BINS, tau))
        ta, fa = float(ht[i:].sum()), float(hf[i:].sum())
        act = ta + fa
        out[f"eff_{tag}"] = ta / max(n_t, 1)
        out[f"far_{tag}"] = fa / max(act, 1)
    # MATCHED-EFFICIENCY working points (2026-07-05 consistency audit): the
    # kernels put their activation levels at different absolute scales (the
    # erf family has doubled couplings), so any fixed tau compares them at
    # different points of their own distributions.  far at a fixed POOLED
    # efficiency is the threshold-free cross-kernel comparison.
    for tgt, tag in ((0.99, "m99"), (0.999, "m999")):
        ok = np.where(tpr >= tgt)[0]
        i = int(ok.max()) if len(ok) else 0
        ta, fa = float(tpr[i] * n_t), float(fpr[i] * n_f)
        out[f"tau_{tag}"] = float(BINS[i])
        out[f"eff_{tag}"] = float(tpr[i])
        out[f"far_{tag}"] = fa / max(ta + fa, 1e-300)
    return out


# ---------------------------------------------------------------- figures
def attractor_levels(w: float) -> dict:
    """Exact activation levels of ideal structures at coupling weight w.

    Isolated (unconnected) segment, coupled pair, and 4-chain end/interior for
    A = (gamma+delta)I - w*Adj, b = delta.  w=1 is the true step kernel;
    w=2 is the erf family's hard-edge limit (theta_d -> 0)."""
    s = GAMMA + DELTA
    lv = {"iso": DELTA / s}
    lv["pair"] = float(np.linalg.solve(np.array([[s, -w], [-w, s]]),
                                       np.full(2, DELTA))[0])
    adj = np.eye(4, k=1) + np.eye(4, k=-1)
    x4 = np.linalg.solve(s * np.eye(4) - w * adj, np.full(4, DELTA))
    lv["chain end"], lv["chain mid"] = float(x4[0]), float(x4[1])
    return lv


def fig_scores(pools, T=200):
    fig, axes = plt.subplots(3, 2, figsize=(9, 8.6), sharex=True)
    lv2 = attractor_levels(2.0)   # both columns are the erf (doubled) family
    for i, (ss, sr, pair) in enumerate(PAIRS):
        for k, td in enumerate([1e-6, KINK[pair]]):
            ax = axes[i, k]
            ht, hf, _ = pools.get((pair, td, T, "classical"), (None, None, None))
            if ht is None:
                ax.set_axis_off(); continue
            for name, v in lv2.items():
                ax.axvline(v, color="#79b791", lw=0.8, ls="-.", alpha=0.8, zorder=0)
                if i == 0:
                    ax.text(v, 2e5, f" {name}", fontsize=5.5, color="#3f7f52",
                            rotation=90, va="top")
            ax.stairs(np.maximum(ht, 0.5), BINS, color=C_TRUE, lw=1.4, label="true")
            ax.stairs(np.maximum(hf, 0.5), BINS, color=C_FALSE, lw=1.4, label="false")
            st = analyse_pool(ht, hf)
            for tau, ls, lab in ((TAU_ABS, "-", r"$\tau$=0.35"),
                                 (st["tau_J"], "--", r"$\tau_J$"),
                                 (st["tau_wp99"], ":", r"$\tau_{wp99}$")):
                ax.axvline(tau, color="#52514e", ls=ls, lw=1.0)
            ax.set_yscale("log")
            ax.set_xlim(-0.1, 1.7)
            ttl = ("step ($\\theta_d$=1e-6)" if td == 1e-6
                   else f"erf $\\theta_d=\\varepsilon/3$ ({td*1e3:.3g} mrad)")
            ax.set_title(f"{pair} · {ttl}   AUC={st['auc']:.4f}  J*={st['youden_J']:.3f}",
                         fontsize=8.5)
            if k == 0:
                ax.set_ylabel("segments / bin")
            if i == 0 and k == 0:
                ax.legend(loc="upper right", fontsize=8)
                ax.text(TAU_ABS, ax.get_ylim()[1] * 0.5, "  0.35 | --$\\tau_J$ | ··$\\tau_{wp99}$",
                        fontsize=6.5, color="#52514e")
    axes[-1, 0].set_xlabel("segment activation score")
    axes[-1, 1].set_xlabel("segment activation score")
    fig.suptitle(f"ERF pooled activation spectra, classical, T={T} — true vs false by kernel and noise pair", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    footer(fig)
    fig.savefig(FIGS / f"erf_youden_scores_T{T}.png"); plt.close(fig)


def fig_roc(pools, T=200):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), sharey=True)
    for ax, (ss, sr, pair) in zip(axes, PAIRS):
        for td in TDS:
            ht, hf, _ = pools.get((pair, td, T, "classical"), (None, None, None))
            if ht is None:
                continue
            fpr, tpr, taus = roc_from_hists(ht, hf)
            st = analyse_pool(ht, hf)
            if not np.isfinite(st["auc"]):
                continue                      # pool fully excluded by the gate
            ax.plot(np.clip(fpr, 1e-6, None), tpr, color=TD_COLOR[td], lw=1.6,
                    label=f"$\\theta_d$={td:g}")
            for tau, mk in ((TAU_ABS, "o"), (st["tau_J"], "^"), (st["tau_EER"], "s")):
                if not np.isfinite(tau):
                    continue
                i = min(int(np.searchsorted(BINS, tau)), len(fpr) - 1)
                ax.plot(max(fpr[i], 1e-6), tpr[i], mk, color=TD_COLOR[td], ms=4.5,
                        mec="white", mew=0.5)
        ax.plot([1e-6, 1], [1, 0], color="#c3c2b7", lw=0.8, ls=":")   # EER locus
        ax.set_xscale("log"); ax.set_xlim(1e-6, 1); ax.set_ylim(0.55, 1.005)
        ax.set_title(f"{pair} ($\\sigma_s$={ss:g}, $\\sigma_r$={sr:g})", fontsize=9)
        ax.set_xlabel("false-positive rate (per false segment)")
    axes[0].set_ylabel("true-segment efficiency (TPR)")
    axes[0].legend(fontsize=7.5, loc="lower right", title="kernel width", title_fontsize=7.5)
    axes[1].text(2e-6, 0.60, "markers: ● $\\tau$=0.35  ▲ $\\tau_J$  ■ $\\tau_{EER}$",
                 fontsize=7.5, color="#52514e")
    fig.suptitle(f"ERF classical ROC by kernel width, T={T} (log FPR; dotted = EER locus)", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    footer(fig)
    fig.savefig(FIGS / f"erf_youden_roc_T{T}.png"); plt.close(fig)


def fig_tau_vs_T(df):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)
    for ax, (ss, sr, pair) in zip(axes, PAIRS):
        for td in (1e-6, KINK[pair], 1e-3):
            lab = ("step" if td == 1e-6
                   else "$\\varepsilon/3$" if td == KINK[pair] else f"{td:g}")
            g = df[(df.pair == pair) & (df.theta_d == td) & (df.solver == "classical")].sort_values("n_trk")
            ax.plot(g.n_trk, g.tau_J, "-o", color=TD_COLOR[td], ms=3.5, lw=1.5,
                    label=f"$\\tau_J$, $\\theta_d$={lab}")
            ax.plot(g.n_trk, g.tau_wp99, ":s", color=TD_COLOR[td], ms=3, lw=1.2)
        ax.axhline(TAU_ABS, color="#52514e", lw=1.0, ls="--")
        ax.text(11, TAU_ABS + 0.012, "abs $\\tau$=0.35", fontsize=7, color="#52514e")
        ax.set_xscale("log"); ax.set_xlabel("T (tracks)")
        ax.set_title(pair, fontsize=9)
    axes[0].set_ylabel(r"threshold $\tau$")
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[1].text(11, 0.02, "solid ● = Youden $\\tau_J$ · dotted ■ = wp99 (1% true quantile)",
                 fontsize=7.5, color="#52514e")
    fig.suptitle("Where the optimal threshold sits vs multiplicity — Youden and wp99 against the fixed 0.35 cut (classical)", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    footer(fig)
    fig.savefig(FIGS / "erf_youden_tau_vs_T.png"); plt.close(fig)


def fig_separability(df):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    metrics = [("auc", "AUC", (0.95, 1.0005)), ("youden_J", "Youden J*", (0.6, 1.02)),
               ("eer", "EER", None)]
    for ax, (col, lab, ylim) in zip(axes, metrics):
        for _, (_, _, pair) in enumerate(PAIRS):
            for T, ls in ((200, "-"), (1000, "--")):
                g = df[(df.pair == pair) & (df.n_trk == T) & (df.solver == "classical")].sort_values("theta_d")
                ax.plot(g.theta_d, g[col], ls, marker="o", ms=3.5, lw=1.5,
                        color=PAIR_RAMP[pair],
                        label=f"{pair}, T={T}" if col == "auc" else None)
        ax.set_xscale("log")
        if col == "eer":
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel(r"kernel width $\theta_d$ (rad)")
        ax.set_title(lab, fontsize=9.5)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Separability of pooled true/false scores vs kernel width (classical; solid T=200, dashed T=1000)", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    footer(fig)
    fig.savefig(FIGS / "erf_youden_separability_vs_td.png"); plt.close(fig)


def fig_quantum(pools, df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    # (a,b) heavy-pair pooled spectra at T=200: classical vs 1BQF (rescaled)
    for ax, td, ttl in ((axes[0, 0], 1e-6, "step ($\\theta_d$=1e-6)"),
                        (axes[0, 1], KINK["heavy"],
                         "erf $\\theta_d=\\varepsilon/3$ (2.22 mrad)")):
        for solver, ls in (("classical", "-"), ("quantum", "--")):
            got = pools.get(("heavy", td, 200, solver))
            if got is None:
                continue
            ht, hf, _ = got
            ax.stairs(np.maximum(ht, 0.5), BINS, color=C_TRUE, lw=1.4, ls=ls)
            ax.stairs(np.maximum(hf, 0.5), BINS, color=C_FALSE, lw=1.4, ls=ls)
        ax.axvline(TAU_ABS, color="#52514e", lw=1.0, ls="--")
        ax.set_yscale("log"); ax.set_xlim(-0.1, 1.7)
        ax.set_title(f"heavy pair, T=200 · {ttl}", fontsize=9)
        ax.set_xlabel("activation score"); ax.set_ylabel("segments / bin")
    handles = [Line2D([], [], color=C_TRUE, lw=1.4, label="true"),
               Line2D([], [], color=C_FALSE, lw=1.4, label="false"),
               Line2D([], [], color="#52514e", lw=1.2, ls="-", label="classical"),
               Line2D([], [], color="#52514e", lw=1.2, ls="--", label="1BQF (rescaled)")]
    axes[0, 0].legend(handles=handles, fontsize=7.5, loc="upper right")
    # (c) AUC vs T, classical vs quantum, step + wide erf, heavy pair
    ax = axes[1, 0]
    for solver, color in (("classical", C_CLASS), ("quantum", C_QUANT)):
        for td, ls in ((1e-6, "-"), (KINK["heavy"], "--")):
            g = df[(df.pair == "heavy") & (df.theta_d == td) & (df.solver == solver)].sort_values("n_trk")
            if not len(g):
                continue
            ax.plot(g.n_trk, g.auc, ls, marker="o", ms=3.5, lw=1.5, color=color,
                    label=f"{'1BQF' if solver=='quantum' else solver}, "
                          f"{'step' if td==1e-6 else 'erf ε/3'}")
    ax.set_xscale("log"); ax.set_xlabel("T (tracks)"); ax.set_ylabel("AUC")
    ax.set_title("AUC vs T — heavy pair", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    # (d) false rate paid at the wp99 point vs T
    ax = axes[1, 1]
    for solver, color in (("classical", C_CLASS), ("quantum", C_QUANT)):
        for td, ls in ((1e-6, "-"), (KINK["heavy"], "--")):
            g = df[(df.pair == "heavy") & (df.theta_d == td) & (df.solver == solver)].sort_values("n_trk")
            if not len(g):
                continue
            ax.plot(g.n_trk, np.clip(g.far_wp99, 5e-5, None), ls, marker="o", ms=3.5,
                    lw=1.5, color=color)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("T (tracks)"); ax.set_ylabel("false rate at $\\tau_{wp99}$")
    ax.set_title("false rate paid at the wp99 working point — heavy pair", fontsize=9)
    fig.suptitle("Classical vs 1BQF on pooled scores (quantum on the classical signal scale, rescale_to_signal; store spans T≤1000 for both)", y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    footer(fig)
    fig.savefig(FIGS / "erf_youden_quantum.png"); plt.close(fig)


def fig_operating_points(df, T=200):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    marks = (("abs", "o", r"$\tau$=0.35"), ("J", "^", r"$\tau_J$"), ("wp99", "*", r"$\tau_{wp99}$"))
    for ax, (ss, sr, pair) in zip(axes, PAIRS):
        for td in TDS:
            g = df[(df.pair == pair) & (df.theta_d == td) & (df.n_trk == T) & (df.solver == "classical")]
            if not len(g):
                continue
            r = g.iloc[0]
            xs = [max(r[f"far_{t}"], 2e-5) for t, _, _ in marks]
            ys = [r[f"eff_{t}"] for t, _, _ in marks]
            ax.plot(xs, ys, "-", color=TD_COLOR[td], lw=0.9, alpha=0.8)
            for (t, mk, _), x, y in zip(marks, xs, ys):
                ax.plot(x, y, mk, color=TD_COLOR[td], ms=6 if mk == "*" else 4.5,
                        mec="white", mew=0.4)
        ax.set_xscale("log"); ax.set_xlabel("false rate (n_false_active / n_active)")
        ax.set_title(f"{pair}", fontsize=9)
        ax.set_xlim(1e-5, 1)
    axes[0].set_ylabel("segment efficiency")
    handles = ([Line2D([], [], color=TD_COLOR[td], lw=1.6, label=f"$\\theta_d$={td:g}") for td in TDS]
               + [Line2D([], [], color="#52514e", ls="", marker=mk, label=lab) for _, mk, lab in marks])
    axes[0].legend(handles=handles, fontsize=6.4, loc="lower left", ncol=2)
    fig.suptitle(f"The threshold menu at T={T} (classical): what Youden / wp99 buy against the fixed 0.35 cut", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    footer(fig)
    fig.savefig(FIGS / f"erf_youden_operating_points_T{T}.png"); plt.close(fig)


def fig_matched(df):
    """THE fair cross-kernel comparison: pooled far at matched efficiency."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), sharey=True)
    for ax, (ss, sr, pair) in zip(axes, PAIRS):
        for td, lab, color in ((1e-6, "step ($\\theta_d$=1e-6)", "#52514e"),
                               (KINK[pair], "erf $\\theta_d=\\varepsilon/3$", "#08519c"),
                               (1e-3, "erf $\\theta_d$=1e-3", "#6baed6")):
            g = df[(df.pair == pair) & (df.solver == "classical")
                   & np.isclose(df.theta_d, td, rtol=1e-3)].sort_values("n_trk")
            if not len(g):
                continue
            ax.plot(g.n_trk, np.clip(g.far_m99, 5e-5, None), "-o", ms=4, lw=1.6,
                    color=color, label=lab)
            ax.plot(g.n_trk, np.clip(g.far_m999, 5e-5, None), ":o", ms=2.5, lw=1.0,
                    color=color, alpha=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("T (tracks)")
        ax.set_title(pair, fontsize=9.5)
    axes[0].set_ylabel("pooled false rate at matched efficiency")
    axes[0].legend(fontsize=7)
    axes[1].text(11, 6e-5, "solid: eff=99% · dotted: eff=99.9% · pooled per-segment, "
                 "gated (max$|x|\\leq$50)", fontsize=7.5, color="#52514e")
    fig.suptitle("Threshold-free kernel comparison: pooled far at MATCHED efficiency "
                 "(classical; each kernel at its own τ)", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    footer(fig)
    fig.savefig(FIGS / "erf_youden_matched_eff.png")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    pools = pool_scores()
    rows = []
    for (pair, td, T, solver), (ht, hf, excl) in sorted(pools.items()):
        ss, sr = next((s, r) for s, r, lab in PAIRS if lab == pair)
        rows.append(dict(pair=pair, sigma_scatt=ss, sigma_res=sr, theta_d=td,
                         n_trk=T, solver=solver, **analyse_pool(ht, hf),
                         n_solves=int(excl[1]), n_excluded=int(excl[0]),
                         excluded_frac=float(excl[0] / max(excl[1], 1))))
    df = pd.DataFrame(rows).sort_values(["pair", "theta_d", "n_trk", "solver"])
    out = RES / "erf_youden_eer.csv"
    df.to_csv(out, index=False)
    print(f"[csv] {len(df)} pool rows -> {out}")

    fig_scores(pools, T=200)
    fig_roc(pools, T=200)
    fig_tau_vs_T(df)
    fig_separability(df)
    fig_quantum(pools, df)
    fig_operating_points(df, T=200)
    fig_matched(df)
    print("[figs] 7 figures ->", FIGS)

    # headline numbers for the write-up
    for pair in ("clean", "moderate", "heavy"):
        for td in (1e-6, KINK[pair], 1e-3):
            r = df[(df.pair == pair) & (df.theta_d == td) & (df.n_trk == 200) & (df.solver == "classical")]
            if len(r):
                r = r.iloc[0]
                print(f"  T=200 {pair:9} td={td:g}: AUC={r.auc:.4f} J*={r.youden_J:.3f} "
                      f"tau_J={r.tau_J:.3f} EER={r.eer:.4f} | eff/far @0.35="
                      f"{r.eff_abs:.3f}/{r.far_abs:.3f} @J={r.eff_J:.3f}/{r.far_J:.3f} "
                      f"@wp99={r.eff_wp99:.3f}/{r.far_wp99:.3f} | far@m99={r.far_m99:.3f} "
                      f"@m999={r.far_m999:.3f}")


if __name__ == "__main__":
    main()
