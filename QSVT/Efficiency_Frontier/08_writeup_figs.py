#!/usr/bin/env python3
"""QSVT XIV write-up figures — the LaTeX paper's segment-metrics style.

Conventions (George 2026-09-02; this supersedes the 2026-08-26 note that read
the classical solver at a fixed absolute threshold — that threshold was a bug):
  * EVERY solver, the classical inversion included, is read at its
    efficiency-first working point: the lowest threshold that keeps 99% of
    the true segments (eff_e990 / far_e990 columns).  No curve anywhere is
    read at a fixed threshold.
  * The QSVT is one solver.  Its variants are labelled by polynomial degree
    (held at 40 here) and by the Hamiltonian the problem is set up on
    (base A, A + occupancy term, A + bifurcation term, both).

Figures (paper rc style, make_paper_figures palette):
  F1 figures/xiv_metrics_vs_T_{clean,moderate,heavy}.png
     (a) segment efficiency vs T, (b) segment false rate vs T — classical,
     1BQF, and the degree-40 QSVT on each Hamiltonian, T = 200..1000.
  F2 figures/xiv_metrics_composite.png — the three regimes stacked.
  F3 figures/xiv_loss_budget.png — stacked loss channels + twin fraction.
Numbers: outputs/08_writeup_numbers.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "outputs", HERE / "figures"
FIG.mkdir(exist_ok=True)

GREEN, RED, GREY, BLUE, ORANGE = "#3d8a4f", "#d84a49", "#8f8d86", "#2a78d6", "#e08a2e"
INK, PURPLE = "#33322e", "#7b5ea7"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

D_HEAD = 40.0
# colour follows the Hamiltonian; every QSVT legend entry names degree + Hamiltonian
HAM_STYLE = {
    "base":            (BLUE,   "-",  "o", "base $A$"),
    "occ_a0.05":       (GREEN,  "-",  "s", "$A$ + occupancy $\\alpha{=}0.05$"),
    "occ_a0.10":       ("#2f6b3e", "--", "s", "$A$ + occupancy $\\alpha{=}0.10$"),
    "fork_b0.5":       (ORANGE, "-",  "D", "$A$ + bifurcation $\\beta{=}0.5$"),
    "occ0.05_fork0.5": (PURPLE, "-",  "^", "$A$ + occupancy + bifurcation"),
    "erf":             ("#b0762a", ":",  "v", "erf-kernel $A$"),
}
HAM_PLAIN = {k: v[3].replace("$", "").replace("\\alpha{=}", "alpha=")
             .replace("\\beta{=}", "beta=") for k, v in HAM_STYLE.items()}


def qsvt_label(ham):
    return f"QSVT $d{{=}}{int(D_HEAD)}$, {HAM_STYLE[ham][3]}"


def load_frames():
    frames = []
    for fn in ["01_frontier_clean_moderate.csv", "02_setout_frontier.csv",
               "02_fork_frontier.csv", "04_heavy_frontier.csv",
               "04_heavy_fork.csv", "04_heavy_base_ref.csv",
               "03_highT_frontier.csv",
               # full degree scan at T=200 (all regimes, all Hamiltonians,
               # matched 1BQF included) + the high-T matched 1BQF reference
               "06_degree_scan.csv", "06b_highT_onebqf.csv"]:
        p = OUT / fn
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "setout" not in d.columns:
            d["setout"] = "base"
        if "T" not in d.columns:
            d["T"] = 200
        d["src"] = fn
        frames.append(d)
    p1b = OUT / "03_1bqf_reference.csv"
    if p1b.exists():
        b = pd.read_csv(p1b)
        b["family"] = "onebqf_cos"
        b["src"] = "03_1bqf_reference.csv"
        frames.append(b)
    df = pd.concat(frames, ignore_index=True)
    df["family"] = df["family"].replace({"fitted_ridge": "fitted_moment"})
    return df


def med(g, col):
    return g[col].median() if len(g) else np.nan


NOISE_TXT = {
    "clean":    "$\\sigma_{\\rm scatt}{=}10^{-4}$, $\\sigma_{\\rm res}{=}0$, drop 0",
    "moderate": "$\\sigma_{\\rm scatt}{=}10^{-4}$, $\\sigma_{\\rm res}{=}10\\,\\mu$m, drop 1%",
    "heavy":    "$\\sigma_{\\rm scatt}{=}10^{-4}$, $\\sigma_{\\rm res}{=}20\\,\\mu$m, drop 1%",
}
WP_TXT = ("every solver at its efficiency-first working point: the lowest "
          "threshold keeping 99% of the true segments")


def param_box(fig, regime, sub, extra=""):
    """George 2026-09-01: every figure carries its parameters."""
    eps = sub["eps"].dropna()
    eps_txt = f"formula $\\varepsilon{{=}}{eps.median()*1e3:.2f}$ mrad" \
        if len(eps) else "formula $\\varepsilon$"
    txt = (f"{NOISE_TXT.get(regime, regime)} · {eps_txt} · "
           "$\\gamma{=}3$, $\\delta{=}1$, step kernel · "
           "occupancy $\\alpha{=}0.05$ · bifurcation $\\beta{=}0.5$, "
           "$\\varepsilon_B{=}\\varepsilon$ · QSVT polynomial: degree 40, "
           "coefficients least-squares fit per Hamiltonian in Chebyshev-moment "
           "space (ridge weight chosen for best efficiency at false rate "
           "$\\leq$ 1%) · " + WP_TXT + (" · " + extra if extra else ""))
    fig.text(0.5, -0.035, txt, ha="center", va="top", fontsize=8.2,
             color="#55534d", wrap=True)


def series_classical(sub, Ts):
    T_, e_, f_ = [], [], []
    for T in Ts:
        g = sub[(sub.setout == "base") & (sub["T"] == T)
                & (sub.family == "classical_invA")]
        if len(g):
            T_.append(T); e_.append(med(g, "eff_e990")); f_.append(med(g, "far_e990"))
    return T_, e_, f_


def series_onebqf(sub, Ts):
    T_, e_, f_ = [], [], []
    for T in Ts:
        g = sub[(sub.setout == "base") & (sub["T"] == T)
                & (sub.family.isin(["1bqf_cos", "onebqf_cos"]))]
        if len(g):
            T_.append(T); e_.append(med(g, "eff_e990")); f_.append(med(g, "far_e990"))
    return T_, e_, f_


def series_qsvt(sub, Ts, ham):
    T_, e_, f_ = [], [], []
    for T in Ts:
        g = sub[(sub.setout == ham) & (sub["T"] == T)
                & (sub.family == "fitted_moment") & (sub.degree == D_HEAD)]
        if len(g):
            T_.append(T); e_.append(med(g, "eff_e990")); f_.append(med(g, "far_e990"))
    return T_, e_, f_


def draw_regime(axE, axF, sub, Ts, hams, rows, regime, lw=2.0):
    cT, ce, cf = series_classical(sub, Ts)
    axE.plot(cT, ce, "--", color=INK, lw=lw, marker="x", ms=6, label="classical")
    axF.plot(cT, np.maximum(cf, 2e-4), "--", color=INK, lw=lw, marker="x",
             ms=6, label="classical")
    for T, e, f in zip(cT, ce, cf):
        rows.append(dict(regime=regime, T=T, hamiltonian="base",
                         series="classical", eff=e, far=f))
    bT, be, bf = series_onebqf(sub, Ts)
    if bT:
        axE.plot(bT, be, "-.", color=RED, lw=lw - 0.2, marker="+", ms=7,
                 label="1BQF")
        axF.plot(bT, np.maximum(bf, 2e-4), "-.", color=RED, lw=lw - 0.2,
                 marker="+", ms=7, label="1BQF")
        for T, e, f in zip(bT, be, bf):
            rows.append(dict(regime=regime, T=T, hamiltonian="base",
                             series="1BQF", eff=e, far=f))
    for ham in hams:
        col, ls, mk, _ = HAM_STYLE[ham]
        xT, xe, xf = series_qsvt(sub, Ts, ham)
        if xT:
            axE.plot(xT, xe, ls, color=col, lw=lw, marker=mk, ms=5,
                     label=qsvt_label(ham))
            axF.plot(xT, np.maximum(xf, 2e-4), ls, color=col, lw=lw,
                     marker=mk, ms=5, label=qsvt_label(ham))
            for T, e, f in zip(xT, xe, xf):
                rows.append(dict(regime=regime, T=T, hamiltonian=HAM_PLAIN[ham],
                                 series=f"QSVT degree {int(D_HEAD)}", eff=e, far=f))


def fig_metrics_vs_T(df):
    rows = []
    for regime in ("clean", "moderate", "heavy"):
        sub = df[df.regime == regime]
        Ts = sorted(sub["T"].unique())
        if not Ts:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                                 constrained_layout=True)
        draw_regime(axes[0], axes[1], sub, Ts, list(HAM_STYLE), rows, regime)
        axes[0].set_ylabel("segment efficiency")
        axes[0].set_ylim(0.90, 1.005)
        axes[0].axhline(0.99, color=GREY, lw=0.7, ls=":")
        axes[0].set_title("(a) segment efficiency", loc="left")
        axes[1].set_yscale("log")
        axes[1].set_ylim(2e-4, 1.2)
        axes[1].set_ylabel("segment false rate")
        axes[1].set_title("(b) segment false rate", loc="left")
        for ax in axes:
            ax.set_xlabel("tracks per event $T$")
            ax.set_xscale("log")
            ax.set_xticks(Ts)
            ax.set_xticklabels([str(t) for t in Ts])
            ax.grid(alpha=0.25, lw=0.5)
        axes[1].legend(loc="lower right", ncol=2)
        fig.suptitle(f"Segment metrics against track density — {regime} noise "
                     "(every solver at its efficiency-first working point; "
                     "QSVT at degree 40)", fontsize=12)
        param_box(fig, regime, sub,
                  extra="$T$ reps: 10 (200) / 5 (400) / 3 (700, 1000) · "
                        "1BQF = matched cosine $|\\cos(\\pi\\lambda/2s')|$ "
                        "on the same events")
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"xiv_metrics_vs_T_{regime}.{ext}", dpi=150,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] figures/xiv_metrics_vs_T_{regime}.png (T = {Ts})")
    pd.DataFrame(rows).to_csv(OUT / "08_writeup_numbers.csv", index=False)
    print("[saved] outputs/08_writeup_numbers.csv")


def fig_metrics_composite(df):
    """Paper composite: 3 regimes x (eff, far) in one fig-6-style block."""
    hams = ["base", "occ_a0.05", "fork_b0.5", "occ0.05_fork0.5"]
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.2),
                             constrained_layout=True)
    for row, regime in enumerate(("clean", "moderate", "heavy")):
        sub = df[df.regime == regime]
        Ts = sorted(sub["T"].unique())
        axE, axF = axes[row]
        draw_regime(axE, axF, sub, Ts, hams, [], regime, lw=1.9)
        axE.set_ylim(0.985, 1.001)
        axE.set_ylabel(f"{regime}\nsegment efficiency")
        axF.set_yscale("log")
        axF.set_ylim(2e-4, 1.4)
        axF.set_ylabel("segment false rate")
        for ax in (axE, axF):
            ax.set_xscale("log")
            ax.set_xticks(Ts)
            ax.set_xticklabels([str(t) for t in Ts])
            ax.grid(alpha=0.25, lw=0.5)
        if row == 0:
            axE.set_title("(a) segment efficiency (every solver at its "
                          "efficiency-first working point)", loc="left")
            axF.set_title("(b) segment false rate", loc="left")
            axF.legend(loc="lower right", fontsize=7.8, ncol=2)
        if row == 2:
            for ax in (axE, axF):
                ax.set_xlabel("tracks per event $T$")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"xiv_metrics_composite.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/xiv_metrics_composite.png")


def fig_loss_budget():
    p = OUT / "05_loss_budget.csv"
    if not p.exists():
        print("no loss budget csv"); return
    f4 = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.4, 1]})
    g = (f4[f4.family == "fitted_moment"]
         .groupby(["regime", "setout", "degree"])[["eff", "L1", "L2", "L3"]]
         .median().reset_index())
    # one bar per (regime, Hamiltonian): the degree with the highest
    # efficiency at false rate <= 1% (median over events)
    best = g.loc[g.groupby(["regime", "setout"]).eff.idxmax()]
    best = best[best.regime.isin(["moderate", "heavy"])]
    best = best.sort_values(["regime", "eff"])
    lab = [f"{r.regime} · {HAM_PLAIN.get(r.setout, r.setout)} · $d{{=}}{int(r.degree)}$"
           for _, r in best.iterrows()]
    y = np.arange(len(best))
    left = np.zeros(len(best))
    for ch, col, name in [("L1", INK, "twin loss (irreducible)"),
                          ("L2", BLUE, "response-ripple loss"),
                          ("L3", ORANGE, "threshold-overlap loss")]:
        axes[0].barh(y, best[ch], left=left, color=col, height=0.7, label=name)
        left += best[ch].values
    axes[0].set_yticks(y); axes[0].set_yticklabels(lab, fontsize=8.5)
    axes[0].set_xlabel("efficiency loss at false rate $\\leq$ 1 %")
    axes[0].set_title("(a) where the lost efficiency lives (QSVT, best degree "
                      "per Hamiltonian)", loc="left")
    axes[0].legend(loc="lower right", fontsize=8)
    tw = (f4.groupby(["regime", "setout"]).twin_frac.median().reset_index())
    tw = tw[tw.regime.isin(["moderate", "heavy"])].sort_values("twin_frac")
    axes[1].barh(np.arange(len(tw)), tw.twin_frac, color=GREY, height=0.65)
    axes[1].set_yticks(np.arange(len(tw)))
    axes[1].set_yticklabels([f"{r.regime} · {HAM_PLAIN.get(r.setout, r.setout)}"
                             for _, r in tw.iterrows()], fontsize=8.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("twin fraction of true segments")
    axes[1].set_title("(b) the Hamiltonian sets the twin population", loc="left")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5, axis="x")
    fig.text(0.5, -0.035,
             "$T{=}200$ · moderate + heavy noise as in the degree scan · "
             "QSVT coefficients refit per Hamiltonian and degree · bar = the "
             "degree with the highest median efficiency at false rate "
             "$\\leq$ 1% for that Hamiltonian",
             ha="center", va="top", fontsize=8.2, color="#55534d", wrap=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"xiv_loss_budget.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/xiv_loss_budget.png")


if __name__ == "__main__":
    df = load_frames()
    fig_metrics_vs_T(df)
    fig_metrics_composite(df)
    fig_loss_budget()
