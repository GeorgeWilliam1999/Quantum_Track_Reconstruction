#!/usr/bin/env python3
"""Degree-scan figures for QSVT XIV + the paper (George 2026-09-01).

From outputs/06_degree_scan.csv (T=200, 10 reps, all regimes, operators
base / +occupancy / +fork / +both; fitted-moment + production-comb families;
matched 1BQF cosine + classical references on the same events):

  F1 figures/xiv_degree_scan_{moderate,heavy}.png
     (a) segment efficiency @ far <= 1% vs polynomial degree,
     (b) segment false rate @ matched eff 0.99 vs degree —
     fitted response per operator, per-rep dots + median lines; the 1BQF
     (a degree-1 filter) plotted at d=1 per operator; classical base-A
     reference line.  Clean is not drawn: every operator sits at 1.000
     for every degree >= 4 (stated in the parameter box).
  F2 figures/xiv_degree_marginal.png
     the MARGINAL efficiency gain from one degree step to the next
     (moderate + heavy panels) — the derivative of the sigmoid.
  F3 figures/xiv_degree_comb.png
     the same scan for the UNFITTED production line comb — what happens
     when the response is not refitted to the modified operator.

Numbers: outputs/10_degree_numbers.csv
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

GREEN, RED, GREY, BLUE, ORANGE = "#3d8a4f", "#d84a49", "#8f8d86", "#2a78d6", "#e08a2e"
INK, PURPLE = "#33322e", "#7b5ea7"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

OPS = {
    "base":            (BLUE,   "o", "base $A$"),
    "occ_a0.05":       (GREEN,  "s", "occupancy $\\alpha{=}0.05$"),
    "fork_b0.5":       (ORANGE, "D", "bifurcation $\\beta{=}0.5$"),
    "occ0.05_fork0.5": (PURPLE, "^", "occ.+bifur."),
}
NOISE_TXT = {
    "moderate": "$\\sigma_{\\rm scatt}{=}10^{-4}$, $\\sigma_{\\rm res}{=}10\\,\\mu$m, drop 1%",
    "heavy":    "$\\sigma_{\\rm scatt}{=}10^{-4}$, $\\sigma_{\\rm res}{=}20\\,\\mu$m, drop 1%",
}


def param_box(fig, txt):
    fig.text(0.5, -0.035, txt, ha="center", va="top", fontsize=8.2,
             color="#55534d")


def box_txt(regime, sub):
    eps = sub["eps"].dropna().median() * 1e3
    return (f"$T{{=}}200$, 10 reps · {NOISE_TXT[regime]} · formula "
            f"$\\varepsilon{{=}}{eps:.2f}$ mrad · $\\gamma{{=}}3$, "
            "$\\delta{=}1$, step kernel · fitted = moment-space ridge, "
            "$\\mu$ by eff@far$\\leq$1% · clean regime omitted: every "
            "operator reads 1.000 (median of 10 reps) at every degree $\\geq$ 4")


def degree_panels(df, family, fname, title_extra=""):
    for regime in ("moderate", "heavy"):
        sub = df[(df.regime == regime)]
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                                 constrained_layout=True)
        # classical base-A reference
        gcls = sub[(sub.setout == "base") & (sub.family == "classical_invA")]
        for ax, col in ((axes[0], "eff_f010"), (axes[1], "far_e990")):
            v = gcls[col].median()
            ax.axhline(v, color=INK, ls="--", lw=1.6,
                       label=f"classical (base $A$): {v:.3f}")
        for so, (c, mk, lab) in OPS.items():
            g = sub[(sub.setout == so) & (sub.family == family)]
            if not len(g):
                continue
            degs = sorted(g.degree.dropna().unique())
            for ax, col in ((axes[0], "eff_f010"), (axes[1], "far_e990")):
                m = [g[g.degree == d][col].median() for d in degs]
                ax.plot(degs, m, "-", color=c, lw=2.0, marker=mk, ms=4.5,
                        label=lab, zorder=3)
                ax.scatter(g.degree, g[col], s=6, color=c, alpha=0.22,
                           lw=0, zorder=2)
            # matched 1BQF (degree-1 filter) on the same operator
            g1 = sub[(sub.setout == so) & (sub.family == "onebqf_cos")]
            if len(g1):
                axes[0].scatter([1.55], [g1.eff_f010.median()], marker="*",
                                s=110, color=c, edgecolor=INK, lw=0.5,
                                zorder=4)
                axes[1].scatter([1.55], [max(g1.far_e990.median(), 2.5e-4)],
                                marker="*", s=110, color=c, edgecolor=INK,
                                lw=0.5, zorder=4)
        axes[0].set_ylabel("segment efficiency @ far $\\leq$ 1%")
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].set_title("(a) efficiency at a 1% false-rate budget",
                          loc="left")
        axes[1].set_yscale("log")
        axes[1].set_ylim(2.5e-4, 1.3)
        axes[1].set_ylabel("segment false rate @ eff 0.99")
        axes[1].set_title("(b) false rate at the matched 99% efficiency point",
                          loc="left")
        for ax in axes:
            ax.set_xscale("log")
            ticks = [1.55, 4, 8, 16, 32, 64, 160]
            ax.set_xticks(ticks)
            ax.set_xticklabels(["1BQF\n($\\star$)", "4", "8", "16", "32",
                                "64", "160"])
            ax.set_xlabel("polynomial degree $d$")
            ax.grid(alpha=0.25, lw=0.5)
        axes[0].legend(loc="lower right", fontsize=8.2)
        fig.suptitle(f"Polynomial degree against segment metrics — {regime} "
                     f"noise{title_extra}", fontsize=12)
        param_box(fig, box_txt(regime, sub))
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"{fname}_{regime}.{ext}", dpi=150,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] figures/{fname}_{regime}.png")


def fig_marginal(df):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                             constrained_layout=True)
    for ax, regime in zip(axes, ("moderate", "heavy")):
        sub = df[(df.regime == regime) & (df.family == "fitted_moment")]
        for so, (c, mk, lab) in OPS.items():
            g = sub[sub.setout == so]
            degs = np.array(sorted(g.degree.dropna().unique()))
            m = np.array([g[g.degree == d].eff_f010.median() for d in degs])
            dm = np.diff(m)
            ax.plot(degs[1:], dm, "-", color=c, lw=1.9, marker=mk, ms=4.5,
                    label=lab)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([8, 16, 32, 64, 160])
        ax.set_xticklabels(["8", "16", "32", "64", "160"])
        ax.set_xlabel("polynomial degree $d$")
        ax.set_title(f"({'ab'[list(axes).index(ax)]}) {regime} noise",
                     loc="left")
        ax.grid(alpha=0.25, lw=0.5)
        if regime == "moderate":
            # occupancy's d=4->8 onset step (+0.94) is off this scale: the
            # zoom shows the flat tail that carries the message
            ax.set_ylim(-0.022, 0.055)
            ax.annotate("occupancy onset $d{=}4{\\to}8$: $+0.94$ (off scale)",
                        xy=(8, 0.052), fontsize=8.2, color=GREEN,
                        ha="left", va="top")
    axes[0].set_ylabel("marginal efficiency gain per degree step\n"
                       "$\\Delta$ eff @ far $\\leq$ 1%")
    axes[0].legend(fontsize=8.2)
    fig.suptitle("The marginal value of one more degree step (fitted "
                 "response, $T{=}200$)", fontsize=12)
    param_box(fig, "moderate + heavy noise as in the degree scan · marginal "
                   "gain between CONSECUTIVE degrees of the (non-uniformly "
                   "spaced) scan grid {4, 8, ..., 48, 56, 64, 80, 84, 96, 120, 160} "
                   "· median of 10 reps")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"xiv_degree_marginal.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/xiv_degree_marginal.png")


if __name__ == "__main__":
    df = pd.read_csv(OUT / "06_degree_scan.csv")
    degree_panels(df, "fitted_moment", "xiv_degree_scan",
                  " (response refit at every degree)")
    fig_marginal(df)
    degree_panels(df, "comb_prod", "xiv_degree_comb",
                  " (UNFITTED production comb — no refit)")
    med = (df.groupby(["regime", "setout", "family", "degree"], dropna=False)
             [["eff_f010", "far_e990", "eff_e990"]].median().reset_index())
    med.to_csv(OUT / "10_degree_numbers.csv", index=False)
    print("[saved] outputs/10_degree_numbers.csv")
