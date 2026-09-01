#!/usr/bin/env python3
"""Fig-2/fig-3 analogues for the modified Hamiltonians (George 2026-09-01).

F1 figures/xiv_responses_modified.png — the paper's response_functions.pdf
   (fig 3) analogue: the FITTED response |p(lambda)| per operator, drawn
   against that operator's exact motif lines (closed forms, 09_modified_atlas):
   base d=40 / +fork d=40 / +occupancy d=80 / +both d=80, moderate T=200,
   rep-0 universal response (the production recipe).
F2 figures/xiv_motif_lines_modified.png — the fig-2 bottom-row analogue:
   the exact eigenvalue lines of the canonical motifs under each operator,
   base positions ghosted, with the same-role k-clique top line
   lambda = s + 4a + 2a(k-1) that sets the occupancy span wall.

Line vocabulary (all exact):
  chains m=1..4 (m=4 = true track P4): lambda_k = s + 4a - 2 cos(k pi/(m+1))
  isolated: s + 4a
  same-role pair (outside window): s + 2a, s + 6a
  competing pair (inside window):  s + 4a -+ (2a + beta)
  same-role k-clique at one hit:   s + 2a (x(k-1)),  s + 4a + 2a(k-1)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT, FIG, CACHE = HERE / "outputs", HERE / "figures", HERE / "outputs" / "cache"

GREEN, RED, GREY, BLUE, ORANGE = "#3d8a4f", "#d84a49", "#8f8d86", "#2a78d6", "#e08a2e"
INK, PURPLE = "#33322e", "#7b5ea7"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

S, DELTA = 4.0, 1.0
OPS = [("base", 0.0, 0.0, 40, "base $A$  (fitted $d{=}40$)"),
       ("fork_b0.5", 0.0, 0.5, 40, "$A+\\beta B_{\\rm fork}$, $\\beta{=}0.5$  (fitted $d{=}40$)"),
       ("occ_a0.05", 0.05, 0.0, 80, "$A_{\\rm occ}$, $\\alpha{=}0.05$  (fitted $d{=}80$)"),
       ("occ0.05_fork0.5", 0.05, 0.5, 80, "$A_{\\rm occ}+\\beta B_{\\rm fork}$  (fitted $d{=}80$)")]


def chain_lines(m, a):
    k = np.arange(1, m + 1)
    return S + 4 * a - 2 * np.cos(k * np.pi / (m + 1))


def motif_lines(a, beta):
    """(true lines, false lines, labels-for-false) under (alpha, beta)."""
    true = list(chain_lines(4, a))                       # P4 track
    false = {"isolated": [S + 4 * a],
             "false pair (chain $m{=}2$)": list(chain_lines(2, a)),
             "false triple (chain $m{=}3$)": list(chain_lines(3, a))}
    if beta > 0:
        false["competing pair (window)"] = [S + 4 * a - (2 * a + beta),
                                            S + 4 * a + (2 * a + beta)]
    if a > 0:
        false["same-role pair"] = [S + 2 * a, S + 6 * a]
    return true, false


def response_curve(tag, deg, rep=0, regime="moderate", full=False):
    z = np.load(CACHE / f"06_coef_{regime}_{tag}_rep{rep}.npz")
    lo, hi = float(z["lo"]), float(z["hi"])
    c = z[f"fitted_c_d{deg}"]
    c0, c1 = 0.5 * (hi + lo), 0.5 * (hi - lo) * 1.02
    top = hi if full else min(hi, 8.0)
    lam = np.linspace(max(lo, 1.2), top, 6000)
    u = (lam - c0) / c1
    p = np.abs(np.polynomial.chebyshev.chebval(u, c))
    return lam, p / p.max(), (lo, hi)


def fig_responses():
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.6), constrained_layout=True)
    for ax, (tag, a, beta, deg, title) in zip(axes.ravel(), OPS):
        lam, p, dom = response_curve(tag, deg)
        true, false = motif_lines(a, beta)
        for v in true:
            ax.axvline(v, color=GREEN, ls=":", lw=1.5, alpha=0.9, zorder=1)
        for i, (nm, vals) in enumerate(false.items()):
            for v in vals:
                ax.axvline(v, color=RED, ls="--", lw=1.0, alpha=0.65, zorder=1)
        ax.plot(lam, p, color=INK, lw=1.8, zorder=3)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlim(1.6, 6.9)
        ax.set_ylim(0, 1.06)
        ax.grid(alpha=0.2, lw=0.4)
        ax.annotate(f"domain $[{dom[0]:.2f},\\,{dom[1]:.1f}]$",
                    xy=(0.985, 0.93), xycoords="axes fraction", ha="right",
                    fontsize=8.2, color="#55534d")
        if dom[1] > 10:
            # occupancy operators: the fit lives on a span-widened domain —
            # show the whole response so the motif window is seen in context
            ins = ax.inset_axes([0.58, 0.55, 0.40, 0.33])
            lamf, pf, _ = response_curve(tag, deg, full=True)
            ins.plot(lamf, pf, color=INK, lw=0.7)
            ins.axvspan(1.6, 6.9, color=ORANGE, alpha=0.18, lw=0)
            ins.set_title("full domain (motif window shaded)", fontsize=7.2,
                          pad=2)
            ins.tick_params(labelsize=6.5)
            ins.set_yticks([])
    for ax in axes[1]:
        ax.set_xlabel("$\\lambda$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$|p(\\lambda)|$ (normalised)")
    fig.suptitle("The fitted response against each operator's motif lines "
                 "(moderate noise, $T{=}200$, rep-0 universal response)",
                 fontsize=12.5)
    fig.text(0.5, -0.025,
             "green dotted: the four true-track ($P_4$) lines · red dashed: "
             "false-motif lines (isolated, chains $m{=}2,3$, competing pair, "
             "same-role pair) · $\\sigma_{\\rm scatt}{=}10^{-4}$, "
             "$\\sigma_{\\rm res}{=}10\\,\\mu$m, drop 1%, formula "
             "$\\varepsilon{=}3.18$ mrad · $\\gamma{=}3$, $\\delta{=}1$ · "
             "occupancy $\\alpha{=}0.05$, fork $\\beta{=}0.5$, "
             "$\\varepsilon_B{=}\\varepsilon$ · fitted = moment-space ridge, "
             "$\\mu$ by eff@far$\\leq$1%", ha="center", va="top",
             fontsize=8.2, color="#55534d")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"xiv_responses_modified.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/xiv_responses_modified.png")


def fig_motif_lines():
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 7.8), sharex=True,
                             constrained_layout=True)
    base_true, base_false = motif_lines(0.0, 0.0)
    base_all = base_true + [v for vs in base_false.values() for v in vs]
    for ax, (tag, a, beta, _, title) in zip(axes, OPS):
        true, false = motif_lines(a, beta)
        if a > 0 or beta > 0:
            for v in base_all:
                ax.vlines(v, 0, 1.0, color=GREY, lw=1.0, alpha=0.45)
        for v in true:
            ax.vlines(v, 0, 1.0, color=GREEN, lw=2.4)
        for nm, vals in false.items():
            for v in vals:
                ax.vlines(v, 0, 0.72, color=RED, lw=1.7)
        if a > 0:
            # the same-role k-clique top line that sets the span wall
            for k, xfrac in ((10, 0.42), (50, 0.62)):
                lam = S + 4 * a + 2 * a * (k - 1)
                if lam < 9.6:
                    ax.vlines(lam, 0, 0.5, color=PURPLE, lw=1.7, ls="-")
                    ax.annotate(f"$k{{=}}{k}$ clique", xy=(lam, 0.52),
                                fontsize=7.8, color=PURPLE, ha="center")
            ax.annotate("$\\lambda_{\\max} = s+4\\alpha+2\\alpha(k{-}1)$ "
                        "$\\to$ the span wall", xy=(0.99, 0.8),
                        xycoords="axes fraction", ha="right", fontsize=8.4,
                        color=PURPLE)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.set_ylabel(title.split("(")[0].strip(), fontsize=9.2, rotation=0,
                      ha="right", va="center", labelpad=8)
    axes[-1].set_xlabel("$\\lambda$")
    axes[-1].set_xlim(2.2, 9.7)
    fig.suptitle("The motif lines under each operator (exact closed forms; "
                 "grey = base-$A$ positions)", fontsize=12.5)
    fig.text(0.5, -0.02,
             "green: true-track $P_4$ lines $s{+}4\\alpha{-}2\\cos(k\\pi/5)$ · "
             "red: false motifs — isolated $s{+}4\\alpha$, chains "
             "$m{=}2,3$, competing pair $s{+}4\\alpha{\\mp}(2\\alpha{+}\\beta)$, "
             "same-role pair $s{+}2\\alpha$, $s{+}6\\alpha$ · purple: same-role "
             "$k$-clique top line (occupancy span mechanism) · "
             "$s{=}4$, $\\alpha{=}0.05$, $\\beta{=}0.5$", ha="center",
             va="top", fontsize=8.2, color="#55534d")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"xiv_motif_lines_modified.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/xiv_motif_lines_modified.png")


if __name__ == "__main__":
    fig_responses()
    fig_motif_lines()
