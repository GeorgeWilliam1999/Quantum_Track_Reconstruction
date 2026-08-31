#!/usr/bin/env python3
"""QSVT XIV write-up figures — the LaTeX paper's segment-metrics style.

Conventions (locked, George 2026-08-26):
  * classical = its fixed absolute tau (attractor + 0.10; 0.35 at gamma=3) —
    the eff_fixed_tau / far_fixed_tau columns where present, else the
    Stage-1 eff_fixed/far_fixed columns;
  * every quantum solver = its wp99 high-efficiency working point
    (eff_e990 / far_e990) — per-solver taus, as always;
  * polynomial degree HELD at the production d=40; the Hamiltonian set-out is
    the variable.

Figures (paper rc style, make_paper_figures palette):
  F1 figures/xiv_metrics_vs_T_{clean,moderate,heavy}.png
     (a) segment efficiency vs T, (b) segment false rate vs T — classical
     fixed-tau reference + quantum wp99 per set-out, T = 200..1000.
  F2 figures/xiv_setout_T200.png — the T=200 set-out comparison (both panels).
  F3 figures/xiv_loss_budget.png — stacked L1-L3 + twin fraction (mechanism).
Numbers: outputs/08_writeup_numbers.csv
"""
from __future__ import annotations

import re
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

SETOUT_STYLE = {          # colour follows the OPERATOR
    "base":            (BLUE,   "-",  "o", "base $A$"),
    "occ_a0.05":       (GREEN,  "-",  "s", "occupancy $\\alpha{=}0.05$"),
    "occ_a0.10":       ("#2f6b3e", "--", "s", "occupancy $\\alpha{=}0.10$"),
    "fork_b0.5":       (ORANGE, "-",  "D", "bifurcation $\\beta{=}0.5$"),
    "occ0.05_fork0.5": (PURPLE, "-",  "^", "occ.+bifur. $\\alpha{=}.05,\\beta{=}.5$"),
    "erf":             ("#b0762a", ":",  "v", "erf kernel"),
}
D_HEAD = 40.0


def load_frames():
    frames = []
    for fn in ["01_frontier_clean_moderate.csv", "02_setout_frontier.csv",
               "02_fork_frontier.csv", "04_heavy_frontier.csv",
               "04_heavy_fork.csv", "04_heavy_base_ref.csv",
               "03_highT_frontier.csv"]:
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
    # the 1BQF store reference (base A, wp99 columns already in shape)
    p1b = OUT / "03_1bqf_reference.csv"
    if p1b.exists():
        b = pd.read_csv(p1b)
        b["family"] = "onebqf_cos"
        b["src"] = "03_1bqf_reference.csv"
        frames.append(b)
    df = pd.concat(frames, ignore_index=True)
    # older runners recorded the classical fixed-tau point in the notes text
    # ("fixed-tau 0.350: eff 0.9838 far 0.0226") — recover it into columns
    if "eff_fixed_tau" not in df.columns:
        df["eff_fixed_tau"] = np.nan
        df["far_fixed_tau"] = np.nan
    m = (df.family == "classical_invA") & df["eff_fixed_tau"].isna() \
        & df["notes"].astype(str).str.contains("fixed-tau")
    pat = re.compile(r"eff ([0-9.]+) far ([0-9.]+)")
    for i in df.index[m]:
        g = pat.search(str(df.at[i, "notes"]))
        if g:
            df.at[i, "eff_fixed_tau"] = float(g.group(1))
            df.at[i, "far_fixed_tau"] = float(g.group(2))
    df["family"] = df["family"].replace({"fitted_ridge": "fitted_moment"})
    return df


def med(g, col):
    return g[col].median() if len(g) else np.nan


def fig_metrics_vs_T(df):
    rows = []
    for regime in ("clean", "moderate", "heavy"):
        sub = df[df.regime == regime]
        Ts = sorted(sub["T"].unique())
        if not Ts:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                                 constrained_layout=True)
        # classical fixed-tau reference (base A)
        ce, cf, cT = [], [], []
        for T in Ts:
            g = sub[(sub.setout == "base") & (sub["T"] == T)
                    & (sub.family == "classical_invA")]
            e = med(g, "eff_fixed_tau") if "eff_fixed_tau" in g and \
                g["eff_fixed_tau"].notna().any() else med(g, "eff_fixed")
            f = med(g, "far_fixed_tau") if "far_fixed_tau" in g and \
                g["far_fixed_tau"].notna().any() else med(g, "far_fixed")
            if np.isfinite(e):
                cT.append(T); ce.append(e); cf.append(f)
            rows.append(dict(regime=regime, T=T, setout="base",
                             series="classical_fixed_tau", eff=e, far=f))
        axes[0].plot(cT, ce, "--", color=INK, lw=2.0, marker="x", ms=6,
                     label="classical (fixed $\\tau$)")
        axes[1].plot(cT, np.maximum(cf, 2e-4), "--", color=INK, lw=2.0,
                     marker="x", ms=6, label="classical (fixed $\\tau$)")
        # 1BQF wp99 reference where it exists (Stage-1 T=200 + heavy base ref)
        be, bf, bT = [], [], []
        for T in Ts:
            g = sub[(sub.setout == "base") & (sub["T"] == T)
                    & (sub.family.isin(["1bqf_cos", "onebqf_cos"]))]
            if len(g):
                bT.append(T); be.append(med(g, "eff_e990"))
                bf.append(med(g, "far_e990"))
                rows.append(dict(regime=regime, T=T, setout="base",
                                 series="1bqf_wp99", eff=be[-1], far=bf[-1]))
        if bT:
            axes[0].plot(bT, be, "-.", color=RED, lw=1.8, marker="+", ms=7,
                         label="1BQF (wp99)")
            axes[1].plot(bT, np.maximum(bf, 2e-4), "-.", color=RED, lw=1.8,
                         marker="+", ms=7, label="1BQF (wp99)")
        # fitted d=40 per set-out at wp99
        for so, (col, ls, mk, lab) in SETOUT_STYLE.items():
            xe, xf, xT = [], [], []
            for T in Ts:
                g = sub[(sub.setout == so) & (sub["T"] == T)
                        & (sub.family == "fitted_moment")
                        & (sub.degree == D_HEAD)]
                if len(g):
                    xT.append(T); xe.append(med(g, "eff_e990"))
                    xf.append(med(g, "far_e990"))
                    rows.append(dict(regime=regime, T=T, setout=so,
                                     series="fitted_d40_wp99",
                                     eff=xe[-1], far=xf[-1]))
            if xT:
                axes[0].plot(xT, xe, ls, color=col, lw=2.0, marker=mk, ms=5,
                             label=lab)
                axes[1].plot(xT, np.maximum(xf, 2e-4), ls, color=col, lw=2.0,
                             marker=mk, ms=5, label=lab)
        axes[0].set_ylabel("segment efficiency")
        axes[0].set_ylim(0.90, 1.005)
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
                     "(fitted response, $d{=}40$; quantum at wp99, classical "
                     "at fixed $\\tau$)", fontsize=12)
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"xiv_metrics_vs_T_{regime}.{ext}", dpi=150,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] figures/xiv_metrics_vs_T_{regime}.png "
              f"(T = {Ts})")
    pd.DataFrame(rows).to_csv(OUT / "08_writeup_numbers.csv", index=False)


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
    best = g.loc[g.groupby(["regime", "setout"]).eff.idxmax()]
    best = best[best.regime.isin(["moderate", "heavy"])]
    best = best.sort_values(["regime", "eff"])
    lab = [f"{r.regime[:3]} · {r.setout}" for _, r in best.iterrows()]
    y = np.arange(len(best))
    left = np.zeros(len(best))
    for ch, col, name in [("L1", INK, "L1 twins (irreducible)"),
                          ("L2", BLUE, "L2 ripple"),
                          ("L3", ORANGE, "L3 threshold overlap")]:
        axes[0].barh(y, best[ch], left=left, color=col, height=0.7, label=name)
        left += best[ch].values
    axes[0].set_yticks(y); axes[0].set_yticklabels(lab, fontsize=8.5)
    axes[0].set_xlabel("efficiency loss at far $\\leq$ 1 %")
    axes[0].set_title("(a) where the lost efficiency lives (fitted response)",
                      loc="left")
    axes[0].legend(loc="lower right", fontsize=8)
    tw = (f4.groupby(["regime", "setout"]).twin_frac.median().reset_index())
    tw = tw[tw.regime.isin(["moderate", "heavy"])].sort_values("twin_frac")
    axes[1].barh(np.arange(len(tw)), tw.twin_frac, color=GREY, height=0.65)
    axes[1].set_yticks(np.arange(len(tw)))
    axes[1].set_yticklabels([f"{r.regime[:3]} · {r.setout}"
                             for _, r in tw.iterrows()], fontsize=8.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("twin fraction of true segments")
    axes[1].set_title("(b) the operator sets the twin population", loc="left")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5, axis="x")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"xiv_loss_budget.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/xiv_loss_budget.png")


if __name__ == "__main__":
    df = load_frames()
    fig_metrics_vs_T(df)
    fig_loss_budget()
