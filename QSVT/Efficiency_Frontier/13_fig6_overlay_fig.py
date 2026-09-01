#!/usr/bin/env python3
"""The figure-6 overlay: the paper's clean-benchmark curves + the
modified-operator QSVT series, every curve's convention named.

Figure-6 curves (from the same seg_store aggregation the paper uses):
  classical, fixed tau = 0.35            (as in figure 6)
  line comb, fixed tau = 0.35            (as in figure 6 - efficiency sags)
  1BQF, matched 99%-efficiency threshold (as in figure 6 - far explodes)
  line comb, matched 99%-efficiency      (ADDED: the comb read at the SAME
                                          convention as the 1BQF curve)
New series (outputs/12_fig6_overlay.csv - same events, same fixed
epsilon = 2 mrad, fitted d=40 refit per operator, matched 99% eff.):
  base A / +occupancy(0.05) / +fork(0.5) / +both

Anchor: the driver's base-A production comb at matched 99% eff. must
reproduce the store's comb wp curve - printed on every run.

Output: figures/xiv_fig6_overlay.png/pdf + outputs/13_fig6_overlay_numbers.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/"
                "Toy_Characterisation/Segment_level_studies")
import seg_store as Sst                                       # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "outputs", HERE / "figures"

GREEN, RED, GREY, BLUE, ORANGE = "#3d8a4f", "#d84a49", "#8f8d86", "#2a78d6", "#e08a2e"
INK, PURPLE = "#33322e", "#7b5ea7"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 8.2, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

OPS = {
    "base":            (BLUE,   "o", "QSVT fitted, base $A$ (matched 99% eff.)"),
    "occ_a0.05":       (GREEN,  "s", "QSVT fitted, $+$occupancy $\\alpha{=}0.05$ (matched 99% eff.)"),
    "fork_b0.5":       (ORANGE, "D", "QSVT fitted, $+$bifurcation $\\beta{=}0.5$ (matched 99% eff.)"),
    "occ0.05_fork0.5": (PURPLE, "^", "QSVT fitted, both terms (matched 99% eff.)"),
}


def main():
    M = Sst.fixed_eps_metrics()
    M = M[M["n_trk"] <= 1000]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)

    # ── figure 6's own curves, conventions named ─────────────────────────
    store = [("classical", "#1b7837", "x", "--",
              "classical (fixed $\\tau{=}0.35$)", False),
             ("qsvt", "#6a3d9a", "D", "--",
              "line comb (fixed $\\tau{=}0.35$)", False),
             ("qsvt", "#6a3d9a", "D", "-",
              "line comb (matched 99% eff.)", True),
             ("quantum", "#d6604d", "*", "-",
              "1BQF (matched 99% eff.)", True)]
    comb_wp = None
    for k, col, mk, ls, lab, use_wp in store:
        d = Sst.agg_by_ntrk(M, k, 3.0, 0.0)
        if not d:
            continue
        if use_wp and "se_wp_m" in d:
            se, see, fr, fre = (d["se_wp_m"], d["se_wp_e"],
                                d["fr_wp_m"], d["fr_wp_e"])
        elif use_wp:
            continue
        else:
            se, see, fr, fre = d["se_m"], d["se_e"], d["fr_m"], d["fr_e"]
        if use_wp and k == "qsvt":
            comb_wp = dict(tc=np.asarray(d["tc"]), fr=np.asarray(fr))
        axes[0].errorbar(d["tc"], se, yerr=see, fmt=mk + ls, color=col,
                         capsize=2.5, ms=7 if mk == "*" else 5, lw=1.6,
                         alpha=0.85 if ls == "-" else 0.45, label=lab)
        axes[1].errorbar(d["tc"], np.maximum(np.asarray(fr, float), 1e-3),
                         yerr=fre, fmt=mk + ls, color=col, capsize=2.5,
                         ms=7 if mk == "*" else 5, lw=1.6,
                         alpha=0.85 if ls == "-" else 0.45, label=lab)
        for t, s_, f_ in zip(d["tc"], se, fr):
            rows.append(dict(series=lab, source="seg_store fixed-eps campaign",
                             T=t, eff_pct=s_, far_pct=f_))

    # ── the new modified-operator series ─────────────────────────────────
    df = pd.read_csv(OUT / "12_fig6_overlay.csv")
    f = df[df.family == "fitted_moment"]
    for so, (col, mk, lab) in OPS.items():
        g = f[f.setout == so]
        Ts = sorted(g["T"].unique())
        se_m = [100 * g[g["T"] == t].eff_e990.mean() for t in Ts]
        se_e = [100 * g[g["T"] == t].eff_e990.std() / np.sqrt(max(len(g[g["T"] == t]), 1)) for t in Ts]
        fr_m = [100 * g[g["T"] == t].far_e990.mean() for t in Ts]
        fr_e = [100 * g[g["T"] == t].far_e990.std() / np.sqrt(max(len(g[g["T"] == t]), 1)) for t in Ts]
        axes[0].errorbar(Ts, se_m, yerr=se_e, fmt=mk + "-", color=col,
                         capsize=2.5, ms=5, lw=1.9, label=lab)
        axes[1].errorbar(Ts, np.maximum(np.asarray(fr_m), 1e-3), yerr=fr_e,
                         fmt=mk + "-", color=col, capsize=2.5, ms=5, lw=1.9,
                         label=lab)
        for t, s_, f_ in zip(Ts, se_m, fr_m):
            rows.append(dict(series=lab, source="12_fig6_overlay.csv",
                             T=t, eff_pct=s_, far_pct=f_))

    # ── anchor: driver base classical fixed-tau vs the store classical ──
    # (machinery-independent; validates events + acceptance + metrics. The
    # driver's own comb row is NOT plotted or anchored: at T>=400 the 2 mrad
    # clean events develop tangle structure pushing lam_min below 2, so a
    # comb rebuilt on the measured per-rep domain is a different realization
    # from the store campaign's production comb solver.)
    dc = Sst.agg_by_ntrk(M, "classical", 3.0, 0.0)
    st = dict(zip(dc["tc"], dc["fr_m"]))
    cc = df[(df.family == "classical_invA") & (df.setout == "base")]
    print("== anchor: base classical fixed-tau far%, driver vs store ==")
    for t in sorted(cc["T"].unique()):
        print(f"   T={t:5d}: ours {100*cc[cc['T']==t].far_fixed_tau.mean():7.3f}%"
              f"  store {st.get(t, float('nan')):7.3f}%")

    axes[0].set_ylabel("segment efficiency  [%]")
    axes[0].set_ylim(60, 102)
    axes[0].axhline(100, color=GREY, lw=0.7, ls="--")
    axes[0].set_title("(a) segment efficiency", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("segment false rate  [%]")
    axes[1].set_ylim(1e-3, 120)
    axes[1].set_title("(b) segment false rate", loc="left")
    for ax in axes:
        ax.set_xlabel("tracks per event $T$")
        ax.set_xscale("log")
        ax.grid(alpha=0.25, lw=0.5)
    axes[1].legend(loc="lower right", ncol=1, framealpha=0.92)
    fig.suptitle("The clean benchmark (fixed $\\varepsilon{=}2$ mrad) — "
                 "figure 6's curves with the modified-operator QSVT series",
                 fontsize=12.2)
    fig.text(0.5, -0.035,
             "one configuration throughout: $\\sigma_{\\rm scatt}{=}10^{-4}$, "
             "$\\sigma_{\\rm res}{=}0$, no hit drop, FIXED "
             "$\\varepsilon{=}2$ mrad, $\\gamma{=}3$, $\\delta{=}1$, step "
             "kernel · store curves: 3-rep mean$\\pm$sem, campaign events · "
             "new series: same events, fitted $d{=}40$ refit per operator, "
             "3-rep mean$\\pm$sem · each curve's threshold convention is in "
             "its legend entry — fixed $\\tau$ and matched-efficiency curves "
             "are NOT comparable to each other", ha="center", va="top",
             fontsize=8.0, color="#55534d")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"xiv_fig6_overlay.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "13_fig6_overlay_numbers.csv", index=False)
    print("[saved] figures/xiv_fig6_overlay.png + outputs/13_fig6_overlay_numbers.csv")


if __name__ == "__main__":
    main()
