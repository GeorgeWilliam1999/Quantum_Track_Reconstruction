#!/usr/bin/env python3
"""
Standard segment efficiency / false rate vs track count T, from the qtrk
store metrics view (formula epsilon, absolute tau = 0.35, quantum rescaled
on the classical signal support).

Two family views, matching the epsilon-sensitivity scan's fixed points:

  eff_fr_vs_T_sres_family.png   : curves per sigma_res   at sigma_scatt = 1e-4
  eff_fr_vs_T_sscatt_family.png : curves per sigma_scatt at sigma_res   = 0.01

Classical = 20 reps everywhere; quantum (1BQF statevector) plotted where
>= 2 reps exist in the store (full to T = 200, partial at T = 400).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures" / "epsilon_sensitivity"
FIGDIR.mkdir(parents=True, exist_ok=True)

SR_FIX, SS_FIX = 0.01, 1e-4
CMAP = plt.get_cmap("viridis")

df = pd.read_csv("/data/bfys/gscriven/qtrk_store/manifest/metrics.csv")
df = df[df["study"] == "Epsilon_study_2"]

# HEADLINE = wp99 high-efficiency working point (per-solve tau holds ~99% of true
# segments; segment_*_wp columns from build_metrics.py >= 2026-06-14).  The fixed
# tau=0.35 cut pinned the 1BQF at the ~75% plateau; we report the working point.
WP = "segment_efficiency_wp" in df.columns
EFF = "segment_efficiency_wp" if WP else "segment_efficiency"
FAR = "segment_false_rate_wp" if WP else "segment_false_rate"
TAU_LAB = r"wp99 ($\varepsilon$-eff $\geq 99\%$)" if WP else r"$\tau=0.35$"

for figname, famcol, famvals, sel, fixed_lab in (
    ("eff_fr_vs_T_sres_family", "sigma_res", [0.0, 0.01, 0.02, 0.05],
     df["sigma_scatt"] == SS_FIX,
     rf"$\sigma_{{\rm scatt}}={SS_FIX:g}$ rad fixed"),
    ("eff_fr_vs_T_sscatt_family", "sigma_scatt", [1e-4, 3e-4, 5e-4],
     df["sigma_res"] == SR_FIX,
     rf"$\sigma_{{\rm res}}={SR_FIX}$ mm fixed"),
):
    sub = df[sel]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    cols = [CMAP(x) for x in np.linspace(0.0, 0.85, len(famvals))]
    for v, col in zip(famvals, cols):
        lab = (rf"$\sigma_r={v:g}$ mm" if famcol == "sigma_res"
               else rf"$\sigma_s={v:g}$")
        for ax, ycol in zip(axes, (EFF, FAR)):
            for solver, ls, mk in (("classical", "-", "o"),
                                   ("quantum", "--", "s")):
                g = (sub[(sub[famcol] == v) & (sub["solver"] == solver)]
                     .groupby("n_trk")[ycol].agg(["mean", "sem", "count"]))
                g = g[g["count"] >= 2]
                if g.empty:
                    continue
                ax.errorbar(g.index, g["mean"], yerr=g["sem"], fmt=mk, ls=ls,
                            ms=5, capsize=3, color=col,
                            mfc=(col if solver == "classical" else "none"),
                            label=f"{lab} ({'classical' if solver == 'classical' else '1BQF'})")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"track count $T$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("segment efficiency")
    axes[1].set_ylabel("segment false rate")
    axes[0].set_ylim(-0.03, 1.06)
    axes[1].set_ylim(-0.03, 1.06)
    axes[0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle(rf"Segment metrics vs $T$ — {fixed_lab}  (formula "
                 rf"$\varepsilon$, {TAU_LAB}, 20 classical / 3 quantum "
                 rf"reps, qtrk store)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGDIR / f"{figname}.png", dpi=160)
    plt.close(fig)
    print(f"[ok] {figname}.png")

# console summary for the write-up
for famcol, famvals, sel in (
    ("sigma_res", [0.0, 0.01, 0.02, 0.05], df["sigma_scatt"] == SS_FIX),
    ("sigma_scatt", [1e-4, 3e-4, 5e-4], df["sigma_res"] == SR_FIX),
):
    sub = df[sel]
    summ_cols = ["segment_efficiency", "segment_false_rate"]
    if WP:
        summ_cols += ["segment_efficiency_wp", "segment_false_rate_wp", "tau_wp"]
    for solver in ("classical", "quantum"):
        g = (sub[sub["solver"] == solver]
             .groupby([famcol, "n_trk"])[summ_cols]
             .mean().round(3))
        print(f"\n=== {famcol} family, {solver} "
              f"(fixed tau=0.35 vs wp99) ===")
        print(g.to_string())
