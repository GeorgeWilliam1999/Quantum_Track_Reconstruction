#!/usr/bin/env python3
"""Stage 4 figure — segment-level metrics, QSVT III style.

George (2026-08-26): "We know that for 1BQF, HHL and classical we can get >95%
segment efficiency, but at the cost of an exploding false rate.  The question
this study is looking to answer is: by adjusting the polynomial, or by
adjusting the Hamiltonian (which means refitting the polynomial) with
occupancy and bifurcation terms, is there a noticeable improvement in segment
efficiency?"

So the figure is the two-sided segment-metric pair of QSVT III
(wp1_fixed_tau / wp1_working_points), with the sweep axis replaced by this
experiment's two axes:

  (a) SEGMENT EFFICIENCY at a fixed false-rate budget (far <= 1%)
      -- how much efficiency you can buy at an affordable fake rate;
  (b) SEGMENT FALSE RATE at a fixed efficiency target (eff = 0.99, the
      efficiency-first working point) -- what >99% efficiency actually costs.

Panel (b) is exactly the QSVT III "working points" plot; panel (a) is its
complement, and together they are the answer to the question above.  Classical
(HHL-equivalent inversion) and the 1BQF are drawn as REFERENCE lines, not as
competing series: they are the baseline the question is posed against.

Reads every frontier CSV this experiment has produced; writes
  figures/segment_metrics_{setouts,degrees}.png/.pdf
  outputs/07_segment_metrics.csv   (the numbers behind the figure)
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

FAM_LAB = {"fitted": "fitted response (refit per set-out)",
           "comb_prod": "production comb (fixed, hw 0.18)",
           "normalized_half_comb": "normalized ±½ comb",
           "sharp_comb": "sharp comb (hw 0.06)",
           "band_inverse": "band-limited inverse"}
FAM_COL = {"fitted": "#1baf7a", "comb_prod": "#4a3aa7",
           "normalized_half_comb": "#eb6834", "sharp_comb": "#8a63d2",
           "band_inverse": "#2a78d6"}
QF = ["fitted", "comb_prod", "normalized_half_comb"]
CLS_COL, BQF_COL = "#52514e", "#e34948"

SETOUT_LAB = {
    "base": "base A\n(no extra term)", "occ_a0.05": "occupancy\nα=0.05",
    "occ_a0.10": "occupancy\nα=0.10", "fork_b0.5_mod": "bifurcation\nβ=0.5",
    "fork_b1.0_mod": "bifurcation\nβ=1.0", "fork_b0.5": "bifurcation\nβ=0.5",
    "occ0.05_fork0.5": "occupancy+bifur.\nα=.05, β=.5",
    "erf": "erf kernel", "eps_s2.12": "ε-scale 2.12",
    "eps_s1.5": "ε-scale 1.5", "gamma1": "γ=1", "gamma2": "γ=2"}
ORDER = ["base", "occ_a0.05", "occ_a0.10", "fork_b0.5_mod", "fork_b0.5",
         "fork_b1.0_mod", "occ0.05_fork0.5", "erf", "eps_s2.12", "gamma2",
         "gamma1", "eps_s1.5"]


def load_all():
    frames = []
    specs = [("01_frontier_clean_moderate.csv", "base"),
             ("02_setout_frontier.csv", None), ("02_setout_dstab.csv", None),
             ("02_fork_frontier.csv", None), ("04_heavy_frontier.csv", None),
             ("04_heavy_fork.csv", None), ("04_heavy_base_ref.csv", None),
             ("03_highT_frontier.csv", None)]
    for fn, default_setout in specs:
        p = OUT / fn
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "setout" not in d.columns:
            d["setout"] = default_setout
        d["src"] = fn
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["family"] = df["family"].replace({"fitted_moment": "fitted",
                                         "fitted_ridge": "fitted"})
    if "T" not in df.columns:
        df["T"] = 200
    return df


def refs(regime, T=200):
    """Matched references: classical inversion and the 1BQF, both on the BASE
    Hamiltonian, at the SAME T, from the same events and machinery.

    NB (2026-08-26 fix): an earlier version pooled classical across every
    set-out (so the fork's rescue of classical leaked into the baseline) and
    took the 1BQF from the high-T store rows (T=400-1000) while comparing
    against QSVT at T=200.  Both made the baseline look worse than it is.
    """
    out = {}
    df = load_all()
    base = df[(df.setout == "base") & (df.regime == regime) & (df["T"] == T)]
    for key, fam in [("classical", "classical_invA"),
                     ("1bqf", "1bqf_cos"), ("1bqf", "onebqf_cos")]:
        g = base[base.family == fam]
        if len(g) and key not in out:
            out[key] = (g.eff_f010.median(), g.far_e990.median())
    return out


EFF_FLOOR = 0.90       # panel (a) view floor; configs below are marked as collapsed


def panel(ax, df, regime, T, col, log=False, floor=None):
    sets = [s for s in ORDER if ((df.setout == s) & (df.regime == regime)
                                 & (df["T"] == T)).any()]
    for k, fam in enumerate(QF):
        xs, ys = [], []
        for i, so in enumerate(sets):
            g = df[(df.setout == so) & (df.regime == regime) & (df["T"] == T)
                   & (df.family == fam)].dropna(subset=[col])
            if not len(g):
                continue
            # best degree by this panel's own metric, then that degree's reps
            med = g.groupby("degree")[col].median()
            bd = med.idxmax() if col == "eff_f010" else med.idxmin()
            v = g[g.degree == bd][col].values
            off = (k - 1) * 0.24
            vis = v if floor is None else v[v >= floor]
            ax.plot(np.full(len(vis), i + off), vis, ".", ms=4, alpha=0.4,
                    color=FAM_COL[fam], zorder=3)
            m = np.median(v)
            if floor is not None and m < floor:
                # collapsed: mark at the view floor rather than squashing the scale
                yv = floor + 0.004 + 0.011 * k
                ax.plot([i + off], [yv], "v", ms=8, color=FAM_COL[fam],
                        zorder=5, clip_on=False,
                        label=FAM_LAB[fam] if i == 0 else None)
                ax.annotate(f"{m:.2f}", (i + off + 0.16, yv), va="center",
                            fontsize=6.5, color=FAM_COL[fam])
                continue
            ax.plot([i + off - 0.1, i + off + 0.1], [m, m], "-", lw=2.6,
                    color=FAM_COL[fam], zorder=4,
                    label=FAM_LAB[fam] if i == 0 else None)
            xs.append(i + off); ys.append(m)
        # NB no connector line: the x axis is categorical, not a sweep
    r = refs(regime, T)
    off_view = []
    for key, cc, ls in [("classical", CLS_COL, "--"), ("1bqf", BQF_COL, "-.")]:
        if key not in r:
            continue
        val = r[key][0 if col == "eff_f010" else 1]
        lab = "classical inversion" if key == "classical" else "1BQF (1-bit notch)"
        if not np.isfinite(val):
            continue
        if floor is not None and val < floor:
            off_view.append((lab, val, cc))          # e.g. classical 0.50 at far<=1%
            continue
        ax.axhline(val, color=cc, ls=ls, lw=1.4, zorder=1, label=lab)
    if off_view:
        txt = "\n".join(f"{lab}: {v:.2f}  (below view)" for lab, v, _ in off_view)
        ax.text(0.985, 0.035, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=CLS_COL,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#b7b3a8",
                          lw=0.8, alpha=0.95))
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([SETOUT_LAB.get(s, s).replace("\n", " ") for s in sets],
                       fontsize=8, rotation=28, ha="right")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    if log:
        ax.set_yscale("log")
    return sets


def main():
    df = load_all()
    rows = []
    for regime, T in [("moderate", 200), ("heavy", 200)]:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
        sets = panel(axes[0], df, regime, T, "eff_f010", floor=EFF_FLOOR)
        axes[0].axhline(0.95, color="#b7b3a8", lw=1, zorder=0)
        axes[0].set_ylim(EFF_FLOOR, 1.005)
        axes[0].set_ylabel("segment efficiency")
        axes[0].set_title("(a) efficiency you can buy at far ≤ 1 %   "
                          "(▼ = collapsed below view)", loc="left", fontsize=11)
        axes[0].legend(fontsize=7.5, loc="lower left", ncol=1, framealpha=0.95)
        panel(axes[1], df, regime, T, "far_e990", log=True)
        axes[1].set_ylim(1e-4, 1.5)
        axes[1].set_ylabel("segment false rate")
        axes[1].set_title("(b) what ≥99 % efficiency costs in fakes",
                          loc="left", fontsize=11)
        axes[1].legend(fontsize=7.5, loc="lower left", ncol=2)
        fig.suptitle(f"Segment-level metrics vs Hamiltonian set-out — "
                     f"{regime} noise, T={T}, 10 reps "
                     f"(dots = reps, bar = median; best degree per family)",
                     fontsize=12)
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"segment_metrics_setouts_{regime}.{ext}",
                        dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] figures/segment_metrics_setouts_{regime}.png")

        for so in ORDER:
            for fam in QF:
                g = df[(df.setout == so) & (df.regime == regime)
                       & (df["T"] == T) & (df.family == fam)]
                if not len(g):
                    continue
                me = g.groupby("degree").eff_f010.median()
                mf = g.groupby("degree").far_e990.median()
                rows.append(dict(regime=regime, T=T, setout=so, family=fam,
                                 best_d_eff=me.idxmax(),
                                 eff_at_far1pct=me.max(),
                                 best_d_far=mf.idxmin(),
                                 far_at_eff99=mf.min(), n_reps=g.rep.nunique()))
        r = refs(regime, T)
        for k, (e, f) in r.items():
            rows.append(dict(regime=regime, T=T, setout="(reference)", family=k,
                             best_d_eff=np.nan, eff_at_far1pct=e,
                             best_d_far=np.nan, far_at_eff99=f, n_reps=np.nan))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "07_segment_metrics.csv", index=False)
    print(f"\n{len(tab)} rows -> outputs/07_segment_metrics.csv\n")
    for regime in ("moderate", "heavy"):
        t = tab[tab.regime == regime].sort_values("eff_at_far1pct",
                                                  ascending=False)
        print(f"== {regime} T=200: segment efficiency @ far<=1% "
              f"| false rate @ eff 0.99 ==")
        for _, r in t.iterrows():
            print(f"  {r.setout:16s} {r.family:22s} "
                  f"eff {r.eff_at_far1pct:.4f} (d={r.best_d_eff}) | "
                  f"far {r.far_at_eff99:.4f} (d={r.best_d_far})")
        print()


if __name__ == "__main__":
    main()
