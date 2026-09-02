#!/usr/bin/env python3
"""Figure-6 companion: the clean benchmark, every solver at the efficiency-first
working point.

Convention (George 2026-09-02): the only working point is the lowest threshold
that admits the true solutions — here the threshold just below the 1% quantile
of each solver's true-segment amplitudes, so 99% of true segments survive by
construction and the false rate is the price.  The fixed absolute threshold
tau = attractor + 0.10 (0.35 at gamma = 3) used by the older figure 6 was a bug
and is not drawn anywhere.

Series, all on the same stored fixed-epsilon = 2 mrad clean events:
  classical, QSVT (degree 40, base A, the benchmark campaign's polynomial),
  1BQF  — from the benchmark campaign store (wp columns of seg_store);
  QSVT degree 40 refit on base A / A + occupancy / A + bifurcation / both
        — outputs/12_fig6_overlay.csv (same events, reps 0-2).

The QSVT is one solver; its series are labelled by polynomial degree and by the
Hamiltonian the problem is set up on.  Where two series share degree and
Hamiltonian (the benchmark polynomial and the refit, both degree 40 on base A)
the label states how the coefficients were chosen.

Anchor (printed): the driver's base-A classical rows reproduce the store's
classical rows rep-for-rep (reps 0-2 at every T) at the efficiency-first point.

Outputs: figures/xiv_fig6_efficiency_first.{png,pdf}
         outputs/13_fig6_efficiency_first_numbers.csv
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
INK, PURPLE, BROWN = "#33322e", "#7b5ea7", "#8c6d31"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 8.4, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

FAR_FLOOR = 1e-3          # a measured 0% is drawn at this floor (log axis)
EFF_LIM = (90.0, 100.6)   # George 2026-09-02: zoom to the 90-100% band

# benchmark-campaign solvers: (store key, colour, marker, label)
STORE = [("classical", INK,   "x", "classical"),
         ("qsvt",      BROWN, "D", "QSVT $d{=}40$, base $A$ (benchmark line-comb coefficients)"),
         ("quantum",   RED,   "*", "1BQF")]
# refit QSVT, degree 40, per Hamiltonian
HAMS = {
    "base":            (BLUE,   "o", "QSVT $d{=}40$, base $A$ (coefficients refit)"),
    "occ_a0.05":       (GREEN,  "s", "QSVT $d{=}40$, $A$ + occupancy $\\alpha{=}0.05$"),
    "fork_b0.5":       (ORANGE, "D", "QSVT $d{=}40$, $A$ + bifurcation $\\beta{=}0.5$"),
    "occ0.05_fork0.5": (PURPLE, "^", "QSVT $d{=}40$, $A$ + occupancy + bifurcation"),
}
CONFIG_TXT = ("one configuration throughout: $\\sigma_{\\rm scatt}{=}10^{-4}$, "
              "$\\sigma_{\\rm res}{=}0$, no hit drop, $\\varphi_{\\max}{=}0.2$, "
              "fixed $\\varepsilon{=}2$ mrad, $\\gamma{=}3$, $\\delta{=}1$, step kernel")
REPS_TXT = ("benchmark campaign: mean$\\pm$sem over event repetitions "
            "(classical 5–20, QSVT 5–10, 1BQF 3–10 per $T$)")


def store_series(M, key):
    d = Sst.agg_by_ntrk(M, key, 3.0, 0.0)
    if not d or "se_wp_m" not in d:
        return None
    return dict(T=np.asarray(d["tc"], float), eff=np.asarray(d["se_wp_m"]),
                eff_e=np.asarray(d["se_wp_e"]), far=np.asarray(d["fr_wp_m"]),
                far_e=np.asarray(d["fr_wp_e"]))


def driver_series(df, setout, family="fitted_moment"):
    g = df[(df.family == family) & (df.setout == setout)]
    Ts = np.array(sorted(g["T"].unique()), float)

    def agg(col):
        m = np.array([100 * g[g["T"] == t][col].mean() for t in Ts])
        e = np.array([100 * g[g["T"] == t][col].std(ddof=1)
                      / np.sqrt(len(g[g["T"] == t])) for t in Ts])
        return m, np.nan_to_num(e)
    eff, eff_e = agg("eff_e990")
    far, far_e = agg("far_e990")
    return dict(T=Ts, eff=eff, eff_e=eff_e, far=far, far_e=far_e)


def draw(axes, s, color, mk, ls, label, lw=1.7, alpha=0.9, rows=None, source=""):
    ms = 8 if mk == "*" else 5
    axes[0].errorbar(s["T"], s["eff"], yerr=s["eff_e"], fmt=mk + ls,
                     color=color, capsize=2.5, ms=ms, lw=lw, alpha=alpha,
                     label=label)
    axes[1].errorbar(s["T"], np.maximum(s["far"], FAR_FLOOR), yerr=s["far_e"],
                     fmt=mk + ls, color=color, capsize=2.5, ms=ms, lw=lw,
                     alpha=alpha, label=label)
    if rows is not None:
        for t, e_, f_ in zip(s["T"], s["eff"], s["far"]):
            rows.append(dict(series=label, source=source, T=t,
                             eff_pct=e_, far_pct=f_))


def fig_efficiency_first(M, df, rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)
    for key, col, mk, lab in STORE:
        s = store_series(M, key)
        if s is None:
            continue
        draw(axes, s, col, mk, "--", lab, rows=rows, alpha=0.75,
             source="benchmark campaign (seg_store wp columns)")
    for ham, (col, mk, lab) in HAMS.items():
        s = driver_series(df, ham)
        draw(axes, s, col, mk, "-", lab, lw=2.0, rows=rows,
             source="12_fig6_overlay.csv (same events, reps 0-2)")
    axes[0].axhline(99, color=GREY, lw=0.7, ls=":")
    axes[0].text(10.5, 99.08, "99% of true segments kept", fontsize=8,
                 color="#55534d")
    axes[0].set_ylabel("segment efficiency  [%]")
    axes[0].set_ylim(*EFF_LIM)
    axes[0].axhline(100, color=GREY, lw=0.7, ls="--")
    axes[0].set_title("(a) segment efficiency", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("segment false rate  [%]")
    axes[1].set_ylim(FAR_FLOOR * 0.7, 130)
    axes[1].set_title("(b) segment false rate", loc="left")
    for ax in axes:
        ax.set_xlabel("tracks per event $T$")
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100, 200, 400, 700, 1000])
        ax.set_xticklabels(["10", "20", "50", "100", "200", "400", "700",
                            "1000"])
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(loc="lower left", ncol=1, framealpha=0.92)
    fig.suptitle("The clean benchmark (fixed $\\varepsilon{=}2$ mrad) — every "
                 "solver at its efficiency-first working point", fontsize=12.2)
    fig.text(0.5, -0.035,
             CONFIG_TXT + " · " + REPS_TXT + " · refit series: the same "
             "campaign events, reps 0–2, mean$\\pm$sem, QSVT coefficients "
             "least-squares fit per Hamiltonian (degree 40) · the threshold of "
             "every curve is set per event so that 99% of the true segments "
             "survive; panel (a) shows the convention holding, panel (b) "
             "carries the comparison · a measured 0% false rate is drawn at "
             "the $10^{-3}$ floor",
             ha="center", va="top", fontsize=8.0, color="#55534d", wrap=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"xiv_fig6_efficiency_first.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/xiv_fig6_efficiency_first.png/.pdf")


def print_anchor(M, df):
    cc = df[(df.family == "classical_invA") & (df.setout == "base")]
    Ms = M[(M.solver == "classical") & (M.gamma == 3.0) & (M.hit_ineff == 0.0)]
    print("== anchor: new classical solves vs benchmark campaign, rep-for-rep, "
          "efficiency-first false rate ==")
    worst = 0.0
    for T in sorted(cc["T"].unique()):
        for r in sorted(cc[cc["T"] == T].rep.unique()):
            d = cc[(cc["T"] == T) & (cc.rep == r)].iloc[0]
            s = Ms[(Ms.n_trk == T) & (Ms.rep == r)]
            if not len(s):
                print(f"   T={T:5d} rep{r}: no store row")
                continue
            s = s.iloc[0]
            dv = 100 * d.far_e990 - 100 * s.segment_false_rate_wp
            worst = max(worst, abs(dv))
            print(f"   T={T:5d} rep{r}: {100*d.far_e990:7.3f} vs "
                  f"{100*s.segment_false_rate_wp:7.3f}")
    print(f"   worst |difference| = {worst:.4f} pp")


def main():
    M = Sst.fixed_eps_metrics()
    M = M[M["n_trk"] <= 1000]
    df = pd.read_csv(OUT / "12_fig6_overlay.csv")
    rows = []
    fig_efficiency_first(M, df, rows)
    print_anchor(M, df)
    pd.DataFrame(rows).to_csv(OUT / "13_fig6_efficiency_first_numbers.csv",
                              index=False)
    print("[saved] outputs/13_fig6_efficiency_first_numbers.csv")


if __name__ == "__main__":
    main()
