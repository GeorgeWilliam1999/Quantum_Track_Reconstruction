#!/usr/bin/env python3
"""Figure-6 companions: the clean benchmark read at ONE convention per figure.

Figure 6 of the paper mixes two threshold conventions between its curves
(line comb at fixed tau = 0.35, 1BQF efficiency-first).  These two figures
separate them, each on the same fixed-epsilon = 2 mrad campaign events:

  A  figures/xiv_fig6_fixed_tau   every solver at the FIXED threshold
       tau = attractor + 0.10 = 0.35: classical, line comb, 1BQF
       (all three from the seg_store campaign aggregation the paper uses;
       the 1BQF at fixed tau sits near 75% efficiency - below the zoomed
       axis, annotated - which is why figure 6 reads it efficiency-first).
  B  figures/xiv_fig6_matched     every solver at the MATCHED 99%-efficiency
       working point: classical, line comb, 1BQF (store wp columns) + the
       fitted d=40 response on base A / +occupancy / +bifurcation / +both
       (outputs/12_fig6_overlay.csv: same events, reps 0-2).

The fitted response is a ridge fit toward the classical solution on true
segments and zero on false ones, with the ridge weight chosen by a matched
criterion - it has no absolute activation scale, so it is read ONLY at the
matched working point and never appears in figure A.

Anchor (printed): the driver's base-A classical rows reproduce the store's
classical rows rep-for-rep (reps 0-2 at every T) at both conventions.

Outputs: the two figures (png+pdf) + outputs/13_fig6_companion_numbers.csv
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

# store solvers: (store key, colour, marker, label stem)
STORE = [("classical", INK,   "x", "classical solve"),
         ("qsvt",      BROWN, "D", "line comb (production QSVT)"),
         ("quantum",   RED,   "*", "1BQF")]
# driver operators (fitted d=40, matched 99% eff.)
OPS = {
    "base":            (BLUE,   "o", "QSVT fitted $d{=}40$, base $A$"),
    "occ_a0.05":       (GREEN,  "s", "QSVT fitted $d{=}40$, $A+$occupancy $\\alpha{=}0.05$"),
    "fork_b0.5":       (ORANGE, "D", "QSVT fitted $d{=}40$, $A+$bifurcation $\\beta{=}0.5$"),
    "occ0.05_fork0.5": (PURPLE, "^", "QSVT fitted $d{=}40$, $A+$both terms"),
}
CONFIG_TXT = ("one configuration throughout: $\\sigma_{\\rm scatt}{=}10^{-4}$, "
              "$\\sigma_{\\rm res}{=}0$, no hit drop, $\\varphi_{\\max}{=}0.2$, "
              "FIXED $\\varepsilon{=}2$ mrad, $\\gamma{=}3$, $\\delta{=}1$, step kernel")
REPS_TXT = ("campaign mean$\\pm$sem over reps (classical 5–20, line comb 5–10, "
            "1BQF 3–10 per $T$)")


def store_series(M, key, wp):
    d = Sst.agg_by_ntrk(M, key, 3.0, 0.0)
    if not d:
        return None
    if wp:
        if "se_wp_m" not in d:
            return None
        return dict(T=np.asarray(d["tc"], float), eff=np.asarray(d["se_wp_m"]),
                    eff_e=np.asarray(d["se_wp_e"]), far=np.asarray(d["fr_wp_m"]),
                    far_e=np.asarray(d["fr_wp_e"]))
    return dict(T=np.asarray(d["tc"], float), eff=np.asarray(d["se_m"]),
                eff_e=np.asarray(d["se_e"]), far=np.asarray(d["fr_m"]),
                far_e=np.asarray(d["fr_e"]))


def driver_series(df, setout, family):
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


def draw(axes, s, color, mk, ls, label, lw=1.7, alpha=0.9, rows=None,
         convention="", source=""):
    ms = 8 if mk == "*" else 5
    axes[0].errorbar(s["T"], s["eff"], yerr=s["eff_e"], fmt=mk + ls,
                     color=color, capsize=2.5, ms=ms, lw=lw, alpha=alpha,
                     label=label)
    axes[1].errorbar(s["T"], np.maximum(s["far"], FAR_FLOOR), yerr=s["far_e"],
                     fmt=mk + ls, color=color, capsize=2.5, ms=ms, lw=lw,
                     alpha=alpha, label=label)
    if rows is not None:
        for t, e_, f_ in zip(s["T"], s["eff"], s["far"]):
            rows.append(dict(convention=convention, series=label,
                             source=source, T=t, eff_pct=e_, far_pct=f_))


def finish(fig, axes, title, footer, fname, legend=(1, "upper left")):
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
    axes[legend[0]].legend(loc=legend[1], ncol=1, framealpha=0.92)
    fig.suptitle(title, fontsize=12.2)
    fig.text(0.5, -0.035, footer, ha="center", va="top", fontsize=8.0,
             color="#55534d", wrap=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] figures/{fname}.png/.pdf")


def fig_fixed_tau(M, rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)
    offscale = []
    for key, col, mk, stem in STORE:
        s = store_series(M, key, wp=False)
        if s is None:
            continue
        lab = f"{stem} (fixed $\\tau{{=}}0.35$)"
        draw(axes, s, col, mk, "--", lab, rows=rows,
             convention="fixed tau 0.35", source="seg_store fixed-eps campaign")
        if np.nanmax(s["eff"]) < EFF_LIM[0]:
            offscale.append((stem, float(np.nanmin(s["eff"])),
                             float(np.nanmax(s["eff"]))))
    for stem, lo, hi in offscale:
        axes[0].annotate(
            f"{stem} at fixed $\\tau$: {lo:.0f}–{hi:.0f}% at every $T$ "
            "(below this axis — its cosine response leaves a quarter of the\n"
            "true segments under $\\tau$; this is why figure 6 reads it "
            "efficiency-first)",
            xy=(30, EFF_LIM[0] + 0.15), xytext=(11, 92.4), fontsize=8.0,
            color=RED, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    finish(fig, axes,
           "The clean benchmark (fixed $\\varepsilon{=}2$ mrad) — figure 6's "
           "solvers at ONE fixed threshold $\\tau{=}0.35$",
           CONFIG_TXT + " · " + REPS_TXT + " · $\\tau{=}0.35$ = attractor "
           "$\\delta/(\\gamma{+}\\delta)$ + 0.10 applied to every solver; "
           "1BQF activations rescaled to the classical signal-support norm "
           "(campaign convention) · a measured 0% false rate is drawn at the "
           "$10^{-3}$ floor · no fitted-response series: the fitted "
           "polynomial has no absolute scale and is read only at a matched "
           "working point (companion figure)",
           "xiv_fig6_fixed_tau")


def fig_matched(M, df, rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)
    for key, col, mk, stem in STORE:
        s = store_series(M, key, wp=True)
        if s is None:
            continue
        draw(axes, s, col, mk, "--", f"{stem} (matched 99% eff.)", rows=rows,
             alpha=0.75, convention="matched 99% efficiency",
             source="seg_store fixed-eps campaign (wp columns)")
    for so, (col, mk, lab) in OPS.items():
        s = driver_series(df, so, "fitted_moment")
        draw(axes, s, col, mk, "-", f"{lab} (matched 99% eff.)", lw=2.0,
             rows=rows, convention="matched 99% efficiency",
             source="12_fig6_overlay.csv (reps 0-2)")
    axes[0].axhline(99, color=GREY, lw=0.7, ls=":")
    axes[0].text(10.5, 99.08, "99% target", fontsize=8, color="#55534d")
    finish(fig, axes,
           "The clean benchmark (fixed $\\varepsilon{=}2$ mrad) — every "
           "solver at the matched 99%-efficiency working point",
           CONFIG_TXT + " · store curves: " + REPS_TXT + " · fitted series: "
           "the same campaign events, reps 0–2, mean$\\pm$sem, response "
           "refit per operator (moment-space ridge, $d{=}40$) · the "
           "threshold of every curve is set per event so that 99% of the "
           "true segments survive; panel (a) shows the convention holding, "
           "panel (b) carries the comparison · a measured 0% false rate is "
           "drawn at the $10^{-3}$ floor",
           "xiv_fig6_matched", legend=(0, "lower left"))


def print_anchor(M, df):
    cc = df[(df.family == "classical_invA") & (df.setout == "base")]
    Ms = M[(M.solver == "classical") & (M.gamma == 3.0) & (M.hit_ineff == 0.0)]
    print("== anchor: driver base-A classical vs store classical, rep-for-rep ==")
    worst = 0.0
    for T in sorted(cc["T"].unique()):
        for r in sorted(cc[cc["T"] == T].rep.unique()):
            d = cc[(cc["T"] == T) & (cc.rep == r)].iloc[0]
            s = Ms[(Ms.n_trk == T) & (Ms.rep == r)]
            if not len(s):
                print(f"   T={T:5d} rep{r}: no store row");
                continue
            s = s.iloc[0]
            dv = [100 * d.far_fixed_tau - 100 * s.segment_false_rate,
                  100 * d.far_e990 - 100 * s.segment_false_rate_wp]
            worst = max(worst, *map(abs, dv))
            print(f"   T={T:5d} rep{r}: fixed-tau far {100*d.far_fixed_tau:7.3f} "
                  f"vs {100*s.segment_false_rate:7.3f} | matched far "
                  f"{100*d.far_e990:7.3f} vs {100*s.segment_false_rate_wp:7.3f}")
    print(f"   worst |difference| = {worst:.4f} pp")


def main():
    M = Sst.fixed_eps_metrics()
    M = M[M["n_trk"] <= 1000]
    df = pd.read_csv(OUT / "12_fig6_overlay.csv")
    rows = []
    fig_fixed_tau(M, rows)
    fig_matched(M, df, rows)
    print_anchor(M, df)
    pd.DataFrame(rows).to_csv(OUT / "13_fig6_companion_numbers.csv", index=False)
    print("[saved] outputs/13_fig6_companion_numbers.csv")


if __name__ == "__main__":
    main()
