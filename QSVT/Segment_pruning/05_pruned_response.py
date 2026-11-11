#!/usr/bin/env python3
"""Re-tune the filters for the pruned spectrum (paper corpus, T = 400 and 1000).

After the decoupled prune the isolated line at lambda = s is gone (only the
middle mode of three-segment chains remains there), so

  * the one-bit filter's notch at s (t = pi / s) does nothing useful any more:
    scan the evolution time t and read the exact one-bit readout
    p_i = [cos^2(At/2) u]_i^2 + [sin(At) u / 2]_i^2 on the pruned systems;
  * the comb no longer has to hold a 1e-3 null against a line that outnumbers
    the tracks 400 to 1: scan the degree for the current comb and for a
    design whose fit weight sits on the surviving false lines (the pairs at
    s -/+ 1, the chain modes at s -/+ sqrt 2 and s) instead of uniformly.

Everything is exact (per-component eigendecomposition from
04_activation_spectrum.event_data), on the two pruned variants: decoupled
segments dropped (deg1) and components under three dropped (comp3).  Metrics
in the full index space with the study convention (common.metrics).

Outputs: outputs/pruned_onebit_tscan.csv, outputs/pruned_comb_scan.csv,
figures fig9_onebit_tscan, fig10_comb_degree_scan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as _cheb

from common import COMB_DEGREE, FIG, OUT, S, T_ONEBIT, metrics
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse
import importlib
act = importlib.import_module("04_activation_spectrum")

TS = [400, 1000]
REPS = range(10)
T_GRID = np.round(np.r_[np.linspace(0.05, 0.30, 6)[:-1], np.linspace(0.30, 1.60, 27)], 4)
DEGREES = [8, 12, 16, 20, 24, 30, 40]
LINES = tuple(S - 2 * np.cos(k * np.pi / 5) for k in (1, 2, 3, 4))
FALSE_LINES = (S - 1, S + 1, S - np.sqrt(2), S + np.sqrt(2), S)
C_VAR = {"full": "#7F7F7F", "deg1": "#D55E00", "comp3": "#8B0000"}
L_VAR = {"full": "full matrix", "deg1": "decoupled dropped", "comp3": "components < 3 dropped"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})


def design_pruned_comb(degree, domain, hw=0.18, notch_w=30.0, notch_hw=0.06, ngrid=6000):
    """Line comb fitted with extra weight on the surviving false lines."""
    lo, hi = domain
    x = np.linspace(lo, hi, ngrid)
    y = np.zeros_like(x)
    for l in LINES:
        y += np.exp(-((x - l) / hw) ** 2)
    y = y / x
    w = np.ones_like(x)
    for f in FALSE_LINES:
        w += notch_w * np.exp(-((x - f) / notch_hw) ** 2)
    p = _cheb.Chebyshev.fit(x, y, degree, domain=list(domain), w=w)
    mx = float(np.max(np.abs(p(x))))
    return p / (mx / 0.95) if mx > 1 else p


def apply_response(d, f, variant):
    """Amplitudes x = |f(A) b| on the pruned system, scattered to full space.
    ``variant == "full"`` adds every decoupled segment at its exact value
    |f(s)| (a 1 x 1 block), which is the full-matrix solve."""
    n_full = len(d["truth"])
    x = np.zeros(n_full)
    if variant == "full":
        x[~d["coupled"]] = abs(float(f(np.array([S]))[0]))
    keep = d["comp3_kept"] if variant == "comp3" else None
    for idx, w, V, proj in d["groups"]:
        if keep is not None and not keep[idx[0]]:
            continue
        x[d["kept"][idx]] = np.abs(V @ (f(w) * proj))
    return x


def onebit_amplitudes(d, t, variant):
    n_full = len(d["truth"])
    x = np.zeros(n_full)
    if variant == "full":
        x[~d["coupled"]] = np.sqrt(np.cos(S * t / 2) ** 4 + 0.25 * np.sin(S * t) ** 2)
    keep = d["comp3_kept"] if variant == "comp3" else None
    for idx, w, V, proj in d["groups"]:
        if keep is not None and not keep[idx[0]]:
            continue
        yc = V @ (np.cos(w * t / 2) ** 2 * proj)
        ys = V @ (np.sin(w * t) * proj)
        x[d["kept"][idx]] = np.sqrt(yc ** 2 + (0.5 * ys) ** 2)
    return x


def main():
    rows_t, rows_c = [], []
    for T in TS:
        ds = [act.event_data(T, rep) for rep in REPS]
        bounds = ds[0]["bounds"]
        lam_lo, lam_hi = bounds
        span = lam_hi - lam_lo
        dom = (min(lam_lo - 0.02 * span, S - 3.8), max(lam_hi + 0.02 * span, S + 3.8))
        for rep, d in zip(REPS, ds):
            truth, coupled = d["truth"], d["coupled"]
            for variant in ("full", "deg1", "comp3"):
                wp_mask = d["comp3"] if variant == "comp3" else coupled
                for t in T_GRID:
                    m = metrics(onebit_amplitudes(d, float(t), variant), truth, wp_mask)
                    rows_t.append(dict(T=T, rep=rep, variant=variant, t=float(t), **m))
                for deg in DEGREES:
                    for design, poly in (("current comb", design_line_comb_inverse(degree=deg, s=S, domain=dom)),
                                         ("comb weighted on surviving false lines",
                                          design_pruned_comb(deg, dom))):
                        m = metrics(apply_response(d, poly, variant), truth, wp_mask)
                        rows_c.append(dict(T=T, rep=rep, variant=variant, degree=deg, design=design, **m))
            print(f"T={T} rep={rep} done", flush=True)
    dt = pd.DataFrame(rows_t)
    dc = pd.DataFrame(rows_c)
    dt.to_csv(OUT / "pruned_onebit_tscan.csv", index=False)
    dc.to_csv(OUT / "pruned_comb_scan.csv", index=False)

    # fig9: one-bit t scan
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), sharex=True)
    for j, T in enumerate(TS):
        for variant in ("full", "deg1", "comp3"):
            g = dt[(dt["T"] == T) & (dt.variant == variant)].groupby("t")
            for i, col in enumerate(["eff_f010", "far_e990"]):
                m = g[col].median()
                axes[i, j].plot(m.index, m.values, color=C_VAR[variant], marker="o", ms=2.5,
                                lw=3 if variant == "full" else 1.4, alpha=0.5 if variant == "full" else 1,
                                label=L_VAR[variant])
        for i in range(2):
            axes[i, j].axvline(T_ONEBIT, color="#555555", ls=":", lw=0.8)
            axes[i, j].text(T_ONEBIT, 0.02, " t = pi/s (notch at s)", transform=axes[i, j].get_xaxis_transform(),
                            fontsize=6.5, color="#555555")
        axes[0, j].set_title(f"T = {T}", fontsize=9)
        axes[1, j].set_xlabel("evolution time t")
        axes[1, j].set_yscale("log")
    axes[0, 0].set_ylabel("efficiency at false rate <= 1 %")
    axes[1, 0].set_ylabel("false rate at 99 % efficiency")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("one-bit filter, exact readout, on the pruned systems", fontsize=9)
    fig.savefig(FIG / "fig9_onebit_tscan.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig9_onebit_tscan.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

    # fig10: comb degree scan
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), sharex=True)
    LS = {"current comb": "-", "comb weighted on surviving false lines": "--"}
    for j, T in enumerate(TS):
        for variant in ("full", "deg1", "comp3"):
            for design in LS:
                g = dc[(dc["T"] == T) & (dc.variant == variant) & (dc.design == design)].groupby("degree")
                for i, col in enumerate(["eff_f010", "far_e990"]):
                    m = g[col].median()
                    axes[i, j].plot(m.index, m.values, LS[design], color=C_VAR[variant], marker="o", ms=3,
                                    lw=3 if variant == "full" else 1.4, alpha=0.5 if variant == "full" else 1,
                                    label=f"{design}, {L_VAR[variant]}")
        axes[0, j].set_title(f"T = {T}", fontsize=9)
        axes[1, j].set_xlabel("polynomial degree")
        axes[1, j].set_yscale("log")
    axes[0, 0].set_ylabel("efficiency at false rate <= 1 %")
    axes[1, 0].set_ylabel("false rate at 99 % efficiency")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("QSVT comb on the pruned systems", fontsize=9)
    fig.savefig(FIG / "fig10_comb_degree_scan.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig10_comb_degree_scan.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

    # summary tables
    print("\n== one-bit (median over reps): at t = pi/s; at the t with the highest eff@1%; "
          "at the t with the lowest far99 among t with eff@1% >= 0.95")
    g = dt.groupby(["T", "variant", "t"])[["eff_f010", "far_e990"]].median().reset_index()
    for (T, v), gg in g.groupby(["T", "variant"]):
        ref = gg.iloc[(gg.t - T_ONEBIT).abs().argmin()]
        be = gg.iloc[gg.eff_f010.argmax()]
        ok = gg[gg.eff_f010 >= 0.95]
        bf = ok.iloc[ok.far_e990.argmin()] if len(ok) else None
        line = (f"T={T} {v:5s}: pi/s eff@1%={ref.eff_f010:.3f} far99={ref.far_e990:.3f} | "
                f"max eff@1% at t={be.t:.2f}: {be.eff_f010:.3f} (far99 {be.far_e990:.3f})")
        if bf is not None:
            line += f" | min far99 with eff@1%>=0.95 at t={bf.t:.2f}: far99={bf.far_e990:.3f} eff@1%={bf.eff_f010:.3f}"
        print(line)
    print("\n== comb degree scan (median over reps)")
    g = dc.groupby(["T", "variant", "design", "degree"])[["eff_f010", "far_e990"]].median()
    print(g.to_string())


if __name__ == "__main__":
    main()
