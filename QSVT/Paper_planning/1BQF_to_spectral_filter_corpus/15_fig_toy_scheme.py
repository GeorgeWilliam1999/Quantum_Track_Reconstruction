#!/usr/bin/env python3
"""Paper 1, schematic figure (asked for by M. Merk, Oct 2026): candidate segments in a small event.

(a) three straight tracks from one vertex through five planes; every pair of hits on adjacent
    planes is a candidate segment; the true segments are active (S_i = 1, green), every other
    candidate is inactive (S_j = 0, grey)
(b) two candidates that share one hit: the kink angle theta_ij between them, and the window
    theta_max that decides compatibility (window drawn much wider than the real 1.7 mrad)

This is a schematic: the geometry (plane spacing 33 mm, cone 0.2 rad) is the toy detector's,
the three slopes are typed in, and no noise is applied. Nothing in it is a result.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
import numpy as np
from matplotlib.patches import Arc, Polygon

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures" / "outputs_5um"
FIG.mkdir(parents=True, exist_ok=True)
GREEN, RED, GREY = "#2e7d32", "#c62828", "#9e9e9e"

DZ = 33.0
Z = DZ * np.arange(1, 6)                      # five planes
SLOPES = np.array([0.16, 0.03, -0.11])        # x per z, inside the 0.2 rad cone
HITS = np.outer(SLOPES, Z)                    # HITS[track, plane] = x


def panel_event(ax):
    for z in Z:
        ax.axvline(z, color="0.88", lw=7, zorder=0)
    ax.plot(0, 0, "k*", ms=7, zorder=3)
    ax.text(0, -4.5, "vertex", ha="center", va="top", fontsize=7.5)
    n_true = n_false = 0
    for k in range(4):                        # gap between plane k and k+1
        for i in range(3):
            for j in range(3):
                z0, z1, x0, x1 = Z[k], Z[k + 1], HITS[i, k], HITS[j, k + 1]
                if i == j:
                    ax.plot([z0, z1], [x0, x1], color=GREEN, lw=1.8, zorder=2)
                    n_true += 1
                else:
                    ax.plot([z0, z1], [x0, x1], color=GREY, lw=0.7, ls="--", zorder=1)
                    n_false += 1
    for i in range(3):
        ax.plot([0, Z[0]], [0, HITS[i, 0]], color=GREEN, lw=0.8, ls=":", zorder=1)
    ax.plot(np.tile(Z, 3), HITS.ravel(), "o", ms=3.4, color="k", zorder=4)
    # labels: one active, one inactive, one isolated false segment
    zm, xm = 0.5 * (Z[1] + Z[2]), 0.5 * (HITS[0, 1] + HITS[0, 2])
    ax.annotate(r"$S_i=1$ (active)", xy=(zm, xm), xytext=(zm - 2, xm + 9), fontsize=8, color=GREEN,
                ha="center", arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7))
    zf, xf = 0.5 * (Z[1] + Z[2]), 0.5 * (HITS[0, 1] + HITS[2, 2])
    ax.annotate(r"$S_j=0$ (inactive)", xy=(zf, xf), xytext=(zf + 2, xf - 12), fontsize=8, color="0.35",
                ha="center", arrowprops=dict(arrowstyle="-", color="0.35", lw=0.7))
    zi = 150.0; xi = HITS[0, 3] + (HITS[1, 4] - HITS[0, 3]) * (zi - Z[3]) / DZ   # on plane-4 track 0 -> plane-5 track 1
    ax.annotate("isolated false\nsegment: no\ncompatible\nneighbour", xy=(zi, xi), xytext=(199, xi),
                fontsize=7.5, color=RED, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    for k, z in enumerate(Z):
        ax.text(z, 31, f"plane {k + 1}", ha="center", va="bottom", fontsize=7.5, color="0.4")
    ax.text(-6, -25, f"3 tracks, {n_true + n_false} candidate segments, {n_true} true",
            fontsize=7.5, va="bottom", ha="left")
    ax.set_xlim(-8, 201); ax.set_ylim(-28, 36)
    ax.set_xlabel("$z$ (mm)"); ax.set_ylabel("$x$ (mm)")
    ax.set_title("(a) candidate segments", loc="left", fontsize=9)


def panel_kink(ax):
    # the hit of track 0 on plane 3; the segment that enters it from plane 2; two candidates that leave it
    z1, z2, z3 = Z[1], Z[2], Z[3]
    xin, xhit = HITS[0, 1], HITS[0, 2]
    x_true, x_false = HITS[0, 3], HITS[1, 3]
    for z in (z1, z2, z3):
        ax.axvline(z, color="0.88", lw=7, zorder=0)
    ax.plot([z1, z2], [xin, xhit], color=GREEN, lw=2.0, zorder=2)
    ax.plot([z2, z3], [xhit, x_true], color=GREEN, lw=2.0, zorder=2)
    ax.plot([z2, z3], [xhit, x_false], color=GREY, lw=1.0, ls="--", zorder=1)
    # extrapolation of the incoming segment and the (exaggerated) window around it
    s = (xhit - xin) / (z2 - z1)
    zx = np.array([z2, z3 + 6]); xx = xhit + s * (zx - z2)
    ax.plot(zx, xx, color="k", lw=0.6, ls=":", zorder=1)
    half = 0.07                                # drawn half-angle; the real theta_max is 1.7 mrad
    L = z3 + 6 - z2
    win = Polygon([[z2, xhit], [z2 + L, xhit + (s + half) * L], [z2 + L, xhit + (s - half) * L]],
                  closed=True, color="#fff3c4", zorder=0)
    ax.add_patch(win)
    ax.text(z3 + 7, xhit + s * L - 1.5, r"window $\theta<\theta_{\max}$" "\n(not to scale)", fontsize=7.5,
            va="top")
    # the kink angle between the extrapolation and the false candidate
    ang_ext = np.degrees(np.arctan(s)); ang_false = np.degrees(np.arctan((x_false - xhit) / (z3 - z2)))
    r = 14
    ax.add_patch(Arc((z2, xhit), 2 * r, 2 * r, angle=0, theta1=ang_false, theta2=ang_ext, color=RED, lw=1))
    am = np.radians(0.5 * (ang_ext + ang_false))
    ax.text(z2 + (r + 3) * np.cos(am), xhit + (r + 3) * np.sin(am), r"$\theta_{ij}$", color=RED, fontsize=9,
            ha="left", va="center")
    for z, x in ((z1, xin), (z2, xhit), (z3, x_true), (z3, x_false)):
        ax.plot(z, x, "o", ms=4, color="k", zorder=4)
    ax.text(z2 - 1.5, xhit + 1.2, "shared hit", ha="right", va="bottom", fontsize=7.5)
    ax.text(z2 + 2, 25.0, "true continuation:\n" r"$\theta<\theta_{\max}$, $f=1$",
            fontsize=7.5, color=GREEN, ha="left", va="bottom")
    ax.text(0.5 * (z2 + z3) + 4, 0.5 * (xhit + x_false) - 1.0, "false candidate:\n" r"$\theta>\theta_{\max}$, $f=0$",
            fontsize=7.5, color="0.35", ha="left", va="top")
    for k, z in zip((2, 3, 4), (z1, z2, z3)):
        ax.text(z, 31.2, f"plane {k}", ha="center", va="bottom", fontsize=7.5, color="0.4")
    ax.set_xlim(z1 - 8, z3 + 30); ax.set_ylim(1, 34)
    ax.set_xlabel("$z$ (mm)"); ax.set_ylabel("$x$ (mm)")
    ax.set_title("(b) two candidates that share a hit", loc="left", fontsize=9)


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
    panel_event(axs[0]); panel_kink(axs[1])
    fig.tight_layout(w_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig0_toy_scheme.{ext}", dpi=200)
    print("wrote", FIG / "fig0_toy_scheme.pdf")
