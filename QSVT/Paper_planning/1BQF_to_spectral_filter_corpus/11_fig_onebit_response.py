#!/usr/bin/env python3
"""Paper 1, Figure 2 companion panel: why the one-bit filter keeps what it keeps.

|cos(lambda t / 2)|, t = pi / s, s = 4, on the eigenvalue axis, with the bright lines of
each configuration marked and the factor the filter keeps printed at each line. Lines are
the exact graph spectra (s - 2cos(k pi/5) for a complete track, bright k = 1, 3; s -+ 1
for a pair; s -+ sqrt 2 for a triple; s -+ sqrt m for a hub with m prongs; s for an
isolated segment). Only bright lines are drawn. Kept factors are computed, not typed.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
S = 4.0
T = np.pi / S


def kept(lam):
    return abs(np.cos(lam * T / 2))


ROWS = [
    ("complete track", [S - 2 * np.cos(np.pi / 5), S - 2 * np.cos(3 * np.pi / 5)], "#2e7d32"),
    ("isolated false segment", [S], "#c62828"),
    ("false pair", [S - 1, S + 1], "#c62828"),
    ("false triple", [S - np.sqrt(2), S + np.sqrt(2)], "#c62828"),
    ("hub with 3 prongs", [S - np.sqrt(3), S + np.sqrt(3)], "#c62828"),
    ("hub with 5 prongs", [S - np.sqrt(5), S + np.sqrt(5)], "#c62828"),
]


def main():
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.0, 1.25]))
    lam = np.linspace(1.4, 6.6, 800)
    ax.plot(lam, kept(lam), color="#1565c0", lw=2)
    ax.axvline(S, color="0.5", lw=0.8, ls=":")
    ax.text(S + 0.05, 0.03, "zero at $\\lambda = \\lambda_c$", fontsize=8, color="0.3")
    ax.set_ylabel("kept factor $|\\cos(\\lambda t/2)|$")
    ax.set_ylim(0, 1.05)
    ax.set_title("the one-bit filter: $t=\\pi/\\lambda_c$, one zero, on the isolated line", fontsize=9, loc="left")
    out = []
    for r, (label, lines, col) in enumerate(ROWS):
        y = len(ROWS) - 1 - r
        for l in lines:
            k = kept(l)
            bx.plot([l, l], [y - 0.32, y + 0.32], color=col, lw=3)
            bx.text(l, y + 0.36, f"{k:.2f}", ha="center", fontsize=7)
            ax.plot([l], [k], "o", color=col, ms=4)
            out.append((label, round(l, 4), round(k, 4)))
    bx.set_yticks(range(len(ROWS)))
    bx.set_yticklabels([r[0] for r in ROWS][::-1], fontsize=8)
    bx.set_ylim(-0.6, len(ROWS) - 0.2)
    bx.set_xlabel("eigenvalue $\\lambda$ of the block ($\\lambda_c = \\alpha + \\beta = 4$)")
    bx.set_title("bright lines of each configuration, with the factor the filter keeps", fontsize=9, loc="left")
    fig.tight_layout()
    d = HERE / "figures"
    d.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"fig2_onebit_response.{ext}", dpi=200)
    with open(d / "fig2_onebit_response_factors.csv", "w") as f:
        f.write("configuration,eigenvalue,kept_factor\n")
        for row in out:
            f.write(",".join(map(str, row)) + "\n")
    print(open(d / "fig2_onebit_response_factors.csv").read())


if __name__ == "__main__":
    main()
