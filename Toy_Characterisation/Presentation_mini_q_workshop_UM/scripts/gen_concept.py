"""
Conceptual / schematic diagrams for the deck (pure matplotlib, no data):

  pipeline_architecture   toy + qtrk_pipeline data flow (geometry -> ... -> metrics)
  hamiltonian_schematic   hits -> segments -> triplet kink -> A x = b
  hhl_vs_1bqf_circuit     side-by-side circuit structure + cost annotation
  eigenvalue_filter       HHL 1/lambda reweighting vs 1BQF single-bit band filter
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import _common as cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

BLUE, RED, ORANGE, GREEN, GREY, PURPLE = (cm.C["classical"], cm.C["quantum"],
    cm.C["erf"], cm.C["true"], "#777777", cm.C["accent"])


def _box(ax, x, y, w, h, text, fc, ec="k", fs=11, tc="white", style="round"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"{style},pad=0.02,rounding_size=0.04",
                       fc=fc, ec=ec, lw=1.6, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, fontweight="bold")


def _arrow(ax, x0, y0, x1, y1, color="k", lw=2.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                 mutation_scale=16, color=color, lw=lw, zorder=1))


# ---------------------------------------------------------------------------
def fig_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(12.2, 5.4)); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 50)
    # top row: generation
    _box(ax, 2, 38, 17, 8, "Detector geometry\n5 planes, $\\pm$40 mm", BLUE, fs=10)
    _box(ax, 23, 38, 18, 8, "State-vector\nevent generator", BLUE, fs=10)
    _box(ax, 45, 38, 20, 8, "Event\nhits + truth tracks\n(scatter, resolution, drop)", BLUE, fs=9)
    _arrow(ax, 19, 42, 23, 42); _arrow(ax, 41, 42, 45, 42)
    # store
    _box(ax, 69, 38, 27, 8, "qtrk_store\nevents/  (gzip JSON, float64)", GREY, fs=9.5)
    _arrow(ax, 65, 42, 69, 42)
    # middle: hamiltonian
    _box(ax, 45, 24, 20, 8, "Hamiltonian $A,\\mathbf{b}$\n$A_{ij}=-C(\\theta)$, sparse", ORANGE, fs=9.5, tc="k")
    _arrow(ax, 55, 38, 55, 32, color=GREY)
    ax.text(56.5, 35, "regenerate\non demand", fontsize=8, color=GREY, style="italic")
    # solvers
    _box(ax, 16, 11, 22, 9, "Classical solver\n$A^{-1}\\mathbf{b}$  (LU / CG)", BLUE, fs=10)
    _box(ax, 56, 11, 26, 9, "1-Bit Quantum Filter\n(OneBitHHL, statevector/shots)", RED, fs=9.5)
    _arrow(ax, 50, 24, 30, 20, color=BLUE); _arrow(ax, 60, 24, 67, 20, color=RED)
    # outputs
    _box(ax, 16, 1.5, 22, 6.5, "activation $\\mathbf{x}_C$", BLUE, fs=10)
    _box(ax, 56, 1.5, 26, 6.5, "activation $\\mathbf{x}_Q$ + $P_{\\rm anc}$", RED, fs=10)
    _arrow(ax, 27, 11, 27, 8, color=BLUE); _arrow(ax, 69, 11, 69, 8, color=RED)
    # metrics (right)
    _box(ax, 84, 9, 14, 13, "Metrics VIEW\n$x>0.35$\nseg / track\n$\\cos\\theta_{QC}$", GREEN, fs=9)
    _arrow(ax, 82, 4.7, 84, 12, color=GREEN); _arrow(ax, 38, 4.7, 84, 13, color=GREEN, lw=1.2)
    ax.text(50, 47.6, "Toy VELO + qtrk_pipeline: decoupled, deterministic, sparse end-to-end",
            ha="center", fontsize=13, fontweight="bold")
    cm.savefig(fig, "pipeline_architecture")


# ---------------------------------------------------------------------------
def fig_hamiltonian_schematic():
    fig, ax = plt.subplots(figsize=(11.6, 5.2)); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 50)
    # planes with hits (left)
    zx = [8, 24, 40, 56, 72]
    rng = np.random.default_rng(3)
    pts = {}
    for i, x in enumerate(zx):
        ax.plot([x, x], [6, 40], color="#bbbbbb", lw=8, alpha=0.5, zorder=0)
        ys = np.sort(rng.uniform(10, 36, 4))
        pts[i] = [(x, y) for y in ys]
        for (px, py) in pts[i]:
            ax.add_patch(Circle((px, py), 0.9, fc="#333", zorder=3))
    # one true track (segments) across planes
    track_y = [12, 17, 22, 27, 32]
    for i in range(4):
        x0, y0 = zx[i], track_y[i]; x1, y1 = zx[i+1], track_y[i+1]
        _arrow(ax, x0, y0, x1, y1, color=GREEN, lw=2.4)
    ax.add_patch(Circle((zx[2], track_y[2]), 1.5, fill=False, ec=RED, lw=2, zorder=4))
    # kink-angle inset
    ax.annotate("middle hit shared\n$\\Rightarrow$ triplet,\nkink angle $\\theta$",
                (zx[2], track_y[2]), xytext=(30, 44), fontsize=9, color=RED,
                ha="center", arrowprops=dict(arrowstyle="->", color=RED))
    for i, x in enumerate(zx):
        ax.text(x, 3.5, f"plane {i}", ha="center", fontsize=8, color=GREY)
    ax.text(40, 47.5, "Hits $\\to$ segments $\\to$ triplets", ha="center", fontsize=13, fontweight="bold")
    ax.text(7, 41.5, "segment = directed hit$\\to$hit", fontsize=9, color=GREEN)

    # equations (right)
    ax.text(88, 40, r"$A\,\mathbf{x} = \mathbf{b}$", ha="center", fontsize=22, fontweight="bold")
    ax.text(88, 33, r"$\mathbf{b}=\delta\mathbf{1}$", ha="center", fontsize=13)
    ax.text(88, 27.5, r"$A_{ii}=\gamma+\delta$", ha="center", fontsize=13)
    ax.text(88, 22.5, r"$A_{ij}=-C(\theta_{ij})$", ha="center", fontsize=13)
    ax.text(88, 17, "attractive coupling on\nsegments sharing a hit", ha="center", fontsize=9, color=GREY)
    ax.text(88, 9.5, r"$x_i>0.35 \Rightarrow$ active", ha="center", fontsize=12, color=RED, fontweight="bold")
    ax.axvline(78, 6, 44, color="#dddddd")
    cm.savefig(fig, "hamiltonian_schematic")


# ---------------------------------------------------------------------------
def _wire(ax, y, x0, x1, label):
    ax.plot([x0, x1], [y, y], color="k", lw=1.3, zorder=1)
    ax.text(x0 - 0.4, y, label, ha="right", va="center", fontsize=9)


def _gate(ax, x, y, w, h, text, fc, fs=9, tc="k"):
    ax.add_patch(Rectangle((x - w/2, y - h/2), w, h, fc=fc, ec="k", lw=1.3, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4, color=tc)


def fig_hhl_vs_1bqf_circuit():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.4))
    for ax in (axA, axB):
        ax.axis("off"); ax.set_xlim(-2.5, 12.5); ax.set_ylim(-1, 9)

    # ---- HHL (left) ----
    ax = axA
    yt = [7.4, 6.6, 5.8]  # time register (n qubits)
    ys, ya = 4.2, 2.6
    for y in yt: _wire(ax, y, 0, 11, "")
    ax.text(-0.4, 6.6, "time\n($n$ qubits)", ha="right", va="center", fontsize=9)
    _wire(ax, ys, 0, 11, "system $|b\\rangle$")
    _wire(ax, ya, 0, 11, "ancilla")
    for y in yt: _gate(ax, 1, y, 0.7, 0.5, "H", "#eee")
    _gate(ax, 3, (yt[0]+ys)/2, 1.5, (yt[0]-ys)+0.7, "QPE\n$e^{iAt}$\n(dense)", BLUE, fs=8, tc="white")
    _gate(ax, 5, (yt[0]+yt[-1])/2, 1.2, (yt[0]-yt[-1])+0.6, "IQFT", "#ddd", fs=8)
    # RY ladder (eigenvalue inversion)
    _gate(ax, 7.2, (yt[0]+ya)/2, 1.7, (yt[0]-ya)+0.7, "$R_y(2\\arcsin\\frac{c}{\\lambda})$\n$1/\\lambda$ inversion", RED, fs=7.5, tc="white")
    _gate(ax, 9.4, (yt[0]+ys)/2, 1.3, (yt[0]-ys)+0.6, "QPE$^\\dagger$", BLUE, fs=8, tc="white")
    _gate(ax, 10.7, ya, 0.7, 0.5, "M", "#fdd")
    axA.set_title("Original HHL", fontsize=14, color=BLUE)
    axA.text(5, 0.4, "$n{\\sim}4$–$8$ time qubits · dense $e^{iAt}$ · full $1/\\lambda$ ladder\n"
                     "depth grows with eigenvalue precision · $O(N^2)$ unitaries",
             ha="center", fontsize=9, color="#333")

    # ---- 1BQF (right) ----
    ax = axB
    y1, ys, ya = 6.6, 4.2, 2.6
    _wire(ax, y1, 0, 11, "time (1 qubit)")
    _wire(ax, ys, 0, 11, "system $|b\\rangle$")
    _wire(ax, ya, 0, 11, "ancilla")
    _gate(ax, 1, y1, 0.7, 0.5, "H", "#eee")
    _gate(ax, 1, ys, 0.7, 0.5, "H", "#eee")
    _gate(ax, 3, (y1+ys)/2, 1.7, (y1-ys)+0.7, "QPE\n$e^{iAt}$\nsparse Givens\n$O(A_{\\rm nnz})$", ORANGE, fs=7.5, tc="k")
    _gate(ax, 5, y1, 1.0, 0.5, "IQFT$_1$", "#ddd", fs=8)
    # single-bit filter: X-CX-X
    _gate(ax, 6.6, y1, 0.6, 0.5, "X", "#eee", fs=8)
    ax.plot([7.6, 7.6], [y1, ya], color=RED, lw=1.5, zorder=2)
    ax.add_patch(Circle((7.6, y1), 0.12, fc=RED, zorder=4))
    _gate(ax, 7.6, ya, 0.55, 0.45, r"$\oplus$", "#fdd", fs=10)
    _gate(ax, 8.6, y1, 0.6, 0.5, "X", "#eee", fs=8)
    _gate(ax, 9.9, (y1+ys)/2, 1.2, (y1-ys)+0.7, "QPE$^\\dagger$", ORANGE, fs=8)
    _gate(ax, 10.7, ya, 0.6, 0.5, "M", "#fdd")
    ax.text(7.6, y1+0.7, "single phase bit", ha="center", fontsize=8, color=RED)
    axB.set_title("1-Bit Quantum Filter (1BQF)", fontsize=14, color=RED)
    axB.text(5, 0.4, "1 time qubit · sparse $e^{iAt}$ ($O(A_{\\rm nnz})$ gates) · one CX 'filter'\n"
                     "no $1/\\lambda$ ladder · shallow, fixed-depth eigenvalue split",
             ha="center", fontsize=9, color="#333")
    fig.suptitle("Eigenvalue inversion (HHL) $\\Rightarrow$ eigenvalue FILTER (1BQF)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    cm.savefig(fig, "hhl_vs_1bqf_circuit")


# ---------------------------------------------------------------------------
def fig_eigenvalue_filter():
    """Spectrum on the lambda axis: bulk at gamma+delta, true band spread out.
    HHL applies a smooth 1/lambda amplitude; 1BQF applies a single-bit pass band."""
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    lam = np.linspace(0.3, 8, 600)
    gd = 4.0
    # mock spectrum: big bulk delta at gamma+delta, small true band lower & higher
    ax.axvline(gd, color=GREY, lw=10, alpha=0.25)
    ax.text(gd, 1.06, r"bulk: false/isolated  $\lambda=\gamma+\delta$", color=GREY,
            ha="center", fontsize=9)
    for lc in [1.0, 1.6, 6.4, 7.1]:
        ax.axvline(lc, color=GREEN, lw=2, alpha=0.7)
    ax.text(1.3, 0.5, "true-segment\neigen-band", color=GREEN, ha="center", fontsize=9)
    # HHL 1/lambda weighting (normalised)
    hhl = (1.0 / lam); hhl = hhl / hhl.max()
    ax.plot(lam, hhl, color=BLUE, lw=2.4, label=r"HHL: amplitude $\propto 1/\lambda$")
    # 1BQF single-bit acceptance.  t = pi/(gamma+delta), phase phi = lambda*t/(2pi)
    # = lambda/(2(gamma+delta)).  Post-select ancilla=1 <=> time-bit=0 <=> phi near
    # 0 or 1 <=> lambda near 0 or 2(gamma+delta).  So the single bit REJECTS the
    # bulk at lambda=gamma+delta (phi=1/2) and KEEPS the spread true-segment band.
    t = np.pi / gd
    phi = lam * t / (2 * np.pi)
    accept = np.cos(np.pi * phi) ** 2             # notch at the bulk, pass the wings
    ax.plot(lam, accept, color=RED, lw=2.6, label="1BQF: single-bit filter")
    ax.annotate("bulk rejected", (gd, 0.05), xytext=(gd, 0.34), color=RED,
                ha="center", fontsize=9, arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel(r"eigenvalue  $\lambda$ of $A$")
    ax.set_ylabel("filter amplitude (normalised)")
    ax.set_title("HHL reweights every $\\lambda$ by $1/\\lambda$;\n"
                 "1BQF's single bit rejects the bulk and keeps the true band")
    ax.legend(loc="upper center"); ax.set_ylim(0, 1.18)
    cm.savefig(fig, "eigenvalue_filter")


if __name__ == "__main__":
    print("== conceptual diagrams ==")
    fig_pipeline_architecture()
    fig_hamiltonian_schematic()
    fig_hhl_vs_1bqf_circuit()
    fig_eigenvalue_filter()
    print("done.")
