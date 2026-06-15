"""make_schematic.py — the labelled worked-example figure for the bifurcation note.

Draws ONE figure with three panels that take you from picture -> matrix:
  (a) the detector schematic: 5 module planes, the hits (with ids), the true
      4-segment chain s1-s2-s3-s4, and a false fork f that shares s1's START hit
      AND lies inside the angular acceptance window (theta < eps) -- i.e. a
      genuinely-competing near-collinear continuation, the only kind the
      (eps-windowed) bifurcation term penalises;
  (b) the two segment graphs the picture defines — the continuation graph C
      (opposite-side hit sharing, the -1 chain couplings) and the eps-windowed
      fork graph B_eps (same-side hit sharing within eps, the +beta coupling);
  (c) the assembled off-diagonal Hamiltonian A' = (gamma+delta) I - C + beta B_eps,
      annotated symbolically so you can read every matrix entry off the picture.

This is a STANDALONE schematic (hand-built 5-segment cluster, not a stored
event) so the algebra in bifurcation_hamiltonian.md is exactly reproducible.
Run with the Q_env kernel; writes outputs/schematic_worked_example.{png,pdf}.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Arc
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

GAMMA, DELTA = 3.0, 1.0
BLUE, RED, GREY = "#1f6fb4", "#c0392b", "#888888"

# ---- the worked example -----------------------------------------------------
# hits: id -> (z plane, y).  True track is a straight line (zero kink);
# h1' is the extra plane-1 hit that creates the false fork.  h1' sits CLOSE to
# h1 so f leaves h0 nearly collinear with s1 (small mutual angle, inside eps):
# that is the genuinely-competing continuation the eps-windowed term targets.
hits = {
    "h0": (0, 0.00), "h1": (1, 0.42), "h2": (2, 0.84),
    "h3": (3, 1.26), "h4": (4, 1.68), "h1'": (1, 0.14),
}
# segments: name -> (start_hit, end_hit)
segs = {"s1": ("h0", "h1"), "s2": ("h1", "h2"),
        "s3": ("h2", "h3"), "s4": ("h3", "h4"), "f": ("h0", "h1'")}
names = list(segs)                              # s1,s2,s3,s4,f
ix = {n: i for i, n in enumerate(names)}
n = len(names)

# continuation C: end(i) == start(j)  (share a MIDDLE hit, opposite sides)
C = np.zeros((n, n), int)
for a in names:
    for b in names:
        if a != b and segs[a][1] == segs[b][0]:
            C[ix[a], ix[b]] = C[ix[b], ix[a]] = 1
# fork B_eps: share START hit or share END hit (same side) AND near-collinear.
# In this hand cluster the only same-side pair is (s1, f) at h0, and it is
# near-collinear, so it is inside the window.
B = np.zeros((n, n), int)
for a in names:
    for b in names:
        if ix[a] < ix[b] and (segs[a][0] == segs[b][0] or segs[a][1] == segs[b][1]):
            B[ix[a], ix[b]] = B[ix[b], ix[a]] = 1

# ============================================================================
fig = plt.figure(figsize=(11.2, 8.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.28, wspace=0.22)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])

# ---- (a) detector schematic -------------------------------------------------
for z in range(5):
    axA.axvline(z, color=GREY, lw=1.0, ls=":", zorder=0)
    axA.text(z, -1.15, f"plane $M_{z}$", ha="center", va="top", fontsize=10, color=GREY)

def arrow(ax, p, q, color, ls="-", lw=2.4):
    ax.add_patch(FancyArrowPatch(hits[p], hits[q], arrowstyle="-|>",
                 mutation_scale=16, color=color, lw=lw, ls=ls,
                 shrinkA=7, shrinkB=7, zorder=2))

# true chain (blue) + labels at segment midpoints
for sname in ["s1", "s2", "s3", "s4"]:
    p, q = segs[sname]
    arrow(axA, p, q, BLUE)
    mx, my = (hits[p][0] + hits[q][0]) / 2, (hits[p][1] + hits[q][1]) / 2
    axA.text(mx, my + 0.13, f"${sname}$", color=BLUE, fontsize=12,
             ha="center", va="bottom", fontweight="bold")
# false fork (red dashed), near-collinear with s1
arrow(axA, "h0", "h1'", RED, ls="--")
axA.text(0.62, 0.01, "$f$", color=RED, fontsize=12, ha="center",
         va="top", fontweight="bold")

# the small mutual angle theta between s1 and f at h0 (inside the eps window)
a_f = np.degrees(np.arctan2(hits["h1'"][1], 1.0))
a_s1 = np.degrees(np.arctan2(hits["h1"][1], 1.0))
axA.add_patch(Arc(hits["h0"], 1.5, 1.5, angle=0, theta1=a_f, theta2=a_s1,
                  color="darkorange", lw=1.8, zorder=4))
amid = np.radians((a_f + a_s1) / 2)
axA.text(0.92 * np.cos(amid), 0.92 * np.sin(amid), r"$\theta<\varepsilon$",
         color="darkorange", fontsize=10.5, ha="left", va="center")

# hits
for h, (z, y) in hits.items():
    col = RED if h == "h1'" else "black"
    axA.plot(z, y, "o", ms=9, color=col, zorder=3)
    off = (10, -12) if h == "h1'" else (7, 9)
    axA.annotate(f"${h}$", (z, y), textcoords="offset points",
                 xytext=off, fontsize=10.5, color=col)

# highlight the shared start hit h0
axA.add_patch(Circle(hits["h0"], 0.10, fill=False, ec="darkorange", lw=2.4, zorder=5))
axA.annotate("shared start-hit $h_0$ and near-collinear ($\\theta<\\varepsilon$)\n"
             "$\\Rightarrow$ competing continuations: fork $(s_1,f)$ is penalised",
             hits["h0"], textcoords="offset points", xytext=(10, -64),
             ha="center", fontsize=9.6, color="darkorange",
             arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5))

axA.set_title("(a)  Detector schematic: a true 4-segment chain $s_1\\!-\\!s_2\\!-\\!s_3\\!-\\!s_4$ "
              "and a false fork $f$ — same start-hit, near-collinear", fontsize=11.5, loc="left")
axA.set_xlim(-0.6, 4.6); axA.set_ylim(-1.55, 2.05)
axA.set_xlabel("beam / plane index  $z$", fontsize=10)
axA.set_ylabel("transverse  $y$", fontsize=10)
axA.set_yticks([]); axA.set_xticks(range(5))
for s in ["top", "right", "left"]:
    axA.spines[s].set_visible(False)

# ---- (b) the two segment graphs --------------------------------------------
pos = {"s1": (0, 0), "s2": (1.1, 0), "s3": (2.2, 0), "s4": (3.3, 0), "f": (0, -1.25)}
# continuation edges (solid black, -1)
for a in names:
    for b in names:
        if ix[a] < ix[b] and C[ix[a], ix[b]]:
            (x1, y1), (x2, y2) = pos[a], pos[b]
            axB.plot([x1, x2], [y1, y2], "-", color="black", lw=2.2, zorder=1)
            axB.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, "$-1$", ha="center",
                     fontsize=11, color="black")
# fork edges (dashed red, +beta)
for a in names:
    for b in names:
        if ix[a] < ix[b] and B[ix[a], ix[b]]:
            (x1, y1), (x2, y2) = pos[a], pos[b]
            axB.plot([x1, x2], [y1, y2], "--", color=RED, lw=2.2, zorder=1)
            axB.text((x1 + x2) / 2 - 0.32, (y1 + y2) / 2, r"$+\beta$", ha="center",
                     fontsize=11, color=RED)
for nm, (x, y) in pos.items():
    col = RED if nm == "f" else BLUE
    axB.add_patch(Circle((x, y), 0.20, color=col, zorder=2))
    axB.text(x, y, f"${nm}$", color="white", ha="center", va="center",
             fontsize=10.5, fontweight="bold", zorder=3)
axB.plot([], [], "-", color="black", lw=2.2, label=r"continuation $C$  (share middle hit, $-1$)")
axB.plot([], [], "--", color=RED, lw=2.2,
         label=r"fork $B_\varepsilon$  (same-side hit, angle $<\varepsilon$, $+\beta$)")
axB.legend(loc="lower right", fontsize=8.4, frameon=False)
axB.set_title("(b)  The two graphs the picture defines", fontsize=11.5, loc="left")
axB.set_xlim(-0.7, 3.9); axB.set_ylim(-1.9, 0.7)
axB.axis("off")

# ---- (c) the assembled matrix A' -------------------------------------------
axC.set_title(r"(c)  $A' = (\gamma{+}\delta)\,I - C + \beta B_\varepsilon$   "
              r"(here $\gamma{+}\delta=4$)", fontsize=11.5, loc="left")
cell = 1.0
for i in range(n):
    for j in range(n):
        if i == j:
            txt, fc, tc = r"$\gamma{+}\delta$", "#dfeaf5", BLUE
        elif C[i, j]:
            txt, fc, tc = "$-1$", "#eeeeee", "black"
        elif B[i, j]:
            txt, fc, tc = r"$+\beta$", "#f6dcd8", RED
        else:
            txt, fc, tc = "$0$", "white", "#bbbbbb"
        axC.add_patch(Rectangle((j, n - 1 - i), cell, cell, facecolor=fc,
                                edgecolor="#cccccc", lw=1.0))
        axC.text(j + 0.5, n - 1 - i + 0.5, txt, ha="center", va="center",
                 fontsize=10.5, color=tc)
for k, nm in enumerate(names):
    axC.text(k + 0.5, n + 0.18, f"${nm}$", ha="center", va="bottom", fontsize=10)  # col labels
    axC.text(-0.18, n - 1 - k + 0.5, f"${nm}$", ha="right", va="center", fontsize=10)  # row labels
axC.text(n / 2, -0.45, r"$\mathbf{b}'=\delta\mathbf{1}$  (off-diagonal form);   "
         r"full form: diag $\to\gamma{+}\delta{+}2\beta$, $\mathbf{b}''=(\delta{+}\beta)\mathbf{1}$",
         ha="center", va="top", fontsize=9.2)
axC.set_xlim(-0.9, n + 0.2); axC.set_ylim(-0.9, n + 0.7)
axC.set_aspect("equal"); axC.axis("off")

fig.suptitle("Bifurcation term — from labelled picture to Hamiltonian matrix",
             fontsize=13, fontweight="bold", x=0.02, ha="left")
fig.savefig(OUT / "schematic_worked_example.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "schematic_worked_example.pdf", bbox_inches="tight")
print("wrote", OUT / "schematic_worked_example.png")
print("theta(s1,f) =", round(a_s1 - a_f, 2), "deg  (illustrative; real eps=2 mrad)")
print("C=\n", C, "\nB=\n", B)
