"""
Two figures requested for the deck:

  A_block_structure.png  How the Hamiltonian A = (gamma+delta)I - C is block-
                         diagonal over clusters, with one block per error type
                         (isolated / pair / bridge / hub / true track), showing
                         which produce BAD (notch) eigenvalues vs GOOD ones.
  noise_types.png        The two detector noises the toy injects: multiple
                         scattering (a cone around the track, 1 sigma) and
                         measurement resolution (a 1 sigma x/y smearing of each hit).

Both are built to match the package's actual model:
 - scattering kicks the slopes (tx,ty) by N(0, sigma_scatt) at each plane -> the
   trajectory random-walks downstream into a cone;
 - resolution smears each measured hit (x,y) by N(0, sigma_res).
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import _common as cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

GREEN, RED, ORANGE, BLUE, GREY = "#1b7837", "#d6604d", "#e08214", "#1f78b4", "#888888"
S = 4.0   # gamma+delta (notch position) at gamma=3, delta=1
TVAL = np.pi / S
filt = lambda lam: np.cos(lam * TVAL / 2)   # 1-bit filter

# canonical cluster blocks (A = (gamma+delta)I - C); name, matrix, truth, outcome
BLOCKS = [
    ("isolated false\n(size 1)", np.array([[4.]]), False, "erased"),
    ("pair false\n($P_2$)", np.array([[4., -1], [-1, 4]]), False, "rejected"),
    ("bridge false\n($P_3$)", np.array([[4., -1, 0], [-1, 4, -1], [0, -1, 4]]), False, "survives"),
    ("TRUE track\n($P_4$)", np.array([[4., -1, 0, 0], [-1, 4, -1, 0],
                                      [0, -1, 4, -1], [0, 0, -1, 4]]), True, "kept"),
    ("hub false\n($K_{1,3}$)", np.array([[4., -1, -1, -1], [-1, 4, 0, 0],
                                         [-1, 0, 4, 0], [-1, 0, 0, 4]]), False, "survives"),
]


# ---------------------------------------------------------------------------
def fig_A_block_structure():
    fig = plt.figure(figsize=(13.6, 7.4))
    gs = fig.add_gridspec(2, len(BLOCKS), height_ratios=[1.35, 1.0], hspace=0.55, wspace=0.35)

    # ---- top-left: the full block-diagonal A schematic -------------------
    axBig = fig.add_subplot(gs[0, :3])
    # build a small example A: many 1x1 isolated, then a P4, P3, K(1,3)
    diag_iso = [4.0] * 8
    comps = [np.array([[4.]])] * 8 + [BLOCKS[3][1], BLOCKS[2][1], BLOCKS[4][1]]
    sizes = [c.shape[0] for c in comps]
    N = sum(sizes)
    Afull = np.zeros((N, N)); off = 0; spans = []
    for c in comps:
        s = c.shape[0]; Afull[off:off+s, off:off+s] = c; spans.append((off, s)); off += s
    # display: blue = diagonal (gamma+delta), red = coupling (-1, stands out)
    disp = np.full((N, N), np.nan)
    disp[Afull > 0] = -1.0    # diagonal s  -> blue
    disp[Afull < 0] = 1.0     # off-diagonal -1 -> red
    axBig.imshow(disp, cmap="bwr", vmin=-1.6, vmax=1.6, interpolation="nearest")
    # outline + label the structured blocks
    labels = {8: ("track $P_4$", GREEN), 9: ("bridge $P_3$", RED), 10: ("hub $K_{1,3}$", ORANGE)}
    for k, (o, s) in enumerate(spans):
        ec = "0.7" if k < 8 else labels[k][1]
        axBig.add_patch(Rectangle((o-0.5, o-0.5), s, s, fill=False, ec=ec,
                                  lw=1.0 if k < 8 else 2.4))
        if k in labels:
            axBig.annotate(labels[k][0], (o+s/2-0.5, o-0.7), color=labels[k][1],
                           fontsize=9, ha="center", va="bottom", fontweight="bold")
    axBig.annotate("the isolated bulk (~99.5%):\neach a $1\\times1$ block $[\\gamma+\\delta]$\n= one BAD (notch) eigenvalue",
                   (3.5, 3.5), (8.6, 6.2), fontsize=8.5, color="0.3",
                   arrowprops=dict(arrowstyle="->", color="0.5"))
    axBig.set_title("$A=(\\gamma+\\delta)\\,I-C$  is block-diagonal over clusters",
                    fontsize=12, fontweight="bold")
    axBig.set_xticks([]); axBig.set_yticks([])
    axBig.text(0.5, -0.09, "blue = diagonal $\\gamma+\\delta$   ·   red = coupling $-C(\\theta)=-1$",
               transform=axBig.transAxes, ha="center", fontsize=9)

    # ---- top-right: the derivation + filter ------------------------------
    axTxt = fig.add_subplot(gs[0, 3:]); axTxt.axis("off")
    axTxt.text(0.5, 0.97, "From the Hamiltonian to the spectrum", ha="center",
               va="top", fontsize=12, fontweight="bold")
    lines = [
        r"$A\,\mathbf{x}=\mathbf{b}=\delta\mathbf{1}$,    $A_{ii}=\gamma+\delta$,    $A_{ij}=-C(\theta_{ij})$",
        r"$C$ = 0/1 compatibility adjacency (share a hit, $\theta<\varepsilon$)",
        r"$\Rightarrow\;\; A=(\gamma+\delta)I-C$    (block-diagonal over clusters)",
        r"each cluster's eigenvalues:   $\lambda=(\gamma+\delta)-\mu(C)$",
        "",
        r"$\bullet$  $\mu(C)=0\;\Leftrightarrow\;\lambda=\gamma+\delta$ = the NOTCH = BAD",
        r"$\bullet$  isolated segment $=[\gamma+\delta]$  $\to$ sits on the notch",
        r"$\bullet$  a hub $K_{1,m}$ has $m-1$ bad eigenvalues",
        r"$\bullet$  quantum 1-bit filter $f(\lambda)=\cos(\frac{\pi}{2}\frac{\lambda}{\gamma+\delta})$",
        r"     erases the notch ($f=0$), keeps everything off it",
    ]
    y = 0.86
    for ln in lines:
        axTxt.text(0.02, y, ln, fontsize=10.5, va="top",
                   color="k" if not ln.startswith(r"$\bullet$") else "0.15")
        y -= 0.095

    # ---- bottom row: each canonical block --------------------------------
    for j, (name, A, istrue, outcome) in enumerate(BLOCKS):
        ax = fig.add_subplot(gs[1, j])
        n = A.shape[0]
        ax.imshow(np.where(A > 0, -1.0, np.where(A < 0, 1.0, np.nan)),
                  cmap="bwr", vmin=-1.6, vmax=1.6, interpolation="nearest")
        for r in range(n):
            for cc in range(n):
                if A[r, cc] != 0:
                    ax.text(cc, r, f"{A[r, cc]:.0f}", ha="center", va="center",
                            fontsize=8, color="white")
        w = np.linalg.eigvalsh(A); nbad = int(np.sum(np.abs(w - S) < 1e-9))
        ec = GREEN if istrue else (RED if outcome == "survives" else GREY)
        for sp in ax.spines.values():
            sp.set_edgecolor(ec); sp.set_linewidth(2.4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=8.5, color=ec, fontweight="bold")
        eig_txt = "{" + ", ".join(f"{v:.2f}" for v in w) + "}"
        sub = (f"$\\lambda=${eig_txt}\n{nbad} bad (notch)\n"
               f"quantum: {outcome}")
        ax.text(0.5, -0.30, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.6, color="0.15")

    fig.suptitle("Encoding the errors in $A$: one block per cluster type — "
                 "good (off-notch) vs bad (notch) eigenvalues",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.subplots_adjust(bottom=0.16, top=0.9)
    cm.savefig(fig, "A_block_structure")


# ---------------------------------------------------------------------------
def fig_noise_types():
    fig, (axS, axR) = plt.subplots(1, 2, figsize=(13.2, 5.3))
    zpl = np.array([33., 66., 99., 132., 165.])
    dz = 33.0

    # ===== (a) multiple scattering: a cone around the track ===============
    # slopes random-walk: tx kicked by N(0, sigma) at each plane crossing.
    sig_disp = 0.020          # exaggerated for visibility
    rng = np.random.default_rng(4)
    z = np.concatenate([[0.0], zpl])
    def walk(sig, n):
        X = np.zeros((n, len(z))); tx = np.zeros(n)
        for k in range(1, len(z)):
            X[:, k] = X[:, k-1] + tx * (z[k]-z[k-1])
            tx = tx + rng.normal(0, sig, n)     # scatter AT the plane
        return X
    XX = walk(sig_disp, 4000)
    band = XX.std(0)
    axR_unscat = np.zeros_like(z)
    axS.fill_between(z, -band, band, color=BLUE, alpha=0.18, label=r"$\pm1\sigma$ scattering cone")
    axS.plot(z, axR_unscat, "--", color="k", lw=1.6, label="no scattering (straight)")
    for i in range(5):
        axS.plot(z, walk(sig_disp, 1)[0], "-", color=BLUE, lw=1.0, alpha=0.6)
    for zp in zpl:
        axS.axvline(zp, color="0.85", lw=6, zorder=0)
    # kink angle annotation at first plane
    axS.annotate(r"kink $\theta\sim\mathcal{N}(0,\sigma_{\rm scatt})$ at each plane",
                 (zpl[1], band[2]), (70, band.max()*0.7), fontsize=9, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE))
    axS.set_xlabel("z (mm) — beam direction"); axS.set_ylabel("lateral position x (mm)")
    axS.set_title("(a) Multiple scattering $\\sigma_{\\rm scatt}$:\nthe track wanders into a cone "
                  "(slopes $t_x,t_y$ kicked at each plane)", fontsize=10.5, fontweight="bold")
    axS.legend(loc="upper left", fontsize=9)
    axS.text(0.98, 0.03, "illustrative (exaggerated); physical $\\sigma_{\\rm scatt}\\sim10^{-4}$ rad",
             transform=axS.transAxes, ha="right", fontsize=8, style="italic", color="0.4")
    axS.set_xticks(zpl); axS.set_xticklabels([f"p{i+1}" for i in range(5)])

    # ===== (b) measurement resolution: 1-sigma x/y smearing ===============
    sig_res = 0.05            # mm, realistic upper end; window zoomed to a few mm
    rng2 = np.random.default_rng(1)
    true_xy = np.array([0.0, 0.0])
    meas = rng2.normal(true_xy, sig_res, size=(300, 2))
    axR.scatter(meas[:, 0], meas[:, 1], s=10, color=RED, alpha=0.35,
                label="measured hits")
    for k in (1, 2):
        axR.add_patch(Circle(true_xy, k*sig_res, fill=False, ec=BLUE, lw=1.8,
                             ls="--" if k == 2 else "-",
                             label=f"${k}\\sigma_{{\\rm res}}$" ))
    axR.scatter([0], [0], marker="*", s=260, color=GREEN, ec="k", zorder=5,
                label="true hit position")
    axR.set_aspect("equal")
    axR.set_xlabel("x on module (mm)"); axR.set_ylabel("y on module (mm)")
    lim = 3.2*sig_res
    axR.set_xlim(-lim, lim); axR.set_ylim(-lim, lim)
    axR.set_title("(b) Measurement resolution $\\sigma_{\\rm res}$:\neach hit smeared by "
                  "$\\mathcal{N}(0,\\sigma_{\\rm res})$ in $x$ and $y$", fontsize=10.5, fontweight="bold")
    axR.legend(loc="upper right", fontsize=8.5)
    axR.text(0.5, -0.16, r"$\sigma_{\rm res}$ tilts each segment by $\sim\arctan(\sigma_{\rm res}/\Delta z)$"
             "  →  feeds the acceptance $\\varepsilon$",
             transform=axR.transAxes, ha="center", fontsize=9, color="0.3")

    fig.suptitle("The two detector noises the toy injects — both widen the triplet kink angle, "
                 "set by $\\varepsilon=\\sqrt{2(s\\sigma_{\\rm scatt})^2+12\\arctan^2(s\\sigma_{\\rm res}/\\Delta z)+\\dots}$",
                 fontsize=11.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    cm.savefig(fig, "noise_types")


if __name__ == "__main__":
    print("== mechanism figures ==")
    fig_A_block_structure()
    fig_noise_types()
    print("done.")
