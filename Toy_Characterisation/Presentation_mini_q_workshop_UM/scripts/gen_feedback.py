"""
Figures for the deck-feedback round (2026-06-10):

  activation_spectrum_400   single-panel T=400 clean event classical activation
                            spectrum (the bulk at 0.25, the true plateau, tau)
  hhl_states                HHL stage-by-stage with the statevector at every stage
                            (P4 track block, analytic - exact)
  A_blocks_segment_space    per cluster type: segment-space schematic ABOVE its
                            A-submatrix, so each block is visualisable
  quantum_2x2_matched_eff   quantum 2x2 vs T at the MATCHED-EFFICIENCY operating
                            point (full-norm rescale at tau=0.35 <=> tau_Q~0.17):
                            high efficiency / high false rate
  per_class_matched_T400    per-class treatment (T=400) in the same matched-eff
                            convention
  fork_matrix               the bifurcation (Denby-Peterson fork) term in the
                            matrix: A' = (gamma+delta)I - C + beta*B on a hub
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import _common as cm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import qtrk_pipeline as qp

S = 4.0; TAU = 0.35
GREEN, RED, GREY, BLUE, ORNG, PURP = "#1b7837", "#d6604d", "#888888", "#1f78b4", "#e08214", "#6a3d9a"
EPS_SET = 0.002      # the fixed 2 mrad acceptance used in Segment_level_studies


def _event_T(T, rep=0):
    ev, _ = qp.ensure_event(n_trk=T, rep=rep, sigma_scatt=1e-4, sigma_res=0.0)
    return ev


# ---------------------------------------------------------------------------
def fig_activation_spectrum_400():
    ev = _event_T(400)
    ham = qp.build_hamiltonian(ev, epsilon=EPS_SET, kernel="step")
    sol, _ = qp.solve_classical(ham)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    x = np.asarray(sol, float)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    bins = np.linspace(0.0, 0.62, 125)
    ax.hist(x[~truth], bins=bins, color=GREY, alpha=0.85,
            label=f"false segments ({int((~truth).sum()):,})")
    ax.hist(x[truth], bins=bins, color=GREEN, alpha=0.9,
            label=f"true segments ({int(truth.sum()):,})")
    ax.axvline(0.25, ls=":", color=RED, lw=1.8)
    ax.text(0.247, 2e5, r"false attractor $\delta/(\delta+\gamma)=0.25$",
            rotation=90, va="top", ha="right", fontsize=10, color=RED)
    ax.axvline(TAU, color="k", lw=2)
    ax.text(0.353, 2e5, r"$\tau=0.35$", rotation=90, va="top", fontsize=11)
    ax.axvspan(4/11, 5/11, color=GREEN, alpha=0.08)
    ax.text(0.41, 4e4, "true plateau\n$[4/11,\\,5/11]$", ha="center", fontsize=10, color=GREEN)
    ax.set_yscale("log")
    ax.set_xlabel("classical activation  $x_i$"); ax.set_ylabel("segments")
    ax.set_title("One clean 400-track event: the classical activation spectrum\n"
                 "(640,000 segments — the 0.25 false bulk vs the true plateau, split by $\\tau$)")
    ax.legend(loc="upper right")
    cm.savefig(fig, "activation_spectrum_400")


# ---------------------------------------------------------------------------
def fig_hhl_states():
    """HHL on the P4 true-track block, statevector at every stage (analytic)."""
    A = np.array([[4., -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 4]])
    b = np.ones(4); b = b / np.linalg.norm(b)
    w, U = np.linalg.eigh(A)
    beta = U.T @ b                          # eigen-amplitudes of |b>
    Cmin = w.min()
    rot = Cmin / w                           # ancilla=1 amplitude factor C/lambda
    amp1 = beta * rot                        # kept branch (unnormalised)
    x_exact = np.linalg.solve(A, np.ones(4)); xs = x_exact / np.linalg.norm(x_exact)
    x_hhl = U @ amp1; x_hhl = x_hhl / np.linalg.norm(x_hhl)

    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.9))
    stages = ["1) prepare $|b\\rangle$", "2) QPE: eigenvalues on the clock",
              "3) ancilla rotation $\\propto 1/\\lambda$", "4) uncompute + post-select"]
    # (1) |b> in the segment basis
    ax = axes[0]
    ax.bar(range(4), b, color=BLUE)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"$s_{i}$" for i in range(4)])
    ax.set_ylabel("amplitude"); ax.set_ylim(0, 0.85)
    ax.set_title(stages[0] + "\n(uniform over segments)", fontsize=10)
    # (2) eigen-decomposition: beta_k at lambda_k
    ax = axes[1]
    ax.bar(w, np.abs(beta), width=0.22, color=PURP)
    for lam, bk in zip(w, beta):
        ax.text(lam, abs(bk) + 0.02, f"$\\lambda$={lam:.2f}", ha="center", fontsize=8)
    ax.set_xlabel("eigenvalue $\\lambda_k$ (in the clock register)")
    ax.set_ylim(0, 1.0)
    ax.set_title(stages[1] + "\n$|b\\rangle=\\sum_k\\beta_k|u_k\\rangle|\\lambda_k\\rangle$", fontsize=10)
    # (3) controlled rotation: amplitudes * C/lambda on ancilla=1
    ax = axes[2]
    ax.bar(w - 0.07, np.abs(beta), width=0.14, color=PURP, alpha=0.45, label=r"before, $|\beta_k|$")
    ax.bar(w + 0.07, np.abs(amp1), width=0.14, color=RED, label=r"ancilla=1: $|\beta_k|\,C/\lambda_k$")
    ax.set_xlabel("eigenvalue $\\lambda_k$"); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8); ax.set_title(stages[2] + "\nlow $\\lambda$ boosted", fontsize=10)
    # (4) post-selected state vs exact solution
    ax = axes[3]
    ax.bar(np.arange(4) - 0.17, x_hhl, width=0.34, color=RED, label="HHL output")
    ax.bar(np.arange(4) + 0.17, xs, width=0.34, color="k", alpha=0.55, label=r"exact $A^{-1}b$")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"$s_{i}$" for i in range(4)])
    ax.set_ylim(0, 0.85); ax.legend(fontsize=8)
    ax.set_title(stages[3] + "\n$\\sum_k\\frac{\\beta_k}{\\lambda_k}|u_k\\rangle \\propto A^{-1}b$", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("HHL stage by stage on a true-track block ($P_4$): the statevector at every step",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    cm.savefig(fig, "hhl_states")


# ---------------------------------------------------------------------------
def _draw_mini(ax, kind):
    """Mini segment-space schematic (z ->, x up) for one cluster type."""
    zp = [0, 1, 2, 3, 4]
    for z in zp:
        ax.axvline(z, color="0.88", lw=5, zorder=0)
    T1, T2 = "#6a3d9a", "#ff7f00"

    def seg(z0, x0, z1, x1, true=True, lw=2.6):
        ax.plot([z0, z1], [x0, x1], "-" if true else "--",
                color=GREEN if true else RED, lw=lw, zorder=3)

    def hit(z, x, c="k"):
        ax.scatter([z], [x], s=42, color=c, ec="k", lw=0.5, zorder=4)

    if kind == "isolated":
        hit(1, 0.4, T1); hit(2, -0.4, T2); seg(1, 0.4, 2, -0.4, False)
    elif kind == "pair":
        hit(0, 0.5, T1); hit(1, 0.0, T2); hit(2, -0.5, T1)
        seg(0, 0.5, 1, 0.0, False); seg(1, 0.0, 2, -0.5, False)
    elif kind == "bridge":
        for z in zp:                       # two real tracks (faint)
            hit(z, 0.55, T1); hit(z, -0.55, T2)
        for z in range(4):
            seg(z, 0.55, z+1, 0.55, True, lw=1.2); seg(z, -0.55, z+1, -0.55, True, lw=1.2)
        seg(1, 0.55, 2, -0.55, False); seg(2, -0.55, 3, 0.55, False)
    elif kind == "track":
        for z in zp:
            hit(z, 0.0, T1)
        for z in range(4):
            seg(z, 0.0, z+1, 0.0, True)
    elif kind == "hub":
        hit(1, 0.0, T1); hit(0, 0.3, T1)
        seg(0, 0.3, 1, 0.0, False)
        for xe, col in [(0.55, T2), (0.0, "#33a02c"), (-0.55, "#b15928")]:
            hit(2, xe, col); seg(1, 0.0, 2, xe, False)
    ax.set_xlim(-0.4, 4.4); ax.set_ylim(-0.95, 0.95)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def fig_A_blocks_segment_space():
    BLOCKS = [
        ("isolated false", "isolated", np.array([[4.]]), "erased ✓"),
        ("pair false ($P_2$)", "pair", np.array([[4., -1], [-1, 4]]), "rejected by $\\tau$"),
        ("bridge false ($P_3$)", "bridge", np.array([[4., -1, 0], [-1, 4, -1], [0, -1, 4]]), "SURVIVES"),
        ("TRUE track ($P_4$)", "track", np.array([[4., -1, 0, 0], [-1, 4, -1, 0],
                                                  [0, -1, 4, -1], [0, 0, -1, 4]]), "kept ✓"),
        ("hub false ($K_{1,3}$)", "hub", np.array([[4., -1, -1, -1], [-1, 4, 0, 0],
                                                   [-1, 0, 4, 0], [-1, 0, 0, 4]]), "SURVIVES"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(14.6, 5.6),
                             gridspec_kw=dict(height_ratios=[1.0, 1.25], hspace=0.12, wspace=0.3))
    for j, (name, kind, A, outcome) in enumerate(BLOCKS):
        axT, axM = axes[0, j], axes[1, j]
        _draw_mini(axT, kind)
        col = GREEN if "TRUE" in name else (RED if "SURVIVES" in outcome else GREY)
        axT.set_title(name, fontsize=10.5, color=col, fontweight="bold")
        n = A.shape[0]
        axM.imshow(np.where(A > 0, -1.0, np.where(A < 0, 1.0, np.nan)),
                   cmap="bwr", vmin=-1.6, vmax=1.6, interpolation="nearest")
        for r in range(n):
            for c2 in range(n):
                if A[r, c2] != 0:
                    axM.text(c2, r, f"{A[r, c2]:.0f}", ha="center", va="center",
                             fontsize=9, color="white")
        w = np.linalg.eigvalsh(A); nbad = int(np.sum(np.abs(w - S) < 1e-9))
        axM.set_xticks([]); axM.set_yticks([])
        for s_ in axM.spines.values():
            s_.set_edgecolor(col); s_.set_linewidth(2.2)
        lamtxt = "{" + ", ".join(f"{v:.2f}".rstrip("0").rstrip(".") for v in w) + "}"
        axM.set_xlabel(f"$\\lambda=${lamtxt}\n{nbad} on the notch — quantum: {outcome}",
                       fontsize=8.2)
    fig.suptitle("Each cluster in segment space (top) and its block of $A$ (bottom):  "
                 "$A=(\\gamma+\\delta)I-C$ is block-diagonal over clusters",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.015, "green solid = true segment · red dashed = false segment · dot colour = which real track the hit belongs to",
             ha="center", fontsize=9.5, color="0.35")
    fig.subplots_adjust(top=0.86, bottom=0.13)
    cm.savefig(fig, "A_blocks_segment_space")


# ---------------------------------------------------------------------------
def _fixed_eps_quantum_rows():
    M = pd.read_csv(cm.METRICS_CSV)
    q = M[(M.solver == "quantum") & (M.kernel == "step") &
          (M.eps_provenance == "set") & (np.isclose(M.epsilon, EPS_SET, atol=1e-4)) &
          (np.isclose(M.sigma_scatt, 1e-4)) & (M.sigma_res == 0.0) & (M.hit_ineff == 0.0)]
    if "gamma" in q.columns:
        q = q[np.isclose(q.gamma, 3.0)]
    return q


def fig_quantum_2x2_matched_eff():
    q = _fixed_eps_quantum_rows()
    rows = []
    cacheC = {}
    for _, r in q.iterrows():
        key = (r.event_key, r.ham_key)
        if key not in cacheC:
            ev = qp.load_event(qp.event_path(r.event_key))
            ham = qp.build_hamiltonian(ev, epsilon=float(r.epsilon), kernel="step",
                                       gamma=float(r.gamma))
            solC, _ = qp.solve_classical(ham)
            truth = np.asarray(qp.truth_from_event(ev), bool)
            cacheC[key] = (solC, truth)
        solC, truth = cacheC[key]
        solQ = qp.rescale_to(np.asarray(qp.load_solution(r.sol_key)["sol"], float),
                             np.asarray(solC, float))           # FULL-norm rescale
        m = qp.metrics_at(solQ, truth, TAU)
        rows.append(dict(T=int(r.n_trk),
                         eff=m["segment_efficiency"], far=m["segment_false_rate"],
                         n_true=int(truth.sum()), n_false=int((~truth).sum()),
                         act_true=int(m.get("n_true_active", round(m["segment_efficiency"]*truth.sum()))),
                         act_false=int(m.get("n_false_active",
                                             round(m["segment_false_rate"]/max(1e-12,(1-m["segment_false_rate"]))
                                                   * m["segment_efficiency"]*truth.sum()) if m["segment_false_rate"]<1 else 0))))
        print(f"  T={int(r.n_trk):4d}  eff={m['segment_efficiency']:.3f} far={m['segment_false_rate']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    g = df.groupby("T").agg(["mean", "sem"])
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4))
    Ts = g.index.values
    axes[0, 0].errorbar(Ts, g[("eff", "mean")]*100, yerr=g[("eff", "sem")]*100,
                        marker="o", capsize=3, color="#c2185b")
    axes[0, 0].set_ylabel("Segment efficiency (%)"); axes[0, 0].set_ylim(90, 101)
    axes[0, 0].set_title("i) Segment efficiency — stays $\\approx$100%")
    axes[0, 1].errorbar(Ts, g[("far", "mean")]*100, yerr=g[("far", "sem")]*100,
                        marker="s", capsize=3, color="#c2185b")
    axes[0, 1].set_ylabel("Segment false rate (%)")
    axes[0, 1].set_title("ii) Segment false rate — grows with $T$")
    axes[1, 0].loglog(Ts, g[("n_true", "mean")], "o-", color=BLUE, label="true segments")
    axes[1, 0].loglog(Ts, g[("n_false", "mean")], "s-", color=RED, label="false segments")
    axes[1, 0].set_ylabel("segment counts"); axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_title("iii) Segment counts")
    axes[1, 1].loglog(Ts, g[("act_true", "mean")], "o-", color=BLUE, label="true active")
    axes[1, 1].loglog(Ts, np.maximum(g[("act_false", "mean")], 0.3), "s-", color=RED, label="false active")
    axes[1, 1].set_ylabel("active segments"); axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_title("iv) Active segments")
    for ax in axes.ravel():
        ax.set_xlabel("Number of tracks"); ax.grid(alpha=0.3)
    fig.suptitle("Quantum (1BQF) segment metrics at the MATCHED-EFFICIENCY operating point\n"
                 "(full-norm rescale at $\\tau=0.35$ $\\Leftrightarrow$ lowering $\\tau_Q$: efficiency $\\approx$100%, false rate grows)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    cm.savefig(fig, "quantum_2x2_matched_eff")


# ---------------------------------------------------------------------------
def fig_per_class_matched_T400():
    q = _fixed_eps_quantum_rows()
    r = q[q.n_trk == 400].iloc[0]
    ev = qp.load_event(qp.event_path(r.event_key))
    ham = qp.build_hamiltonian(ev, epsilon=float(r.epsilon), kernel="step", gamma=float(r.gamma))
    solC, _ = qp.solve_classical(ham)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    solQ = qp.rescale_to(np.asarray(qp.load_solution(r.sol_key)["sol"], float),
                         np.asarray(solC, float))
    A = ham.A.tocsr(); n = ham.n_segments
    Cm = (S * sp.identity(n, format="csr") - A); Cm.setdiag(0); Cm.eliminate_zeros()
    Cm = (abs(Cm) > 1e-9).astype(np.int8)
    _, lab = connected_components(Cm, directed=False)
    csize = np.bincount(lab); sz = csize[lab]
    classes = ["isolated\n(false)", "pair\n(false)", "coupled $\\geq$3\n(false)", "TRUE\n(track)"]

    def cls(i):
        if truth[i]:
            return 3
        if sz[i] == 1:
            return 0
        if sz[i] == 2:
            return 1
        return 2
    clab = np.array([cls(i) for i in range(n)])
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    for axi, sol, nm in [(axs[0], np.asarray(solC, float), "CLASSICAL  ($1/\\lambda$)"),
                         (axs[1], np.asarray(solQ, float), "QUANTUM 1BQF — matched efficiency")]:
        for j in range(4):
            selj = clab == j
            idx = rng.choice(np.where(selj)[0], size=min(int(selj.sum()), 400), replace=False)
            xs = j + rng.uniform(-0.28, 0.28, len(idx))
            col = GREEN if j == 3 else RED
            axi.scatter(xs, sol[idx], s=10, color=col, alpha=0.5, ec="none")
            act = (sol[selj] > TAU).mean() * 100
            axi.text(j, 1.45, f"{act:.0f}% active", ha="center", fontsize=9.5)
        axi.axhline(TAU, color="k", ls="--", lw=1.4)
        axi.text(3.42, TAU + 0.03, "$\\tau$", fontsize=12)
        axi.set_xticks(range(4)); axi.set_xticklabels(classes, fontsize=9)
        axi.set_title(nm, fontweight="bold"); axi.set_ylim(-0.06, 1.55)
    axs[0].set_ylabel("segment activation")
    fig.suptitle("Matched-efficiency operating point ($T{=}400$): the quantum keeps ALL true segments —\n"
                 "and with them the coupled false clusters (high efficiency $\\Rightarrow$ high false rate)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    cm.savefig(fig, "per_class_matched_T400")


# ---------------------------------------------------------------------------
def fig_fork_matrix():
    fig = plt.figure(figsize=(13.6, 4.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.35)
    # (a) segment-space fork schematic
    ax = fig.add_subplot(gs[0, 0])
    _draw_mini(ax, "hub")
    # arcs marking fork pairs between the 3 outgoing segments
    ax.annotate("", xy=(1.55, 0.30), xytext=(1.55, -0.30),
                arrowprops=dict(arrowstyle="<->", color=ORNG, lw=2,
                                connectionstyle="arc3,rad=0.45"))
    ax.text(1.18, 0.0, r"fork pairs: $+\beta$", color=ORNG, fontsize=11, rotation=90, va="center")
    ax.set_title("a fork: competing continuations\nsharing one hit", fontsize=11, fontweight="bold")
    # matrices
    Abase = np.array([[4., -1, -1, -1], [-1, 4, 0, 0], [-1, 0, 4, 0], [-1, 0, 0, 4]])
    Bf = np.zeros((4, 4)); Bf[1, 2] = Bf[2, 1] = Bf[1, 3] = Bf[3, 1] = Bf[2, 3] = Bf[3, 2] = 1.0
    beta = 1.0
    Afork = Abase + beta * Bf
    mats = [(Abase, "$A=(\\gamma{+}\\delta)I-C$\n(hub $K_{1,3}$)"),
            (beta * Bf, "$+\\,\\beta B$\nfork adjacency (share a hit,\nnot collinear)"),
            (Afork, "$A'=(\\gamma{+}\\delta)I-C+\\beta B$\n($\\beta=1$)")]
    for k, (Mt, title) in enumerate(mats):
        axm = fig.add_subplot(gs[0, k + 1])
        disp = np.full_like(Mt, np.nan, dtype=float)
        disp[np.isclose(Mt, 4)] = -1.0          # diagonal -> blue
        disp[Mt < 0] = 1.0                       # -C -> red
        disp[(Mt > 0) & ~np.isclose(Mt, 4)] = 0.45   # +beta -> light red/orange
        disp[np.isclose(Mt, 5)] = -1.0           # (diag+2beta variant safety)
        axm.imshow(disp, cmap="bwr", vmin=-1.6, vmax=1.6, interpolation="nearest")
        for rr in range(4):
            for cc2 in range(4):
                v = Mt[rr, cc2]
                if v != 0:
                    axm.text(cc2, rr, f"{v:+.0f}" if (rr != cc2 or k == 1) else f"{v:.0f}",
                             ha="center", va="center", fontsize=10,
                             color="white" if abs(v) >= 1 and not (0 < v < 4 and rr != cc2) else "k")
        axm.set_xticks([]); axm.set_yticks([])
        axm.set_title(title, fontsize=10)
        if k == 2:
            w = np.linalg.eigvalsh(Afork)
            axm.set_xlabel("$\\lambda=\\{" + ", ".join(f"{v:.2f}" for v in w) + "\\}$", fontsize=9)
    fig.suptitle("The bifurcation term: penalise competing continuations — "
                 "$\\varepsilon$-windowed $B_\\varepsilon$ keeps it sparse and 1BQF-safe",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    cm.savefig(fig, "fork_matrix")


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"all"}
    print("== feedback figures ==")
    if which & {"all", "spec"}:     fig_activation_spectrum_400()
    if which & {"all", "hhl"}:      fig_hhl_states()
    if which & {"all", "blocks"}:   fig_A_blocks_segment_space()
    if which & {"all", "fork"}:     fig_fork_matrix()
    if which & {"all", "matched"}:  fig_quantum_2x2_matched_eff()
    if which & {"all", "perclass"}: fig_per_class_matched_T400()
    print("done.")
