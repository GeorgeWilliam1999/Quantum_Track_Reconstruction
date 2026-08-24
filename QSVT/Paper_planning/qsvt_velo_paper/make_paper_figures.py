#!/usr/bin/env python3
"""Paper figures for 'From one bit to a fitted polynomial' (review pass, 2026-07-19).

Each subcommand produces one figure into figures/.  Run `all_fast` for the
figures that need no event data, `heavy` for the ones built from the heavy
T=200 rep-0 event, `threshold` for the clean T=400 tau-sweep, `fitpaper` for
the paper version of the fit figure, `erf` and `run3crop` for the repaired
imported figures.

Review items covered: 1/5 (motif gallery + measured pile-up), 6 (response
functions), 7 (threshold trade-off), 10 (resources restack is LaTeX-side),
11 (tangle gallery), 14 (no 'wp99' tokens in labels), 15 (no unfitted-comb
series), 17/18 (no footers), 21 (spectral-twin diagram), 23 (readout
probability), 25 (real-geometry pile-up).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

S = 4.0                       # gamma + delta at the main-text working point
P4 = S - 2.0 * np.cos(np.arange(1, 5) * np.pi / 5.0)
GREEN, RED, GREY, BLUE, ORANGE = "#3d8a4f", "#d84a49", "#8f8d86", "#2a78d6", "#e08a2e"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 300, "font.size": 11,
    "axes.labelsize": 11.5, "axes.titlesize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] figures/{name}.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Motif drawing helpers (hit-space cartoons; abstract units)
# ─────────────────────────────────────────────────────────────────────────────

def draw_motif(ax, segs, hit_colour="#33322e", true_mask=None, lw=2.4):
    """segs: list of ((z1,x1),(z2,x2)); true_mask: which segs are true."""
    pts = {}
    for k, (p, q) in enumerate(segs):
        col = GREEN if (true_mask is not None and true_mask[k]) else RED
        ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=col, lw=lw,
                solid_capstyle="round", zorder=2)
        pts[p] = None
        pts[q] = None
    P = np.array(list(pts))
    ax.plot(P[:, 0], P[:, 1], "o", color=hit_colour, ms=6.5, zorder=3)
    ax.set_xlim(-0.35, 2.55)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")


MOTIFS = [
    ("isolated segment", [((0, 0), (1, 0.45))], None,
     [S], "$\\lambda = s$"),
    ("pair (one continuation)", [((0, -0.35), (1, 0.0)), ((1, 0.0), (2, 0.5))],
     None, [S - 1, S + 1], "$\\lambda = s \\mp 1$"),
    ("triple (fork)", [((0, -0.3), (1, 0.05)), ((1, 0.05), (2, 0.6)),
                       ((1, 0.05), (2, -0.35))],
     None, [S - np.sqrt(2), S, S + np.sqrt(2)], "$\\lambda = s,\\ s \\mp \\sqrt{2}$"),
    ("star ($m{=}3$ continuations)",
     [((0, -0.3), (1, 0.0)), ((1, 0.0), (2, 0.62)),
      ((1, 0.0), (2, 0.12)), ((1, 0.0), (2, -0.55))],
     None, [S - np.sqrt(3), S, S, S + np.sqrt(3)], "$\\lambda = s,\\ s,\\ s \\mp \\sqrt{3}$"),
    ("true track ($P_4$ chain)",
     [((0, -0.5), (0.5, -0.25)), ((0.5, -0.25), (1.0, 0.0)),
      ((1.0, 0.0), (1.5, 0.25)), ((1.5, 0.25), (2.0, 0.5))],
     [1, 1, 1, 1], list(P4), "$\\lambda = s - 2\\cos(k\\pi/5)$"),
]


def fig_motifs():
    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.15], hspace=0.08)
    for i, (title, segs, tmask, lines, lab) in enumerate(MOTIFS):
        ax = fig.add_subplot(gs[0, i])
        draw_motif(ax, segs, true_mask=tmask)
        ax.set_title(title, fontsize=10.5, pad=2)
    axl = fig.add_subplot(gs[1, :])
    y = 0
    yticklabels = []
    for title, segs, tmask, lines, lab in MOTIFS:
        col = GREEN if tmask else RED
        for lam in lines:
            axl.plot([lam, lam], [y - 0.30, y + 0.30], "-", color=col, lw=3.0)
        axl.text(7.62, y, lab, fontsize=10.5, va="center", color="#33322e")
        yticklabels.append(title.split(" (")[0])
        y += 1
    axl.axvline(S, color=GREY, lw=0.8, ls=":")
    axl.text(S, y - 0.28, "$\\lambda = s$ (isolated line)", fontsize=9.5,
             color=GREY, ha="center", va="bottom")
    for lam in P4:
        axl.axvline(lam, color=GREEN, lw=0.7, ls=":", alpha=0.55)
    axl.set_yticks(range(len(MOTIFS)), yticklabels, fontsize=10.5)
    axl.set_ylim(-0.6, len(MOTIFS) - 0.1)
    axl.set_xlim(1.6, 9.4)
    axl.set_xlabel("eigenvalue $\\lambda$   ($s = \\gamma + \\delta = 4$)")
    axl.grid(axis="y", visible=False)
    save(fig, "motif_gallery")


def fig_twin():
    fig = plt.figure(figsize=(12.0, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.35], hspace=0.32,
                          wspace=0.12)

    ax = fig.add_subplot(gs[0, 0])
    draw_motif(ax, [((0, -0.3), (1, 0.0)), ((1, 0.0), (2, 0.35))],
               true_mask=[1, 1])
    ax.set_title("true fragment: 3 surviving hits\nof a broken track (2 segments)",
                 fontsize=10, pad=2, color=GREEN)
    ax = fig.add_subplot(gs[1, 0])
    draw_motif(ax, [((0, -0.55), (1, 0.1)), ((1, 0.1), (2, 0.9))],
               true_mask=[0, 0])
    ax.set_title("false pair: accidental continuation\nof two fake segments",
                 fontsize=10, pad=6, color=RED)

    for r, col in ((0, GREEN), (1, RED)):
        ax = fig.add_subplot(gs[r, 1])
        ax.axis("off")
        ax.text(0.30, 0.52, "$A_{\\rm block}=$", fontsize=12.5, ha="center",
                va="center", color="#33322e")
        ax.text(0.52, 0.52, "(", fontsize=30, ha="center", va="center",
                color="#33322e")
        ax.text(0.62, 0.66, "$s$", fontsize=11.5, ha="center", va="center")
        ax.text(0.74, 0.66, "$-1$", fontsize=11.5, ha="center", va="center")
        ax.text(0.62, 0.38, "$-1$", fontsize=11.5, ha="center", va="center")
        ax.text(0.74, 0.38, "$s$", fontsize=11.5, ha="center", va="center")
        ax.text(0.84, 0.52, ")", fontsize=30, ha="center", va="center",
                color="#33322e")
        ax.text(0.57, 0.10, "identical $2\\times2$ block", fontsize=9.5,
                ha="center", color=col)

    axs = fig.add_subplot(gs[:, 2])
    for k, (lam, col, lab, yo) in enumerate(
            [(S - 1, None, None, 0), (S + 1, None, None, 0)]):
        axs.plot([lam, lam], [0.12, 0.88], "-", color=GREEN, lw=5.5, alpha=0.9,
                 zorder=2)
        axs.plot([lam, lam], [0.12, 0.88], "--", color=RED, lw=2.2, zorder=3)
    axs.set_xlim(2.2, 5.8)
    axs.set_ylim(0, 1)
    axs.set_yticks([])
    axs.set_xticks([S - 1, S, S + 1],
                   ["$s-1$", "$s$", "$s+1$"])
    axs.axvline(S, color=GREY, lw=0.7, ls=":")
    axs.set_title("identical spectrum: $\\lambda = s \\mp 1$\n"
                  "(green true / red-dashed false, exactly superimposed)",
                  fontsize=10.5)
    axs.set_xlabel("eigenvalue $\\lambda$")
    save(fig, "spectral_twins")


def fig_responses():
    lam = np.linspace(0.0, 8.0, 3000)
    t = np.pi / S
    f_notch = np.cos(lam * t / 2.0)

    from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse
    comb = design_line_comb_inverse(degree=40, s=S, hw=0.18, domain=(0.0, 8.0))
    f_comb = np.abs(comb(lam))
    f_comb = f_comb / f_comb.max()

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
    pops = [(S, GREY, "isolated false ($\\lambda=s$)"),
            (S - 1, RED, "pair"), (S + 1, RED, None),
            (S - np.sqrt(2), ORANGE, "triple"), (S + np.sqrt(2), ORANGE, None)]

    ax = axes[0]
    ax.plot(lam, np.abs(f_notch), color=BLUE, lw=2.2,
            label="1BQF response $|\\cos(\\lambda t/2)|$,  $t=\\pi/s$")
    ax.plot(lam, f_notch, color=BLUE, lw=0.9, alpha=0.35)
    for lv, col, lab in pops:
        ax.axvline(lv, color=col, lw=1.4 if col is GREY else 1.1,
                   ls="-" if col is GREY else "--",
                   label=lab, alpha=0.85)
    for lv in P4:
        ax.axvline(lv, color=GREEN, lw=1.1, ls=":")
    ax.axvline(P4[0], color=GREEN, lw=1.1, ls=":",
               label="true-track lines $s-2\\cos(k\\pi/5)$")
    ax.annotate("the single zero lands exactly on the\nisolated-false line: "
                "those segments are erased",
                xy=(S, 0.02), xytext=(5.6, 0.42), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="#33322e", lw=0.9))
    ax.annotate("coupled false motifs sit off the zero\nand pass with "
                "large weight", xy=(S - 1, np.abs(np.cos((S - 1) * t / 2))),
                xytext=(0.35, 0.75), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="#a83232", lw=0.9))
    ax.set_ylabel("$|f(\\lambda)|$")
    ax.set_ylim(-0.02, 1.06)
    ax.legend(loc="upper right", fontsize=8.8, ncol=1)
    ax.set_title("(a) the one-bit filter: one zero, spent on the largest false population",
                 loc="left", fontsize=11)

    ax = axes[1]
    ax.plot(lam, f_comb, color=BLUE, lw=2.2,
            label="line comb $|p(\\lambda)|$ (degree 40, half-width 0.18)")
    for lv, col, lab in pops:
        ax.axvline(lv, color=col, lw=1.4 if col is GREY else 1.1,
                   ls="-" if col is GREY else "--", alpha=0.85)
    for lv in P4:
        ax.axvline(lv, color=GREEN, lw=1.1, ls=":")
    ax.annotate("narrow passes at the four true-track lines only;\n"
                "isolated line, pair and triple lines all rejected",
                xy=(P4[1], 0.62), xytext=(5.3, 0.72), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="#2c6e3f", lw=0.9))
    ax.set_xlabel("eigenvalue $\\lambda$")
    ax.set_ylabel("$|p(\\lambda)|$")
    ax.set_ylim(-0.02, 1.06)
    ax.legend(loc="upper left", fontsize=9.2)
    ax.set_title("(b) the QSVT line comb: a zero (or a pass) wherever the spectrum needs one",
                 loc="left", fontsize=11)
    save(fig, "response_functions")


def fig_readout():
    import pandas as pd
    df = pd.read_csv("/data/bfys/gscriven/Quantum_Track_Reconstruction/QSVT/"
                     "Segment_level_studies/outputs/depth_and_qubits/degree_sweep.csv")
    d40 = df[df.degree == 40].groupby("T").mean(numeric_only=True).reset_index()
    Ts = np.geomspace(40, 1200, 200)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    ax.loglog(d40["T"], d40["P"], "o", color=BLUE, ms=7,
              label="measured, line comb ($d=40$)")
    ax.loglog(Ts, 0.44 / Ts, "-", color=BLUE, lw=1.4, alpha=0.7,
              label="$P_{\\rm anc} = 0.44/T$")
    ax.loglog(Ts, 0.25 / Ts, "--", color=RED, lw=1.4, alpha=0.7,
              label="1BQF estimate $0.25/T$")
    ax.set_xlabel("tracks per event $T$")
    ax.set_ylabel("post-selection success $P_{\\rm anc}$")
    ax.legend()
    ax.set_title("(a) success probability of one repetition", loc="left",
                 fontsize=11)

    ax = axes[1]
    ax.loglog(d40["T"], d40["total_depth"], "s", color=BLUE, ms=7,
              label="measured amplified walk calls ($d=40$)")
    scale = float(d40["total_depth"].iloc[0] / np.sqrt(d40["T"].iloc[0]))
    ax.loglog(Ts, scale * np.sqrt(Ts), "-", color=BLUE, lw=1.4, alpha=0.7,
              label="$\\propto d\\sqrt{T}$")
    ax.set_xlabel("tracks per event $T$")
    ax.set_ylabel("total walk calls per solve")
    ax.legend()
    ax.set_title("(b) walk-call budget after amplitude amplification",
                 loc="left", fontsize=11)
    save(fig, "readout_probability")


def fig_erf():
    import pandas as pd
    df = pd.read_csv("/data/bfys/gscriven/Quantum_Track_Reconstruction/"
                     "Toy_Characterisation/ERF/results/erf_exact_evolution_check.csv")
    df["label"] = df["pair"] + "\n" + df["cell"]
    order = ["moderate\nstep2x", "moderate\nkink", "heavy\nstep2x", "heavy\nkink"]
    df = df.set_index("label").reindex(order).reset_index()
    xs = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    w = 0.27
    ax.bar(xs - w, df.far_wp99_exact, w * 0.92, color=BLUE,
           label="exact evolution (no Trotter)")
    ax.bar(xs, df.far_wp99_circuit, w * 0.92, color=RED,
           label="implemented circuit (per-pair product formula)")
    ax.bar(xs + w, df.far_wp99_classical, w * 0.92, color="#5c5b55",
           label="classical (same events)")
    ax.set_xticks(xs, df.label)
    ax.set_ylabel("false rate at the efficiency-first\nworking point (99% true survival)")
    ax.legend(fontsize=9)
    ax.set_title("(a) circuit vs exact evolution, erf angular-cost Hamiltonian",
                 loc="left", fontsize=11)

    ax = axes[1]
    ax.bar(xs - w, df.cos_exact_circuit, w * 0.92, color=BLUE,
           label="cos(exact, circuit)")
    ax.bar(xs, df.cos_exact_classical, w * 0.92, color="#5c5b55",
           label="cos(exact, classical)")
    ax.bar(xs + w, df.cos_circuit_classical, w * 0.92, color=GREY,
           label="cos(circuit, classical)")
    ax.set_xticks(xs, df.label)
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_title("(b) where the readouts disagree: uniform kernels agree, "
                 "weighted kernels are Trotter-dominated", loc="left",
                 fontsize=11)
    save(fig, "erf_exact_evolution_paper")


def fig_run3crop():
    from PIL import Image
    src = ("/data/bfys/gscriven/Quantum_Track_Reconstruction/QSVT/Run3/"
           "outputs/run3_activation_spectra.png")
    im = Image.open(src)
    w, h = im.size
    im2 = im.crop((0, int(0.055 * h), w, h))
    im2.save(FIG / "run3_activation_spectra.png")
    print(f"[saved] figures/run3_activation_spectra.png (cropped from {src})")


def fig_resources_compare():
    """Table-2 companion: measured resource scaling vs track count for
    HHL (structure-level estimate band), the 1BQF, and the DSS-QETU comb."""
    import pandas as pd
    base = "/data/bfys/gscriven/Quantum_Track_Reconstruction/QSVT/DSS/outputs/"
    gw = pd.read_csv(base + "dss_width.csv")
    gg = pd.read_csv(base + "dss_gates.csv")
    NC_LO, NC_HI = 6, 8
    calls_lo, calls_hi = 2 * (2 ** NC_LO - 1), 2 * (2 ** NC_HI - 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7),
                             constrained_layout=True)

    ax = axes[0]
    ax.fill_between(gw["T"], gw["n_s"] + NC_LO + 1, gw["n_s"] + NC_HI + 1,
                    color=RED, alpha=0.25, label="HHL (clock $n_c=6$--$8$)")
    ax.plot(gw["T"], gw["QSVT_LCU"], ":", color=GREY, lw=1.6,
            label="comb, dense-dilation LCU (simulation only)")
    ax.plot(gw["T"], gw["BQF_1"], "o-", color=RED, lw=1.8, ms=6,
            label="1BQF")
    ax.plot(gw["T"], gw["DSS_QETU"], "s--", color=BLUE, lw=1.8, ms=6,
            label="comb (DSS--QETU)")
    ax.set_xscale("log")
    ax.set_xlabel("tracks per event $T$")
    ax.set_ylabel("logical qubits")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("(a) register width: $\\lceil\\log_2 4T^2\\rceil$ plus a "
                 "constant", loc="left", fontsize=11)

    ax = axes[1]
    ax.fill_between(gg["T"], calls_lo * gg["call_CX"],
                    calls_hi * gg["call_CX"], color=RED, alpha=0.25,
                    label="HHL estimate ($2(2^{n_c}{-}1)$ calls)")
    ax.loglog(gg["T"], gg["bqf_total"], "o-", color=RED, lw=1.8, ms=6,
              label="1BQF (2 calls)")
    ax.loglog(gg["T"], gg["comb_global"], "s--", color=BLUE, lw=1.8, ms=6,
              label="comb, global ($d=40$ calls)")
    ax.loglog(gg["T"], gg["comb_percluster_max_instance"], "D", color=GREEN,
              ms=7, ls="none", label="comb, largest single cluster")
    ax.set_xlabel("tracks per event $T$")
    ax.set_ylabel("two-qubit (CX) gates per shot")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("(b) measured gate cost per shot", loc="left", fontsize=11)
    save(fig, "resource_comparison")


def fig_occproofcrop():
    """Strip the baked suptitle and provenance footer from the occupancy
    proof (the panels themselves are untouched study output)."""
    from PIL import Image
    src = ("/data/bfys/gscriven/Quantum_Track_Reconstruction/"
           "Toy_Characterisation/Bifurification/figures/dp_occupancy_proof.png")
    im = Image.open(src)
    w, h = im.size
    im2 = im.crop((0, int(0.030 * h), w, int(0.972 * h)))
    im2.save(FIG / "dp_occupancy_proof.png")
    print(f"[saved] figures/dp_occupancy_proof.png (cropped from {src})")


# ─────────────────────────────────────────────────────────────────────────────
# Heavy-data figures (T=200 heavy rep 0 and clean T=400)
# ─────────────────────────────────────────────────────────────────────────────

def _load_04():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fit04", "/data/bfys/gscriven/Quantum_Track_Reconstruction/QSVT/"
        "Codesign/04_fit_comb_to_measured_spectrum.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _heavy_system(m4, rep=0, beta=0.0, gmode="ref"):
    from dp_matrix_characterisation import NOISES
    from lhcb_velo_toy.analysis import compute_epsilon
    nz = NOISES["heavy"]
    eps = float(compute_epsilon(nz["sigma_res"], nz["sigma_scatt"]))
    return m4.build_system(rep, beta, gmode, eps, nz), eps, nz


def fig_tangles():
    """§6: what the heavy-noise compatibility graph actually looks like."""
    import qtrk_pipeline as qp
    from dp_matrix_characterisation import NOISES
    from lhcb_velo_toy.analysis import compute_epsilon
    from scipy.sparse.csgraph import connected_components

    nz = NOISES["heavy"]
    eps = float(compute_epsilon(nz["sigma_res"], nz["sigma_scatt"]))
    ev, _ = qp.ensure_event(n_trk=200, rep=0, **nz)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                               gamma=3.0, delta=1.0)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    seg_hits = np.asarray(ham._segment_to_hit_ids)
    A = ham.A.tocsr()
    Off = A.copy()
    Off.setdiag(0)
    Off.eliminate_zeros()
    ncomp, lab = connected_components(Off, directed=False)
    sizes = np.bincount(lab)
    nontriv = sizes[sizes > 1]

    hid = sorted({int(h) for h in seg_hits.ravel()})
    hz = {h: ev.get_hit_by_id(h).z for h in hid}
    hx = {h: ev.get_hit_by_id(h).x for h in hid}

    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(ncomp + 1))

    def comp_segs(c):
        return order[bounds[c]:bounds[c + 1]]

    cands = np.flatnonzero(sizes > 1)
    n_true_in = np.array([truth[comp_segs(c)].sum() for c in cands])
    n_tot = sizes[cands]
    biggest = cands[np.argmax(n_tot)]
    pure_false_pair = cands[(n_tot == 2) & (n_true_in == 0)][0]
    pure_false_tri = cands[(n_tot == 3) & (n_true_in == 0)][0]
    mixed = cands[(n_true_in >= 2) & (n_tot - n_true_in >= 2)]
    mixed_pick = mixed[np.argmin(sizes[mixed])] if len(mixed) else biggest

    fig = plt.figure(figsize=(12.6, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(nontriv, bins=np.arange(1.5, nontriv.max() + 1.5), color=BLUE,
            alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("segments per connected component")
    ax.set_ylabel("components")
    ax.set_title(f"(a) {len(nontriv)} small components, largest "
                 f"{nontriv.max()};\n{int((sizes[lab] == 1).sum()):,} "
                 "isolated segments besides", loc="left", fontsize=10.5)

    panels = [(pure_false_pair, "(b) pure-false pair"),
              (pure_false_tri, "(c) pure-false triple"),
              (mixed_pick, "(d) contaminated true chain"),
              (biggest, "(e) the largest tangle")]
    slots = [gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1]]
    for (c, title), slot in zip(panels, slots):
        ax = fig.add_subplot(slot)
        segs = comp_segs(c)
        hh = sorted({int(h) for h in seg_hits[segs].ravel()})
        zs = np.array([hz[h] for h in hh])
        xs_ = np.array([hx[h] for h in hh])
        coef = np.polyfit(zs, xs_, 1)
        detr = {h: (hx[h] - np.polyval(coef, hz[h])) * 1e3 for h in hh}
        for si in segs:
            h1, h2 = seg_hits[si]
            col = GREEN if truth[si] else RED
            ax.plot([hz[h1], hz[h2]], [detr[h1], detr[h2]], "-", color=col,
                    lw=2.0, alpha=0.85)
        ax.plot([hz[h] for h in hh], [detr[h] for h in hh], "o",
                color="#33322e", ms=4.5)
        nt = int(truth[segs].sum())
        ax.set_title(f"{title}  ({len(segs)} segments, {nt} true)",
                     loc="left", fontsize=10.5)
        ax.set_xlabel("z  [mm]")
        ax.set_ylabel("x $-$ straight-line trend  [$\\mu$m]")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(0.02, 0.92,
            "green: true segments\nred: false segments\ndots: hits\n\n"
            "x is shown relative to the straight line\n"
            "through each component's hits: the\n"
            "$\\mu$m-scale kinks are what the angular\n"
            "acceptance $\\varepsilon = 6.3$ mrad admits",
            fontsize=10.5, va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f4f3ef", ec="#c9c7bf"))
    save(fig, "tangle_gallery")


def fig_pileup(m4=None):
    """§6/§9 support: measured false pile-up on the motif lines (heavy rep 0)."""
    m4 = m4 or _load_04()
    sysd_pack, eps, nz = _heavy_system(m4)
    sysd = sysd_pack
    mt = m4.mode_table(sysd)
    s_p = sysd["s_p"]
    edges = np.arange(sysd["lam_min"] - 0.02, sysd["lam_max"] + 0.04, 0.02)
    lam, wb, vt, vf = mt[:, 0], mt[:, 1], mt[:, 2], mt[:, 3]
    wsol = (wb / np.maximum(np.abs(lam), 1e-9)) ** 2

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.hist(lam, bins=edges, weights=wsol * vt, histtype="stepfilled",
            alpha=0.5, color=GREEN, label="true formation weight")
    ax.hist(lam, bins=edges, weights=wsol * vf, histtype="stepfilled",
            alpha=0.55, color=RED, label="false formation weight")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)
    ax.set_xlabel("eigenvalue $\\lambda$")
    ax.set_ylabel("solution weight per mode  [log]")
    for l4 in s_p - 2 * np.cos(np.arange(1, 5) * np.pi / 5):
        ax.axvline(l4, color=GREEN, lw=0.9, ls=":")
    for lf, nm in [(s_p - 1, "pair line $s-1$"),
                   (s_p - np.sqrt(2), "triple line $s-\\sqrt{2}$"),
                   (s_p + 1, None), (s_p + np.sqrt(2), None)]:
        ax.axvline(lf, color=RED, lw=0.9, ls="--")
        if nm:
            ax.text(lf - 0.05, 2e2, nm, fontsize=9.5, color="#a83232",
                    rotation=90, va="top", ha="right")
    ax.axvline(s_p, color=GREY, lw=1.2)
    ax.text(s_p + 0.04, 2e2, "isolated line $s$", fontsize=9.5, color=GREY,
            rotation=90, va="top")
    ax.legend(loc="upper right")
    save(fig, "measured_pileup")


def fig_threshold():
    """Item 7: the efficiency / false-rate trade-off against the threshold.

    Uses the STORED T=400 rep-0 solutions of the canonical fixed-acceptance
    campaign (same store rows as the headline figures), so every curve here is
    consistent with the paper's quoted numbers.
    """
    import pandas as pd
    import qtrk_pipeline as qp
    from qtrk_pipeline.metrics import rescale_to_signal

    ev, _ = qp.ensure_event(n_trk=400, rep=0, sigma_scatt=1e-4, sigma_res=0.0,
                            phi_max=0.2, hit_ineff=0.0)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    man = pd.read_csv(os.environ["QTRK_STORE"] + "/manifest/solutions.csv")
    sub = man[(man.n_trk == 400) & (man.rep == 0) & (man.gamma == 3.0) &
              (man.hit_ineff == 0.0) & (man.epsilon == 0.002)]

    def stored(solver):
        for k in sub[sub.solver == solver].sol_key:
            if qp.solution_exists(k):
                return np.abs(np.asarray(qp.load_solution(k)["sol"],
                                         float))[:len(truth)]
        raise RuntimeError(f"no stored {solver} solution")

    x_cls = stored("classical")
    x_1b = rescale_to_signal(stored("quantum"), x_cls, 0.35)
    x_comb = rescale_to_signal(stored("qsvt"), x_cls, 0.35)
    taus = np.linspace(0.02, 0.55, 240)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                             constrained_layout=True)
    for x, col, lab in [(x_cls, "#33322e", "classical"),
                        (x_1b, RED, "1BQF"),
                        (x_comb, BLUE, "line comb")]:
        eff = [(x[truth] > t).mean() for t in taus]
        far = [((x > t) & ~truth).sum() / max(int((x > t).sum()), 1)
               for t in taus]
        tau99 = float(np.quantile(x[truth], 0.01)) - 1e-12
        axes[0].plot(taus, eff, "-", color=col, lw=2.0, label=lab)
        axes[1].plot(taus, far, "-", color=col, lw=2.0, label=lab)
        for ax in axes:
            ax.axvline(tau99, color=col, lw=1.0, ls="--", alpha=0.6)
    for ax, ylab, ttl in [
            (axes[0], "segment efficiency",
             "(a) efficiency against the threshold"),
            (axes[1], "segment false rate",
             "(b) false rate against the threshold")]:
        ax.axvline(0.35, color=GREY, lw=1.6)
        ax.text(0.353, 0.60, "fixed $\\tau=0.35$", rotation=90, fontsize=9.5,
                color=GREY, va="center",
                transform=ax.get_xaxis_transform())
        ax.set_xlabel("threshold $\\tau$")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, loc="left", fontsize=11)
    axes[1].set_yscale("log")
    axes[0].legend(loc="center left", fontsize=9.5)
    axes[1].text(0.02, 0.02,
                 "dashed verticals: each solver's efficiency-first\n"
                 "threshold (99% of true segments kept)", fontsize=9,
                 transform=axes[1].transAxes)
    save(fig, "threshold_tradeoff")


def fig_fitpaper():
    """Paper version of the fit figure: no footer, no unfitted-comb series."""
    import pandas as pd
    m4 = _load_04()
    sysd, eps, nz = _heavy_system(m4)
    truth, n, s_p = sysd["truth"], sysd["n"], sysd["s_p"]
    edges = np.arange(sysd["lam_min"] - m4.BIN_W,
                      sysd["lam_max"] + 2 * m4.BIN_W, m4.BIN_W)
    nb = len(edges) - 1
    x_cls = m4.amplitudes(sysd, lambda l: 1.0 / l)
    x_1b = m4.amplitudes(sysd, lambda l: np.abs(np.cos(np.pi * l / (2 * s_p))))

    D, seg_rows, tru_rows = m4.design_matrix(sysd, edges)
    y_true = x_cls[seg_rows[tru_rows]]
    DT, DF = D[tru_rows], D[~tru_rows]
    iso_row = np.zeros(nb)
    iso_row[np.clip(np.searchsorted(edges, s_p) - 1, 0, nb - 1)] = m4.DELTA
    wiso = np.sqrt(float(np.count_nonzero(sysd["iso"] & ~truth))) / 30.0
    DF_full = np.vstack([DF, wiso * iso_row])
    GF, GT = DF_full.T @ DF_full, DT.T @ DT
    bT = DT.T @ y_true
    reg = 1e-8 * np.eye(nb)
    best = None
    for mu in m4.MUS:
        p_b = np.linalg.solve(GF + mu * GT + reg, mu * bT)
        p_b = p_b / max(np.abs(p_b).max(), 1e-12)
        x_o = m4.amplitudes(sysd, m4.binned_response(p_b, edges))
        fe = m4.far_at_eff(x_o, truth, (0.985,))[0.985]
        if best is None or fe[1] < best[0]:
            best = (fe[1], mu, p_b, x_o)
    _, mu_b, p_opt, x_opt = best
    mt = m4.mode_table(sysd)

    df = pd.read_csv("/data/bfys/gscriven/Quantum_Track_Reconstruction/QSVT/"
                     "Codesign/outputs/fork_noisy/fit_comb.csv")
    dfo = pd.read_csv("/data/bfys/gscriven/Quantum_Track_Reconstruction/QSVT/"
                      "Codesign/outputs/fork_noisy/occupancy_fitted_check.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6),
                             constrained_layout=True)

    ax = axes[0, 0]
    lam, wb, vt, vf = mt[:, 0], mt[:, 1], mt[:, 2], mt[:, 3]
    wsol = (wb / np.maximum(np.abs(lam), 1e-9)) ** 2
    ax.hist(lam, bins=edges, weights=wsol * vt, histtype="stepfilled",
            alpha=0.5, color=GREEN, label="true formation weight")
    ax.hist(lam, bins=edges, weights=wsol * vf, histtype="stepfilled",
            alpha=0.55, color=RED, label="false formation weight")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)
    ax.set_ylabel("solution weight per mode  [log]")
    ax.set_xlabel("eigenvalue $\\lambda$")
    for l4 in s_p - 2 * np.cos(np.arange(1, 5) * np.pi / 5):
        ax.axvline(l4, color=GREEN, lw=0.7, ls=":")
    for lf in (s_p - 1, s_p - np.sqrt(2), s_p + 1, s_p + np.sqrt(2)):
        ax.axvline(lf, color=RED, lw=0.7, ls="--")
    ax2 = ax.twinx()
    lg = np.linspace(edges[0], edges[-1], 1500)
    ax2.plot(lg, np.abs(m4.binned_response(p_opt, edges)(lg)),
             color=BLUE, lw=1.8, label="fitted response")
    ax2.set_ylim(0, 1.12)
    ax2.grid(False)
    ax2.set_ylabel("|fitted response| (normalised)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    ax.set_title("(a) the measurement: false weight piles up on the motif "
                 "lines;\nthe fitted response puts its roots there",
                 loc="left", fontsize=11)

    ax = axes[0, 1]
    for xv, cc, lab, lw in [(x_cls, "#33322e", "classical", 1.8),
                            (x_1b, RED, "1BQF", 1.4),
                            (x_opt, BLUE, "fitted response", 2.2)]:
        e_, f_ = m4.roc(xv, truth)
        ax.plot(np.maximum(f_, 2e-3), e_, "-", color=cc, lw=lw, label=lab)
    for tgt, ls in [(0.99, ":"), (0.985, "--")]:
        ax.axhline(tgt, color="#79776f", lw=0.8, ls=ls)
    ax.set_xscale("log")
    ax.set_ylim(0.90, 1.005)
    ax.set_xlabel("false rate")
    ax.set_ylabel("segment efficiency")
    ax.legend(loc="lower right")
    ax.set_title("(b) the frontier after fitting: the fragment floor bites\n"
                 "only above efficiency $\\approx 0.99$", loc="left",
                 fontsize=11)

    ax = axes[1, 0]
    order = ["beta0", "b0.5-gstar", "b1-gstar"]
    names = ["no fork ($\\beta=0$)", "fork $\\beta=0.5$ ($\\gamma^*$)",
             "fork $\\beta=1$ ($\\gamma^*$)"]
    agg = df.groupby("config").mean(numeric_only=True).reindex(order)
    series = [("far_cls_e985", "#33322e", "classical"),
              ("far_1b_e985", RED, "1BQF"),
              ("far_opt_e985", BLUE, "fitted response")]
    width = 0.24
    for k, (col, cc, lab) in enumerate(series):
        ax.bar(np.arange(len(order)) + (k - 1.0) * width, agg[col],
               width - 0.02, color=cc, label=lab)
    ax.set_xticks(np.arange(len(order)), names, fontsize=10)
    ax.set_ylabel("false rate at matched efficiency 0.985")
    ax.legend(fontsize=9)
    ax.set_title("(c) matched-efficiency comparison below the floor\n"
                 "(mean over three events)", loc="left", fontsize=11)

    ax = axes[1, 1]
    d0 = df[df.config == "beta0"]
    base = dfo[dfo.system == "base"]
    degs = [40, 80, 160, 240, 480, 960]
    ax.semilogy(degs, [base[f"far_fit{d}_e985"].mean() for d in degs], "o-",
                color=BLUE, label="directly fitted degree-$d$ response")
    ax.axhline(d0.far_opt_e985.mean(), color=GREEN, ls="--",
               label="binned (unlimited-degree) response")
    ax.axhline(d0.far_cls_e985.mean(), color="#33322e", ls=":",
               label="classical")
    gen = d0[d0.rep > 0]
    if len(gen) and "far_gen_e985" in gen:
        ax.axhline(gen.far_gen_e985.mean(), color=ORANGE, ls="-.",
                   label="event-0 fit applied to events 1–2")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("false rate at matched efficiency 0.985  [log]")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.set_title("(d) realisability (direct degree-$d$ fit) and "
                 "generalisation,\nno-fork system", loc="left", fontsize=11)
    save(fig, "fit_comb_paper")


def fig_clean2x2():
    """One readable headline figure replacing the two side-by-side 2x2 grids."""
    sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/"
                    "Toy_Characterisation/Segment_level_studies")
    import seg_store as Sst
    M = Sst.fixed_eps_metrics()
    M = M[M["n_trk"] <= 1000]
    solvers = [("classical", "#1b7837", "o", "classical (fixed $\\tau=0.35$)"),
               ("qsvt", "#6a3d9a", "D", "line comb (fixed $\\tau=0.35$)"),
               ("quantum", "#d6604d", "*",
                "1BQF (efficiency-first threshold)")]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8),
                             constrained_layout=True)
    for k, col, mk, lab in solvers:
        d = Sst.agg_by_ntrk(M, k, 3.0, 0.0)
        if not d:
            continue
        if k == "quantum" and "se_wp_m" in d:
            se, see, fr, fre = (d["se_wp_m"], d["se_wp_e"],
                                d["fr_wp_m"], d["fr_wp_e"])
            axes[0].plot(d["tc"], d["se_m"], mk + "--", color=col, alpha=0.30,
                         ms=5, label="1BQF (fixed $\\tau$: endpoint cut)")
            axes[1].plot(d["tc"], d["fr_m"], mk + "--", color=col, alpha=0.30,
                         ms=5)
        else:
            se, see, fr, fre = d["se_m"], d["se_e"], d["fr_m"], d["fr_e"]
        axes[0].errorbar(d["tc"], se, yerr=see, fmt=mk + "-", color=col,
                         capsize=3, mec="k", mew=0.5, label=lab,
                         ms=9 if mk == "*" else 6)
        axes[1].errorbar(d["tc"], fr, yerr=fre, fmt=mk + "-", color=col,
                         capsize=3, mec="k", mew=0.5, label=lab,
                         ms=9 if mk == "*" else 6)
    axes[0].set_ylabel("segment efficiency  [%]")
    axes[0].set_ylim(60, 102)
    axes[0].axhline(100, color=GREY, lw=0.7, ls="--")
    axes[1].set_ylabel("segment false rate  [%]")
    for ax in axes:
        ax.set_xlabel("tracks per event $T$")
    axes[0].legend(fontsize=9.5, loc="lower left")
    axes[0].set_title("(a) segment efficiency", loc="left", fontsize=11)
    axes[1].set_title("(b) segment false rate", loc="left", fontsize=11)
    save(fig, "clean_headline")


def fig_occfit():
    """§7 correction: the occupancy no-go dissolves under a fitted response."""
    import pandas as pd
    dfo = pd.read_csv("/data/bfys/gscriven/Quantum_Track_Reconstruction/QSVT/"
                      "Codesign/outputs/fork_noisy/occupancy_fitted_check.csv")
    degs = [40, 80, 160, 240, 480, 960]
    dfo["far_fitbest_e985"] = dfo[[f"far_fit{d}_e985" for d in degs]].min(axis=1)
    dfo["far_fit40_"] = dfo["far_fit40_e985"]
    order = ["base", "occ_a0.05", "occ_a0.3"]
    names = ["base $A$", "$A_{\\rm occ}$, $\\alpha=0.05$",
             "$A_{\\rm occ}$, $\\alpha=0.3$"]
    agg = dfo.groupby("system").mean(numeric_only=True).reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                             constrained_layout=True)
    ax = axes[0]
    w = 0.27
    xs = np.arange(len(order))
    ax.bar(xs - w, agg.far_cls_e985, w * 0.9, color="#33322e",
           label="classical")
    ax.bar(xs, agg.far_fit40_e985, w * 0.9, color="#7fb0e0",
           label="fitted response, degree 40")
    ax.bar(xs + w, agg.far_fitbest_e985, w * 0.9, color=BLUE,
           label="fitted response, best degree $\\leq 960$")
    ax.set_xticks(xs, names)
    ax.set_ylabel("false rate at matched efficiency 0.985")
    ax.legend(fontsize=9.5)
    ax.set_title("(a) the occupancy term helps a fitted response\n"
                 "(mean over three events, heavy noise, $T=200$)",
                 loc="left", fontsize=11)

    ax = axes[1]
    reps = sorted(dfo["rep"].unique())
    w = 0.38
    for k, (sysname, col, lab) in enumerate(
            [("base", "#33322e", "base $A$"),
             ("occ_a0.05", BLUE, "$A_{\\rm occ}$, $\\alpha=0.05$")]):
        d = dfo[dfo.system == sysname].set_index("rep")
        ax.bar(np.array(reps) + (k - 0.5) * w,
               d.loc[reps, "far_fitbest_e985"], w * 0.92, color=col,
               label=f"fitted, {lab}")
    ax.set_xticks(reps, [f"event {r}" for r in reps])
    ax.set_ylabel("false rate at matched efficiency 0.985")
    ax.legend(fontsize=9.5)
    ax.set_title("(b) event 1 is fragment-floor-bound on the base system;\n"
                 "the occupancy coupling dissolves the floor", loc="left",
                 fontsize=11)
    save(fig, "occupancy_fitted")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cmds = sys.argv[1:] or ["all_fast"]
    if "all_fast" in cmds:
        cmds = ["motifs", "twin", "responses", "readout", "erf", "run3crop"]
    if "heavy" in cmds:
        i = cmds.index("heavy")
        cmds[i:i + 1] = ["tangles", "pileup", "threshold", "fitpaper"]
    for c in cmds:
        {"motifs": fig_motifs, "twin": fig_twin, "responses": fig_responses,
         "readout": fig_readout, "erf": fig_erf, "run3crop": fig_run3crop,
         "tangles": fig_tangles, "pileup": fig_pileup,
         "threshold": fig_threshold, "fitpaper": fig_fitpaper,
         "occfit": fig_occfit, "clean2x2": fig_clean2x2,
         "occproofcrop": fig_occproofcrop,
         "rescompare": fig_resources_compare}[c]()
