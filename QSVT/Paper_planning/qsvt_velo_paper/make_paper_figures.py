#!/usr/bin/env python3
"""Paper figures for 'From one bit to a fitted polynomial' (review pass, 2026-07-19;
convention pass 2026-09-02).

Conventions (George 2026-09-02): every solver, the classical inversion included,
is read at its efficiency-first working point — the lowest threshold that keeps
99% of the true segments.  The fixed absolute threshold tau = 0.35 was a bug and
is drawn nowhere.  The QSVT is one solver; its series are labelled by polynomial
degree and by the Hamiltonian the problem is set up on.

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
            label="QSVT polynomial $|p(\\lambda)|$, degree 40 (line comb, half-width 0.18)")
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
    ax.set_title("(b) the QSVT polynomial: a zero (or a pass) wherever the spectrum needs one",
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
              label="measured, QSVT ($d=40$)")
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
              label="measured amplified walk calls (QSVT, $d=40$)")
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


def fig_epsilon():
    """Section 2: where the acceptance formula comes from.

    (a) the geometry of one measured kink: a scattering kick at the shared
        plane plus three independent hit smears (schematic, not to scale);
    (b) the resulting 3D kink is Rayleigh-distributed, measured on real
        generated events against the closed form, with the acceptance drawn.

    Derivation and Monte-Carlo validation:
    Toy_Characterisation/Epsilon_study_2/{theory.md,gen_epsilon_derivation_figs.py}.
    """
    sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/"
                    "Toy_Characterisation/_shared")
    import helpers
    from lhcb_velo_toy.analysis.segment_metrics import (
        compute_epsilon, DEFAULT_DZ, DEFAULT_SCALE)

    DZ, SC = DEFAULT_DZ, DEFAULT_SCALE
    SIG_R, SIG_S = 0.02, 1e-4                      # heavy configuration
    geom = helpers.make_geometry()

    def true_kinks(event):
        pos = {h.hit_id: np.array([h.x, h.y, h.z]) for h in event.hits}
        out = []
        for trk in event.tracks:
            ids = trk.hit_ids
            for k in range(1, len(ids) - 1):
                v1 = pos[ids[k]] - pos[ids[k - 1]]
                v2 = pos[ids[k + 1]] - pos[ids[k]]
                c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                out.append(np.arccos(np.clip(c, -1.0, 1.0)))
        return out

    kinks = []
    for rep in range(6):
        ev = helpers.safe_generate(200, seed=90_000 + rep, geom=geom,
                                   measurement_error=SIG_R,
                                   collision_noise=SIG_S)
        kinks += true_kinks(ev)
    kinks = np.asarray(kinks)
    sig_p = np.sqrt(SIG_S ** 2 + 6.0 * (SIG_R / DZ) ** 2)
    eps = compute_epsilon(SIG_R, SIG_S)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))

    # ---- (a) schematic -----------------------------------------------------
    ax = axes[0]
    for xz, lab in ((0, "plane $k{-}1$"), (1, "plane $k$"), (2, "plane $k{+}1$")):
        ax.axvline(xz, color="0.82", lw=7, zorder=0)
        ax.text(xz, 1.42, lab, ha="center", fontsize=10)
    ax.text(0.5, 1.28, r"$\Delta z$", ha="center", fontsize=10, color="0.4")
    ax.annotate("", xy=(1.0, 1.24), xytext=(0.0, 1.24),
                arrowprops=dict(arrowstyle="<->", color="0.4", lw=1.0))
    ax.plot([-0.35, 2.3], [-0.14, 0.92], ls="--", color="0.45", lw=1.4,
            label="undisturbed straight trajectory")
    x_tr = np.array([-0.35, 0, 1, 2, 2.3])
    y_tr = np.array([-0.14, 0.0, 0.40, 1.05, 1.245])
    ax.plot(x_tr[:3], y_tr[:3], color=BLUE, lw=2.2,
            label="true trajectory (scattering kick at each plane)")
    ax.plot(x_tr[2:], y_tr[2:], color=BLUE, lw=2.2)
    true_pts = [(0, 0.0), (1, 0.40), (2, 1.05)]
    meas_pts = [(0, 0.10), (1, 0.33), (2, 1.13)]
    for (zt, yt), (zm, ym) in zip(true_pts, meas_pts):
        ax.plot(zt, yt, "o", color=BLUE, ms=7, zorder=5)
        ax.plot(zm, ym, "s", color=RED, ms=8, zorder=6)
        ax.annotate("", xy=(zm, ym), xytext=(zt, yt),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.plot([], [], "o", color=BLUE, label="true crossing point")
    ax.plot([], [], "s", color=RED,
            label=r"recorded hit ($\pm\sigma_{\rm res}$ in $x$ and $y$)")
    ax.plot([p[0] for p in meas_pts], [p[1] for p in meas_pts], color=RED,
            lw=2.2, label=r"measured segments, kink $\theta$ at plane $k$")
    ax.annotate(r"$\theta$", xy=(1.24, 0.60), fontsize=13, color=RED)
    ax.annotate(r"scattering kick $\sim\mathcal{N}(0,\sigma_{\rm scatt}^2)$"
                "\nper projection, at the plane",
                xy=(1.02, 0.42), xytext=(0.38, 0.86), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))
    ax.set_xlim(-0.5, 2.75); ax.set_ylim(-0.35, 1.62)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8.2, frameon=True)
    ax.set_title("(a) one measured kink: one scattering kick, three hit "
                 "smears\n(schematic, angles hugely exaggerated)",
                 loc="left", fontsize=11)

    # ---- (b) Rayleigh ------------------------------------------------------
    ax = axes[1]
    hi = max(eps * 1.25, np.quantile(kinks, 0.999))
    ax.hist(kinks, bins=70, range=(0, hi), density=True, color="#9ecae1",
            edgecolor="none", label=f"measured true kinks ({len(kinks)})")
    g = np.linspace(0, hi, 600)
    ax.plot(g, g / sig_p ** 2 * np.exp(-g ** 2 / (2 * sig_p ** 2)), color="#33322e",
            lw=2.0,
            label=r"Rayleigh($\sigma_p$), $\sigma_p^2=\sigma_{\rm scatt}^2"
                  r"+6\sigma_{\rm res}^2/\Delta z^2$")
    ax.axvline(eps, color=GREEN, lw=2.0,
               label=rf"$\varepsilon=3\sqrt{{2}}\,\sigma_p={eps*1e3:.2f}$ mrad"
                     "\n" + r"(misses $e^{-9}=1.2\times10^{-4}$ of true kinks)")
    ax.set_xlabel(r"3D kink angle $\theta$ at the shared hit [rad]")
    ax.set_ylabel("density")
    ax.set_xlim(0, hi)
    ax.legend(fontsize=8.4, loc="upper right")
    ax.set_title("(b) the kink is Rayleigh, and its tail sets the acceptance",
                 loc="left", fontsize=11)
    print(f"    [eps fig] sigma_p={sig_p:.3e}  eps={eps:.4e} "
          f"({eps*1e3:.3f} mrad)  measured mean={kinks.mean():.4e}  "
          f"Rayleigh mean={sig_p*np.sqrt(np.pi/2):.4e}  "
          f"fraction above eps={float((kinks>eps).mean()):.2e}")
    save(fig, "epsilon_derivation")


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
    HHL (structure-level estimate band), the 1BQF, and the QSVT (DSS-QETU)."""
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
            label="QSVT, dense-dilation LCU (simulation only)")
    ax.plot(gw["T"], gw["BQF_1"], "o-", color=RED, lw=1.8, ms=6,
            label="1BQF")
    ax.plot(gw["T"], gw["DSS_QETU"], "s--", color=BLUE, lw=1.8, ms=6,
            label="QSVT (DSS--QETU)")
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
              label="QSVT, global ($d=40$ calls)")
    ax.loglog(gg["T"], gg["comb_percluster_max_instance"], "D", color=GREEN,
              ms=7, ls="none", label="QSVT, largest single cluster")
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
    # Put the quantum amplitudes on the classical scale by norm-matching on the
    # classical solver's OWN efficiency-first support (the segments it keeps at
    # its 99% working point).  The earlier anchor was the classical support at
    # the fixed cut 0.35 -- the retired convention leaking into a plotting scale.
    # Quantile metrics are unchanged by any rescale; only the horizontal
    # placement of the curves depends on it.
    from lhcb_velo_toy.analysis.working_points import working_point_threshold
    tau_cls_wp = working_point_threshold(x_cls, truth, 0.99)
    x_1b = rescale_to_signal(stored("quantum"), x_cls, tau_cls_wp)
    x_comb = rescale_to_signal(stored("qsvt"), x_cls, tau_cls_wp)
    taus = np.linspace(0.02, 0.55, 240)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                             constrained_layout=True)
    for x, col, lab in [(x_cls, "#33322e", "classical"),
                        (x_1b, RED, "1BQF (package circuit, stored)"),
                        (x_comb, BLUE, "QSVT ($d=40$)")]:
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
        ax.set_xlabel("threshold $\\tau$")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, loc="left", fontsize=11)
    axes[1].set_yscale("log")
    axes[0].legend(loc="center left", fontsize=9.5)
    axes[1].text(0.02, 0.02,
                 "dashed verticals: each solver's efficiency-first\n"
                 "working point (the lowest threshold keeping 99%\n"
                 "of its true segments); quantum amplitudes norm-matched\n"
                 "to the classical solver on its own working-point support",
                 fontsize=8.6, transform=axes[1].transAxes)
    save(fig, "threshold_tradeoff")


def fig_fitpaper():
    """Paper version of the fit figure: no footer; QSVT series only (no line-comb-coefficient series)."""
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
             color=BLUE, lw=1.8, label="QSVT polynomial (fitted)")
    ax2.set_ylim(0, 1.12)
    ax2.grid(False)
    ax2.set_ylabel("|QSVT polynomial| (normalised)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    ax.set_title("(a) the measurement: false weight piles up on the motif "
                 "lines;\nthe fitted QSVT polynomial puts its roots there",
                 loc="left", fontsize=11)

    ax = axes[0, 1]
    for xv, cc, lab, lw in [(x_cls, "#33322e", "classical", 1.8),
                            (x_1b, RED, "1BQF", 1.4),
                            (x_opt, BLUE, "QSVT (fitted polynomial)", 2.2)]:
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
              ("far_opt_e985", BLUE, "QSVT (fitted polynomial)")]
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
                color=BLUE, label="QSVT, degree $d$ (coefficients fit directly)")
    ax.axhline(d0.far_opt_e985.mean(), color=GREEN, ls="--",
               label="QSVT, unlimited degree (binned fit)")
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
    """The headline figure: every solver at its efficiency-first working point
    (George 2026-09-02: the fixed threshold was a bug; nothing is drawn at it)."""
    sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/"
                    "Toy_Characterisation/Segment_level_studies")
    import seg_store as Sst
    M = Sst.fixed_eps_metrics()
    M = M[M["n_trk"] <= 1000]
    solvers = [("classical", "#1b7837", "o", "classical"),
               ("qsvt", "#6a3d9a", "D", "QSVT ($d=40$, base $A$)"),
               ("quantum", "#d6604d", "*", "1BQF")]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8),
                             constrained_layout=True)
    for k, col, mk, lab in solvers:
        d = Sst.agg_by_ntrk(M, k, 3.0, 0.0)
        if not d or "se_wp_m" not in d:
            continue
        se, see, fr, fre = (d["se_wp_m"], d["se_wp_e"],
                            d["fr_wp_m"], d["fr_wp_e"])
        axes[0].errorbar(d["tc"], se, yerr=see, fmt=mk + "-", color=col,
                         capsize=3, mec="k", mew=0.5, label=lab,
                         ms=9 if mk == "*" else 6)
        axes[1].errorbar(d["tc"], np.maximum(fr, 1e-3), yerr=fre,
                         fmt=mk + "-", color=col, capsize=3, mec="k", mew=0.5,
                         label=lab, ms=9 if mk == "*" else 6)
    axes[0].set_ylabel("segment efficiency  [%]")
    axes[0].set_ylim(90, 100.6)
    axes[0].axhline(100, color=GREY, lw=0.7, ls="--")
    axes[0].axhline(99, color=GREY, lw=0.7, ls=":")
    axes[0].text(10.5, 99.08, "99% of true segments kept", fontsize=8.5,
                 color="#55534d")
    axes[1].set_ylabel("segment false rate  [%]")
    axes[1].set_yscale("log")
    axes[1].set_ylim(7e-4, 130)
    for ax in axes:
        ax.set_xlabel("tracks per event $T$")
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100, 200, 400, 700, 1000])
        ax.set_xticklabels(["10", "20", "50", "100", "200", "400", "700",
                            "1000"])
    axes[0].legend(fontsize=9.5, loc="lower left")
    axes[0].set_title("(a) segment efficiency", loc="left", fontsize=11)
    axes[1].set_title("(b) segment false rate (a measured 0% is drawn at "
                      "$10^{-3}$)", loc="left", fontsize=11)
    save(fig, "clean_headline")



def fig_hhl_onebit():
    """Section 3.5: HHL, the 1BQF, the classical solver and the QSVT on the
    figure-12 events and axes, every solver at its efficiency-first working
    point.  HHL is the exact finite-clock emulation of
    QSVT/Paper_planning/section3_onebit/01_hhl_segment_metrics.py (validated
    against a correctly wired circuit to 1e-12); the package's own HHL circuit
    is not the comparator (to-do)."""
    import pandas as pd
    src = (HERE.parent / "section3_onebit" / "outputs" / "01_hhl_metrics.csv")
    d = pd.read_csv(src)
    d["far"] = d["far"].astype(float); d["eff"] = d["eff"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)

    def series(sub, label, col, mk, ls="-", ms=6, z=3, hollow=False):
        g = sub.groupby("T").agg(eff=("eff", "mean"), effe=("eff", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0),
                                 far=("far", "mean"), fare=("far", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)).reset_index()
        kw = dict(mfc="white", mec=col, mew=1.6) if hollow else dict(mec="k", mew=0.5)
        axes[0].errorbar(g["T"], 100 * g["eff"], yerr=100 * g["effe"], fmt=mk + ls, color=col, capsize=3,
                         ms=ms, label=label, zorder=z, **kw)
        axes[1].errorbar(g["T"], 100 * np.maximum(g["far"], 1e-5), yerr=100 * g["fare"], fmt=mk + ls, color=col,
                         capsize=3, ms=ms, label=label, zorder=z, **kw)

    series(d[d.solver == "classical"], "classical", "#1b7837", "o", ms=11, z=4, hollow=True)
    series(d[d.solver == "qsvt"], "QSVT ($d=40$, base $A$)", "#6a3d9a", "D")
    series(d[d.solver == "quantum"], "1BQF (package circuit, stored)", "#d6604d", "*", ms=9)
    hh = d[d.solver == "hhl"]
    for n_c, col, ls in ((4, "#9ecae1", ":"), (6, "#4292c6", "--"), (8, "#08306b", "-")):
        series(hh[(hh.n_c == n_c) & (hh.gain == "textbook")], f"HHL, {n_c}-bit clock, $C=\\lambda_{{\\min}}$", col, "s", ls=ls, ms=5, z=2)
    series(hh[(hh.n_c == 8) & (hh.gain == "package")], "HHL, 8-bit clock, package gain (saturated)", "#bdbdbd", "v", ls="-", ms=5, z=1)

    axes[0].set_ylabel("segment efficiency  [%]")
    axes[0].axhline(100, color=GREY, lw=0.7, ls="--"); axes[0].axhline(99, color=GREY, lw=0.7, ls=":")
    axes[0].set_ylim(90, 100.6)
    axes[1].set_ylabel("segment false rate  [%]"); axes[1].set_yscale("log"); axes[1].set_ylim(7e-4, 130)
    for ax in axes:
        ax.set_xlabel("tracks per event $T$"); ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100, 200, 400, 700, 1000]); ax.set_xticklabels(["10", "20", "50", "100", "200", "400", "700", "1000"])
    axes[0].legend(fontsize=8.6, loc="lower left")
    axes[0].set_title("(a) segment efficiency", loc="left", fontsize=11)
    axes[1].set_title("(b) segment false rate (a measured 0% is drawn at $10^{-3}$)", loc="left", fontsize=11)
    save(fig, "hhl_onebit_headline")


def fig_notch_spectrum():
    """Section 3.2: the one zero, and where the surviving errors live.  The
    kept probability decomposes exactly over eigenmodes; each mode's weight is
    split by its eigenvector into a true and a false share.  The FALSE share is
    histogrammed against lambda before the filter (classical inversion, weight
    beta^2/lambda^2) and after it (one-bit readout, weight beta^2 cos^2).
    Data: section3_onebit/02_surviving_false_spectrum.py."""
    import json
    D = json.loads((HERE.parent / "section3_onebit" / "outputs" / "02_surviving_false_spectrum.json").read_text())
    L = D["lines"]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.0), sharex=True, constrained_layout=True)
    rows = [("clean_T400", "clean event, $T=400$, fixed $\\varepsilon=2$ mrad"),
            ("heavy_T200", "heavy-noise event, $T=200$, $\\varepsilon=6.31$ mrad")]
    for r, (key, title) in enumerate(rows):
        E = D["events"][key]; edges = np.array(E["edges"]); c = 0.5 * (edges[1:] + edges[:-1]); w = edges[1] - edges[0]
        for k, (stage, lab) in enumerate((("before", "before the filter: classical inversion, weight $\\beta_j^2/\\lambda_j^2$"),
                                          ("after", "after the one-bit filter: kept weight $\\beta_j^2\\cos^2(\\lambda_j t/2)$"))):
            ax = axes[r, k]; H = E[stage]
            hf = np.array(H["false_hist"]); ht = np.array(H["true_hist"])
            tot = H["false_total"] + H["true_total"]
            ax.bar(c, hf / tot, width=w, color=RED, alpha=0.85, label="false segments' share", zorder=3)
            ax.bar(c, ht / tot, width=w, color=GREEN, alpha=0.55, label="true segments' share", zorder=2)
            for l in L["true"]: ax.axvline(l, color=GREEN, lw=0.8, ls=":", alpha=0.7)
            ax.axvline(S, color="#33322e", lw=1.0, ls="--", alpha=0.8)
            for l in L["pair"]: ax.axvline(l, color=RED, lw=0.7, ls=":", alpha=0.6)
            for l in L["triple"]: ax.axvline(l, color=ORANGE, lw=0.7, ls=":", alpha=0.6)
            ax.set_yscale("log"); ax.set_ylim(1e-6, 1.5); ax.set_xlim(0.0, 8.0)
            ax.set_title(("(%s) " % "abcd"[2 * r + k]) + lab, loc="left", fontsize=10)
            fam = H["false_by_family"]; ftot = max(H["false_total"], 1e-300)
            if stage == "before":
                txt = ("false weight on $\\lambda=s$: %.2f%%\nfalse : true weight  %.0f : 1"
                       % (100 * H["false_at_s"] / ftot, H["false_total"] / max(H["true_total"], 1e-300)))
            else:
                rest = 100 * (1 - (fam["pair"] + fam["triple"] + fam["hub"] + fam["true"]) / ftot)
                txt = ("false weight on $\\lambda=s$: 0  (%.0e)\nsurvivors: pairs %.0f%%, triples %.0f%%, hubs %.0f%%,\n"
                       "on the true lines %.1f%%, elsewhere %.0f%%\nfalse : true weight  %.2f : 1"
                       % (H["false_at_s"], 100 * fam["pair"] / ftot, 100 * fam["triple"] / ftot, 100 * fam["hub"] / ftot,
                          100 * fam["true"] / ftot, rest, H["false_total"] / max(H["true_total"], 1e-300)))
            ax.text(0.015, 0.97, txt, transform=ax.transAxes, fontsize=8.6, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c9c6bf", lw=0.6))
            if r == 1: ax.set_xlabel("eigenvalue $\\lambda$   ($s=\\gamma+\\delta=4$)")
            if k == 0: ax.set_ylabel("share of kept weight per 0.02 in $\\lambda$")
        axes[r, 0].text(-0.02, 1.16, title, transform=axes[r, 0].transAxes, fontsize=10.5, fontweight="bold", va="bottom")
    axes[0, 1].legend(fontsize=8.6, loc="upper right")
    axes[1, 0].text(S + 0.08, 4e-6, "$\\lambda=s$", fontsize=8.6, color="#33322e")
    save(fig, "notch_spectrum")

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
           label="QSVT, degree 40")
    ax.bar(xs + w, agg.far_fitbest_e985, w * 0.9, color=BLUE,
           label="QSVT, best degree $\\leq 960$")
    ax.set_xticks(xs, names)
    ax.set_ylabel("false rate at matched efficiency 0.985")
    ax.legend(fontsize=9.5)
    ax.set_title("(a) the occupancy term helps the QSVT\n"
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
               label=f"QSVT, {lab}")
    ax.set_xticks(reps, [f"event {r}" for r in reps])
    ax.set_ylabel("false rate at matched efficiency 0.985")
    ax.legend(fontsize=9.5)
    ax.set_title("(b) event 1 is fragment-floor-bound on the base system;\n"
                 "the occupancy coupling dissolves the floor", loc="left",
                 fontsize=11)
    save(fig, "occupancy_fitted")


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Section-2 graph figures (added 2026-09-02).
#
# Section 2 argues in graph theory throughout but never draws a graph: the
# figures jump from a picture of segments in the detector straight to a list of
# eigenvalues, with the object that actually HAS the eigenvalues never shown.
# These four figures supply the missing rungs, and the fifth is the Gershgorin
# picture behind the positive-definiteness paragraph.
#
# Bright vs dark modes: the bias b = delta*1 is invariant under every
# automorphism of a cluster's graph, so only eigenvectors in the trivial
# representation get excited.  A cluster's other modes carry an eigenvalue but
# no amplitude.  Half of every path's spectrum is dark; a star K(1,m) has only
# two bright lines whatever m.
# ─────────────────────────────────────────────────────────────────────────────

DELTA = 1.0
BRIGHT_TOL = 1e-9


def cluster_modes(adj):
    """(lam, V, overlap, bright) for A = s*I - adj driven by b = delta*1."""
    adj = np.asarray(adj, float)
    n = adj.shape[0]
    A = S * np.eye(n) - adj
    b = DELTA * np.ones(n)
    lam, V = np.linalg.eigh(A)
    for k in range(n):                       # sign convention: largest |v| > 0
        if V[np.argmax(np.abs(V[:, k])), k] < 0:
            V[:, k] *= -1
    ov = V.T @ b
    return lam, V, ov, np.abs(ov) > BRIGHT_TOL


def path_adj(m):
    a = np.zeros((m, m))
    for j in range(m - 1):
        a[j, j + 1] = a[j + 1, j] = 1
    return a


def star_adj(m):
    a = np.zeros((m + 1, m + 1))
    for j in range(1, m + 1):
        a[0, j] = a[j, 0] = 1
    return a


def draw_graph(ax, pos, adj, node_col=None, node_sz=340, labels=None,
               edge_col="#7a7873", lw=1.8, fs=9.5):
    """Abstract compatibility graph: one node per segment, one edge per
    compatible head-to-tail pair."""
    pos = np.asarray(pos, float)
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                ax.plot(pos[[i, j], 0], pos[[i, j], 1], "-", color=edge_col,
                        lw=lw, zorder=1)
    cols = node_col if node_col is not None else ["#33322e"] * n
    ax.scatter(pos[:, 0], pos[:, 1], s=node_sz, c=cols, zorder=3,
               edgecolors="white", linewidths=1.4)
    if labels is not None:
        for k, t in enumerate(labels):
            ax.text(pos[k, 0], pos[k, 1], t, ha="center", va="center",
                    fontsize=fs, color="white", zorder=4, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def _collapse(lam, bright, tol=1e-9):
    """Group coincident eigenvalues so a degenerate line is drawn once, with
    its multiplicity, instead of several markers stacked invisibly."""
    out = []
    for l, br in zip(lam, bright):
        for k, ((l0, br0), m) in enumerate(out):
            if abs(l - l0) < tol and br == br0:
                out[k] = ((l0, br0), m + 1)
                break
        else:
            out.append(((float(l), bool(br)), 1))
    return out


def draw_lines(ax, lam, bright, colour, xlim=(1.5, 6.6), show_s=True,
               p4_guides=True, fs=9):
    """One motif's eigenvalue lines: bright solid, dark hollow."""
    if p4_guides:
        for l in P4:
            ax.axvline(l, color=GREEN, lw=0.7, ls=":", alpha=0.5)
    if show_s:
        ax.axvline(S, color=GREY, lw=0.8, ls=":")
    for (l, br), mult in _collapse(lam, bright):
        if br:
            ax.plot([l, l], [-0.42, 0.42], "-", color=colour, lw=3.0, zorder=3)
        else:
            ax.plot([l], [0], "o", mfc="white", mec=colour, mew=1.6, ms=8.5,
                    zorder=3)
        if mult > 1:
            ax.text(l, 0.50, f"$\\times{mult}$", color=colour, fontsize=8.6,
                    ha="center", va="bottom")
    ax.set_ylim(-0.95, 0.95)
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.tick_params(labelsize=fs)
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(False)


def draw_block(ax, adj, fs=9.5):
    """The sub-block of A = s*I - C this cluster contributes."""
    n = adj.shape[0]
    M = S * np.eye(n) - adj
    ax.imshow(np.where(M == S, 0.55, np.where(M == -1, -0.55, 0.0)),
              cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.add_patch(plt.Rectangle((-0.5, -0.5), n, n, fill=False,
                               edgecolor="#c9c6bf", lw=1.0, zorder=4))
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}" if v != 0 else "$\\cdot$",
                    ha="center", va="center", fontsize=fs,
                    color="#33322e" if v else "#b6b3ac")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    # pad every block into the same 4-cell window, so the drawn size of a block
    # is proportional to the number of segments in the cluster
    pad = (4 - n) / 2.0
    ax.set_xlim(-0.5 - pad, n - 0.5 + pad)
    ax.set_ylim(n - 0.5 + pad, -0.5 - pad)
    for sp in ax.spines.values():
        sp.set_visible(False)


# rows: (name, hit-space segments, true mask, node positions, adjacency, colour)
CORRESPOND = [
    ("isolated segment $P_1$\n(no compatible continuation)",
     [((0, 0), (1, 0.45))], None,
     [(0, 0)], np.zeros((1, 1)), RED),
    ("true fragment $P_2$\n(3 surviving hits of a broken track)",
     [((0, -0.35), (1, 0.0)), ((1, 0.0), (2, 0.5))], [1, 1],
     [(0, 0), (1, 0)], path_adj(2), GREEN),
    ("true track $P_4$\n(5 hits, 4 segments)",
     [((0, -0.5), (0.5, -0.25)), ((0.5, -0.25), (1.0, 0.0)),
      ((1.0, 0.0), (1.5, 0.25)), ((1.5, 0.25), (2.0, 0.5))], [1, 1, 1, 1],
     [(0, 0), (1, 0), (2, 0), (3, 0)], path_adj(4), GREEN),
    ("hub $K_{1,3}$\n(one segment, three continuations)",
     [((0, -0.3), (1, 0.0)), ((1, 0.0), (2, 0.62)),
      ((1, 0.0), (2, 0.12)), ((1, 0.0), (2, -0.55))], None,
     [(0.9, 0), (1.9, 0.75), (1.9, 0), (1.9, -0.75)], star_adj(3), RED),
]


def fig_correspondence():
    """Detector picture -> compatibility graph -> block of A -> eigenvalues."""
    nrow = len(CORRESPOND)
    fig = plt.figure(figsize=(12.6, 2.35 * nrow))
    gs = fig.add_gridspec(nrow, 4, width_ratios=[1.15, 0.95, 0.75, 1.85],
                          hspace=0.42, wspace=0.28)
    heads = ["in the detector", "compatibility graph",
             "its block of $A=sI-C$", "its eigenvalues"]
    for r, (name, segs, tmask, pos, adj, col) in enumerate(CORRESPOND):
        ax = fig.add_subplot(gs[r, 0])
        draw_motif(ax, segs, true_mask=tmask)
        ax.text(-0.30, 0.5, name, transform=ax.transAxes, fontsize=9.6,
                ha="right", va="center", color="#33322e")

        ax = fig.add_subplot(gs[r, 1])
        draw_graph(ax, pos, adj, node_col=[col] * len(pos),
                   labels=[str(k + 1) for k in range(len(pos))])
        ax.set_xlim(-0.75, 3.75); ax.set_ylim(-1.15, 1.15)

        ax = fig.add_subplot(gs[r, 2])
        draw_block(ax, adj)

        ax = fig.add_subplot(gs[r, 3])
        lam, V, ov, bright = cluster_modes(adj)
        draw_lines(ax, lam, bright, col)
        if r == nrow - 1:
            ax.set_xlabel("eigenvalue $\\lambda$  ($s=\\gamma+\\delta=4$)")
        else:
            ax.set_xticklabels([])
        nb = int(bright.sum())
        ax.text(0.995, 0.90, f"{nb} bright, {len(lam)-nb} dark",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="#6b6862")
        if r == 0:
            for c, h in zip(range(4), heads):
                fig.axes[c].set_title(h, fontsize=10.5, pad=9)
    fig.text(0.985, 0.012,
             "solid line: bright mode (non-zero overlap with the bias "
             "$\\mathbf{b}=\\delta\\mathbf{1}$)   ·   open circle: dark mode "
             "(zero overlap, no amplitude)",
             ha="right", fontsize=9, color="#6b6862")
    save(fig, "graph_correspondence")


def fig_track_modes():
    """The four modes of a true track, and the two that actually carry it."""
    adj = path_adj(4)
    lam, V, ov, bright = cluster_modes(adj)
    b = DELTA * np.ones(4)
    x = np.linalg.solve(S * np.eye(4) - adj, b)

    fig = plt.figure(figsize=(12.6, 5.5))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    draw_motif(ax, CORRESPOND[2][1], true_mask=[1, 1, 1, 1])
    ax.set_title("(a) a true track: 5 hits, 4 segments\n"
                 "its graph is the chain $P_4$", fontsize=10, pad=4)
    axi = ax.inset_axes([0.16, 0.02, 0.68, 0.30])
    draw_graph(axi, [(0, 0), (1, 0), (2, 0), (3, 0)], adj,
               node_col=[GREEN] * 4, node_sz=150,
               labels=["1", "2", "3", "4"], fs=7.5)
    axi.set_xlim(-0.6, 3.6); axi.set_ylim(-0.6, 0.6)

    order = [0, 1, 2, 3]
    cells = [gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1]]
    tags = ["(b)", "(c)", "(d)", "(e)"]
    for cell, k, tag in zip(cells, order, tags):
        ax = fig.add_subplot(cell)
        v = V[:, k]
        col = GREEN if bright[k] else GREY
        ax.bar(np.arange(1, 5), v, width=0.55, color=col,
               alpha=1.0 if bright[k] else 0.40,
               edgecolor="none")
        ax.axhline(0, color="#33322e", lw=0.9)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("segment along the track", fontsize=9.5)
        ax.set_ylim(-0.85, 0.85)
        ax.set_ylabel("mode weight", fontsize=9.5)
        state = "BRIGHT" if bright[k] else "dark"
        ax.set_title(f"{tag} $\\lambda={lam[k]:.3f}$   —   {state}\n"
                     f"$\\langle v,\\mathbf{{b}}\\rangle={ov[k]:.3f}$",
                     fontsize=10, pad=4,
                     color="#33322e" if bright[k] else "#8f8d86")
        if not bright[k]:
            ax.text(0.5, 0.90, "antisymmetric under the chain's reflection:\n"
                               "the uniform bias cannot excite it",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8.6, color="#8f8d86")

    ax = fig.add_subplot(gs[1, 2])
    j = np.arange(1, 5)
    c1 = ov[0] / lam[0] * V[:, 0]
    c3 = ov[2] / lam[2] * V[:, 2]
    ax.bar(j - 0.19, c1, width=0.34, color=GREEN, alpha=0.85,
           label=f"$\\lambda={lam[0]:.3f}$  (98.5% of the weight)")
    ax.bar(j + 0.19, c3, width=0.34, color=BLUE, alpha=0.85,
           label=f"$\\lambda={lam[2]:.3f}$  (1.5%)")
    ax.plot(j, x, "o-", color="#33322e", lw=1.6, ms=6.5,
            label="sum $=\\ 4/11,\\ 5/11,\\ 5/11,\\ 4/11$")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("segment along the track", fontsize=9.5)
    ax.set_ylabel("activation", fontsize=9.5)
    ax.set_title("(f) the solution is the two bright modes,\n"
                 "and nothing else", fontsize=10, pad=4)
    ax.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              frameon=False)
    ax.set_ylim(-0.10, 0.62)
    save(fig, "track_modes")


# five hits, four segments; dropping a hit deletes the segments that touch it
FRAGMENTS = [
    ("no hit lost", [], "$P_4$", (4 / 11, 5 / 11, 5 / 11, 4 / 11)),
    ("an end hit lost", [0], "$P_3$", (5 / 14, 3 / 7, 5 / 14)),
    ("the second hit lost", [1], "$P_2$", (1 / 3, 1 / 3)),
    ("the middle hit lost", [2], "$P_1+P_1$", (0.25, 0.25)),
]


def fig_fragmentation():
    """How one dropped hit turns a track into the shorter motifs."""
    fig = plt.figure(figsize=(12.6, 3.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.62], hspace=0.05,
                          wspace=0.22)
    hits = [(k * 0.5, -0.5 + k * 0.25) for k in range(5)]
    for c, (title, drop, gname, levels) in enumerate(FRAGMENTS):
        ax = fig.add_subplot(gs[0, c])
        keep = [k for k in range(4) if k not in drop and k + 1 not in drop]
        for k in range(4):
            p, q = hits[k], hits[k + 1]
            alive = k in keep
            ax.plot([p[0], q[0]], [p[1], q[1]],
                    color=GREEN if alive else "#d9d6cf",
                    lw=2.6 if alive else 1.6,
                    ls="-" if alive else (0, (2, 2)), zorder=2)
        for k, h in enumerate(hits):
            lost = k in drop
            ax.plot([h[0]], [h[1]], "o", ms=7.5, zorder=3,
                    mfc="white" if lost else "#33322e",
                    mec=RED if lost else "#33322e", mew=1.8 if lost else 1.0)
        ax.set_xlim(-0.35, 2.35); ax.set_ylim(-1.0, 1.0)
        ax.axis("off")
        ax.set_title(title, fontsize=10, pad=3)

        ax = fig.add_subplot(gs[1, c])
        nk = len(keep)
        # graph of the surviving segments: adjacent if consecutive and both kept
        adjf = np.zeros((nk, nk))
        for a in range(nk):
            for bq in range(a + 1, nk):
                if keep[bq] - keep[a] == 1:
                    adjf[a, bq] = adjf[bq, a] = 1
        draw_graph(ax, [(i, 0) for i in range(nk)], adjf,
                   node_col=[GREEN] * nk, node_sz=200,
                   labels=[str(keep[i] + 1) for i in range(nk)], fs=8)
        ax.set_xlim(-0.8, 3.8); ax.set_ylim(-0.75, 0.95)
        ax.text(0.5, -0.02, f"{gname}   ·   $x={', '.join(f'{v:.3f}' for v in levels)}$",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color="#33322e")
    save(fig, "fragmentation")


# The motif dictionary.  Eigenvalues are DERIVED from each motif's adjacency
# rather than written down, so the picture and the numbers cannot drift apart.
# "triple" is the graph P_3 -- not called a fork here, because the fork term of
# section 7 is the bifurcation penalty and the two must not share a word.
MOTIFS2 = [
    ("isolated segment\n$P_1$", [((0, 0), (1, 0.45))], None,
     [(1.5, 0)], np.zeros((1, 1)), "$\\lambda = s$"),
    ("pair\n$P_2$", [((0, -0.35), (1, 0.0)), ((1, 0.0), (2, 0.5))], None,
     [(1.0, 0), (2.0, 0)], path_adj(2), "$\\lambda = s \\mp 1$"),
    ("triple\n$P_3$", [((0, -0.3), (1, 0.05)), ((1, 0.05), (2, 0.6)),
                       ((1, 0.05), (2, -0.35))], None,
     [(0.5, 0), (1.5, 0), (2.5, 0)], path_adj(3),
     "$\\lambda = s,\\ s \\mp \\sqrt{2}$"),
    ("hub\n$K_{1,3}$",
     [((0, -0.3), (1, 0.0)), ((1, 0.0), (2, 0.62)),
      ((1, 0.0), (2, 0.12)), ((1, 0.0), (2, -0.55))], None,
     [(1.0, 0), (2.1, 0.7), (2.1, 0.0), (2.1, -0.7)], star_adj(3),
     "$\\lambda = s,\\ s,\\ s \\mp \\sqrt{3}$"),
    ("true track\n$P_4$",
     [((0, -0.5), (0.5, -0.25)), ((0.5, -0.25), (1.0, 0.0)),
      ((1.0, 0.0), (1.5, 0.25)), ((1.5, 0.25), (2.0, 0.5))],
     [1, 1, 1, 1],
     [(0.4, 0), (1.1, 0), (1.8, 0), (2.5, 0)], path_adj(4),
     "$\\lambda = s - 2\\cos(k\\pi/5)$"),
]


def fig_motifs():
    """Motif dictionary: detector picture, graph, and eigenvalue lines."""
    fig = plt.figure(figsize=(12.8, 6.9))
    gs = fig.add_gridspec(3, 5, height_ratios=[0.95, 0.62, 1.45], hspace=0.10)
    for i, (title, segs, tmask, pos, adj, lab) in enumerate(MOTIFS2):
        col = GREEN if tmask else RED
        ax = fig.add_subplot(gs[0, i])
        draw_motif(ax, segs, true_mask=tmask)
        ax.set_title(title, fontsize=10.5, pad=2)
        ax = fig.add_subplot(gs[1, i])
        draw_graph(ax, pos, adj, node_col=[col] * len(pos), node_sz=175,
                   labels=None)
        ax.set_xlim(-0.35, 2.95); ax.set_ylim(-1.0, 1.0)

    axl = fig.add_subplot(gs[2, :])
    y = 0
    yticklabels = []
    for title, segs, tmask, pos, adj, lab in MOTIFS2:
        col = GREEN if tmask else RED
        lam, V, ov, bright = cluster_modes(adj)
        for (l, br), mult in _collapse(lam, bright):
            if br:
                axl.plot([l, l], [y - 0.30, y + 0.30], "-", color=col, lw=3.0,
                         zorder=3)
            else:
                axl.plot([l], [y], "o", mfc="white", mec=col, mew=1.7, ms=9,
                         zorder=3)
            if mult > 1:
                axl.text(l + 0.13, y + 0.02, f"$\\times{mult}$", color=col,
                         fontsize=9, ha="left", va="center")
        axl.text(7.62, y, lab, fontsize=10.5, va="center", color="#33322e")
        yticklabels.append(title.replace("\n", " "))
        y += 1
    axl.axvline(S, color=GREY, lw=0.8, ls=":")
    axl.text(S, y - 0.30, "$\\lambda = s$ (isolated line)", fontsize=9.5,
             color=GREY, ha="center", va="bottom")
    for lam0 in P4:
        axl.axvline(lam0, color=GREEN, lw=0.7, ls=":", alpha=0.55)
    axl.set_yticks(range(len(MOTIFS2)), yticklabels, fontsize=10)
    axl.set_ylim(-0.6, len(MOTIFS2) - 0.1)
    axl.set_xlim(1.6, 9.4)
    axl.set_xlabel("eigenvalue $\\lambda$   ($s = \\gamma + \\delta = 4$)")
    axl.grid(axis="y", visible=False)
    axl.plot([], [], "-", color="#33322e", lw=3.0, label="bright mode")
    axl.plot([], [], "o", mfc="white", mec="#33322e", mew=1.7, ms=9,
             label="dark mode (zero overlap with $\\mathbf{b}=\\delta\\mathbf{1}$)")
    axl.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
               fontsize=9.5)
    save(fig, "motif_gallery")


def fig_gershgorin():
    """The discs behind the positive-definiteness condition, and the knob."""
    import json
    src = (HERE.parent / "section2_graphs" / "outputs" / "03_gershgorin.json")
    D = json.loads(src.read_text())

    fig = plt.figure(figsize=(12.8, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.30)

    # (a) the discs
    ax = fig.add_subplot(gs[0, 0])
    for d, col in ((6, RED), (4, ORANGE), (2, BLUE)):
        ax.add_patch(plt.Circle((S, 0), d, fill=True, facecolor=col,
                                alpha=0.12, edgecolor=col, lw=1.6, zorder=1))
        ax.text(S, d - 0.18, f"$d_i={d}$", color=col, fontsize=9.2,
                ha="center", va="top", zorder=6)
    ax.axvline(0, color="#33322e", lw=1.4, zorder=4)
    ax.text(0.18, -6.2, "$\\lambda=0$", fontsize=9.5, color="#33322e")
    ax.plot([S], [0], "o", color="#33322e", ms=6, zorder=5)
    ax.text(S + 0.30, 0.10, "$d_i=0$: a point at $s=4$", fontsize=9.0,
            ha="left", va="bottom", color="#33322e")
    ax.annotate("", xy=(0, 0), xytext=(S, 0), zorder=6,
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.3))
    ax.text(S / 2, -0.55, "$d_i=s$: the disc\nfirst reaches zero",
            fontsize=8.8, color=ORANGE, ha="center", va="top", zorder=6)
    ax.set_xlim(-3.2, 11.2); ax.set_ylim(-6.6, 6.6)
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.set_xlabel("real axis")
    ax.set_title("(a) Gershgorin discs of $A=sI-C$\n"
                 "centre $s$, radius $d_i$ = number of neighbours",
                 fontsize=10, pad=5)
    ax.grid(False)
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)

    # (b) degree distribution -- the tail, not the bulk, sets the bound
    ax = fig.add_subplot(gs[0, 1])
    for cfg, col in (("clean", GREEN), ("moderate", BLUE), ("heavy", RED)):
        h = D["degree_hist"][cfg]
        c = np.array(h["counts"], float)
        f = c / c.sum()
        k = np.arange(len(c))
        ax.plot(k[f > 0], f[f > 0], "o-", color=col, ms=4.5, lw=1.4,
                label=f"{cfg} ($d_{{\\max}}={h['d_max']}$)")
    ax.axvline(S, color=GREY, lw=1.2, ls="--")
    ax.text(S - 0.12, 2e-5, "$d_i=s$", fontsize=9.0, color="#6b6862",
            ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("number of compatible neighbours $d_i$")
    ax.set_ylabel("fraction of segments")
    ax.set_title("(b) the bound is set by a thin tail\n($T=200$, one event each)",
                 fontsize=10, pad=5)
    ax.legend(fontsize=8.8)

    # (c) the control knob
    ax = fig.add_subplot(gs[0, 2])
    for key, col in (("moderate_T400", BLUE), ("moderate_T700", "#7ea9e0"),
                     ("heavy_T400", ORANGE), ("heavy_T700", RED)):
        if key not in D["gamma_scan"]:
            continue
        g = D["gamma_scan"][key]
        gg, ll = np.array(g["gammas"]), np.array(g["lam_min"])
        ax.plot(gg, ll, "-", color=col, lw=1.8,
                label=f"{g['config']}, $T={g['T']}$")
        ax.plot([g["gamma_crit"]], [0.0], "v", color=col, ms=7, zorder=5)
    ax.axhline(0, color="#33322e", lw=1.0)
    ax.axvline(3.0, color=GREY, lw=1.0, ls=":")
    ax.text(3.10, -0.95, "$\\gamma=3$ used here", fontsize=8.6,
            color="#6b6862", va="bottom", ha="left")
    ax.set_xlabel("self-penalty $\\gamma$")
    ax.set_ylabel("$\\lambda_{\\min}(A)=\\gamma+\\delta-\\lambda_{\\max}(C)$")
    ax.set_title("(c) positive definiteness is a knob:\n"
                 "raising $\\gamma$ always restores it", fontsize=10, pad=5)
    ax.legend(fontsize=8.4, loc="upper left")
    save(fig, "gershgorin_discs")


if __name__ == "__main__":
    cmds = sys.argv[1:] or ["all_fast"]
    if "all_fast" in cmds:
        cmds = ["motifs", "twin", "responses", "readout", "erf", "run3crop",
                "correspondence", "trackmodes", "fragmentation"]
    if "heavy" in cmds:
        i = cmds.index("heavy")
        cmds[i:i + 1] = ["tangles", "pileup", "threshold", "fitpaper"]
    for c in cmds:
        {"motifs": fig_motifs, "twin": fig_twin, "responses": fig_responses,
         "correspondence": fig_correspondence, "trackmodes": fig_track_modes,
         "hhl": fig_hhl_onebit, "notchspec": fig_notch_spectrum,
         "fragmentation": fig_fragmentation, "gershgorin": fig_gershgorin,
         "readout": fig_readout, "erf": fig_erf, "run3crop": fig_run3crop,
         "tangles": fig_tangles, "pileup": fig_pileup,
         "threshold": fig_threshold, "fitpaper": fig_fitpaper,
         "occfit": fig_occfit, "clean2x2": fig_clean2x2,
         "occproofcrop": fig_occproofcrop,
         "epsilon": fig_epsilon,
         "rescompare": fig_resources_compare}[c]()
