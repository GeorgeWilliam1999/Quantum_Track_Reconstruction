#!/usr/bin/env python3
"""08 -- purpose-built figures for the hit-oracle write-up (Block Encoding IV).

Four figures, no new measurements: panels either re-plot the measured CSVs of
studies 04/06/07 + the QSVT degree-sweep store data, or are labelled schematics
of the construction in HIT_ORACLE_DESIGN.md.

  fig08_oracle_geometry.png : (a) schematic event + epsilon-window enumeration
                              (b) sorted-hit / inverse-CDF picture
                              (c) MEASURED window occupancy & alpha vs T (04)
  fig08_oracle_circuit.png  : register-lane block schematic of U_R + costs
  fig08_cost_comparison.png : (a) CX/walk-call vs T  (b) CX/full solve vs T
                              (c) qubits vs T                     (all 06)
  fig08_on_off.png          : (a) classical loading time vs T (07)
                              (b) P_anc vs T (measured degree sweep)
                              (c) amplified walk-call budget (measured)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SWEEP = Path("/data/bfys/gscriven/Quantum_Track_Reconstruction/QSVT/"
             "Segment_level_studies/outputs/depth_and_qubits/degree_sweep.csv")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})

MCOL = {"native": "#33322e", "hit_oracle": "#2a78d6", "szegedy": "#1e9e6a",
        "camps": "#e34948", "dictionary": "#eb6834", "fable": "#9a9890",
        "dilation": "#8e6fad"}
MLAB = {"native": "native 1BQF (Givens)", "hit_oracle": "hit oracle (ours)",
        "szegedy": "szegedy walk (hit-prep)", "camps": "Camps sparse oracle",
        "dictionary": "dictionary (SBM)", "fable": "FABLE family",
        "dilation": "dilation (default QSVT)"}


# ---------------------------------------------------------------- fig 1
def fig_geometry():
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4),
                             gridspec_kw={"width_ratios": [1.25, 1.05, 1.0]})

    # (a) schematic event: planes, one segment, extrapolation, eps-window
    ax = axes[0]
    ax.grid(False)
    zs = [0, 1, 2, 3]
    for z in zs:
        ax.axvline(z, color="#c9c7c0", lw=1.2, zorder=0)
        ax.text(z, -1.45, f"plane {z}", ha="center", fontsize=8,
                color="#79776f")
    slopes = [-0.38, -0.05, 0.22, 0.52]
    x0s = [0.35, -0.15, 0.10, -0.30]
    hits = {}
    for t, (m, x0) in enumerate(zip(slopes, x0s)):
        xs = [x0 + m * z for z in zs]
        ax.plot(zs, xs, "-", color="#dddbd4", lw=1.0, zorder=1)
        for z, x in zip(zs, xs):
            hits[(t, z)] = x
    for (t, z), x in hits.items():
        ax.plot(z, x, "o", color="#79776f", ms=5, zorder=3)
    # highlighted segment on track 2: plane 1 -> 2
    a, b = hits[(2, 1)], hits[(2, 2)]
    ax.plot([1, 2], [a, b], "-", color="#2a78d6", lw=3.0, zorder=4)
    ax.annotate("segment $(a,b)$", (1.5, (a + b) / 2), xytext=(0.45, 1.05),
                fontsize=9, color="#2a78d6",
                arrowprops=dict(arrowstyle="->", color="#2a78d6", lw=1.1))
    # extrapolated line to plane 3 + window
    x_ext = b + (b - a)
    ax.plot([2, 3], [b, x_ext], "--", color="#2a78d6", lw=1.6, zorder=4)
    r = 0.22
    ax.fill_betweenx([x_ext - r, x_ext + r], 2.93, 3.07,
                     color="#2a78d6", alpha=0.18, zorder=2)
    ax.annotate(r"extrapolation", (2.55, b + (x_ext - b) * 0.55),
                xytext=(1.62, -1.05), fontsize=9, color="#2a78d6",
                arrowprops=dict(arrowstyle="->", color="#2a78d6", lw=1.1))
    ax.annotate(r"$\varepsilon$-window, radius $r(\varepsilon)$",
                (3.05, x_ext + r), xytext=(2.0, 1.35), fontsize=9,
                color="#2a78d6",
                arrowprops=dict(arrowstyle="->", color="#2a78d6", lw=1.1))
    # mark in-window candidates
    for t in range(4):
        x = hits[(t, 3)]
        if abs(x - x_ext) < r:
            ax.plot(3, x, "o", color="#1e9e6a", ms=9, mfc="none", mew=2.2,
                    zorder=5)
    ax.text(3.12, x_ext - 0.02, "$w$ candidates\n(slots $\\ell$)",
            fontsize=8.5, color="#1e9e6a", va="center")
    ax.set_xlim(-0.35, 3.75), ax.set_ylim(-1.6, 1.75)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a) the coupling rule is geometry (schematic):\n"
                 "shared hit + kink angle inside the acceptance",
                 loc="left", fontsize=9)

    # (b) sorted hits + inverse-CDF
    ax = axes[1]
    ax.grid(False)
    rng = np.random.default_rng(7)
    xs = np.sort(rng.uniform(0, 1, 14))
    lo, hi = 0.52, 0.71
    inw = (xs >= lo) & (xs <= hi)
    ax.axhline(0.55, color="#c9c7c0", lw=1.0)
    ax.plot(xs[~inw], np.full((~inw).sum(), 0.55), "o", color="#79776f", ms=6)
    ax.plot(xs[inw], np.full(inw.sum(), 0.55), "o", color="#1e9e6a", ms=8)
    ax.fill_betweenx([0.44, 0.66], lo, hi, color="#1e9e6a", alpha=0.15)
    base = int(np.flatnonzero(inw)[0])
    for k in (0, base, len(xs) - 1):
        ax.text(xs[k], 0.47, str(k), ha="center", fontsize=7.5,
                color="#79776f")
    ax.text(xs[2], 0.47, r"$\dots$", ha="center", fontsize=7.5,
            color="#79776f")
    ax.text(xs[11], 0.47, r"$\dots$", ha="center", fontsize=7.5,
            color="#79776f")
    ax.annotate(f"base $=$ inverse-CDF QROM$(x_{{lo}})$ $= {base}$",
                (xs[base], 0.585), xytext=(0.03, 0.87), fontsize=9,
                color="#1e9e6a",
                arrowprops=dict(arrowstyle="->", color="#1e9e6a", lw=1.2))
    ax.annotate("", xy=(hi, 0.70), xytext=(lo, 0.70),
                arrowprops=dict(arrowstyle="<->", color="#1e9e6a", lw=1.2))
    ax.text((lo + hi) / 2, 0.72, "slots $\\ell = 0 \\dots w{-}1$",
            ha="center", fontsize=9, color="#1e9e6a")
    ax.text(0.5, 0.28, "hits sorted by coordinate, once per event "
            "($O(T\\log T)$, classical)\ntable size $5T$ entries — "
            "not $4T^2$ matrix rows", ha="center", fontsize=8.5,
            color="#52514e")
    ax.set_xlim(-0.03, 1.03), ax.set_ylim(0.18, 0.97)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(b) window lookup on the sorted hits:\n"
                 "one QROM table indexes every continuation",
                 loc="left", fontsize=9)

    # (c) measured occupancy / alpha vs T
    ax = axes[2]
    d4 = pd.read_csv(OUT / "04_hit_oracle_window.csv")
    cl = d4[d4.noise == "clean"]
    no = d4[d4.noise == "noisy"]
    ax.plot(cl["T"], cl.w_disc_max, "o-", color="#2a78d6", lw=1.8,
            label="max window occupancy $w$ (clean)")
    ax.plot(cl["T"], cl.alpha_hit_disc, "s--", color="#33322e", lw=1.5,
            label=r"subnormalization $\alpha = w_{pad}$ (clean)")
    ax.plot(no["T"], no.alpha_hit_disc, "x", color="#e34948", ms=9, mew=2.2,
            label=r"$\alpha$ (noisy events)")
    ax.set_xscale("log")
    ax.set_ylim(0, 5.2)
    ax.set_xlabel("tracks per event $T$")
    ax.set_ylabel(r"$w$, $\alpha$")
    ax.text(6, 4.65, "coverage = 1.0 at every point\n"
            "(every accepted coupling reproduced)", fontsize=8.5,
            color="#1e9e6a")
    ax.legend(fontsize=7.5, loc="center left")
    ax.set_title("(c) measured: the window stays $O(1)$\n"
                 r"$\Rightarrow$ $\alpha$ at the occupancy floor to $T=1000$",
                 loc="left", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig08_oracle_geometry.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_circuit():
    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    ax.grid(False)
    ax.set_xlim(0, 100), ax.set_ylim(0, 56)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    lanes = [("layer $g$ — 2 q", 50),
             ("hit $a$ — $\\lceil\\log T'\\rceil$ q", 44),
             ("hit $b$ — $\\lceil\\log T'\\rceil$ q", 38),
             ("slot $\\ell$ — $\\lceil\\log w_{pad}\\rceil$ q", 32),
             ("arithmetic — $3\\times$16-bit words", 26),
             ("accept flag — 1 q", 20)]
    for name, y in lanes:
        ax.plot([13, 88], [y, y], "-", color="#c9c7c0", lw=1.1, zorder=0)
        ax.text(12.4, y, name, ha="right", va="center", fontsize=8.5,
                color="#52514e")

    blocks = [
        ("QROM\ncoords of $a,b$\n(5$T$-entry table)", 17, 26, 50, "#2a78d6"),
        ("fixed-point\nextrapolation\n2 sub + 1 mul-add", 28, 37, 44, "#33322e"),
        ("inverse-CDF\nQROM $\\to$ base;\nadder $+\\ell$", 39, 48, 38, "#2a78d6"),
        ("QROM\ncoord of\ncandidate $c$", 50, 58, 44, "#2a78d6"),
        ("kink angle\n$\\theta(a,b,c)$\narithmetic", 60, 68, 32, "#33322e"),
        ("comparator $\\theta<\\varepsilon$ (step)\n"
         "$R_y(\\arccos k(\\theta))$ (erf)", 70, 80, 26, "#1e9e6a"),
        ("uncompute\n(mirror)", 82, 88, 38, "#79776f"),
    ]
    for txt, x0, x1, ymid, col in blocks:
        bb = FancyBboxPatch((x0, 17), x1 - x0, 34,
                            boxstyle="round,pad=0.4", fc="none", ec=col,
                            lw=1.6, zorder=2)
        ax.add_patch(bb)
        ax.text((x0 + x1) / 2, 35, txt, ha="center", va="center",
                fontsize=8, color=col, zorder=4,
                bbox=dict(fc="white", ec="none", alpha=0.92, pad=1.5))

    costs = ["$2\\times$QROM", "$\\sim$500 CX", "QROM + adder",
             "QROM", "$\\sim$500 CX", "$O(b)$", "$\\times 2$"]
    for (txt, x0, x1, _, col), c in zip(blocks, costs):
        ax.text((x0 + x1) / 2, 13.5, c, ha="center", fontsize=7.5,
                color="#79776f")

    ax.text(50, 7.5, "QROM dominates: $\\approx 8\\cdot 5T$ CX linear "
            "(or $O(\\sqrt{T})$ T-depth select-swap) — every other block is "
            "$T$-independent.\nmodel total per walk call: "
            "$1.6\\cdot10^4$ CX at $T=400$ vs $2.4\\cdot10^6$ entry-priced — "
            "and the same oracle at degree 1 is a hardware-route 1BQF.",
            ha="center", fontsize=9, color="#33322e")
    ax.text(50, 54.2, "$U_R\\,|0\\rangle|j\\rangle = w_{pad}^{-1/2}"
            "\\sum_{\\ell<w_{pad}} |\\mathrm{window}_\\ell(j)\\rangle"
            "|\\mathrm{flag}_\\ell\\rangle|j\\rangle$; "
            "$U = U_L^{\\dagger}\\,\\mathrm{SWAP}\\,U_R$ block-encodes "
            "$C/w_{pad}$ (state-preparation-pair form — no injectivity "
            "assumption)", ha="center", fontsize=9.5, color="#33322e")
    ax.set_title("The hit-level oracle, one walk call: registers and blocks "
                 "(design + cost model; per-block CX below)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig08_oracle_circuit.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_costs():
    d6 = pd.read_csv(OUT / "06_resource_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))

    def curves(ax, col, methods):
        for m in methods:
            d = d6[d6.method == m].sort_values("T").copy()
            d[col] = pd.to_numeric(d[col], errors="coerce")
            d = d[np.isfinite(d[col])]
            if not len(d):
                continue
            ls = {"camps": "--", "dilation": "-."}.get(m, "-")
            ax.plot(d["T"], d[col], ls, color=MCOL[m],
                    lw=2.2 if m == "hit_oracle" else 1.5, label=MLAB[m],
                    alpha=0.8 if m == "dilation" else 1.0)

    ax = axes[0]
    curves(ax, "cx_call", ["fable", "dictionary", "camps", "native",
                           "szegedy", "hit_oracle"])
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("tracks per event $T$")
    ax.set_ylabel("CX per walk call / evolution call")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(a) one call: only the hit-priced routes\nbeat the native "
                 "Givens pass (Camps $\\equiv$ native)", loc="left", fontsize=9)

    ax = axes[1]
    curves(ax, "cx_qsvt", ["fable", "dictionary", "camps", "dilation",
                           "szegedy", "hit_oracle"])
    dn = d6[d6.method == "native"].sort_values("T")
    ax.plot(dn["T"], dn.cx_call, ":", color="#33322e", lw=2.0,
            label="ONE native 1BQF call (reference)")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("tracks per event $T$")
    ax.set_ylabel("CX per full QSVT comb solve (at $d^*$)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(b) full filtered solve: the hit oracle undercuts\na single "
                 "native call, $\\times5.5$ (T=200) $\\to$ $\\times23$ (T=1000)",
                 loc="left", fontsize=9)

    ax = axes[2]
    for m in ["szegedy", "hit_oracle", "fable", "camps", "dictionary"]:
        d = d6[d6.method == m].sort_values("T")
        col = "qubits_qsvt" if np.isfinite(d.qubits_qsvt).any() else "qubits_1bqf"
        ax.plot(d["T"], d[col], "-", color=MCOL[m], lw=2.2
                if m == "hit_oracle" else 1.5, label=MLAB[m])
    dn = d6[d6.method == "native"].sort_values("T")
    ax.plot(dn["T"], dn.qubits_1bqf, "-", color="#33322e", lw=2.0,
            label="native 1BQF floor ($n{+}1$)")
    ax.set_xscale("log")
    ax.set_xlabel("tracks per event $T$")
    ax.set_ylabel("qubits (QSVT solve incl. LCU register)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(c) the price is width: every encoding adds\nqubits over "
                 "the native floor — ours adds $\\approx$45–49", loc="left",
                 fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig08_cost_comparison.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_on_off():
    d7 = pd.read_csv(OUT / "07_classical_cost.csv")
    d6 = pd.read_csv(OUT / "06_resource_curves.csv")
    sw = pd.read_csv(SWEEP)
    d40 = sw[sw.degree == 40].groupby("T").mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))

    ax = axes[0]
    for m in ["dilation", "fable", "dictionary", "camps", "szegedy",
              "native", "hit_oracle"]:
        d = d7[d7.method == m].sort_values("T")
        if not len(d):
            continue
        ax.plot(d["T"], d.t_total, "-", color=MCOL[m],
                lw=2.2 if m == "hit_oracle" else 1.4, label=MLAB[m],
                ls="-." if m == "dilation" else "-")
    mem = {m: d7[(d7.method == m) & (d7["T"] == 1000)].mem_bytes.iloc[0]
           for m in ["hit_oracle", "native", "fable"]}
    ax.text(1050, d7[(d7.method == "hit_oracle")]["t_total"].max() * 1.6,
            f"{mem['hit_oracle']/1e3:.0f} KB", fontsize=8, color="#2a78d6")
    ax.text(1050, 12, f"{mem['native']/1e6:.0f} MB", fontsize=8,
            color="#33322e")
    ax.text(1050, d7[(d7.method == 'fable') & (d7['T'] == 1000)]
            .t_total.iloc[0] * 0.5, f"{mem['fable']/1e12:.0f} TB",
            fontsize=8, color="#9a9890")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("tracks per event $T$")
    ax.set_ylabel("classical prep per event (s)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(a) reading ON: per-event classical loading\n(hit oracle: "
                 "no $A$ build, no Lanczos bounds — ms and KB)", loc="left",
                 fontsize=9)

    ax = axes[1]
    Ts = np.geomspace(40, 1200, 100)
    ax.loglog(d40["T"], d40["P"], "o", color="#2a78d6", ms=7,
              label="measured, line comb $d=40$ (store)")
    ax.loglog(Ts, 0.44 / Ts, "-", color="#2a78d6", lw=1.4, alpha=0.7,
              label=r"$P_{\rm anc} = 0.44/T$")
    ax.loglog(Ts, 0.25 / Ts, "--", color="#e34948", lw=1.4, alpha=0.7,
              label=r"1BQF estimate $0.25/T$")
    ax.set_xlabel("tracks per event $T$")
    ax.set_ylabel(r"post-selection success $P_{\rm anc}$")
    ax.legend(fontsize=7.5)
    ax.set_title("(b) reading OFF, step 1: one repetition succeeds\nwith "
                 r"$P_{\rm anc}\sim 1/T$ (identical for every encoding)",
                 loc="left", fontsize=9)

    ax = axes[2]
    ax.loglog(d40["T"], d40.total_depth, "s", color="#2a78d6", ms=7,
              label="measured amplified walk calls ($d=40$)")
    scale = float(d40.total_depth.iloc[0] / np.sqrt(d40["T"].iloc[0]))
    ax.loglog(Ts, scale * np.sqrt(Ts), "-", color="#2a78d6", lw=1.4,
              alpha=0.7, label=r"$\propto d\sqrt{T}$ (amplitude amplification)")
    cx1000 = float(d6[(d6.method == "hit_oracle")
                      & (d6["T"] == 1000)].cx_call.iloc[0])
    cxnat1000 = float(d6[(d6.method == "native")
                         & (d6["T"] == 1000)].cx_call.iloc[0])
    calls1000 = float(d40[d40["T"] == 1000].total_depth.iloc[0])
    ax.annotate("walk calls $\\times$ CX/call at $T=1000$:\n"
                f"hit oracle $\\approx${calls1000 * cx1000:.1e} CX per "
                "amplified sample\nentry-priced "
                f"$\\approx${calls1000 * cxnat1000:.1e} CX",
                xy=(45, 1650), va="top", fontsize=8, color="#33322e")
    ax.set_xlabel("tracks per event $T$")
    ax.set_ylabel("total walk calls per useful sample")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("(c) reading OFF, step 2: the walk-call budget\n"
                 r"$d\cdot T^{1/2}$ after amplification — the encoding sets "
                 "the CX each call costs", loc="left", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig08_on_off.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_geometry()
    fig_circuit()
    fig_costs()
    fig_on_off()
    print("wrote", [f"fig08_{n}.png" for n in
                    ("oracle_geometry", "oracle_circuit",
                     "cost_comparison", "on_off")])
