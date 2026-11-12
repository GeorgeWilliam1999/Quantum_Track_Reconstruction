#!/usr/bin/env python3
"""Tables, kill-criterion verdict and figures for Experiment A (run_readouts.py outputs).

Efficiency and ghost rate are POOLED over the events of a point (sum matched / sum
reconstructible; sum ghosts / sum candidates), as Real_events/scripts/summarize.py does.

Kill criterion (research plan, candidate A): stop if, on Run 3 at the truth-free reading
(I), no readout lifts classical above 0.60 AND QSVT above 0.70 track efficiency with ghost
rate <= 0.10, or if the chosen readout loses more than 0.02 efficiency against get_tracks
(gt) on the toy at any (T, n_pv) point.

Figures (paper 1 style, Focusing_scan/scripts/plot_style.py), figures/:
  A1_run3_group_sizes      largest group per event: gt, cc, ccu against the largest
                           Hamiltonian component (hits); the merging test
  A2_run3_eff_ghost        Run 3 efficiency and ghost rate, solver x readout, reading I
  A3_toy_eff_vs_pv         toy efficiency against n_pv, T = 100, 400, 1000, gt vs ccu, readings I and F
  A4_toy_ghost_vs_pv       same for ghost rate
  A5_resources_vs_tracks   qubits and two-qubit gates per solved event against reconstructible
                           tracks, toy and Run 3 (unchanged by the readout)
  A6_run3_categories       Run 3 per-track efficiency by category, gt vs ccu, reading I
Also appends the no-solver control (control_no_solver.py) to summary.txt when its CSVs exist.
Output: outputs/summary.txt, outputs/run3_table.csv, outputs/toy_table.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Focusing_scan" / "scripts"))
import plot_style as PS  # noqa: E402
from plot_style import plt, METHODS, WIDTH  # noqa: E402

OUT = HERE.parent / "outputs"
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
SOLVERS = ("classical", "1bqf", "qsvt")
RO_LABEL = {"gt": "get_tracks (shared hit)", "gtu": "get_tracks + hit uniqueness",
            "cc": "coupled", "ccu": "coupled + hit uniqueness", "lay": "get_tracks_layered"}
READING = {"I": "truth-free (above isolated value)", "F": "<= 1 % false segments", "P": "paper 1 rule"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def pooled(g):
    return pd.Series(dict(
        efficiency=g.n_matched.sum() / max(g.n_reconstructible.sum(), 1),
        ghost_rate=g.n_ghosts.sum() / max(g.n_candidates.sum(), 1),
        clone_fraction=g.n_clones.sum() / max(g.n_matched.sum() + g.n_clones.sum(), 1),
        seg_eff=g.seg_eff.mean(), seg_far=g.seg_far.mean(),
        max_group_med=g.max_group_hits.median(), max_group_max=g.max_group_hits.max(),
        n_events=len(g)))


def rounds(P):
    return np.maximum(1.0, np.pi / (4 * np.sqrt(np.maximum(P, 1e-300))))


def resources(d):
    """Per event: qubits and two-qubit gates per solved event (Givens price, paper 1 calls)."""
    e = d.drop_duplicates("event").copy()
    e["q_1bqf"] = e.seg_ns + 2
    e["q_qsvt"] = e.seg_ns + 4
    e["cx_1bqf"] = 2 * rounds(e.seg_P_1b) * e.seg_cx_givens
    e["cx_qsvt"] = 160 * rounds(e.seg_P_q) * e.seg_cx_givens
    return e


lines = []
P = print


def out(s=""):
    lines.append(s)
    P(s)


# ── Run 3 ────────────────────────────────────────────────────────────────────
dr = pd.read_csv(OUT / "real_readouts_events.csv")
tr = pd.read_csv(OUT / "real_readouts_tracks.csv")
r3 = dr.groupby(["reading", "solver", "readout"]).apply(pooled).reset_index()
r3.to_csv(OUT / "run3_table.csv", index=False)
ne = dr.event.nunique()
out(f"Run 3 (Gauss Sim10 min bias, {ne} events; EventValidator purity >= 0.7, reco >= 3 hits; pooled)")
for rd in ("I", "F", "P"):
    out(f"\nreading {rd}: {READING[rd]}")
    out(f"{'solver':10s} {'readout':30s} {'eff':>6s} {'ghost':>6s} {'clone':>6s} {'segeff':>6s} {'segfar':>6s} {'maxgrp med/max':>15s}")
    for _, r in r3[r3.reading == rd].iterrows():
        out(f"{r.solver:10s} {RO_LABEL[r.readout]:30s} {r.efficiency:6.3f} {r.ghost_rate:6.3f} {r.clone_fraction:6.3f} "
            f"{r.seg_eff:6.3f} {r.seg_far:6.3f} {r.max_group_med:7.0f}/{r.max_group_max:<7.0f}")
e = dr.drop_duplicates("event")
gI = dr[(dr.reading == "I")]
out("\nMerging test (reading I, per event, hits): largest Hamiltonian component median "
    f"{e.ham_max_comp_hits.median():.0f} (max {e.ham_max_comp_hits.max()})")
for s in SOLVERS:
    for ro in ("gt", "cc", "ccu"):
        x = gI[(gI.solver == s) & (gI.readout == ro)].max_group_hits
        out(f"  {s:10s} {ro:4s} largest group median {x.median():6.0f}  max {x.max():5d}")

# kill criterion, Run 3 part
I = r3[r3.reading == "I"].set_index(["solver", "readout"])
out("\nKill criterion, Run 3 (reading I): classical > 0.60 and QSVT > 0.70, ghosts <= 0.10")
passing = []
for ro in RO_LABEL:
    c, q = I.loc[("classical", ro)], I.loc[("qsvt", ro)]
    ok = c.efficiency > 0.60 and q.efficiency > 0.70 and c.ghost_rate <= 0.10 and q.ghost_rate <= 0.10
    out(f"  {RO_LABEL[ro]:30s} classical {c.efficiency:.3f}/{c.ghost_rate:.3f}  qsvt {q.efficiency:.3f}/{q.ghost_rate:.3f}  -> {'PASS' if ok else 'fail'}")
    if ok:
        passing.append(ro)

# ── toy ──────────────────────────────────────────────────────────────────────
dt = pd.read_csv(OUT / "toy_readouts_events.csv")
tt = dt.groupby(["T", "n_pv", "reading", "solver", "readout"]).apply(pooled).reset_index()
tt.to_csv(OUT / "toy_table.csv", index=False)
out(f"\nToy sweep ({dt.event.nunique()} events; T = {sorted(dt['T'].unique())}, n_pv = {sorted(dt.n_pv.unique())})")
out("Largest efficiency loss against gt at any (T, n_pv) point, per readout x solver x reading (negative = loss):")
worst = {}
for ro in ("gtu", "cc", "ccu", "lay"):
    for rd in ("I", "F", "P"):
        for s in SOLVERS:
            a = tt[(tt.readout == ro) & (tt.reading == rd) & (tt.solver == s)].set_index(["T", "n_pv"]).efficiency
            b = tt[(tt.readout == "gt") & (tt.reading == rd) & (tt.solver == s)].set_index(["T", "n_pv"]).efficiency
            if a.empty:
                continue
            dlt = (a - b.reindex(a.index))
            k = dlt.idxmin()
            worst[(ro, rd, s)] = dlt.min()
            out(f"  {ro:4s} {rd} {s:10s} min d_eff {dlt.min():+.3f} at T={k[0]}, n_pv={k[1]}; mean {dlt.mean():+.3f}; max gain {dlt.max():+.3f}")
out("\nKill criterion, toy: chosen readout loses <= 0.02 against gt at every point (readings I and F)")
for ro in passing or ["ccu"]:
    w = min(worst[(ro, rd, s)] for rd in ("I", "F") for s in SOLVERS)
    out(f"  {RO_LABEL[ro]:30s} worst {w:+.3f} -> {'PASS' if w >= -0.02 else 'fail'}")

# resources
er = resources(dr)
et = resources(dt)
out("\nResources (unchanged by the readout; Givens price; 1BQF 2 calls, QSVT 160 calls; rounds = max(1, pi/4/sqrt P))")
out(f"  Run 3 median: {er.n_reco.median():.0f} reco tracks, 1BQF {er.q_1bqf.median():.0f} qubits {er.cx_1bqf.median():.2e} gates; "
    f"QSVT {er.q_qsvt.median():.0f} qubits {er.cx_qsvt.median():.2e} gates")
for T in sorted(et["T"].unique()):
    x = et[et["T"] == T]
    out(f"  toy T={T}: 1BQF {x.q_1bqf.median():.0f} qubits {x.cx_1bqf.median():.2e} gates; QSVT {x.q_qsvt.median():.0f} qubits {x.cx_qsvt.median():.2e} gates")
out("\nReadout time (classical, per event, median s) by readout, Run 3 reading I: " + ", ".join(
    f"{ro} {gI[gI.readout == ro].t_readout.median():.3f}" for ro in RO_LABEL))
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")

# ── figures ──────────────────────────────────────────────────────────────────
RO_STYLE = {"gt": ("--", "w"), "ccu": ("-", None)}
# A1 merging test
fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.7), sharey=True)
for ax, s in zip(axs, SOLVERS):
    lab, c, mk = METHODS[s]
    for ro, col, m in (("gt", PS.RED, "x"), ("cc", c, "o"), ("ccu", PS.GREEN, ".")):
        x = gI[(gI.solver == s) & (gI.readout == ro)].sort_values("event")
        ax.scatter(x.ham_max_comp_hits, x.max_group_hits, s=9, marker=m, color=col, label=RO_LABEL[ro], lw=0.8)
    lim = [1, 3e3]
    ax.plot(lim, lim, color=PS.GREY, lw=0.8, ls=":", label="largest Hamiltonian component")
    ax.set(xscale="log", yscale="log", xlim=(3, 300), ylim=(1, 3e3), title=lab,
           xlabel="largest Hamiltonian component [hits]")
    ax.grid(alpha=0.25, which="both")
axs[0].set_ylabel("largest track candidate [hits]")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
save(fig, "A1_run3_group_sizes")

# A2 Run 3 bars
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
ros = list(RO_LABEL)
w = 0.26
for k, s in enumerate(SOLVERS):
    lab, c, mk = METHODS[s]
    v = I.loc[s].reindex(ros)
    for ax, met in zip(axs, ("efficiency", "ghost_rate")):
        ax.bar(np.arange(len(ros)) + (k - 1) * w, v[met], w, color=c, label=lab)
for ax, met, th in zip(axs, ("track efficiency", "ghost rate"), ((0.60, 0.70), (0.10,))):
    ax.set_xticks(np.arange(len(ros)), [RO_LABEL[r] for r in ros], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel(met)
    for t in th:
        ax.axhline(t, color=PS.GREY, ls=":", lw=0.9)
    ax.grid(alpha=0.25, axis="y")
axs[0].set_ylim(0, 1)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
axs[0].set_title(f"Run 3, {ne} events, truth-free reading", loc="left")
fig.tight_layout()
save(fig, "A2_run3_eff_ghost")


# A3/A4 toy vs n_pv
def toy_fig(metric, ylabel, name, ylim):
    fig, axs = plt.subplots(2, 3, figsize=(WIDTH, 4.6), sharex=True, sharey=True)
    for i, rd in enumerate(("I", "F")):
        for j, T in enumerate(sorted(dt["T"].unique())):
            ax = axs[i, j]
            for s in SOLVERS:
                lab, c, mk = METHODS[s]
                for ro in ("gt", "ccu"):
                    x = tt[(tt["T"] == T) & (tt.reading == rd) & (tt.solver == s) & (tt.readout == ro)].sort_values("n_pv")
                    ls, mfc = RO_STYLE[ro]
                    ax.plot(x.n_pv, x[metric], ls=ls, marker=mk, color=c, mfc=mfc or c, ms=3.2, lw=1.1,
                            label=f"{lab}, {RO_LABEL[ro]}" if (i, j) == (0, 0) else None)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)
            ax.set_title(f"T = {T}, {READING[rd]}", loc="left", fontsize=7.5)
            if i == 1:
                ax.set_xlabel("primary vertices")
        axs[i, 0].set_ylabel(ylabel)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, name)


toy_fig("efficiency", "track efficiency", "A3_toy_eff_vs_pv", (0, 1.02))
toy_fig("ghost_rate", "ghost rate", "A4_toy_ghost_vs_pv", (-0.02, 1.02))

# A5 resources
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
for s, qc, cc in (("1bqf", "q_1bqf", "cx_1bqf"), ("qsvt", "q_qsvt", "cx_qsvt")):
    lab, c, mk = METHODS[s]
    for df, m, tag in ((et, mk, "toy"), (er, "x", "Run 3")):
        axs[0].scatter(df.n_reco, df[qc], s=8, color=c, marker=m, label=f"{lab}, {tag}", lw=0.8,
                       facecolors="none" if tag == "toy" else None)
        axs[1].scatter(df.n_reco, df[cc], s=8, color=c, marker=m, lw=0.8, facecolors="none" if tag == "toy" else None)
axs[0].set(xscale="log", xlabel="reconstructible tracks", ylabel="logical qubits (system + ancilla)")
axs[1].set(xscale="log", yscale="log", xlabel="reconstructible tracks", ylabel="two-qubit gates per solved event")
for ax in axs:
    ax.grid(alpha=0.25, which="both")
axs[0].legend(frameon=False, fontsize=6.5)
fig.tight_layout()
save(fig, "A5_resources_vs_tracks")

# A6 categories
cat = {"all reconstructible": np.ones(len(tr), bool),
       "prompt, not e, >= 4 hits, p > 2 GeV": tr.prompt & ~tr.electron & (tr.n_hits >= 4) & (tr.p > 2000),
       "3 hits": tr.n_hits == 3, "|t| > 0.25": tr.t > 0.25, "p < 1 GeV": tr.p < 1000,
       "secondaries": ~tr.prompt, "electrons": tr.electron}
fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
k = 0
w = 0.13
for s in SOLVERS:
    lab, c, mk = METHODS[s]
    for ro in ("gt", "ccu"):
        v = [tr.loc[m, f"found_{s}I_{ro}"].mean() for m in cat.values()]
        ax.bar(np.arange(len(cat)) + (k - 2.5) * w, v, w, color=c, alpha=1.0 if ro == "ccu" else 0.35,
               label=f"{lab}, {RO_LABEL[ro]}")
        k += 1
ax.set_xticks(np.arange(len(cat)), list(cat), fontsize=7, rotation=15)
ax.set(ylabel="track efficiency", ylim=(0, 1))
ax.grid(alpha=0.25, axis="y")
ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper right")
fig.tight_layout()
save(fig, "A6_run3_categories")
out_cat = ["\nRun 3 per-track efficiency by category (reading I): share | " +
           " ".join(f"{s}:{ro}" for s in SOLVERS for ro in ("gt", "ccu"))]
for name, m in cat.items():
    out_cat.append(f"  {name:38s} {m.mean():.3f} | " + " ".join(
        f"{tr.loc[m, f'found_{s}I_{ro}'].mean():.3f}" for s in SOLVERS for ro in ("gt", "ccu")))
P("\n".join(out_cat))
with open(OUT / "summary.txt", "a") as f:
    f.write("\n".join(out_cat) + "\n")

# ── control: no solver (control_no_solver.py) ────────────────────────────────
ctl = []
for mode, tag in (("real", "Run 3"), ("toy", "toy")):
    f = OUT / f"{mode}_control_events.csv"
    if not f.exists():
        continue
    c = pd.read_csv(f)
    keys = ["score"] if mode == "real" else ["score", "T", "n_pv"]
    t = c.groupby(keys).apply(pooled).reset_index()
    ctl.append(f"\nControl, {tag}: every coupled segment active, coupled readout + greedy uniqueness, no solver")
    for _, r in t.iterrows():
        where = "" if mode == "real" else f"T={r['T']:<5d} n_pv={r.n_pv:<3d}"
        ctl.append(f"  score {r.score:5s} {where} eff {r.efficiency:.3f} ghost {r.ghost_rate:.3f}")
P("\n".join(ctl))
with open(OUT / "summary.txt", "a") as f:
    f.write("\n".join(ctl) + "\n")
