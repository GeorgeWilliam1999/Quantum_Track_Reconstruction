#!/usr/bin/env python3
"""Experiment G: shots to reach the exact-output efficiency, and the gate cost once shots count.

Reads outputs/{toy,real}_shots_{events,curves}.csv (01_shots.py).

Definitions:
  curve      efficiency against N for (event, solver, rule, k): mean over the 5 seeds. A seed that
             stopped at saturation (01_shots.py) keeps its last value at larger N.
  exact      the exact-output efficiency of the solver on that event: experiment B's readout
             (hit register) or experiment A's reading I with the coupled readout (segments).
  N90        smallest N of the grid (10 ... 1e8, half decades) with curve >= exact - 0.02.
             Not reached within 1e8 -> NaN, counted in frac_reached.
  gates_per_solved  two-qubit gates for one post-selected shot, section 4.7 pricing (experiment B
             for the hit register; segments: calls x 2 steps x Givens price x rounds, calls 2 / 40).
  gates_total = N90 x gates_per_solved (amplitude amplification inside every shot, as 4.7).
  gates_per_track = gates_total / (curve efficiency at N90 x reconstructible tracks).
Averages: medians over events of a (T, vertices) point (toy) or over the 103 Run 3 events.
Reference lines without shots: classical solver (reading I, coupled readout; computed in
01_shots.py) and the no-solver control of experiment A (score "deg",
Vertexing/Readout_A/outputs/*_control_events.csv).

Reproduction checks (written to outputs/reproduction.txt):
  hit register: the estimate rule at N -> infinity against experiment B per event
      (Cheap_VELO/B_graded_clique_filter/outputs/{toy,run3}_events.csv, scan_<filter>_efficiency);
  segments: the exact reading I + coupled readout here against experiment A per event
      (Vertexing/Readout_A/outputs/{toy,real}_readouts_events.csv, reading I, readout ccu),
      and the estimate rule at N -> infinity against the exact reading.

Outputs: outputs/summary_toy.csv, outputs/summary_run3.csv, outputs/n90_scaling.csv,
outputs/reproduction.txt, figures/G0*.{pdf,png}; the printed tables go to outputs/02_summary.log.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT, FIG = ROOT / "outputs", ROOT / "figures"
FIG.mkdir(exist_ok=True)
QTR = ROOT.parent.parent
sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/Vertexing/Focusing_scan/scripts")
from plot_style import plt, WIDTH  # noqa: E402

SOLV = ["onebit8", "poly6s", "step2", "1bqf", "qsvt"]
LAB = {"onebit8": "hit, one-bit filter on 8I - C", "poly6s": "hit, step polynomial d = 6 (shifted)",
       "step2": "hit, ideal step lines >= 2 (no circuit)", "1bqf": "segment one-bit filter",
       "qsvt": "segment QSVT comb d = 40", "classical": "classical (no shots)",
       "control": "no-solver control (no shots)"}
STY = {"onebit8": ("#1f77b4", "+", "-"), "poly6s": ("#1b9e77", "o", "-"), "step2": ("#b15928", "D", ":"),
       "1bqf": ("#1f77b4", "s", "-."), "qsvt": ("#d95f02", "^", "-."), "classical": ("k", None, "--"),
       "control": ("0.5", None, ":")}
GAP = 0.02


def load(mode):
    e = pd.read_csv(OUT / f"{mode}_shots_events.csv")
    c = pd.read_csv(OUT / f"{mode}_shots_curves.csv")
    return e, c


def fill_curves(c):
    """Full N grid per (event, solver, rule, k, seed); stopped seeds keep their last value."""
    grid = np.sort(c.N.unique())
    keys = ["event", "solver", "rule", "k", "seed"]
    idx = pd.MultiIndex.from_product([grid], names=["N"])
    out = []
    for kv, g in c.groupby(keys, sort=False):
        g = g.set_index("N")[["efficiency", "ghost_rate", "n_matched"]].reindex(grid).ffill()
        g = g.assign(**dict(zip(keys, kv))).reset_index()
        out.append(g)
    f = pd.concat(out, ignore_index=True)
    m = f.groupby(["event", "solver", "rule", "k", "N"])[["efficiency", "ghost_rate", "n_matched"]].mean().reset_index()
    return m


def n90(m, e):
    """Per (event, solver, rule, k): N90, efficiency and ghost rate there, gates."""
    ref = e.set_index(["event", "solver"])
    rows = []
    for (ev, s, r, k), g in m.groupby(["event", "solver", "rule", "k"]):
        er = ref.loc[(ev, s)]
        target = er.ref_efficiency - GAP
        ok = g[g.efficiency >= target - 1e-12].sort_values("N")
        row = dict(event=ev, solver=s, rule=r, k=k, exact=er.ref_efficiency, n_items=er.n_items,
                   n_reco=er.n_reco, P_padded=er.P_padded, qubits=er.qubits, gates_per_solved=er.gates_per_solved,
                   inf_efficiency=er[f"inf_{r}_efficiency"], inf_ghost_rate=er[f"inf_{r}_ghost_rate"],
                   eff_1e8=g.sort_values("N").efficiency.iloc[-1])
        if len(ok):
            f = ok.iloc[0]
            row.update(N90=f.N, eff_N90=f.efficiency, ghost_N90=f.ghost_rate)
            row["gates_total"] = f.N * er.gates_per_solved
            row["gates_per_track"] = row["gates_total"] / max(f.efficiency * er.n_reco, 1e-9)
        else:
            row.update(N90=np.nan)
        for c in ("T", "n_pv"):
            if c in er:
                row[c] = er[c]
        rows.append(row)
    return pd.DataFrame(rows)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", FIG / f"{name}.png")


def med(x):
    x = pd.Series(x).dropna()
    return x.median() if len(x) else np.nan


log = []
P = lambda *a: log.append(" ".join(str(x) for x in a))  # noqa: E731
pd.set_option("display.width", 250)
fmt = lambda v: f"{v:.3g}"  # noqa: E731

# ── toy ──────────────────────────────────────────────────────────────────────
et, ct = load("toy")
mt = fill_curves(ct)
nt = n90(mt, et)
nt.to_csv(OUT / "n90_toy_events.csv", index=False)
ctrl_t = pd.read_csv(QTR / "Vertexing/Readout_A/outputs/toy_control_events.csv")
ctrl_t = ctrl_t[ctrl_t.score == "deg"].set_index("event")
grp = []
for (T, npv, s, r, k), g in nt.groupby(["T", "n_pv", "solver", "rule", "k"]):
    grp.append(dict(T=T, n_pv=npv, solver=s, rule=r, k=k, events=len(g), exact=g.exact.mean(),
                    frac_reached=g.N90.notna().mean(), N90=med(g.N90), eff_N90=g.eff_N90.mean(),
                    ghost_N90=g.ghost_N90.mean(), n_items=g.n_items.median(), qubits=g.qubits.median(),
                    P_padded=g.P_padded.median(), gates_per_solved=med(g.gates_per_solved),
                    gates_total=med(g.gates_total), gates_per_track=med(g.gates_per_track)))
cl = et[et.solver == "classical"].copy()
for (T, npv), g in cl.groupby(["T", "n_pv"]):
    evs = g.event
    grp.append(dict(T=T, n_pv=npv, solver="classical", rule="-", k=0, events=len(g), exact=g.ref_efficiency.mean()))
    cc = ctrl_t.reindex(evs)
    grp.append(dict(T=T, n_pv=npv, solver="control", rule="-", k=0, events=len(cc), exact=cc.efficiency.mean(),
                    ghost_N90=cc.ghost_rate.mean()))
ST = pd.DataFrame(grp)
ST.to_csv(OUT / "summary_toy.csv", index=False)
main = ST[(ST.rule.isin(["est", "-"])) & (ST.k.isin([1, 0]))]
P("TOY: estimate rule, k = 1 (N90 = median over events; gates = median)")
for T in (100, 400, 1000):
    s = main[main["T"] == T]
    P(f"\nT = {T}")
    P(s.pivot_table(index="solver", columns="n_pv", values=["exact", "N90", "frac_reached", "gates_total",
                                                           "gates_per_track"], sort=False).to_string(float_format=fmt))
P("\nTOY, count rule vs estimate rule, k = 1 and 2: median N90 over all toy events")
P(nt.groupby(["solver", "rule", "k"]).agg(N90=("N90", "median"), reached=("N90", lambda x: x.notna().mean()),
                                          inf_eff=("inf_efficiency", "mean"), exact=("exact", "mean"))
  .to_string(float_format=fmt))

# ── Run 3 ────────────────────────────────────────────────────────────────────
er, cr = load("real")
mr = fill_curves(cr)
nr = n90(mr, er)
nr.to_csv(OUT / "n90_run3_events.csv", index=False)
ctrl_r = pd.read_csv(QTR / "Vertexing/Readout_A/outputs/real_control_events.csv")
ctrl_r = ctrl_r[ctrl_r.score == "deg"]
grp = []
for (s, r, k), g in nr.groupby(["solver", "rule", "k"]):
    grp.append(dict(solver=s, rule=r, k=k, events=len(g), exact=g.exact.mean(), frac_reached=g.N90.notna().mean(),
                    N90=med(g.N90), eff_N90=g.eff_N90.mean(), ghost_N90=g.ghost_N90.mean(),
                    n_items=g.n_items.median(), qubits=g.qubits.median(), P_padded=g.P_padded.median(),
                    gates_per_solved=med(g.gates_per_solved), gates_total=med(g.gates_total),
                    gates_per_track=med(g.gates_per_track)))
c = er[er.solver == "classical"]
grp.append(dict(solver="classical", rule="-", k=0, events=len(c), exact=c.ref_efficiency.mean(),
                ghost_N90=c.ref_ghost_rate.mean()))
grp.append(dict(solver="control", rule="-", k=0, events=len(ctrl_r), exact=ctrl_r.efficiency.mean(),
                ghost_N90=ctrl_r.ghost_rate.mean()))
SR = pd.DataFrame(grp)
SR.to_csv(OUT / "summary_run3.csv", index=False)
P("\nRUN 3 (103 events; efficiency = mean over events; N90 and gates = median)")
P(SR.to_string(index=False, float_format=fmt))

# ── N90 scaling with register size ───────────────────────────────────────────
rows = []
allN = pd.concat([nt.assign(set="toy"), nr.assign(set="run3")])
sel = allN[(allN.rule == "est") & (allN.k == 1) & allN.N90.notna()]
for s, g in sel.groupby("solver"):
    for st, gg in [("all", g)] + list(g.groupby("set")):
        if len(gg) < 3 or gg.n_items.nunique() < 2:
            continue
        a, b = np.polyfit(np.log(gg.n_items), np.log(gg.N90), 1)
        rows.append(dict(solver=s, set=st, events=len(gg), exponent=a,
                         median_N90_over_n=float((gg.N90 / gg.n_items).median()),
                         median_N90_over_nlogn=float((gg.N90 / (gg.n_items * np.log(gg.n_items))).median())))
SC = pd.DataFrame(rows)
SC.to_csv(OUT / "n90_scaling.csv", index=False)
P("\nN90 against register size n (estimate rule, k = 1): fit N90 ~ n^a")
P(SC.to_string(index=False, float_format=fmt))

# ── reproduction ─────────────────────────────────────────────────────────────
R = []
Bt = pd.read_csv(QTR / "Cheap_VELO/B_graded_clique_filter/outputs/toy_events.csv")
Bt["event"] = "T" + Bt["T"].astype(str) + "_pv" + Bt.n_pv.astype(str) + "_r" + Bt.rep.astype(str)
Br = pd.read_csv(QTR / "Cheap_VELO/B_graded_clique_filter/outputs/run3_events.csv")
At = pd.read_csv(QTR / "Vertexing/Readout_A/outputs/toy_readouts_events.csv")
Ar = pd.read_csv(QTR / "Vertexing/Readout_A/outputs/real_readouts_events.csv")
for name, e, Bx, Ax in (("toy", et, Bt, At), ("run3", er, Br, Ar)):
    Bx = Bx.set_index("event")
    for s in ("onebit8", "poly6s", "step2"):
        x = e[e.solver == s].set_index("event")
        b = Bx.reindex(x.index)[f"scan_{s}_efficiency"]
        d1 = np.abs(x.inf_est_efficiency - b).max()
        d0 = np.abs(x.ref_efficiency - b).max()
        R.append(f"{name} hit {s}: {len(x)} events; max |exact here - B| = {d0:.2e}; "
                 f"max |estimate rule at N=inf - B| = {d1:.2e}; mean {x.inf_est_efficiency.mean():.4f} vs B {b.mean():.4f}")
    Ax = Ax[(Ax.reading == "I") & (Ax.readout == "ccu")].set_index(["event", "solver"])
    for s in ("classical", "1bqf", "qsvt"):
        x = e[e.solver == s].set_index("event")
        a = Ax.reindex(pd.MultiIndex.from_product([x.index, [s]])).efficiency.values
        ok = ~np.isnan(a)
        d0 = np.abs(x.ref_efficiency.values[ok] - a[ok]).max() if ok.any() else np.nan
        line = (f"{name} segment {s}: {ok.sum()}/{len(x)} events in A; max |exact here - A| = {d0:.2e}; "
                f"mean {x.ref_efficiency.values[ok].mean():.4f} vs A {a[ok].mean():.4f}")
        if s != "classical":
            d1 = np.abs(x.inf_est_efficiency - x.ref_efficiency).max()
            line += f"; max |estimate rule at N=inf - exact| = {d1:.2e}"
        R.append(line)
(OUT / "reproduction.txt").write_text("\n".join(R) + "\n")
P("\nREPRODUCTION")
P("\n".join(R))

# ── figures ──────────────────────────────────────────────────────────────────
mt = mt.merge(et[["event", "T", "n_pv"]].drop_duplicates(), on="event")


def eff_vs_N(ax, m, ttl, refs):
    for s in SOLV:
        for k, ls in ((1, None), (2, ":")):
            g = m[(m.solver == s) & (m.rule == "est") & (m.k == k)].groupby("N").efficiency.mean()
            c, mk, l0 = STY[s]
            ax.plot(g.index, g.values, color=c, marker=mk if k == 1 else None, ms=2.8, lw=1.1 if k == 1 else 0.8,
                    ls=ls or l0, label=(LAB[s] if k == 1 else None))
    for key, val in refs.items():
        c, _, ls = STY[key]
        ax.axhline(val, color=c, ls=ls, lw=0.9, label=LAB[key])
    ax.set(xscale="log", xlabel="shots $N$", ylim=(0, 1.02), title=ttl)
    ax.grid(alpha=0.25, which="both")


fig, axs = plt.subplots(1, 4, figsize=(WIDTH * 1.25, 3.0), sharey=True)
for ax, T in zip(axs[:3], (100, 400, 1000)):
    evs = et[et["T"] == T].event.unique()
    eff_vs_N(ax, mt[mt["T"] == T], f"toy T = {T}, all vertex counts",
             {"classical": et[(et.solver == "classical") & (et["T"] == T)].ref_efficiency.mean(),
              "control": ctrl_t.reindex(evs).efficiency.mean()})
eff_vs_N(axs[3], mr, "Run 3, 103 events", {"classical": er[er.solver == "classical"].ref_efficiency.mean(),
                                         "control": ctrl_r.efficiency.mean()})
axs[0].set_ylabel("track efficiency (estimate rule; dotted k = 2)")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.3, frameon=False,
           bbox_to_anchor=(0.5, -0.15))
fig.tight_layout()
save(fig, "G01_efficiency_vs_shots")


def by_T(metric, ylabel, name, solvers, logy=True):
    fig, axs = plt.subplots(1, 4, figsize=(WIDTH * 1.25, 2.9), sharey=True)
    for i, (ax, npv) in enumerate(zip(axs, (1, 5, 10, 20))):
        for s in solvers:
            g = main[(main.solver == s) & (main.n_pv == npv)].sort_values("T")
            c, mk, ls = STY[s]
            ax.plot(g["T"], g[metric], color=c, marker=mk, ls=ls, ms=3.2, lw=1.1, label=LAB[s] if i == 0 else None)
        ax.set(xscale="log", xlabel="tracks per event $T$")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.set_title(f"{npv} vert{'ex' if npv == 1 else 'ices'}", loc="left")
    axs[0].set_ylabel(ylabel)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6.3, frameon=False,
               bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    save(fig, name)


by_T("N90", "shots to exact efficiency $-$ 0.02 ($N_{90}$)", "G02_N90_vs_T", SOLV)
by_T("gates_total", "two-qubit gates per event, shots included", "G03_total_gates_vs_T",
     ["onebit8", "poly6s", "1bqf", "qsvt"])
by_T("gates_per_track", "two-qubit gates per found track", "G03b_gates_per_track_vs_T",
     ["onebit8", "poly6s", "1bqf", "qsvt"])
by_T("qubits", "logical qubits", "G04_qubits_vs_T", ["onebit8", "poly6s", "1bqf", "qsvt"], logy=False)

# Run 3 bars
fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.9))
sr = SR[(SR.rule == "est") & (SR.k == 1)].set_index("solver").reindex(SOLV)
for ax, col, yl in zip(axs, ("N90", "gates_total", "gates_per_track"),
                       ("median $N_{90}$", "two-qubit gates per event", "two-qubit gates per found track")):
    ax.bar(range(len(SOLV)), sr[col], color=[STY[s][0] for s in SOLV])
    ax.set(yscale="log", xticks=range(len(SOLV)), ylabel=yl)
    ax.set_xticklabels(["hit 1-bit 8I-C", "hit poly d=6", "hit ideal step", "seg 1-bit", "seg QSVT"],
                       rotation=40, ha="right", fontsize=6.5)
    ax.grid(alpha=0.25, axis="y", which="both")
axs[0].set_title("Run 3, estimate rule, k = 1", loc="left")
fig.tight_layout()
save(fig, "G05_run3_shots_and_gates")

# N90 against register size
fig, ax = plt.subplots(figsize=(WIDTH * 0.55, 3.2))
for s in SOLV:
    g = sel[sel.solver == s]
    c, mk, _ = STY[s]
    ax.scatter(g.n_items, g.N90, s=8, color=c, marker=mk or "o", alpha=0.6, label=LAB[s])
x = np.logspace(2.3, 6.8, 50)
ax.plot(x, x * np.log(x), "k--", lw=0.8, label=r"$n \ln n$")
ax.set(xscale="log", yscale="log", xlabel="register items $n$ (hits or segments)", ylabel="$N_{90}$",
       title="toy and Run 3 events, estimate rule, k = 1")
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=5.8)
save(fig, "G06_N90_vs_register_size")

(OUT / "02_summary.log").write_text("\n".join(log) + "\n")
print("\n".join(log))
