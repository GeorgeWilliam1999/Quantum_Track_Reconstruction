#!/usr/bin/env python3
"""Experiment C: tables, kill-criterion verdict and figures from 01_components.py.

Reads outputs/{toy,real}_comp_{events,sizes,curves}.csv and, for the whole-register numbers
on the same events, experiment G (Cheap_VELO/G_shot_budget/outputs/n90_{toy,run3}_events.csv,
estimate rule, k = 1: N90, total gates per event with shots, gates per found track, qubits).

Definitions:
  beta90     smallest beta (shots per circuit = ceil(beta m ln m)) with the seed-mean
             efficiency >= exact - 0.02, per event, solver and hybrid threshold t.
  gates      sum over circuit components of shots x gates per solved shot, at beta90.
  per track  gates / (efficiency at beta90 x reconstructible tracks).
  t          components with <= t items are solved classically (t = 1: every component with a
             coupling is a circuit).
Averages: medians over the events of a (T, vertices) point, or over the 103 Run 3 events.

Kill criterion (plan, candidate C): stop if more than 5 % of the Run 3 events have a component
above 10 qubits, or if the total gates per event (shots included) do not fall below the
whole-register count. Qubits are counted with the ancillas (system + 2 or + 4).

Outputs: outputs/summary_toy.csv, outputs/summary_run3.csv, outputs/exactness.txt,
outputs/kill_criterion.txt, outputs/size_distribution.csv, figures/C0*.{pdf,png};
printed tables in outputs/02_summary.log.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT, FIG = ROOT / "outputs", ROOT / "figures"
FIG.mkdir(exist_ok=True)
G = ROOT.parent / "G_shot_budget" / "outputs"
sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/Vertexing/Focusing_scan/scripts")
from plot_style import plt, WIDTH  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")
SOLV = ["onebit8", "poly6s", "1bqf", "qsvt"]
LAB = {"onebit8": "hit, one-bit filter on 8I - C", "poly6s": "hit, step polynomial d = 6 (shifted)",
       "1bqf": "segment one-bit filter", "qsvt": "segment QSVT comb d = 40"}
COL = {"onebit8": "#1f77b4", "poly6s": "#1b9e77", "1bqf": "#6a3d9a", "qsvt": "#d95f02"}
MK = {"onebit8": "+", "poly6s": "o", "1bqf": "s", "qsvt": "^"}
GAP = 0.02
THRESH = (1, 2, 4, 8)
log = []
P = lambda *a: log.append(" ".join(str(x) for x in a))  # noqa: E731
pd.set_option("display.width", 250)
fmt = lambda v: f"{v:.3g}"  # noqa: E731


def beta90(cur, ev):
    m = cur.groupby(["event", "solver", "t", "beta"]).agg(
        efficiency=("efficiency", "mean"), ghost_rate=("ghost_rate", "mean"), shots=("shots", "mean"),
        gates=("gates", "mean"), n_circuits=("n_circuits", "first")).reset_index()
    ex = ev.set_index(["event", "solver"])
    rows = []
    for (e, s, t), g in m.groupby(["event", "solver", "t"]):
        r = ex.loc[(e, s)]
        ok = g[g.efficiency >= r.exact_efficiency - GAP - 1e-12].sort_values("beta")
        row = dict(event=e, solver=s, t=t, exact=r.exact_efficiency, n_reco=r.n_reco,
                   n_circuits=int(g.n_circuits.iloc[0]))
        if len(ok):
            f = ok.iloc[0]
            row.update(beta90=f.beta, eff=f.efficiency, ghost=f.ghost_rate, shots=f.shots, gates=f.gates,
                       gates_per_track=f.gates / max(f.efficiency * r.n_reco, 1e-9))
        rows.append(row)
    return pd.DataFrame(rows)


def whole(name):
    w = pd.read_csv(G / f"n90_{name}_events.csv")
    w = w[(w.rule == "est") & (w.k == 1)]
    return w.set_index(["event", "solver"])[["N90", "gates_total", "gates_per_track", "qubits", "gates_per_solved"]]


def med(x):
    x = pd.Series(x).dropna()
    return x.median() if len(x) else np.nan


def table(ev, b9, W, keys):
    rows = []
    for k, g in ev.groupby(keys) if keys else [((), ev)]:
        k = k if isinstance(k, tuple) else (k,)
        for s in SOLV:
            e = g[g.solver == s]
            evs = e.event
            w = W.reindex(pd.MultiIndex.from_arrays([evs, [s] * len(evs)]))
            row = dict(zip(keys, k[:len(keys)]), solver=s, events=len(e), exact=e.exact_efficiency.mean(),
                       n_items=e.n_items.median(), circuits_t1=e.n_circ_t1.median(),
                       circuits_t8=e.n_circ_t8.median(), max_items=e.max_items.median(),
                       max_qubits_med=e.max_qubits.median(), max_qubits_max=e.max_qubits.max(),
                       whole_qubits=w.qubits.median(), whole_N90=w.N90.median(),
                       whole_gates=w.gates_total.median(), whole_gates_per_track=w.gates_per_track.median(),
                       t_eigh_med=e.t_eigh.median(), t_eigh_max=e.t_eigh.max())
            for t in THRESH:
                b = b9[(b9.solver == s) & (b9.t == t) & b9.event.isin(evs)]
                row[f"reached_t{t}"] = b.beta90.notna().mean()
                row[f"beta90_t{t}"] = med(b.beta90)
                row[f"shots_t{t}"] = med(b.shots)
                row[f"gates_t{t}"] = med(b.gates)
                row[f"gates_per_track_t{t}"] = med(b.gates_per_track)
                row[f"eff_t{t}"] = b.eff.mean()
            # per-event ratio whole / components (t = 1)
            b = b9[(b9.solver == s) & (b9.t == 1)].set_index("event").reindex(evs)
            row["ratio_whole_over_comp"] = med(w.gates_total.values / b.gates.values)
            rows.append(row)
    return pd.DataFrame(rows)


et = pd.read_csv(OUT / "toy_comp_events.csv")
ct = pd.read_csv(OUT / "toy_comp_curves.csv")
er = pd.read_csv(OUT / "real_comp_events.csv")
cr = pd.read_csv(OUT / "real_comp_curves.csv")
bt, br = beta90(ct, et), beta90(cr, er)
bt.to_csv(OUT / "beta90_toy_events.csv", index=False)
br.to_csv(OUT / "beta90_run3_events.csv", index=False)
Wt, Wr = whole("toy"), whole("run3")
ST = table(et, bt, Wt, ["T", "n_pv"])
SR = table(er, br, Wr, [])
ST.to_csv(OUT / "summary_toy.csv", index=False)
SR.to_csv(OUT / "summary_run3.csv", index=False)

show = ["solver", "exact", "circuits_t1", "circuits_t8", "max_items", "max_qubits_med", "max_qubits_max", "whole_qubits",
        "whole_N90", "shots_t1", "whole_gates", "gates_t1", "gates_t2", "gates_t4", "gates_t8", "ratio_whole_over_comp",
        "whole_gates_per_track", "gates_per_track_t1", "reached_t1", "t_eigh_med"]
for T in (100, 400, 1000):
    for npv in (1, 5, 10, 20):
        s = ST[(ST["T"] == T) & (ST.n_pv == npv)]
        P(f"\nTOY T = {T}, {npv} vertices")
        P(s[show].to_string(index=False, float_format=fmt))
P("\nRUN 3 (103 events)")
P(SR[show].to_string(index=False, float_format=fmt))

# ── exactness ────────────────────────────────────────────────────────────────
X = []
for name, e in (("toy", et), ("run3", er)):
    gref = pd.read_csv(G / f"{'toy' if name == 'toy' else 'real'}_shots_events.csv").set_index(["event", "solver"])
    for s in SOLV:
        x = e[e.solver == s].set_index("event")
        ref = gref.reindex(pd.MultiIndex.from_arrays([x.index, [s] * len(x)])).ref_efficiency.values
        line = (f"{name} {s}: {len(x)} events; max |per-component - whole-register output| = "
                f"{x.max_diff_vs_whole.max():.1e}; max |per-component - independent whole-matrix evaluation| = "
                f"{x.max_diff_vs_independent.max():.1e}")
        if s in ("onebit8", "poly6s"):
            line += f"; max |efficiency - G exact| = {np.nanmax(np.abs(x.exact_efficiency.values - ref)):.1e}"
        else:
            line += (f"; on-set differences {int(x.n_on_diff.sum())}; max |efficiency per-component - whole "
                     f"register, both rounded| = {np.max(np.abs(x.exact_efficiency - x.whole_rounded_efficiency)):.1e}"
                     f"; max |whole register raw - G exact| = {np.nanmax(np.abs(x.whole_raw_efficiency.values - ref)):.1e}"
                     f"; tie effect (rounded - raw): mean {np.mean(x.whole_rounded_efficiency - x.whole_raw_efficiency):+.4f}, "
                     f"range {np.min(x.whole_rounded_efficiency - x.whole_raw_efficiency):+.3f} to "
                     f"{np.max(x.whole_rounded_efficiency - x.whole_raw_efficiency):+.3f}, events changed "
                     f"{int(np.sum(np.abs(x.whole_rounded_efficiency - x.whole_raw_efficiency) > 1e-9))}")
        X.append(line)
(OUT / "exactness.txt").write_text("\n".join(X) + "\n")
P("\nEXACTNESS")
P("\n".join(X))

# ── kill criterion ───────────────────────────────────────────────────────────
K = []
for s in SOLV:
    e = er[er.solver == s]
    f10 = float((e.max_qubits > 10).mean())
    f10s = float((e.max_qubits_sys > 10).mean())
    b = br[(br.solver == s) & (br.t == 1)].set_index("event").reindex(e.event)
    w = Wr.reindex(pd.MultiIndex.from_arrays([e.event, [s] * len(e)]))
    below = float(np.mean(b.gates.values < w.gates_total.values))
    K.append(f"{s}: Run 3 events with a component above 10 qubits (with ancillas) {f10:.3f} "
             f"(system register only {f10s:.3f}; largest {int(e.max_qubits.max())} qubits); "
             f"total gates per event: components (t = 1) median {med(b.gates):.3g} vs whole register "
             f"{med(w.gates_total):.3g}, components lower in {below:.3f} of events -> "
             f"{'KILL' if (f10 > 0.05 or med(b.gates) >= med(w.gates_total)) else 'pass'}")
(OUT / "kill_criterion.txt").write_text("\n".join(K) + "\n")
P("\nKILL CRITERION")
P("\n".join(K))

# ── size distribution ────────────────────────────────────────────────────────
SZ = []
for name, f in (("toy", "toy_comp_sizes.csv"), ("run3", "real_comp_sizes.csv")):
    z = pd.read_csv(OUT / f)
    if name == "toy":
        groups = [(f"toy T={T}", g) for T, g in z.groupby("T")]
    else:
        groups = [("run3", z)]
    for lab, g in groups:
        for (s, lo, hi), gg in g.groupby(["solver", "size_lo", "size_hi"]):
            SZ.append(dict(set=lab, solver=s, size_lo=lo, size_hi=hi, mean_per_event=gg.n_comp.mean()))
SZ = pd.DataFrame(SZ)
SZ.to_csv(OUT / "size_distribution.csv", index=False)
P("\nCOMPONENTS PER EVENT BY SIZE (mean)")
P(SZ.pivot_table(index=["set", "solver"], columns="size_lo", values="mean_per_event").to_string(float_format=fmt))

(OUT / "02_summary.log").write_text("\n".join(log) + "\n")
print("\n".join(log))


# ── figures ──────────────────────────────────────────────────────────────────
def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", FIG / f"{name}.png")


def by_T(cols, ylabel, name, logy=True, ylim=None):
    """cols: list of (column, linestyle, label suffix, marker fill)."""
    fig, axs = plt.subplots(1, 4, figsize=(WIDTH * 1.25, 2.9), sharey=True)
    for i, (ax, npv) in enumerate(zip(axs, (1, 5, 10, 20))):
        for s in SOLV:
            g = ST[(ST.solver == s) & (ST.n_pv == npv)].sort_values("T")
            for col, ls, suf, mfc in cols:
                ax.plot(g["T"], g[col], color=COL[s], marker=MK[s], ls=ls, ms=3.2, lw=1.1,
                        mfc=COL[s] if mfc else "w", label=(LAB[s] + suf) if i == 0 else None)
        ax.set(xscale="log", xlabel="tracks per event $T$")
        if logy:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, which="both")
        ax.set_title(f"{npv} vert{'ex' if npv == 1 else 'ices'}", loc="left")
    axs[0].set_ylabel(ylabel)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=6.0, frameon=False,
               bbox_to_anchor=(0.5, -0.32 if len(cols) > 2 else -0.2))
    fig.tight_layout()
    save(fig, name)


by_T([("max_qubits_med", "-", ": largest component", True), ("whole_qubits", ":", ": whole register (G)", False)],
     "logical qubits (with ancillas)", "C01_qubits_vs_T", logy=False)
by_T([("circuits_t1", "-", ": circuits, every coupled component", True),
      ("circuits_t8", "--", ": circuits, components > 8 items", False)], "circuits per event", "C02_circuits_vs_T")
by_T([("whole_gates", ":", ": whole register (G)", False), ("gates_t1", "-", ": components", True),
      ("gates_t8", "--", ": hybrid, components > 8 items", False)],
     "two-qubit gates per event, shots included", "C03_total_gates_vs_T")
by_T([("whole_gates_per_track", ":", ": whole register (G)", False), ("gates_per_track_t1", "-", ": components", True)],
     "two-qubit gates per found track", "C04_gates_per_track_vs_T")

fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 3.0))
x = np.arange(len(SOLV))
sr = SR.set_index("solver").reindex(SOLV)
ax = axs[0]
ax.bar(x - 0.2, sr.whole_qubits, 0.4, color="0.6", label="whole register (G)")
ax.bar(x + 0.2, sr.max_qubits_med, 0.4, color=[COL[s] for s in SOLV], label="largest component (median)")
ax.scatter(x + 0.2, sr.max_qubits_max, color="k", marker="_", s=60, zorder=3, label="largest component (max)")
ax.axhline(10, color="r", ls="--", lw=0.8)
ax.set(ylabel="logical qubits (with ancillas)", title="(a) width")
ax.legend(fontsize=5.8)
ax = axs[1]
w = 0.2
for j, (col, lab, c) in enumerate((("whole_gates", "whole register (G)", "0.6"), ("gates_t1", "components", None),
                                   ("gates_t8", "hybrid > 8 items", None))):
    ax.bar(x - 0.3 + w * (j + 0.5), sr[col], w, color=c or [COL[s] for s in SOLV], alpha=1 if j < 2 else 0.5, label=lab)
ax.set(yscale="log", ylabel="two-qubit gates per event, shots included", title="(b) gates")
ax.legend(fontsize=5.8)
ax = axs[2]
ax.bar(x - 0.2, sr.whole_gates_per_track, 0.4, color="0.6", label="whole register (G)")
ax.bar(x + 0.2, sr.gates_per_track_t1, 0.4, color=[COL[s] for s in SOLV], label="components")
ax.set(yscale="log", ylabel="two-qubit gates per found track", title="(c) per found track")
ax.legend(fontsize=5.8)
for a in axs:
    a.set_xticks(x)
    a.set_xticklabels(["hit 1-bit 8I-C", "hit poly d=6", "seg 1-bit", "seg QSVT"], rotation=35, ha="right", fontsize=6.5)
    a.grid(alpha=0.25, axis="y", which="both")
fig.suptitle("Run 3, 103 events (medians)", fontsize=8.5, x=0.08, ha="left")
fig.tight_layout()
save(fig, "C05_run3_bars")

fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
for ax, st in zip(axs, ("toy T=1000", "run3")):
    z = SZ[SZ.set == st]
    for s in SOLV:
        g = z[z.solver == s].sort_values("size_lo")
        ax.plot(g.size_lo, g.mean_per_event, color=COL[s], marker=MK[s], ms=3.2, lw=1.1, label=LAB[s])
    ax.set(xscale="log", yscale="log", xlabel="component size (items, lower edge of bin)",
           title=st.replace("run3", "Run 3, 103 events"))
    ax.grid(alpha=0.25, which="both")
axs[0].set_ylabel("components per event")
axs[1].legend(fontsize=6)
fig.tight_layout()
save(fig, "C06_component_sizes")
