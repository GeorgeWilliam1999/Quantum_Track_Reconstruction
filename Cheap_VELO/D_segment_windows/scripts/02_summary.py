#!/usr/bin/env python3
"""Experiment D: tables, kill verdict and figures from 01_windows.py.

Definitions (as experiment C):
  efficiency, ghost rate   pooled: sum matched / sum reconstructible, sum ghosts / sum candidates
                           (exact output, reading I, coupled readout with the deterministic tie-break)
  beta90                   1BQF only: smallest beta (shots per component ceil(beta m ln m)) whose
                           seed-mean efficiency >= exact_rand - 0.02 (exact output with ties broken at
                           random, as shots do)
  gates                    sum over components of shots x gates per solved shot at beta90
  gates per found track    gates / (efficiency at beta90 x reconstructible tracks); medians over events
Reference: the hit register (one-bit filter on 8I - C) of experiment C on the same events:
Cheap_VELO/C_small_circuits/outputs/{beta90_run3_events,real_comp_events}.csv (t = 1) and, for Run 3
categories, Cheap_VELO/B_graded_clique_filter/outputs/run3_tracks.csv (found_scan_onebit8).
Kill criterion (plan, candidate D, as set 12 Nov): stop D if no window keeps Run 3 efficiency >= 0.85
while cutting the 1BQF gates per found track by >= 2x against C's 1.1e5.
Outputs: outputs/summary_run3.csv, outputs/summary_toy.csv, outputs/kill_criterion.txt,
outputs/02_summary.log, figures/D0*.{pdf,png}
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
CDIR = ROOT.parent / "C_small_circuits" / "outputs"
BDIR = ROOT.parent / "B_graded_clique_filter" / "outputs"
sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/Vertexing/Focusing_scan/scripts")
from plot_style import plt, WIDTH  # noqa: E402

warnings.filterwarnings("ignore")
CONFIGS = ["L4", "L2", "W0", "W1", "W2", "W3", "W4", "W5"]
CLAB = {"L4": "eps x4", "L2": "eps x2", "W0": "baseline", "W1": "W1 0.60/10mm", "W2": "W2 0.45/3mm",
        "W3": "W3 0.35/1mm", "W4": "W4 0.30/0.3mm", "W5": "W5 0.25/0.1mm"}
SOLV = ["classical", "1bqf", "qsvt", "control"]
COL = {"classical": "k", "1bqf": "#6a3d9a", "qsvt": "#d95f02", "control": "0.5", "hit": "#1f77b4"}
C_REF = 1.1e5
GAP = 0.02
log = []
P = lambda *a: log.append(" ".join(str(x) for x in a))  # noqa: E731
pd.set_option("display.width", 260)
fmt = lambda v: f"{v:.3g}"  # noqa: E731


def pooled(g):
    return pd.Series(dict(efficiency=g.n_matched.sum() / max(g.n_reconstructible.sum(), 1),
                          ghost_rate=g.n_ghosts.sum() / max(g.n_candidates.sum(), 1)))


def beta90(cur, ev):
    m = cur.groupby(["event", "config", "beta"]).agg(efficiency=("efficiency", "mean"), gates=("gates", "mean"),
                                                     shots=("shots", "mean")).reset_index()
    ref = ev[ev.solver == "1bqf"].set_index(["event", "config"])
    rows = []
    for (e, c), g in m.groupby(["event", "config"]):
        r = ref.loc[(e, c)]
        ok = g[g.efficiency >= r.exact_rand_efficiency - GAP - 1e-12].sort_values("beta")
        row = dict(event=e, config=c, n_reco=r.n_reco)
        if len(ok):
            f = ok.iloc[0]
            row.update(beta90=f.beta, eff90=f.efficiency, shots=f.shots, gates=f.gates,
                       gates_per_track=f.gates / max(f.efficiency * r.n_reco, 1e-9))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(ev, b9, keys):
    rows = []
    for k, g in ev.groupby(keys) if keys else [((), ev)]:
        k = k if isinstance(k, tuple) else (k,)
        for c in CONFIGS:
            gc = g[g.config == c]
            if gc.empty:
                continue
            one = gc.drop_duplicates("event")
            row = dict(zip(keys, k), config=c, events=len(one), n_seg_med=one.n_seg.median(),
                       true_kept=one.true_kept.mean(), comps_med=one.n_comp2.median(),
                       max_comp_med=one.max_comp.median(), max_comp_max=one.max_comp.max(),
                       skipped=int((gc.solver == "skipped").sum()), t_eigh_med=one.t_eigh.median())
            for s in SOLV:
                gs = gc[gc.solver == s]
                if len(gs):
                    pl = pooled(gs)
                    row[f"{s}_eff"], row[f"{s}_ghost"] = pl.efficiency, pl.ghost_rate
            q = gc[gc.solver == "1bqf"]
            row["qubits_1bqf_med"], row["qubits_1bqf_max"] = q.max_qubits.median(), q.max_qubits.max()
            row["qsvt_gates_solved_med"] = gc[gc.solver == "qsvt"].sum_gates_solved.median()
            bb = b9[(b9.config == c) & b9.event.isin(one.event)]
            row.update(reached=bb.beta90.notna().mean(), shots_med=bb.shots.median(), gates_med=bb.gates.median(),
                       gates_per_track_med=bb.gates_per_track.median(), eff_at_beta90=bb.eff90.mean())
            rows.append(row)
    return pd.DataFrame(rows)


# ── Run 3 ────────────────────────────────────────────────────────────────────
er = pd.read_csv(OUT / "real_win_events.csv")
cr = pd.read_csv(OUT / "real_win_curves.csv")
tr = pd.read_csv(OUT / "real_win_tracks.csv")
br = beta90(cr, er)
br.to_csv(OUT / "beta90_run3_events.csv", index=False)
SR = summarise(er, br, [])
SR.to_csv(OUT / "summary_run3.csv", index=False)
cats = {"toy-like": tr.prompt & ~tr.electron & (tr.n_hits >= 4) & (tr.p > 2000), "3-hit": tr.n_hits == 3,
        "secondaries": ~tr.prompt, "p<1GeV": tr.p < 1000}
CAT = []
for c in CONFIGS:
    for s in SOLV:
        col = f"found_{c}_{s}"
        if col in tr:
            CAT.append(dict(config=c, solver=s, **{k: tr.loc[m, col].mean() for k, m in cats.items()}))
CAT = pd.DataFrame(CAT)
CAT.to_csv(OUT / "summary_run3_categories.csv", index=False)
# hit-register reference (C / B, same events)
cb = pd.read_csv(CDIR / "beta90_run3_events.csv")
cb = cb[(cb.solver == "onebit8") & (cb.t == 1)]
ce = pd.read_csv(CDIR / "real_comp_events.csv")
ce = ce[ce.solver == "onebit8"]
btr = pd.read_csv(BDIR / "run3_tracks.csv")
bcats = {"toy-like": btr.prompt & ~btr.electron & (btr.n_hits >= 4) & (btr.p > 2000), "3-hit": btr.n_hits == 3,
         "secondaries": ~btr.prompt, "p<1GeV": btr.p < 1000}
HIT = dict(eff=float((btr.found_scan_onebit8).mean()), gates=cb.gates.median(), gpt=cb.gates_per_track.median(),
           qubits=ce.max_qubits.median(), **{k: btr.loc[m, "found_scan_onebit8"].mean() for k, m in bcats.items()})
P("RUN 3, 103 events: per window configuration")
P(SR.to_string(index=False, float_format=fmt))
P("\nRun 3 efficiency by category (exact output)")
P(CAT.pivot_table(index="config", columns="solver", values=list(cats), sort=False).reindex(CONFIGS)
  .to_string(float_format=fmt))
P(f"\nhit register reference (one-bit 8I - C, scan vertices; C / B): efficiency {HIT['eff']:.3f}, "
  f"toy-like {HIT['toy-like']:.3f}, 3-hit {HIT['3-hit']:.3f}, secondaries {HIT['secondaries']:.3f}, "
  f"p<1GeV {HIT['p<1GeV']:.3f}; gates per event {HIT['gates']:.3g}, per found track {HIT['gpt']:.3g}, "
  f"largest component {HIT['qubits']:.0f} qubits")

# ── toy ──────────────────────────────────────────────────────────────────────
et = pd.read_csv(OUT / "toy_win_events.csv")
ct = pd.read_csv(OUT / "toy_win_curves.csv")
bt = beta90(ct, et)
bt.to_csv(OUT / "beta90_toy_events.csv", index=False)
ST = summarise(et, bt, ["T", "n_pv"])
ST.to_csv(OUT / "summary_toy.csv", index=False)
show = ["config", "n_seg_med", "true_kept", "comps_med", "max_comp_med", "max_comp_max", "skipped", "classical_eff",
        "1bqf_eff", "qsvt_eff", "control_eff", "1bqf_ghost", "qubits_1bqf_max", "gates_med", "gates_per_track_med",
        "reached", "t_eigh_med"]
for T in (100, 400, 1000):
    for npv in (1, 20):
        P(f"\nTOY T = {T}, {npv} vertices")
        P(ST[(ST["T"] == T) & (ST.n_pv == npv)][show].to_string(index=False, float_format=fmt))

# ── kill criterion ───────────────────────────────────────────────────────────
K = []
best = None
for _, r in SR.iterrows():
    ok_eff = r["1bqf_eff"] >= 0.85
    ok_g = r.gates_per_track_med <= C_REF / 2
    K.append(f"{r.config:3s} ({CLAB[r.config]}): 1BQF efficiency {r['1bqf_eff']:.3f}, gates per found track "
             f"{r.gates_per_track_med:.3g} ({C_REF / r.gates_per_track_med:.2f}x below C's 1.1e5), largest component "
             f"median {r.max_comp_med:.0f} / max {r.max_comp_max:.0f} -> {'meets both' if ok_eff and ok_g else 'no'}")
    if ok_eff and ok_g and r.config.startswith("W") and r.config != "W0":
        best = r.config
K.append("VERDICT: " + (f"continue D ({best} meets both)" if best else
                        "STOP D: no window keeps Run 3 efficiency >= 0.85 while cutting gates per found track >= 2x"))
(OUT / "kill_criterion.txt").write_text("\n".join(K) + "\n")
P("\nKILL CRITERION")
P("\n".join(K))
(OUT / "02_summary.log").write_text("\n".join(log) + "\n")
print("\n".join(log))


# ── figures ──────────────────────────────────────────────────────────────────
def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", FIG / f"{name}.png")


x = np.arange(len(CONFIGS))
s = SR.set_index("config").reindex(CONFIGS)
fig, axs = plt.subplots(1, 3, figsize=(WIDTH * 1.15, 3.1))
ax = axs[0]
ax.plot(x, s.n_seg_med, "o-", color="k", ms=3, label="segments kept (median)")
ax.plot(x, s.max_comp_max, "s-", color="#6a3d9a", ms=3, label="largest component (max over events)")
ax.plot(x, s.max_comp_med, "s--", color="#6a3d9a", mfc="w", ms=3, label="largest component (median)")
ax.set(yscale="log", ylabel="segments", title="(a) register and components")
ax.legend(fontsize=5.8)
ax = axs[1]
for sv in SOLV:
    ax.plot(x, s[f"{sv}_eff"], "o-", color=COL[sv], ms=3, label=sv.replace("1bqf", "1BQF").replace("qsvt", "QSVT"))
ax.plot(x, s.true_kept, ":", color="g", label="true segments kept")
ax.axhline(HIT["eff"], color=COL["hit"], ls="--", lw=0.9, label="hit register 8I - C")
ax.axhline(0.85, color="r", ls="--", lw=0.8)
ax.set(ylim=(0, 1.02), ylabel="track efficiency (exact output)", title="(b) efficiency")
ax.legend(fontsize=5.8)
ax = axs[2]
ax.plot(x, s.gates_per_track_med, "s-", color=COL["1bqf"], ms=3, label="1BQF, components, shots included")
ax.axhline(HIT["gpt"], color=COL["hit"], ls="--", lw=0.9, label="hit register 8I - C (C)")
ax.axhline(C_REF / 2, color="r", ls="--", lw=0.8, label="kill line: C / 2")
ax.set(yscale="log", ylabel="two-qubit gates per found track", title="(c) cost")
ax.legend(fontsize=5.8)
for a in axs:
    a.set_xticks(x)
    a.set_xticklabels([CLAB[c] for c in CONFIGS], rotation=50, ha="right", fontsize=6)
    a.grid(alpha=0.25, which="both")
fig.suptitle("Run 3, 103 events", fontsize=8.5, x=0.06, ha="left")
fig.tight_layout()
save(fig, "D01_run3_windows")

fig, ax = plt.subplots(figsize=(WIDTH * 0.55, 3.3))
for c in CONFIGS:
    r = s.loc[c]
    ax.scatter(r.gates_per_track_med, r["1bqf_eff"], color=COL["1bqf"], s=14)
    ax.annotate(CLAB[c], (r.gates_per_track_med, r["1bqf_eff"]), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
ax.scatter(HIT["gpt"], HIT["eff"], color=COL["hit"], marker="D", s=14)
ax.annotate("hit register 8I - C", (HIT["gpt"], HIT["eff"]), fontsize=5.5, xytext=(3, -8), textcoords="offset points")
ax.axhline(0.85, color="r", ls="--", lw=0.8)
ax.axvline(C_REF / 2, color="r", ls="--", lw=0.8)
ax.set(xscale="log", xlabel="two-qubit gates per found track (1BQF, shots included)", ylabel="track efficiency",
       title="Run 3: the trade-off (D passes only above and left of both red lines)")
ax.grid(alpha=0.25, which="both")
save(fig, "D02_run3_tradeoff")

fig, axs = plt.subplots(1, 4, figsize=(WIDTH * 1.25, 2.9), sharey=True)
for ax, cat in zip(axs, cats):
    t = CAT.pivot_table(index="config", columns="solver", values=cat).reindex(CONFIGS)
    for sv in SOLV:
        ax.plot(x, t[sv], "o-", color=COL[sv], ms=3, label=sv.replace("1bqf", "1BQF").replace("qsvt", "QSVT"))
    ax.axhline(HIT[cat], color=COL["hit"], ls="--", lw=0.9, label="hit register 8I - C")
    ax.set(title=cat, ylim=(0, 1.02), xticks=x)
    ax.set_xticklabels([CLAB[c] for c in CONFIGS], rotation=55, ha="right", fontsize=5.5)
    ax.grid(alpha=0.25)
axs[0].set_ylabel("track efficiency, Run 3")
axs[0].legend(fontsize=5.8)
fig.tight_layout()
save(fig, "D03_run3_categories")

fig, axs = plt.subplots(1, 4, figsize=(WIDTH * 1.25, 2.9), sharey=True)
for i, (ax, npv) in enumerate(zip(axs, (1, 5, 10, 20))):
    for c, ls in (("L4", ":"), ("W0", "-"), ("W2", "--"), ("W4", "-.")):
        g = ST[(ST.config == c) & (ST.n_pv == npv)].sort_values("T")
        ax.plot(g["T"], g.gates_per_track_med, ls, marker="s", color=COL["1bqf"], ms=3, label=CLAB[c] if i == 0 else None)
    ax.set(xscale="log", yscale="log", xlabel="tracks per event $T$", title=f"{npv} vert{'ex' if npv == 1 else 'ices'}")
    ax.grid(alpha=0.25, which="both")
axs[0].set_ylabel("1BQF gates per found track")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.3, frameon=False,
           bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
save(fig, "D04_toy_gates_per_track_vs_T")

fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
for ax, col, yl in zip(axs, ("max_comp_max", "1bqf_eff"), ("largest component (segments, max)", "1BQF efficiency")):
    for T, mk in ((100, "o"), (400, "s"), (1000, "^")):
        g = ST[(ST["T"] == T)].groupby("config")[col].max() if col.startswith("max") else \
            ST[(ST["T"] == T)].groupby("config")[col].mean()
        ax.plot(x, g.reindex(CONFIGS), "-" + mk, ms=3, label=f"T = {T}")
    ax.set(ylabel=yl, xticks=x, yscale="log" if col.startswith("max") else "linear")
    ax.set_xticklabels([CLAB[c] for c in CONFIGS], rotation=50, ha="right", fontsize=6)
    ax.grid(alpha=0.25, which="both")
axs[0].legend(fontsize=6)
fig.suptitle("toy, all vertex counts", fontsize=8.5, x=0.06, ha="left")
fig.tight_layout()
save(fig, "D05_toy_components_and_efficiency")
