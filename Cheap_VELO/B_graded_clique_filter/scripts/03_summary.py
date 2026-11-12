#!/usr/bin/env python3
"""Experiment B: tables, kill-criterion verdicts and figures from 01_toy_sweep.py and 02_run3.py.

Baselines on the same events, read from the earlier outputs (not rerun):
  toy   Vertexing/Focusing_scan/outputs/full_sweep_events.csv, full_sweep_tracks.csv
  Run 3 Vertexing/Real_events/outputs/real_events.csv, real_tracks.csv
  classical / one-bit filter at <= 1 % segment false rate (classicalF, 1bqfF), QSVT comb degree 40
  at <= 1 % false (qsvt), the focused-cell hit register (hit); on Run 3 also the truth-free
  readings (classicalI, qsvtI: every segment above the isolated value).
Segment gates with the section 4.7 pricing (10_trotter_padding.py): calls x 2 product steps x
seg_cx_givens x max(1, pi / (4 sqrt(P n / 2^n))), calls = 2 (one-bit filter) and 40 (comb of
degree 40: one call per degree, the same rule as the step polynomials; paper 1 counted 160 calls
for its phase-sequence route, which would be 4x more). P = exact one-bit-filter P (seg_P_1b) and
exact comb P (seg_P_q) over the real segments. Qubits ceil(log2 n) + 2 / + 4.
Averages: toy = mean over the events of a (T, vertices) point, as in section 8; Run 3 =
median over the 103 events for P, gates and qubits (as the Run 3 README), and pooled over
truth tracks for efficiency (EventValidator per event, then the mean over events for the event
metrics; the per-category numbers pool all truth tracks).

Outputs: outputs/summary_toy.csv, outputs/summary_run3.csv, outputs/summary_hits.csv,
outputs/summary_run3_categories.csv, outputs/kill_criteria.txt, figures/B*.{pdf,png}
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT, FIGD = ROOT / "outputs", ROOT / "figures"
FS = Path("/data/bfys/gscriven/Quantum_Track_Reconstruction/Vertexing/Focusing_scan")
REV = Path("/data/bfys/gscriven/Quantum_Track_Reconstruction/Vertexing/Real_events")
sys.path.insert(0, str(FS / "scripts"))
from plot_style import plt, METHODS, WIDTH  # noqa: E402

FILT = ["ideal34", "step2", "band", "poly6", "poly10", "poly16", "poly6s", "poly10s", "poly16s", "onebit", "onebit8"]
POLY = ["poly6", "poly10", "poly16", "poly6s", "poly10s", "poly16s"]
LAB = {"ideal34": "ideal {3, 4}", "step2": "ideal step, lines >= 2", "band": "soft band > 1.5",
       "poly6": "polynomial d = 6", "poly10": "polynomial d = 10", "poly16": "polynomial d = 16",
       "poly6s": "polynomial d = 6, shifted", "poly10s": "polynomial d = 10, shifted",
       "poly16s": "polynomial d = 16, shifted", "onebit": "one-bit filter on 4I - C",
       "onebit8": "one-bit filter on 8I - C"}
BASE = {"classicalF": "classical, <= 1 % false", "1bqfF": "one-bit filter (segments), <= 1 % false",
        "qsvt": "QSVT comb d = 40, <= 1 % false", "hit": "hit register, focused cells"}
BASE_RUN3 = {**BASE, "classicalI": "classical, truth-free", "qsvtI": "QSVT comb, truth-free"}
STYLE = {  # filter: (colour, marker, linestyle)
    "ideal34": ("#6a3d9a", "D", ":"), "step2": ("#b15928", "D", ":"), "band": ("#e7298a", "v", ":"),
    "poly6": ("#1b9e77", "o", "-"), "poly10": ("#66a61e", "s", "-"), "poly16": ("#a6761d", "^", "-"),
    "poly6s": ("#1b9e77", "o", "--"), "poly10s": ("#66a61e", "s", "--"), "poly16s": ("#a6761d", "^", "--"),
    "onebit": ("#1f77b4", "x", "-"), "onebit8": ("#1f77b4", "+", "--")}
BSTYLE = {"classicalF": ("k", "o", "-"), "1bqfF": ("#1f77b4", "s", "-."), "qsvt": ("#d95f02", "^", "-."),
          "hit": ("#6a3d9a", "D", "-."), "classicalI": ("k", "o", "--"), "qsvtI": ("#d95f02", "^", "--")}


def rounds(P):
    P = np.asarray(P, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(P > 0, np.maximum(1.0, np.pi / (4 * np.sqrt(P))), np.nan)


def seg_cost(d):
    pad = d.seg_n / 2.0 ** d.seg_ns
    out = pd.DataFrame(index=d.index)
    out["1bqfF_P_padded"] = d.seg_P_1b * pad
    out["qsvt_P_padded"] = d.seg_P_q * pad
    out["1bqfF_gates"] = 2 * 2 * d.seg_cx_givens * rounds(out["1bqfF_P_padded"])
    out["qsvt_gates"] = 40 * 2 * d.seg_cx_givens * rounds(out["qsvt_P_padded"])
    out["1bqfF_qubits"] = d.seg_ns + 2
    out["qsvt_qubits"] = d.seg_ns + 4
    return out


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGD / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", FIGD / f"{name}.png")


# ── load ─────────────────────────────────────────────────────────────────────
toy = pd.read_csv(OUT / "toy_events.csv")
ttr = pd.read_csv(OUT / "toy_tracks.csv")
ref = pd.read_csv(FS / "outputs" / "full_sweep_events.csv")
ref = ref[ref["T"].isin([100, 400, 1000]) & ref.n_pv.isin([1, 5, 10, 20])]
reft = pd.read_csv(FS / "outputs" / "full_sweep_tracks.csv")
reft = reft[reft["T"].isin([100, 400, 1000]) & reft.n_pv.isin([1, 5, 10, 20])]
toy = toy.merge(pd.concat([ref, seg_cost(ref)], axis=1), on=["T", "n_pv", "rep"], how="left", suffixes=("", "_ref"))
assert len(toy) == 52 and toy.seg_n.notna().all()
r3 = pd.read_csv(OUT / "run3_events.csv")
rtr = pd.read_csv(OUT / "run3_tracks.csv")
rref = pd.read_csv(REV / "outputs" / "real_events.csv")
rreft = pd.read_csv(REV / "outputs" / "real_tracks.csv")
r3 = r3.merge(pd.concat([rref, seg_cost(rref)], axis=1), on="event", how="left", suffixes=("", "_ref"))
assert len(r3) == 103 and r3.seg_n.notna().all()
# track tables: same truth tracks in the same order per event -> align by (event key, row order)
for a, b, k in ((ttr, reft, ["T", "n_pv", "rep"]), (rtr, rreft, ["event"])):
    a["_i"] = a.groupby(k).cumcount()
    b["_i"] = b.groupby(k).cumcount()
ttr = ttr.merge(reft[["T", "n_pv", "rep", "_i", "n_hits"] + [f"found_{b}" for b in BASE]],
                on=["T", "n_pv", "rep", "_i"], suffixes=("", "_ref"))
rtr = rtr.merge(rreft[["event", "_i", "n_hits"] + [f"found_{b}" for b in BASE_RUN3]],
                on=["event", "_i"], suffixes=("", "_ref"))
assert (ttr.n_hits == ttr.n_hits_ref).all() and (rtr.n_hits == rtr.n_hits_ref).all()

# ── tables ───────────────────────────────────────────────────────────────────
rows = []
G = toy.groupby(["T", "n_pv"])
for (T, npv), g in G:
    for vs in ("scan", "truth"):
        for f in FILT:
            p = f"{vs}_{f}"
            rows.append(dict(T=T, n_pv=npv, vertices=vs, method=f, label=LAB[f], P=g[f"{p}_P"].mean(),
                             P_padded=g[f"{p}_P_padded"].mean(),
                             qubits=g[f"{p}_qubits"].mean() if f"{p}_qubits" in g else np.nan,
                             gates=g[f"{p}_gates"].mean() if f"{p}_gates" in g else np.nan,
                             efficiency=g[f"{p}_efficiency"].mean(), ghost_rate=g[f"{p}_ghost_rate"].mean(),
                             clone_fraction=g[f"{p}_clone_fraction"].mean()))
    for b in BASE:
        rows.append(dict(T=T, n_pv=npv, vertices="scan" if b == "hit" else "-", method=b, label=BASE[b],
                         P_padded=g[f"{b}_P_padded"].mean() if f"{b}_P_padded" in g else np.nan,
                         qubits=g[f"{b}_qubits"].mean() if f"{b}_qubits" in g else np.nan,
                         gates=g[f"{b}_gates"].mean() if f"{b}_gates" in g else np.nan,
                         efficiency=g[f"{b}_efficiency"].mean(), ghost_rate=g[f"{b}_ghost_rate"].mean(),
                         clone_fraction=g[f"{b}_clone_fraction"].mean()))
ST = pd.DataFrame(rows)
ST.to_csv(OUT / "summary_toy.csv", index=False)

rows = []
for vs in ("scan", "truth"):
    for f in FILT:
        p = f"{vs}_{f}"
        rows.append(dict(vertices=vs, method=f, label=LAB[f], P=r3[f"{p}_P"].median(),
                         P_padded=r3[f"{p}_P_padded"].median(), P_padded_mean=r3[f"{p}_P_padded"].mean(),
                         qubits=r3[f"{p}_qubits"].median() if f"{p}_qubits" in r3 else np.nan,
                         gates=r3[f"{p}_gates"].median() if f"{p}_gates" in r3 else np.nan,
                         efficiency=r3[f"{p}_efficiency"].mean(), ghost_rate=r3[f"{p}_ghost_rate"].mean(),
                         clone_fraction=r3[f"{p}_clone_fraction"].mean()))
for b in BASE_RUN3:
    rows.append(dict(vertices="scan" if b == "hit" else "-", method=b, label=BASE_RUN3[b],
                     P_padded=r3[f"{b.rstrip('I')}_P_padded"].median() if f"{b.rstrip('I')}_P_padded" in r3 else np.nan,
                     qubits=r3[f"{b.rstrip('I')}_qubits"].median() if f"{b.rstrip('I')}_qubits" in r3 else np.nan,
                     gates=r3[f"{b.rstrip('I')}_gates"].median() if f"{b.rstrip('I')}_gates" in r3 else np.nan,
                     efficiency=r3[f"{b}_efficiency"].mean(), ghost_rate=r3[f"{b}_ghost_rate"].mean(),
                     clone_fraction=r3[f"{b}_clone_fraction"].mean()))
SR = pd.DataFrame(rows)
SR.to_csv(OUT / "summary_run3.csv", index=False)


def hbin(n):
    return np.where(n > 5, ">5", n.astype(str))


rows = []
for name, t, cols in (("toy T=100", ttr[ttr["T"] == 100], None), ("toy T=400", ttr[ttr["T"] == 400], None),
                      ("toy T=1000", ttr[ttr["T"] == 1000], None), ("run3", rtr, None)):
    t = t.assign(hb=hbin(t.n_hits))
    meths = [f"scan_{f}" for f in FILT] + [f"truth_{f}" for f in FILT] + list(BASE_RUN3 if name == "run3" else BASE)
    for hb, g in t.groupby("hb"):
        for m in meths:
            rows.append(dict(set=name, hits=hb, method=m, n_tracks=len(g), efficiency=g[f"found_{m}"].mean()))
SH = pd.DataFrame(rows)
SH.to_csv(OUT / "summary_hits.csv", index=False)

CATS = {"all": lambda t: np.ones(len(t), bool), "3 hits": lambda t: t.n_hits == 3,
        "|t| > 0.25": lambda t: t.t > 0.25, "p < 1 GeV": lambda t: t.p < 1000,
        "secondaries": lambda t: ~t.prompt, "electrons": lambda t: t.electron,
        "toy-like": lambda t: t.prompt & ~t.electron & (t.n_hits >= 4) & (t.p > 2000)}
rows = []
for c, fn in CATS.items():
    g = rtr[fn(rtr)]
    for m in [f"scan_{f}" for f in FILT] + [f"truth_{f}" for f in FILT] + list(BASE_RUN3):
        rows.append(dict(category=c, n_tracks=len(g), frac=len(g) / len(rtr), method=m,
                         efficiency=g[f"found_{m}"].mean()))
SC = pd.DataFrame(rows)
SC.to_csv(OUT / "summary_run3_categories.csv", index=False)

# check: the baseline categories reproduce the Run 3 README (hit 0.90 toy-like, 0.03 3-hit)
pd.set_option("display.width", 250)
fmt = lambda v: f"{v:.3g}"  # noqa: E731
print("TOY (scan vertices; baselines on the same events)")
for T in (100, 400, 1000):
    s = ST[(ST["T"] == T) & (ST.vertices != "truth")]
    print(f"\nT = {T}")
    print(s.pivot_table(index="method", columns="n_pv", values=["P_padded", "efficiency", "ghost_rate", "gates"],
                        sort=False).to_string(float_format=fmt))
print("\nTOY, truth vertices")
for T in (100, 400, 1000):
    s = ST[(ST["T"] == T) & (ST.vertices == "truth")]
    print(f"\nT = {T}")
    print(s.pivot_table(index="method", columns="n_pv", values=["P_padded", "efficiency", "ghost_rate", "gates"],
                        sort=False).to_string(float_format=fmt))
print("\nRUN 3")
print(SR.to_string(index=False, float_format=fmt))
print("\nefficiency by hits per track")
print(SH.pivot_table(index=["set", "method"], columns="hits", values="efficiency", sort=False).to_string(float_format=fmt))
print("\nRun 3 categories")
print(SC.pivot_table(index="method", columns="category", values="efficiency", sort=False).to_string(float_format=fmt))

# ── kill criteria ────────────────────────────────────────────────────────────
L = []
sp = SR[SR.method.isin(POLY)]
for vs in ("scan", "truth"):
    b = sp[sp.vertices == vs].sort_values("P_padded").iloc[-1]
    L.append(f"K1 Run 3 best polynomial (degree <= 16), {vs} vertices: {b.method} padded P median "
             f"{b.P_padded:.3f} (mean {b.P_padded_mean:.3f}) -> {'KILL (< 0.30)' if b.P_padded < 0.30 else 'pass (>= 0.30)'}")
    best_spec = sp[(sp.vertices == vs) & ~sp.method.str.endswith("s")].sort_values("P_padded").iloc[-1]
    L.append(f"   same, symmetric domain only: {best_spec.method} {best_spec.P_padded:.3f}")
tl = SC[SC.category == "toy-like"].set_index("method").efficiency
for vs in ("scan", "truth"):
    vals = {f: tl[f"{vs}_{f}"] for f in POLY + ["onebit", "onebit8", "band", "step2", "ideal34"]}
    worst_poly = min(vals[f] for f in POLY)
    L.append(f"K2 Run 3 toy-like efficiency, {vs} vertices: polynomials {min(vals[f] for f in POLY):.3f}-"
             f"{max(vals[f] for f in POLY):.3f}, one-bit s=4 {vals['onebit']:.3f}, s=8 {vals['onebit8']:.3f}, band {vals['band']:.3f}, "
             f"step {vals['step2']:.3f}, ideal {{3,4}} {vals['ideal34']:.3f}; focused cells {tl['hit']:.3f} "
             f"-> {'KILL (< 0.85)' if worst_poly < 0.85 else 'pass (>= 0.85)'}")
h3 = SH[(SH.set.str.startswith("toy")) & (SH.hits == "3")]
gh = ST[ST.vertices == "scan"].groupby("method").ghost_rate.max()
for f in POLY + ["onebit", "onebit8", "step2", "band"]:
    e3 = h3[h3.method == f"scan_{f}"].set_index("set").efficiency
    kill = (e3.min() < 0.5) and (gh[f] > 0.02)
    L.append(f"K3 toy 3-hit efficiency, scan vertices, {f}: T=100/400/1000 "
             f"{e3.get('toy T=100', np.nan):.3f}/{e3.get('toy T=400', np.nan):.3f}/{e3.get('toy T=1000', np.nan):.3f}; "
             f"max ghost rate over points {gh[f]:.4f} -> {'KILL' if kill else 'pass'}")
e3i = h3[h3.method == "scan_ideal34"].set_index("set").efficiency
L.append(f"   reference ideal {{3,4}} 3-hit efficiency: {e3i.min():.3f}-{e3i.max():.3f}")
txt = "\n".join(L)
print("\n" + txt)
(OUT / "kill_criteria.txt").write_text(txt + "\n")

# ── figures ──────────────────────────────────────────────────────────────────
FIGD.mkdir(exist_ok=True)
SHOW = ["ideal34", "step2", "band", "poly6", "poly16", "poly6s", "poly16s", "onebit", "onebit8"]
NPV = (1, 5, 10, 20)


def panel_T(metric, ylabel, name, filters, bases, ylim=None, logy=False, vs="scan"):
    fig, axs = plt.subplots(1, 4, figsize=(WIDTH * 1.25, 2.9), sharey=True)
    for i, (ax, npv) in enumerate(zip(axs, NPV)):
        for f in filters:
            s = ST[(ST.method == f) & (ST.vertices == vs) & (ST.n_pv == npv)].sort_values("T")
            c, mk, ls = STYLE[f]
            ax.plot(s["T"], s[metric], color=c, marker=mk, ls=ls, ms=3.2, lw=1.1, mfc="w" if ls == "--" else c,
                    label=LAB[f] if i == 0 else None)
        for b in bases:
            s = ST[(ST.method == b) & (ST.n_pv == npv)].sort_values("T")
            if s[metric].isna().all():
                continue
            c, mk, ls = BSTYLE[b]
            ax.plot(s["T"], s[metric], color=c, marker=mk, ls=ls, ms=3.2, lw=1.1, label=BASE[b] if i == 0 else None)
        ax.set(xscale="log", xlabel="tracks per event $T$")
        if logy:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, which="both")
        ax.set_title(f"{npv} vert{'ex' if npv == 1 else 'ices'}", loc="left")
    axs[0].set_ylabel(ylabel)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.3, frameon=False,
               bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    save(fig, name)


panel_T("P_padded", "success probability $P$ (padded register)", "B01_P_vs_T", SHOW, ["1bqfF", "qsvt"],
        ylim=(1e-4, 1.1), logy=True)
CIRC = ["poly6", "poly10", "poly16", "poly6s", "poly10s", "poly16s", "onebit", "onebit8"]
panel_T("gates", "two-qubit gates per solved event", "B02_gates_vs_T", CIRC, ["1bqfF", "qsvt"], logy=True)
panel_T("qubits", "logical qubits", "B03_qubits_vs_T", ["poly6", "onebit"], ["1bqfF", "qsvt"])
panel_T("efficiency", "track efficiency", "B04_efficiency_vs_T", SHOW, list(BASE), ylim=(0, 1.02))
panel_T("ghost_rate", "ghost rate", "B05_ghost_rate_vs_T", SHOW, list(BASE), ylim=(-0.005, 0.1))
panel_T("efficiency", "track efficiency (truth vertices)", "B04b_efficiency_vs_T_truth_vertices", SHOW, list(BASE),
        ylim=(0, 1.02), vs="truth")

# efficiency by hits per track
fig, axs = plt.subplots(1, 4, figsize=(WIDTH * 1.25, 2.9), sharey=True)
order = ["3", "4", "5", ">5"]
for i, (ax, st) in enumerate(zip(axs, ("toy T=100", "toy T=400", "toy T=1000", "run3"))):
    s = SH[SH.set == st]
    bases = BASE_RUN3 if st == "run3" else BASE
    for f in SHOW:
        e = s[s.method == f"scan_{f}"].set_index("hits").efficiency.reindex(order)
        c, mk, ls = STYLE[f]
        ax.plot(range(4), e.values, color=c, marker=mk, ls=ls, ms=3.2, lw=1.1, mfc="w" if ls == "--" else c,
                label=LAB[f] if i == 3 else None)
    for b in bases:
        e = s[s.method == b].set_index("hits").efficiency.reindex(order)
        c, mk, ls = BSTYLE[b]
        ax.plot(range(4), e.values, color=c, marker=mk, ls=ls, ms=3.2, lw=1.1,
                label=BASE_RUN3[b] if i == 3 else None)
    ax.set(xticks=range(4), xticklabels=order, xlabel="hits on the truth track", ylim=(0, 1.02),
           title=st.replace("run3", "Run 3, 103 events"))
    ax.grid(alpha=0.25)
axs[0].set_ylabel("track efficiency (scan vertices)")
fig.legend(*axs[3].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.3, frameon=False,
           bbox_to_anchor=(0.5, -0.2))
fig.tight_layout()
save(fig, "B06_efficiency_by_hits")

# Run 3: P and gates per filter, and efficiency by category
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.2))
ax = axs[0]
for vs, off, mfc in (("scan", -0.15, None), ("truth", 0.15, "w")):
    s = SR[SR.vertices == vs].set_index("method").reindex(FILT)
    ax.bar(np.arange(len(FILT)) + off, s.P_padded, width=0.3, color=[STYLE[f][0] for f in FILT],
           alpha=1.0 if vs == "scan" else 0.45, label=f"{vs} vertices")
ax.axhline(0.30, color="r", ls="--", lw=0.8)
ax.text(len(FILT) - 0.5, 0.31, "kill: 0.30", color="r", fontsize=6.5, ha="right")
ax.set(xticks=range(len(FILT)), ylabel="padded P (median over events)", title="(a) Run 3 success probability")
ax.set_xticklabels([LAB[f].replace("polynomial ", "poly ").replace(", shifted", " sh") for f in FILT],
                   rotation=60, ha="right", fontsize=6.5)
ax.legend(fontsize=6.5)
ax.grid(alpha=0.25, axis="y")
ax = axs[1]
cats = list(CATS)
sel = ["scan_band", "scan_poly16s", "scan_onebit", "scan_onebit8", "scan_ideal34", "hit", "qsvtI", "classicalI"]
cl = {"scan_band": STYLE["band"][0], "scan_poly16s": STYLE["poly16s"][0], "scan_onebit": STYLE["onebit"][0],
      "scan_onebit8": "#a6cee3", "scan_ideal34": STYLE["ideal34"][0], "hit": "0.5", "qsvtI": "#d95f02", "classicalI": "k"}
lb = {"scan_band": LAB["band"], "scan_poly16s": LAB["poly16s"], "scan_onebit": LAB["onebit"],
      "scan_onebit8": LAB["onebit8"], "scan_ideal34": LAB["ideal34"], "hit": BASE["hit"], "qsvtI": BASE_RUN3["qsvtI"], "classicalI": BASE_RUN3["classicalI"]}
w = 0.8 / len(sel)
for k, m in enumerate(sel):
    e = SC[SC.method == m].set_index("category").efficiency.reindex(cats)
    ax.bar(np.arange(len(cats)) - 0.4 + w * (k + 0.5), e, width=w, color=cl[m], label=lb[m])
ax.axhline(0.85, color="r", ls="--", lw=0.8)
ax.set(xticks=range(len(cats)), ylim=(0, 1.02), ylabel="track efficiency", title="(b) Run 3 by category, scan vertices")
ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=7)
ax.legend(fontsize=5.5, ncol=2, loc="upper left")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
save(fig, "B07_run3_P_and_categories")

# the responses themselves (one toy and one Run 3 alpha)
sys.path.insert(0, str(HERE))
import b_lib as B  # noqa: E402
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
for ax, (a, lo, ttl) in zip(axs, ((4.0, -1.0, "toy, alpha = 4"),
                                  (float(r3.scan_alpha.median()), float(r3.scan_lo.median()),
                                   f"Run 3 median, alpha = {r3.scan_alpha.median():.1f}"))):
    xs = np.linspace(-a, a, 2000)
    R = B.responses(a, lo)
    for f in ["poly6", "poly10", "poly16", "poly6s", "poly10s", "poly16s", "onebit", "onebit8"]:
        c, mk, ls = STYLE[f]
        x = xs if not f.endswith("s") else np.linspace(lo, a, 2000)
        ax.plot(x, R[f](x), color=c, ls=ls, lw=1.1, label=LAB[f])
    ax.plot(xs, (xs > 1.5).astype(float), color="0.4", lw=0.8, ls=":", label="ideal step")
    for ln in range(-1, int(a) + 1):
        ax.axvline(ln, color="0.85", lw=0.6, zorder=0)
    ax.set(xlabel="eigenvalue $c$ of the hit operator", title=ttl, ylim=(-1.05, 1.1))
    ax.grid(alpha=0.2)
axs[0].set_ylabel("response $f(c)$")
axs[1].legend(fontsize=5.8, loc="lower right")
fig.tight_layout()
save(fig, "B08_responses")
