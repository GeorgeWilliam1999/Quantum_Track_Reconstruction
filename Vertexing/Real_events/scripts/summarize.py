#!/usr/bin/env python3
"""Tables and figures for the Run 3 simulated events, next to the toy at a similar size.

Reads outputs/real_{events,vertices,tracks}.csv (run_real.py) and, for the toy comparison,
Vertexing/Focusing_scan/outputs/full_sweep_{events,vertices}.csv at T = 200, 5 vertices
(the closest toy point: the real events have a median of ~230 reconstructible tracks and
~5.6 reconstructible PVs).  Writes outputs/real_summary.txt and figures/*.png|pdf.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_lib as R  # noqa: E402
import fs_lib as L  # noqa: E402

OUT, FIG = HERE.parent / "outputs", HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
TOY = Path(R.FS).parent / "outputs"
d = pd.read_csv(OUT / "real_events.csv")
v = pd.read_csv(OUT / "real_vertices.csv")
t = pd.read_csv(OUT / "real_tracks.csv")
toy = pd.read_csv(TOY / "full_sweep_events.csv")
toy = toy[(toy["T"] == 200) & (toy.n_pv == 5)]
toyv = pd.read_csv(TOY / "full_sweep_vertices.csv")
toyv = toyv[(toyv["T"] == 200) & (toyv.n_pv == 5)]
hold = d.event.str.startswith("job_501")
lines = []


def P(s=""):
    lines.append(s)
    print(s)


def pooled(df, k):
    eff = df[f"{k}_n_matched"].sum() / df[f"{k}_n_reconstructible"].sum()
    gh = df[f"{k}_n_ghosts"].sum() / max(df[f"{k}_n_candidates"].sum(), 1)
    cl = df[f"{k}_n_clones"].sum() / max(df[f"{k}_n_candidates"].sum(), 1)
    return eff, gh, cl, df[f"{k}_mean_purity"].mean()


P(f"Run 3 simulated events: {len(d)} (Gauss v56r9 Sim10, 2024 W40-42 md, nu = 7.6, min bias)")
P(f"  VELO hits / event: median {d.n_hits.median():.0f} (range {d.n_hits.min()}-{d.n_hits.max()})")
P(f"  reconstructible tracks (>= 3 hits): median {d.n_reco.median():.0f}; hits per track median "
  f"{d.med_hits_per_track.median():.0f}; prompt {np.average(d.frac_prompt, weights=d.n_reco):.2f}; "
  f"electrons {np.average(d.frac_electron, weights=d.n_reco):.2f}")
P(f"  PVs / event: {d.n_pv.mean():.1f} (reconstructible, >= 4 tracks: {d.n_pv_reco.mean():.1f})")
P(f"  segments / event: median {d.seg_n.median():.0f} (max {d.seg_n.max()}); true {d.seg_n_true.median():.0f}")
P(f"  toy comparison point: T = 200, 5 vertices, {len(toy)} events, {toy.seg_n.median():.0f} segments")
P()
P("Track reconstruction (EventValidator, purity >= 0.7, reconstructible = >= 3 hits), pooled:")
P(f"{'method':44s} {'eff':>6s} {'ghost':>6s} {'clone':>6s} {'purity':>6s} | toy T=200,5 PV eff / ghost")
rows = [("classical, 99 % true segments (paper)", "classical"),
        ("classical, <= 1 % false segments", "classicalF"),
        ("classical, all coupled segments (truth-free)", "classicalI"),
        ("one-bit filter, 99 % true segments (paper)", "1bqf"),
        ("one-bit filter, <= 1 % false segments", "1bqfF"),
        ("QSVT comb, <= 1 % false segments (paper)", "qsvt"),
        ("QSVT comb, above isolated value (truth-free)", "qsvtI"),
        ("hit register at scan peaks (as specified)", "hit"),
        ("hit register at refitted vertices", "hitR"),
        ("hit register at truth vertices", "hitT"),
        ("hit register at truth vertices, 0.5 rad window", "hitTW")]
summ = []
for lab, k in rows:
    e, g, c, pu = pooled(d, k)
    tk = {"classicalI": None, "qsvtI": None, "hitR": None, "hitT": None, "hitTW": None}.get(k, k)
    ts = f"{toy[f'{tk}_efficiency'].mean():.3f} / {toy[f'{tk}_ghost_rate'].mean():.3f}" if tk else "-"
    P(f"{lab:44s} {e:6.3f} {g:6.3f} {c:6.3f} {pu:6.3f} | {ts}")
    summ.append(dict(method=lab, key=k, eff=e, ghost=g, clone=c, purity=pu))
pd.DataFrame(summ).to_csv(OUT / "real_track_summary.csv", index=False)
P()
cats = {"all reconstructible": np.ones(len(t), bool),
        "prompt, not electron, >= 4 hits, p > 2 GeV": t.prompt & ~t.electron & (t.n_hits >= 4) & (t.p > 2000),
        "3 hits": t.n_hits == 3, "|t| > 0.25": t.t > 0.25, "p < 1 GeV": t.p < 1000,
        "not prompt (secondaries)": ~t.prompt, "electrons": t.electron}
keys = ["classicalI", "qsvtI", "qsvt", "hit", "hitR", "hitT", "hitTW"]
P("Track efficiency by category (per truth track):")
P(f"{'category':44s} {'share':>6s} " + " ".join(f"{k:>10s}" for k in keys))
for lab, m in cats.items():
    P(f"{lab:44s} {m.mean():6.3f} " + " ".join(f"{t.loc[m, 'found_' + k].mean():10.3f}" for k in keys))
P()
P("Segment level (pooled over events):")
for k in ("classical", "classicalF", "classicalI", "qsvt", "qsvtI"):
    P(f"  {k:12s} segment efficiency {d[f'{k}_seg_eff'].mean():.3f}, segment false rate {d[f'{k}_seg_far'].mean():.3f}"
      f"  (toy: {toy[f'{k}_seg_eff'].mean():.3f} / {toy[f'{k}_seg_far'].mean():.3f})" if k in toy else
      f"  {k:12s} segment efficiency {d[f'{k}_seg_eff'].mean():.3f}, segment false rate {d[f'{k}_seg_far'].mean():.3f}")
P(f"  events with no <= 1 % false working point (classical): {(d.classicalF_seg_eff < 0.05).mean():.2f}")
P()
P("Primary vertices (truth PVs with >= 4 reconstructible tracks; 1 mm match):")
for lab, m, sel in (("focusing scan, as specified", "focusing", d.index == d.index),
                    ("scan peaks -> track fit -> z clustering (hold-out)", "scan_refit", hold),
                    ("track baseline on classical <= 1 % tracks", "track_baseline", d.index == d.index),
                    ("track baseline on QSVT truth-free tracks", "track_baseline_qsvtI", d.index == d.index)):
    evs = set(d.event[sel])
    r = v[(v.method == m) & (v.n_reco_trk >= 4) & v.event.isin(evs)]
    fk = {"focusing": "fakes", "scan_refit": "refit_fakes", "track_baseline": "base_fakes",
          "track_baseline_qsvtI": "baseQ_fakes"}[m]
    mg = fk.replace("fakes", "merges")
    P(f"  {lab:50s} eff {r.matched.mean():.3f}  fakes/event {d.loc[sel, fk].mean():5.2f}  merges/event "
      f"{d.loc[sel, mg].mean():.2f}  dz rms {np.sqrt(np.nanmean(r.dz ** 2)) * 1e3:4.0f} um, robust "
      f"{1.4826 * np.nanmedian(np.abs(r.dz)) * 1e3:4.0f} um  ({len(evs)} events)")
P(f"  toy T = 200, 5 PV, focusing scan: eff {toyv.matched.mean():.3f}, fakes/event {toy.fakes.mean():.2f}, "
  f"dz rms {np.sqrt(np.nanmean(toyv.dz ** 2)) * 1e3:.0f} um")
r = v[(v.method == "focusing") & (v.n_reco_trk >= 4)]
P("  focusing scan efficiency vs distance to the nearest other PV  <2 / 2-5 / 5-10 / >10 mm: " + " / ".join(
    f"{r[(r.nearest_other >= a) & (r.nearest_other < b)].matched.mean():.2f}" for a, b in ((0, 2), (2, 5), (5, 10), (10, 1e9))))
P()
# ── resources ────────────────────────────────────────────────────────────────
def rounds(p):
    return np.maximum(1.0, np.pi / (4 * np.sqrt(np.maximum(p, 1e-12))))


P("Resources per event (median; Givens price; rounds = max(1, pi/(4 sqrt P))):")
res = []
for lab, df in (("real", d), ("toy T=200 5PV", toy)):
    for route, calls, anc, Pseg in (("2-call filter", 2, 2, "seg_P_1b"), ("QSVT phase sequence", 160, 4, "seg_P_q")):
        gs = calls * rounds(df[Pseg]) * df.seg_cx_givens
        out = dict(set=lab, route=route, seg_qubits=np.median(df.seg_ns + anc), seg_P=np.median(df[Pseg]),
                   seg_gates=np.median(gs))
        for tag, pc in (("hitV", "hitV_P"), ("hitVR", "hitVR_P"), ("hitVT", "hitVT_P"),
                        ("hitV_all", "hitV_P_all"), ("hitVR_all", "hitVR_P_all")):
            reg = tag.split("_")[0]
            if f"{reg}_ns" not in df or pc not in df:
                continue
            out[f"{tag}_qubits"] = np.median(df[f"{reg}_ns"] + anc)
            out[f"{tag}_P"] = np.median(df[pc])
            out[f"{tag}_gates"] = np.median(calls * rounds(df[pc]) * df[f"{reg}_cx_givens"])
        res.append(out)
rs = pd.DataFrame(res)
rs.to_csv(OUT / "real_resources.csv", index=False)
for _, r_ in rs.iterrows():
    P(f"  {r_['set']:14s} {r_.route:20s} segment: {r_.seg_qubits:.0f} qubits, P {r_.seg_P:.1e}, {r_.seg_gates:.2e} gates"
      f" | hit per-vertex: {r_.hitV_qubits:.0f} qubits, P {r_.hitV_P:.2f}, {r_.hitV_gates:.2e} gates"
      + (f" | P on all lines {r_['hitV_all_P']:.2f}, {r_['hitV_all_gates']:.2e} gates" if 'hitV_all_P' in r_ and pd.notna(r_.get('hitV_all_P')) else ""))
P(f"  real hit register, largest component median {d.hitV_max_comp.median():.0f}, coupled pairs {d.hitV_k.median():.0f};"
  f" segment register coupled pairs {d.seg_k.median():.0f}, largest component {d.seg_max_comp.median():.0f}")
(OUT / "real_summary.txt").write_text("\n".join(lines) + "\n")

# ── figures ──────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
# 1. one-event focusing scan, real vs toy
ev, ex = R.load_real(R.dump_files()[1], seed=1)
zs = np.arange(-100, 150, 0.1)
fig, ax = plt.subplots(2, 1, figsize=(7, 4.2))
n = L.scan(ev, zs, 512)
ax[0].plot(zs, n, lw=0.7, color="#1f4e79")
for z_, nr in zip(ev.pv_xyz[:, 2], ex.pv_n_reco):
    ax[0].axvline(z_, color="#c0392b", lw=0.8, alpha=0.7)
    ax[0].text(z_, n.max() * 1.02, str(nr), color="#c0392b", fontsize=6, ha="center")
ax[0].set_title(f"Run 3 simulated event ({len(ev.x)} hits, {len(ev.pv_xyz)} PVs; red = truth PV, label = "
                f"reconstructible tracks)", loc="left", fontsize=8)
ax[0].set_ylabel("N4 (512 grid)")
tev = L.generate_multi_pv(5, 200, seed=30000 + 5000 + 200)
zt = np.arange(-150, 28, 0.1)
nt = L.scan(tev, zt, 512)
ax[1].plot(zt, nt, lw=0.7, color="#1f4e79")
for z_ in tev.pv_xyz[:, 2]:
    ax[1].axvline(z_, color="#c0392b", lw=0.8, alpha=0.7)
ax[1].set_title("toy, T = 200, 5 vertices (section 8 settings)", loc="left", fontsize=8)
ax[1].set_xlabel(r"vertex hypothesis $\zeta$ [mm]")
ax[1].set_ylabel("N4 (512 grid)")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"01_scan_real_vs_toy.{ext}", dpi=200)
# 2. efficiency vs |t|, hits, p
fig, ax = plt.subplots(1, 3, figsize=(9, 2.8))
show = [("classicalI", "classical, coupled", "#7f7f7f"), ("qsvtI", "QSVT, truth-free", "#d95f02"),
        ("hit", "hit register, scan peaks", "#1b9e77"), ("hitTW", "hit register, truth PV, 0.5 rad", "#7570b3")]
for a, (col, bins, lab, logx) in zip(ax, (("t", [0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.5], "|t| = |(tx, ty)|", True),
                                         ("n_hits", [2.5, 3.5, 4.5, 5.5, 7.5, 10.5, 30], "hits on the track", False),
                                         ("p", [0, 500, 1000, 2000, 5000, 10000, 1e6], "p [MeV]", True))):
    c = pd.cut(t[col], bins)
    x = t.groupby(c, observed=True)[col].median()
    for k, name, colr in show:
        a.plot(x, t.groupby(c, observed=True)[f"found_{k}"].mean(), "o-", ms=3, lw=1, color=colr, label=name)
    a.set_xlabel(lab)
    if logx:
        a.set_xscale("log")
    a.set_ylim(0, 1.02)
ax[0].set_ylabel("track efficiency")
ax[0].legend(fontsize=6, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"02_track_eff_vs_kinematics.{ext}", dpi=200)
# 3. vertex residuals
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
for m, colr, lab in (("focusing", "#1b9e77", "focusing scan"), ("scan_refit", "#d95f02", "scan + track-fit refit"),
                     ("track_baseline_qsvtI", "#7570b3", "track baseline (QSVT tracks)")):
    r = v[(v.method == m) & (v.n_reco_trk >= 4) & v.matched]
    ax[0].hist(r.dz * 1e3, bins=np.linspace(-1000, 1000, 41), histtype="step", color=colr, label=lab)
    rr = v[(v.method == m) & (v.n_reco_trk >= 4)]
    cb = pd.cut(rr.n_reco_trk, [3, 9, 19, 49, 1000])
    ax[1].plot(rr.groupby(cb, observed=True).n_reco_trk.median(), rr.groupby(cb, observed=True).matched.mean(), "o-",
               ms=3, color=colr, label=lab)
ax[0].hist(toyv.dz.dropna() * 1e3, bins=np.linspace(-1000, 1000, 41), histtype="step", color="k", ls="--",
           label="toy focusing scan")
ax[0].set_xlabel(r"$z_{found} - z_{true}$ [$\mu$m]")
ax[0].legend(fontsize=6, frameon=False)
ax[1].set_xscale("log")
ax[1].set_xlabel("reconstructible tracks of the PV")
ax[1].set_ylabel("PV efficiency")
ax[1].set_ylim(0, 1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"03_vertices.{ext}", dpi=200)
print("figures written")
