#!/usr/bin/env python3
"""Hit-uniqueness as a CLASSICAL POST-SELECTION filter (George, 2026-07-07 ruling).

The DP matrix study proved per-hit occupancy must never enter the (quantum)
Hamiltonian: the notch already kills the co-hit false bulk exactly, and the
in-matrix coupling revives it (figures/dp_occupancy_proof.png).  George's
ruling: occupancy is a classical-side tool ONLY.  This study builds that tool
— ``qtrk_pipeline.hit_uniqueness_filter`` (same-role slot competition, greedy
amplitude-descending claiming) — and asks four questions:

Q1  ANCHOR — on the exact dp_occupancy_proof event (heavy T=200 rep 0), does
    threshold->uniqueness on the BASE classical solution reproduce (or beat)
    the in-matrix occupancy result far 0.650 -> 0.210 at eff ~0.991, without
    touching the solver?
Q2  COMPOSITION — the full composed frontier (tau swept, filter applied at
    each tau) vs the base and in-matrix-occ tau sweeps, for the classical
    solve AND the exact 1BQF notch readout (three-knob composition: notch for
    the isolated bulk + uniqueness for the co-hit survivors).
Q3  MECHANISM — which false classes die (per-class census at the working
    point): co-hit competitors should die; C-true continuations (no slot
    conflict with a higher-amplitude segment) are the expected survivors.
Q4  ROBUSTNESS & DENSITY — clean/moderate/heavy x T in {100, 200, 400}:
    does the filter hold eff >= 0.99 while cutting far, everywhere?

Role semantics: competition = same-role hit sharing (two segments using h as
start, or as end) — the ``bif.fork_graph`` pairing.  Consecutive true
segments share their interior hit in opposite roles and never compete.  The
``anyrole`` mode (pooled roles, hard per-hit argmax) is kept as the NEGATIVE
CONTROL — it deletes track interiors and its efficiency shows it.

Data: qtrk_pipeline events/solutions only (heavy/moderate/clean = the DP
study operating points, formula epsilon, step kernel, gamma=3, delta=1).
Anchor events reuse results/dp_occupancy_proof_cache.pkl (identical solves).
Outputs: figures/dp_postselect_uniqueness.png,
         results/dp_postselect_uniqueness.csv, results/dp_postselect_uniqueness.log
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import minres

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp_matrix_characterisation as dmc  # noqa: E402  (brings qp on path)
import dp_terms  # noqa: E402
import bif  # noqa: E402

qp = dmc.qp
from qtrk_pipeline import hit_uniqueness_filter, uniqueness_frontier  # noqa: E402

HERE = Path(__file__).resolve().parent
LOG = open(HERE / "results" / "dp_postselect_uniqueness.log", "w")
TARGET_EFF = 0.99


def say(*a):
    print(*a, flush=True)
    print(*a, file=LOG, flush=True)


def selftest():
    """Synthetic 6-segment check of every mode's slot semantics."""
    # hits: track A = segs 0(h0->h1), 1(h1->h2)  (chain: opposite-role share)
    # false seg 2 shares START h0 with seg 0 (fork), lower amplitude -> dies
    # false seg 3 shares END h2 with seg 1 (merge), HIGHER amplitude -> steals
    # isolated segs 4 (h5->h6), 5 (h7->h8)
    seg_hits = np.array([[0, 1], [1, 2], [0, 3], [4, 2], [5, 6], [7, 8]])
    x = np.array([1.0, 0.8, 0.5, 0.9, 0.7, 0.2])
    act = x > 0.1
    g = hit_uniqueness_filter(x, seg_hits, act, mode="greedy")
    assert g.tolist() == [True, False, False, True, True, True], g
    s = hit_uniqueness_filter(x, seg_hits, act, mode="strict")
    assert s.tolist() == [True, False, False, True, True, True], s
    a = hit_uniqueness_filter(x, seg_hits, act, mode="anyrole")
    # anyrole pools roles: h1 shared by segs 0,1 -> seg 1 dies to seg 0
    assert not a[1] and a[0], a
    # chain is safe under role-aware modes even when amplitudes differ
    seg_hits2 = np.array([[0, 1], [1, 2], [2, 3]])
    x2 = np.array([0.5, 0.4, 0.45])
    g2 = hit_uniqueness_filter(x2, seg_hits2, x2 > 0.1, mode="greedy")
    assert g2.all(), g2
    say("[selftest] hit_uniqueness_filter semantics OK")


def wp_metrics(x, truth, seg_hits=None, mode=None, tau=None):
    """(eff, far, tau, n_act) at wp99 (or given tau), optionally filtered."""
    x = np.abs(np.asarray(x).ravel())
    if tau is None:
        tau = qp.working_point_threshold(x, truth)
    act = x > tau
    if mode is not None:
        act = hit_uniqueness_filter(x, seg_hits, act, mode=mode)
    na = int(act.sum())
    eff = int((act & truth).sum()) / max(int(truth.sum()), 1)
    far = int((act & ~truth).sum()) / max(na, 1)
    return eff, far, float(tau), na, act


def composed_wp(x, truth, seg_hits, mode="greedy", n_tau=60, kappa=0.8):
    """Efficiency-first working point of the COMPOSED selector.

    Sweep tau; keep points with composed eff >= TARGET_EFF; among them return
    the (tau, eff, far) with minimal far (efficiency-first, then purity).
    Falls back to the max-eff point if the target is unreachable.
    """
    x = np.abs(np.asarray(x).ravel())
    qs = np.linspace(0.0, 1.0, 400)
    lo = float(np.quantile(x[truth], 0.001)) * 0.25
    taus = np.unique(np.quantile(x, qs))
    taus = taus[(taus >= lo) & (taus < x.max())]
    if len(taus) > n_tau:
        taus = taus[np.linspace(0, len(taus) - 1, n_tau).astype(int)]
    eff, far = uniqueness_frontier(x, seg_hits, truth, taus, mode=mode,
                                   kappa=kappa)
    ok = eff >= TARGET_EFF
    if ok.any():
        j = np.flatnonzero(ok)[np.argmin(far[ok])]
    else:
        j = int(np.argmax(eff))
    return float(taus[j]), float(eff[j]), float(far[j]), (taus, eff, far)


KAPPAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def margin_scan(x, truth, seg_hits, n_tau=60):
    """kappa sweep of the composed margin selector; returns (rows, best).

    best = the efficiency-first choice: among kappas whose composed wp holds
    eff >= TARGET_EFF, the one with minimal far (falls back to max-eff).
    """
    out = []
    for kap in KAPPAS:
        tc, ec, fc, curve = composed_wp(x, truth, seg_hits, mode="margin",
                                        n_tau=n_tau, kappa=kap)
        out.append(dict(kappa=kap, tau=tc, eff=ec, far=fc, curve=curve))
    ok = [r for r in out if r["eff"] >= TARGET_EFF]
    best = (min(ok, key=lambda r: r["far"]) if ok
            else max(out, key=lambda r: r["eff"]))
    return out, best


def frontier(x, truth, n=250):
    """Plain tau-sweep (eff, far) for reference curves."""
    x = np.abs(np.asarray(x).ravel())
    taus = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n)))[::-1]
    nt = max(int(truth.sum()), 1)
    eff, far = [], []
    for t in taus:
        act = x > t
        eff.append(int((act & truth).sum()) / nt)
        far.append(int((act & ~truth).sum()) / max(int(act.sum()), 1))
    return np.asarray(eff), np.asarray(far)


def fresh_case(T, noise, rep, want_occ=True):
    """Solve base (and in-matrix occ) classically for a fresh event."""
    ham, truth, eps = dmc.get_ham(T, noise, rep=rep)
    seg_hits = np.asarray(ham._segment_to_hit_ids)
    A0, b0, _, _ = dp_terms.dp_system(ham, beta=0.0, alpha=0.0,
                                      gamma=dmc.GAMMA, delta=dmc.DELTA)
    x0, _ = minres(A0, b0, rtol=1e-8, maxiter=8000)
    out = dict(x=np.abs(np.asarray(x0).ravel()))
    if want_occ:
        Ao, bo, _, _ = dp_terms.dp_system(ham, beta=0.0, alpha=0.3,
                                          gamma=dmc.GAMMA, delta=dmc.DELTA)
        xo, _ = minres(Ao, bo, rtol=1e-8, maxiter=8000)
        out["x_occ"] = np.abs(np.asarray(xo).ravel())
    return ham, truth, eps, seg_hits, out


def main():
    t0 = time.time()
    selftest()
    rows = []

    # ---------------- anchors from the proof cache (identical solves) --------
    with open(HERE / "results" / "dp_occupancy_proof_cache.pkl", "rb") as fh:
        payload = pickle.load(fh)
    cls200, truth200, main_ev = payload["main"]
    say("[cache] heavy T=200 rep0 + heavy T=100 reps 0-2 loaded")

    ham200, truth200b, eps200 = dmc.get_ham(200, "heavy", rep=0)
    assert (np.asarray(truth200b, bool) == np.asarray(truth200, bool)).all()
    seg200 = np.asarray(ham200._segment_to_hit_ids)

    # anchor validation vs dp_occupancy_proof.csv
    eff_b, far_b, tau_b, na_b, _ = wp_metrics(main_ev["base"]["x"], truth200)
    say(f"[anchor] classical base wp99: eff={eff_b:.4f} far={far_b:.4f} "
        f"(proof CSV: 0.9949 / 0.6497)")
    assert abs(eff_b - 0.994924) < 5e-4 and abs(far_b - 0.649687) < 5e-3

    # Q1: filter at the base wp99 tau + composed working point
    cases = {}
    for mode in ("greedy", "strict", "anyrole"):
        e, f, _, na, act = wp_metrics(main_ev["base"]["x"], truth200,
                                      seg200, mode=mode, tau=tau_b)
        say(f"[Q1 heavy T200 classical] {mode:8s} at base-wp99 tau: "
            f"eff={e:.4f} far={f:.4f} n_act={na}")
        rows.append(dict(noise="heavy", T=200, rep=0, solver="classical",
                         selector=f"filter-{mode}@basewp", eff=e, far=f,
                         tau=tau_b, n_act=na))
        if mode != "anyrole":
            tc, ec, fc, curve = composed_wp(main_ev["base"]["x"], truth200,
                                            seg200, mode=mode)
            cases[f"cls-{mode}"] = curve
            say(f"[Q1 heavy T200 classical] composed-{mode} wp: tau={tc:.4f} "
                f"eff={ec:.4f} far={fc:.4f}")
            rows.append(dict(noise="heavy", T=200, rep=0, solver="classical",
                             selector=f"composed-{mode}", eff=ec, far=fc,
                             tau=tc, n_act=np.nan))
    scan, best = margin_scan(main_ev["base"]["x"], truth200, seg200)
    for r in scan:
        say(f"[Q1 heavy T200 classical] composed-margin k={r['kappa']:.2f}: "
            f"eff={r['eff']:.4f} far={r['far']:.4f}")
        rows.append(dict(noise="heavy", T=200, rep=0, solver="classical",
                         selector=f"composed-margin{r['kappa']:g}", eff=r["eff"],
                         far=r["far"], tau=r["tau"], kappa=r["kappa"],
                         n_act=np.nan))
    rows.append(dict(noise="heavy", T=200, rep=0, solver="classical",
                     selector="composed-margin-best", eff=best["eff"],
                     far=best["far"], tau=best["tau"], kappa=best["kappa"],
                     n_act=np.nan))
    cases["cls-margin"] = best["curve"]
    say(f"[Q1 heavy T200 classical] MARGIN-BEST k={best['kappa']:.2f}: "
        f"eff={best['eff']:.4f} far={best['far']:.4f}")
    for name, lab in [("base", "base"), ("occ", "inmatrix-occ")]:
        e, f, t, na, _ = wp_metrics(main_ev[name]["x"], truth200)
        rows.append(dict(noise="heavy", T=200, rep=0, solver="classical",
                         selector=f"{lab}@wp99", eff=e, far=f, tau=t, n_act=na))

    # Q2 quantum: exact 1BQF notch readout, composed with uniqueness
    xq = main_ev["base"]["xq"]
    eq, fq, tq_, naq, _ = wp_metrics(xq, truth200)
    say(f"[Q2 heavy T200 1BQF] base notch wp99: eff={eq:.4f} far={fq:.4f}")
    rows.append(dict(noise="heavy", T=200, rep=0, solver="1bqf_emu",
                     selector="base@wp99", eff=eq, far=fq, tau=tq_, n_act=naq))
    tcq, ecq, fcq, curve_q = composed_wp(xq, truth200, seg200, mode="greedy")
    say(f"[Q2 heavy T200 1BQF] composed-greedy wp: tau={tcq:.3e} "
        f"eff={ecq:.4f} far={fcq:.4f}")
    rows.append(dict(noise="heavy", T=200, rep=0, solver="1bqf_emu",
                     selector="composed-greedy", eff=ecq, far=fcq, tau=tcq,
                     n_act=np.nan))
    scan_q, best_q = margin_scan(xq, truth200, seg200)
    rows.append(dict(noise="heavy", T=200, rep=0, solver="1bqf_emu",
                     selector="composed-margin-best", eff=best_q["eff"],
                     far=best_q["far"], tau=best_q["tau"],
                     kappa=best_q["kappa"], n_act=np.nan))
    cases["q-margin"] = best_q["curve"]
    say(f"[Q2 heavy T200 1BQF] MARGIN-BEST k={best_q['kappa']:.2f}: "
        f"eff={best_q['eff']:.4f} far={best_q['far']:.4f}")

    # Q3 mechanism: per-class census at the composed classical working point
    tc, ec, fc, _ = composed_wp(main_ev["base"]["x"], truth200, seg200, "greedy")
    x0 = np.abs(np.asarray(main_ev["base"]["x"]).ravel())
    act_base = x0 > tc
    act_filt = hit_uniqueness_filter(x0, seg200, act_base, mode="greedy")
    census = []
    for c in np.unique(cls200):
        m = cls200 == c
        census.append(dict(cls=str(c), n=int(m.sum()),
                           act_thresh=int((act_base & m).sum()),
                           act_composed=int((act_filt & m).sum())))
        say(f"[Q3] {c:14s} n={m.sum():7d}  thresholded={int((act_base & m).sum()):6d}"
            f"  survive-filter={int((act_filt & m).sum()):6d}")
    census = pd.DataFrame(census)

    # heavy T=100 anchors (faint set)
    for r, (cls_f, tr_f, ev_f) in enumerate(payload["faint"]):
        ham_f, tr_chk, _ = dmc.get_ham(100, "heavy", rep=r)
        assert (np.asarray(tr_chk, bool) == np.asarray(tr_f, bool)).all()
        seg_f = np.asarray(ham_f._segment_to_hit_ids)
        for name, lab in [("base", "base@wp99"), ("occ", "inmatrix-occ@wp99")]:
            e, f, t, na, _ = wp_metrics(ev_f[name]["x"], tr_f)
            rows.append(dict(noise="heavy", T=100, rep=r, solver="classical",
                             selector=lab, eff=e, far=f, tau=t, n_act=na))
        tcf, ecf, fcf, _ = composed_wp(ev_f["base"]["x"], tr_f, seg_f, "greedy")
        rows.append(dict(noise="heavy", T=100, rep=r, solver="classical",
                         selector="composed-greedy", eff=ecf, far=fcf, tau=tcf,
                         n_act=np.nan))
        e, f, t, na, _ = wp_metrics(ev_f["base"]["xq"], tr_f)
        rows.append(dict(noise="heavy", T=100, rep=r, solver="1bqf_emu",
                         selector="base@wp99", eff=e, far=f, tau=t, n_act=na))
        tcq2, ecq2, fcq2, _ = composed_wp(ev_f["base"]["xq"], tr_f, seg_f, "greedy")
        rows.append(dict(noise="heavy", T=100, rep=r, solver="1bqf_emu",
                         selector="composed-greedy", eff=ecq2, far=fcq2,
                         tau=tcq2, n_act=np.nan))
        _, bf = margin_scan(ev_f["base"]["x"], tr_f, seg_f)
        rows.append(dict(noise="heavy", T=100, rep=r, solver="classical",
                         selector="composed-margin-best", eff=bf["eff"],
                         far=bf["far"], tau=bf["tau"], kappa=bf["kappa"],
                         n_act=np.nan))
        _, bfq = margin_scan(ev_f["base"]["xq"], tr_f, seg_f)
        rows.append(dict(noise="heavy", T=100, rep=r, solver="1bqf_emu",
                         selector="composed-margin-best", eff=bfq["eff"],
                         far=bfq["far"], tau=bfq["tau"], kappa=bfq["kappa"],
                         n_act=np.nan))
        say(f"[faint heavy T100 rep{r}] cls composed {ecf:.4f}/{fcf:.4f} · "
            f"1bqf composed {ecq2:.4f}/{fcq2:.4f}")

    # ---------------- Q4 robustness & density (fresh solves) -----------------
    for noise, T in [("clean", 200), ("moderate", 200), ("heavy", 400)]:
        tt = time.time()
        ham, truth, eps, seg, out = fresh_case(T, noise, 0)
        say(f"[fresh {noise} T={T}] n_seg={ham.n_segments:,} eps={eps:.5f} "
            f"({time.time()-tt:.0f}s)")
        for key, lab in [("x", "base@wp99"), ("x_occ", "inmatrix-occ@wp99")]:
            e, f, t, na, _ = wp_metrics(out[key], truth)
            rows.append(dict(noise=noise, T=T, rep=0, solver="classical",
                             selector=lab, eff=e, far=f, tau=t, n_act=na))
        tc2, ec2, fc2, _ = composed_wp(out["x"], truth, seg, "greedy",
                                       n_tau=30 if T >= 400 else 60)
        say(f"[fresh {noise} T={T}] composed-greedy wp: eff={ec2:.4f} far={fc2:.4f}")
        rows.append(dict(noise=noise, T=T, rep=0, solver="classical",
                         selector="composed-greedy", eff=ec2, far=fc2, tau=tc2,
                         n_act=np.nan))
        _, bfr = margin_scan(out["x"], truth, seg,
                             n_tau=30 if T >= 400 else 60)
        say(f"[fresh {noise} T={T}] MARGIN-BEST k={bfr['kappa']:.2f}: "
            f"eff={bfr['eff']:.4f} far={bfr['far']:.4f}")
        rows.append(dict(noise=noise, T=T, rep=0, solver="classical",
                         selector="composed-margin-best", eff=bfr["eff"],
                         far=bfr["far"], tau=bfr["tau"], kappa=bfr["kappa"],
                         n_act=np.nan))
        e, f, _, na, _ = wp_metrics(out["x"], truth, seg, mode="anyrole",
                                    tau=qp.working_point_threshold(
                                        np.abs(out["x"]), truth))
        rows.append(dict(noise=noise, T=T, rep=0, solver="classical",
                         selector="filter-anyrole@basewp", eff=e, far=f,
                         tau=np.nan, n_act=na))

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "dp_postselect_uniqueness.csv", index=False)
    census.to_csv(HERE / "results" / "dp_postselect_uniqueness_census.csv",
                  index=False)

    # ---------------- figures -------------------------------------------------
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })
    figdir = HERE / "figures"
    d200c = df[(df.noise == "heavy") & (df["T"] == 200)
               & (df.solver == "classical")]
    d200q = df[(df.noise == "heavy") & (df["T"] == 200)
               & (df.solver == "1bqf_emu")]

    def style_roc(ax):
        ax.set_xscale("log")
        ax.set_xlim(5e-4, 1.0)
        ax.set_ylim(0.55, 1.02)
        ax.axhline(TARGET_EFF, color="#79776f", lw=0.7, ls=":")
        ax.set_xlabel(r"false rate = N$_{false\,act}$ / N$_{act}$")
        ax.set_ylabel("segment efficiency")

    def mark_wp(ax, d, sel, col, mk, dx=1.15, dy=0.012, ha="left"):
        r = d[d.selector == sel].iloc[0]
        x = max(r.far, 5e-4)
        ax.plot(x, r.eff, mk, color=col, ms=12 if mk == "*" else 6, zorder=5)
        ax.annotate(f"{r.eff:.3f} / {r.far:.3f}", (x, r.eff),
                    xytext=(x * dx, r.eff + dy), fontsize=7, color=col, ha=ha)

    def draw_frontier_classical(ax, tag="(a) "):
        e, f = frontier(main_ev["base"]["x"], truth200)
        ax.plot(np.maximum(f, 5e-4), e, "-", color="#33322e", lw=1.8,
                label="base: threshold only")
        e, f = frontier(main_ev["occ"]["x"], truth200)
        ax.plot(np.maximum(f, 5e-4), e, "-", color="#e34948", lw=1.6,
                label="in-matrix occupancy (α=0.3)")
        _, ee, ff = cases["cls-strict"]
        ax.plot(np.maximum(ff, 5e-4), ee, "--", color="#8e6fad", lw=1.2,
                label="threshold → uniqueness (strict)")
        _, ee, ff = cases["cls-greedy"]
        ax.plot(np.maximum(ff, 5e-4), ee, "-", color="#2a78d6", lw=2.2,
                label="threshold → uniqueness (greedy)")
        _, ee, ff = cases["cls-margin"]
        ax.plot(np.maximum(ff, 5e-4), ee, "-", color="#1e9e6a", lw=1.6,
                label="threshold → margin uniqueness (best κ)")
        mark_wp(ax, d200c, "base@wp99", "#33322e", "o",
                dx=0.50, dy=0.012, ha="right")
        mark_wp(ax, d200c, "inmatrix-occ@wp99", "#e34948", "o",
                dx=0.85, dy=-0.050, ha="right")
        mark_wp(ax, d200c, "composed-greedy", "#2a78d6", "*",
                dx=1.25, dy=0.010)
        mark_wp(ax, d200c, "composed-margin-best", "#1e9e6a", "*",
                dx=1.10, dy=-0.055)
        style_roc(ax)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.set_title(tag + "classical, heavy T=200 — post-selection replaces "
                     "the in-matrix term\n(marker labels: eff / far at the "
                     "working point; ★ = composed wp)", loc="left", fontsize=9)

    def draw_frontier_1bqf(ax, tag="(b) "):
        e, f = frontier(xq, truth200)
        ax.plot(np.maximum(f, 5e-4), e, "-", color="#33322e", lw=1.8,
                label="1BQF notch: threshold only")
        _, ee, ff = curve_q
        ax.plot(np.maximum(ff, 5e-4), ee, "-", color="#2a78d6", lw=2.2,
                label="1BQF notch → uniqueness (greedy)")
        _, ee, ff = cases["q-margin"]
        ax.plot(np.maximum(ff, 5e-4), ee, "-", color="#1e9e6a", lw=1.6,
                label="1BQF notch → margin uniqueness (best κ)")
        e, f = frontier(main_ev["occ"]["xq"], truth200)
        ax.plot(np.maximum(f, 5e-4), e, "-", color="#e34948", lw=1.2,
                alpha=0.7, label="1BQF on in-matrix occ system (disproven route)")
        mark_wp(ax, d200q, "base@wp99", "#33322e", "o",
                dx=0.50, dy=0.012, ha="right")
        mark_wp(ax, d200q, "composed-greedy", "#2a78d6", "*",
                dx=1.25, dy=0.010)
        mark_wp(ax, d200q, "composed-margin-best", "#1e9e6a", "*",
                dx=1.10, dy=-0.055)
        style_roc(ax)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.set_title(tag + "1BQF notch readout, heavy T=200 — notch kills the "
                     "isolated bulk,\nthe uniqueness gate kills the co-hit "
                     "survivors", loc="left", fontsize=9)

    def draw_census(ax, tag="(c) "):
        order = ["cohit-true", "C-true", "cohit-false", "fork-true",
                 "TRUE-coupled"]
        cc = census.set_index("cls").loc[order].reset_index()
        ys = np.arange(len(cc))[::-1]
        ax.barh(ys + 0.19, cc.act_thresh, 0.36, color="#9a9890",
                label="active after threshold alone")
        ax.barh(ys - 0.19, cc.act_composed, 0.36,
                color=[dmc.CCOL.get(c, "#33322e") for c in cc.cls],
                label="survive the uniqueness gate")
        for y, (_, r) in zip(ys, cc.iterrows()):
            ax.text(max(r.act_thresh, 0.75) * 1.3, y + 0.19,
                    f"{r.act_thresh:,}", va="center", fontsize=7.5,
                    color="#52514e")
            ax.text(max(r.act_composed, 0.75) * 1.3, y - 0.19,
                    f"{r.act_composed:,}", va="center", fontsize=7.5,
                    color="#33322e", fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlim(0.7, 6000)
        ax.set_yticks(ys, [f"{c}\n(n = {n:,})" for c, n in
                           zip(cc.cls, cc["n"])], fontsize=7.5)
        ax.set_xlabel("active segments at the composed wp (log; grey → colour "
                      "= threshold → gate)")
        ax.legend(fontsize=7.5, loc="center right")
        ax.set_title(tag + "who dies — per-class census at the composed wp "
                     "(heavy T=200, classical)", loc="left", fontsize=9)

    def draw_wp_map(ax, tag="(d) "):
        cells = [("clean", 200, "#79b465"), ("moderate", 200, "#e6b422"),
                 ("heavy", 100, "#eb6834"), ("heavy", 200, "#e34948"),
                 ("heavy", 400, "#8e6fad")]
        for noise, T, col in cells:
            dd = df[(df.noise == noise) & (df["T"] == T)
                    & (df.solver == "classical")]
            for rep in sorted(dd.rep.unique()):
                b = dd[(dd.selector == "base@wp99") & (dd.rep == rep)].iloc[0]
                g = dd[(dd.selector == "composed-greedy")
                       & (dd.rep == rep)].iloc[0]
                x0, x1 = max(b.far, 5e-4), max(g.far, 5e-4)
                ax.plot(x0, b.eff, "o", color=col, ms=5, alpha=0.9)
                ax.plot(x1, g.eff, "*", color=col, ms=12, zorder=5)
                if abs(x1 - x0) > 1e-9 or abs(g.eff - b.eff) > 1e-9:
                    ax.annotate("", xy=(x1, g.eff), xytext=(x0, b.eff),
                                arrowprops=dict(arrowstyle="->", color=col,
                                                lw=1.3, alpha=0.8,
                                                shrinkA=4, shrinkB=7))
            g = dd[dd.selector == "composed-greedy"]
            xl = max(g.far.min(), 5e-4)
            dy = {("clean", 200): 0.008, ("moderate", 200): -0.034}.get(
                (noise, T), -0.022)
            ax.text(xl, g.eff.min() + dy, f"{noise} T={T}", fontsize=7,
                    color=col, ha="left" if xl < 2e-3 else "center")
        style_roc(ax)
        ax.set_ylim(0.80, 1.02)
        ax.set_title(tag + "every cell: threshold-only wp (●) → composed "
                     "uniqueness wp (★),\nclassical solve", loc="left",
                     fontsize=9)

    def draw_robustness(ax_e, ax_f):
        grid = [("clean", 200), ("moderate", 200), ("heavy", 100),
                ("heavy", 200), ("heavy", 400)]
        sels = [("base@wp99", "#33322e", "base wp99 (threshold only)"),
                ("inmatrix-occ@wp99", "#e34948", "in-matrix occupancy wp99"),
                ("composed-greedy", "#2a78d6", "composed uniqueness wp"),
                ("composed-margin-best", "#1e9e6a",
                 "composed margin wp (best κ)")]
        xs = np.arange(len(grid))
        for k, (sel, col, lab) in enumerate(sels):
            off = (k - 1.5) * 0.2
            effs, fars = [], []
            for noise, T in grid:
                dd = df[(df.noise == noise) & (df["T"] == T)
                        & (df.solver == "classical") & (df.selector == sel)]
                effs.append(dd.eff.mean() if len(dd) else np.nan)
                fars.append(dd.far.mean() if len(dd) else np.nan)
            ax_e.bar(xs + off, effs, 0.19, color=col, label=lab)
            ax_f.bar(xs + off, fars, 0.19, color=col)
            for xi, (ev, fv) in enumerate(zip(effs, fars)):
                if np.isfinite(ev):
                    ax_e.text(xi + off, min(ev + 0.004, 1.035), f"{ev:.3f}",
                              ha="center", va="bottom", fontsize=6,
                              color=col, rotation=90)
                if np.isfinite(fv):
                    ax_f.text(xi + off, fv + 0.015, f"{fv:.3f}", ha="center",
                              va="bottom", fontsize=6, color=col, rotation=90)
        ax_e.axhline(TARGET_EFF, color="#79776f", lw=0.7, ls=":")
        ax_e.set_ylim(0.80, 1.10)
        ax_e.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
        ax_e.set_ylabel("efficiency at the wp")
        ax_e.set_xticks([])
        ax_e.legend(fontsize=7.5, ncols=2, loc="upper left",
                    bbox_to_anchor=(0.0, 1.22))
        ax_f.set_ylim(0, 1.12)
        ax_f.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_f.set_ylabel("false rate at the wp")
        ax_f.set_xticks(xs, [f"{n}\nT={T}"
                             + ("\n(3 reps, mean)" if (n, T) == ("heavy", 100)
                                else "") for n, T in grid], fontsize=7.5)

    # Fig 1 — composed frontiers (heavy T=200, classical + 1BQF)
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 5.4))
    draw_frontier_classical(axs[0], tag="")
    draw_frontier_1bqf(axs[1], tag="")
    fig.tight_layout()
    fig.savefig(figdir / "dp_postselect_uniqueness_frontiers.png",
                bbox_inches="tight")
    plt.close(fig)

    # Fig 2 — per-class census
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    draw_census(ax, tag="")
    fig.tight_layout()
    fig.savefig(figdir / "dp_postselect_uniqueness_census.png",
                bbox_inches="tight")
    plt.close(fig)

    # Fig 3 — robustness & density (eff and far, same grid)
    fig, (ax_e, ax_f) = plt.subplots(2, 1, figsize=(9.6, 6.8))
    draw_robustness(ax_e, ax_f)
    fig.suptitle("Robustness & density: working points across noise × T "
                 "(classical solve)", y=0.99, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(figdir / "dp_postselect_uniqueness_robustness.png",
                bbox_inches="tight")
    plt.close(fig)

    # Fig 4 — working-point map (all cells, base → composed)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    draw_wp_map(ax, tag="")
    fig.tight_layout()
    fig.savefig(figdir / "dp_postselect_uniqueness_wpmap.png",
                bbox_inches="tight")
    plt.close(fig)

    # Overview 2x2 (kept filename for PROJECT_STATUS / older links)
    fig = plt.figure(figsize=(13.4, 10.0))
    gs = fig.add_gridspec(2, 2)
    draw_frontier_classical(fig.add_subplot(gs[0, 0]))
    draw_frontier_1bqf(fig.add_subplot(gs[0, 1]))
    draw_census(fig.add_subplot(gs[1, 0]))
    draw_wp_map(fig.add_subplot(gs[1, 1]))
    anyrole = df[df.selector == "filter-anyrole@basewp"]
    note = " · ".join(f"{r.noise} T={r['T']}: eff {r.eff:.2f}"
                      for _, r in anyrole.iterrows())
    fig.suptitle("Hit-uniqueness as classical POST-SELECTION: the occupancy "
                 "physics without the Hamiltonian damage", y=0.995, fontsize=11)
    foot = (r"heavy = ($\sigma_{res}$ 20 µm, drop 1%, $\sigma_{scatt}$ 1e-4) · "
            r"moderate = (10 µm, 1%, 1e-4) · clean = (0, 0, 1e-4) · formula "
            r"$\varepsilon$ · step kernel · $\gamma$=3 $\delta$=1 · far = "
            r"N$_{false\,act}$/N$_{act}$ · composed wp = efficiency-first "
            f"(eff ≥ {TARGET_EFF}) then min far · uniqueness = per-(hit, role) "
            "slot claiming, amplitude-descending (qtrk_pipeline."
            "hit_uniqueness_filter) · anchor solves = dp_occupancy_proof_cache "
            "(identical to the proof figure) · negative control (anyrole "
            f"pooled-slot argmax) kills track interiors: {note} · "
            "provenance: dp_postselect_uniqueness.py")
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    fig.text(0.01, 0.004, foot, fontsize=5.8, color="#52514e", va="bottom",
             wrap=True)
    fig.savefig(figdir / "dp_postselect_uniqueness.png", bbox_inches="tight")
    plt.close(fig)

    say(df[["noise", "T", "rep", "solver", "selector", "eff", "far"]]
        .round(4).to_string(index=False))
    say(f"[done] {time.time()-t0:.0f}s -> figures/dp_postselect_uniqueness{{,_frontiers,_census,_robustness,_wpmap}}.png")


if __name__ == "__main__":
    main()
