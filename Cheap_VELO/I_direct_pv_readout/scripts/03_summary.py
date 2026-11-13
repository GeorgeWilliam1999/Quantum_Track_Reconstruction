#!/usr/bin/env python3
"""Item I/J, step 3: summary tables and figures (paper 1 style).

Inputs: outputs/pv_events.csv, shots_events.csv, resources_events.csv (02_analyse.py) and
cache/*/*.npz meta (01_events.py). Hold-out events only (toy: all 180 sweep events, which
are not the tuning events; Run 3: job_501x, 79 events).

Shots to reproduce the exact readout, per event and item kind:
    K_rep = the smallest K of the grid (2^4 ... 2^18) such that at K and every larger K
    of the grid the sampled readout matches at least as many eligible PVs as the exact
    (K = infinity) readout and has no more fakes. Events that never settle get K_rep = inf
    (counted, reported).
Queries (quantum) = K_rep x queries_per_shot of the event (amplitude amplification at the
    padded flagged fraction, 02_analyse / pv_sampler.sampler_resources).
Classical comparators of the same oracle (not of the best classical algorithm):
    classical sampling = K_rep / P (rejection sampling over the real items only; for triplets
        this needs the coupled-pair list, i.e. the Hamiltonian already built);
    classical sampling, same oracle = K_rep / P_padded (rejection sampling over the same
        index register as the quantum sampler: the like-for-like square-root comparison);
    classical exhaustive = every item of the register read once (doublets: every segment).
Focusing scan: hit-direction evaluations of its coarse pass, n_hits x n_slices (a lower
    bound of its cost; the fine pass adds 201 slices per coarse peak).

Outputs: outputs/summary_pv_toy.csv, summary_pv_real.csv, summary_resources.csv,
         summary.txt; figures/I01 ... I05 (.pdf, .png)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# paper 1 conventions (Vertexing/Focusing_scan/scripts/plot_style.py)
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 7, "axes.titlesize": 9})
WIDTH = 8.6
M = {  # method: (label, colour, marker, linestyle)
    "sampler_t_exact": ("vertex sampler, triplets (quantum, exact output)", "#6a3d9a", "D", "-"),
    "sampler_d_exact": ("vertex sampler, doublets (quantum, exact output)", "#b15928", "d", "-"),
    "scan": ("focusing scan, as specified", "#9e9e9e", "x", ":"),
    "scan_refit": ("focusing scan + refit", "#616161", "v", "--"),
    "trk_classical_fit": ("tracks (classical solver) + vertex fit", "k", "o", "-"),
    "trk_1bqf_fit": ("tracks (one-bit filter) + vertex fit", "#1f77b4", "s", "-"),
    "trk_qsvt_fit": ("tracks (QSVT, degree 40) + vertex fit", "#d95f02", "^", "-"),
    "trk_classical_median": ("tracks (classical) + z0 median", "k", "o", ":"),
    "trk_truth_fit": ("truth tracks + vertex fit (bound)", "#2e7d32", "", "--"),
}
KIND = {"t": ("triplets", "#6a3d9a", "D"), "d": ("doublets", "#b15928", "d")}
NPV_BINS = [0, 1, 5, 10, 15, 20]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def agg_pv(g):
    nres = g.n_res.sum()
    res = np.concatenate([np.array(json.loads(r)) for r in g.res_list]) if len(g) else np.zeros(0)
    mad = 1.4826 * np.median(np.abs(res - np.median(res))) if len(res) else np.nan
    return pd.Series(dict(events=len(g), eff=g.matched_elig.sum() / max(g.n_elig.sum(), 1),
                          eff_all=g.matched.sum() / max(g.n_true.sum(), 1),
                          fakes=g.fakes.mean(), merges=g.merges.mean(),
                          rms_mm=np.sqrt(g.res_sq.sum() / nres) if nres else np.nan, robust_sigma_mm=mad))


def k_rep(s):
    """Per event and kind: smallest K from which the sampled readout reproduces the exact one."""
    pv = pd.read_csv(OUT / "pv_events.csv")
    ex = pv[pv.method.str.startswith("sampler_")].copy()
    ex["kind"] = ex.method.str[8]
    ex = ex.set_index(["event", "set", "kind"])[["matched_elig", "fakes"]]
    rows = []
    for (ev, st, kind), g in s.groupby(["event", "set", "kind"]):
        e = ex.loc[(ev, st, kind)]
        g = g.sort_values("K")
        ok = ((g.matched_elig >= e.matched_elig) & (g.fakes <= e.fakes)).to_numpy()
        # suffix-all: ok at K and every larger K
        suf = np.logical_and.accumulate(ok[::-1])[::-1]
        K = float(g.K.to_numpy()[np.argmax(suf)]) if suf.any() else np.inf
        r = g.iloc[0]
        qps = (g.queries / g.K).iloc[0]
        P = (g.K / g.classical_sampling).iloc[0]
        rows.append(dict(event=ev, set=st, kind=kind, T=r["T"], n_pv=r.n_pv, n_hits=r.n_hits, n_reco=r.n_reco,
                         n_seg=r.n_seg, K_rep=K, queries_per_shot=qps, queries=K * qps,
                         classical_sampling=K / P if P > 0 else np.inf))
    return pd.DataFrame(rows)


def main():
    pv = pd.read_csv(OUT / "pv_events.csv")
    pv = pv[pv.holdout]
    sh = pd.read_csv(OUT / "shots_events.csv")
    sh = sh[sh.holdout]
    rs = pd.read_csv(OUT / "resources_events.csv")
    rs = rs[rs.holdout]
    meta = {}
    for f in (ROOT / "cache").glob("*/*.npz"):
        m = json.loads(str(np.load(f)["meta"]))
        meta[(("real" if f.parent.name == "real" else "toy"), m["event"], f.parent.name)] = m
    tuned = json.load(open(OUT / "tuned.json"))
    lines = [f"tuned sampler settings (02_analyse.py, outputs/tuned.json): {tuned}"]

    # ── PV tables ────────────────────────────────────────────────────────────
    toy = pv[pv.set == "toy"].copy()
    toy["npv_bin"] = pd.cut(toy.n_pv, NPV_BINS, labels=["1", "2-5", "6-10", "11-15", "16-20"])
    tt = toy.groupby(["T", "npv_bin", "method"], observed=True).apply(agg_pv, include_groups=False).reset_index()
    tt.to_csv(OUT / "summary_pv_toy.csv", index=False)
    real = pv[pv.set == "real"]
    tr = real.groupby("method").apply(agg_pv, include_groups=False).reset_index()
    tr.to_csv(OUT / "summary_pv_real.csv", index=False)
    lines += ["", "Run 3 hold-out (79 events), PVs with >= 4 reconstructible tracks, 1 mm match:",
              tr.round(3).to_string(index=False), "",
              "Toy (hold-out sweep), efficiency over PVs with >= 4 reconstructible tracks:"]
    piv = tt.pivot_table(index=["T", "method"], columns="npv_bin", values="eff", observed=True)
    lines += [piv.round(3).to_string(), "", "Toy fakes per event:",
              tt.pivot_table(index=["T", "method"], columns="npv_bin", values="fakes", observed=True).round(2).to_string(),
              "", "Toy z residual rms [mm]:",
              tt.pivot_table(index=["T", "method"], columns="npv_bin", values="rms_mm", observed=True).round(3).to_string()]

    # ── shots, queries, qubits ───────────────────────────────────────────────
    kr = k_rep(sh)
    kr.to_csv(OUT / "krep_events.csv", index=False)
    scan_eval = []
    for _, r in kr[kr.kind == "t"].iterrows():
        key = [k for k in meta if k[1] == r.event and k[0] == r.set and k[2] != "tune"][0]
        m = meta[key]
        n_sl = (28.0 + 150.0) / 0.25 if r.set == "toy" else 500.0 / 0.25
        scan_eval.append(dict(event=r.event, set=r.set, scan_evals=m["n_hits"] * n_sl))
    kr = kr.merge(pd.DataFrame(scan_eval), on=["event", "set"], how="left")
    res = kr.merge(rs[["event", "set", "kind", "q_index", "q_zeta", "q_data", "q_work_model", "P", "P_padded",
                       "rounds", "n_items", "n_flagged", "frac_flag_true", "q_seg_1bqf", "q_seg_qsvt"]],
                   on=["event", "set", "kind"])
    # classical rejection sampling over the SAME index register (same oracle, padded slots)
    res["classical_sampling_same_oracle"] = res.K_rep / res.P_padded
    res.to_csv(OUT / "summary_resources_events.csv", index=False)
    res["npv_bin"] = pd.cut(res.n_pv.where(res.set == "toy", 0).clip(lower=0), [-1] + NPV_BINS,
                            labels=["run3", "1", "2-5", "6-10", "11-15", "16-20"])
    res["Tkey"] = np.where(res.set == "real", "Run 3", res["T"].astype(str))
    fin = res[np.isfinite(res.K_rep)]
    g = res.groupby(["Tkey", "kind"])
    summ = pd.DataFrame(dict(
        events=g.size(), unsettled=g.K_rep.apply(lambda k: int((~np.isfinite(k)).sum())),
        n_reco=g.n_reco.median(), n_hits=g.n_hits.median(), n_seg=g.n_seg.median(),
        q_index=g.q_index.median(), q_zeta=g.q_zeta.median(), q_data=g.q_data.median(),
        q_work_model=g.q_work_model.median(), q_seg_1bqf=g.q_seg_1bqf.median(), q_seg_qsvt=g.q_seg_qsvt.median(),
        P_padded=g.P_padded.median(), rounds=g.rounds.median(), frac_flag_true=g.frac_flag_true.median(),
        K_rep_med=fin.groupby(["Tkey", "kind"]).K_rep.median(),
        K_rep_p90=fin.groupby(["Tkey", "kind"]).K_rep.quantile(0.9),
        queries_med=fin.groupby(["Tkey", "kind"]).queries.median(),
        classical_sampling_med=fin.groupby(["Tkey", "kind"]).classical_sampling.median(),
        classical_sampling_same_oracle_med=fin.groupby(["Tkey", "kind"]).classical_sampling_same_oracle.median(),
        classical_exhaustive_med=g.n_items.median(), doublet_register=g.n_seg.median(),
        scan_evals_med=g.scan_evals.median())).reset_index()
    summ.to_csv(OUT / "summary_resources.csv", index=False)
    pd.set_option("display.width", 250)
    lines += ["", "Qubits, shots and queries (median per event; hold-out):", summ.to_string(index=False, float_format=lambda v: f"{v:.3g}")]
    gq = fin[fin.set == "toy"].groupby(["T", "kind", "npv_bin"], observed=True)
    qv = pd.DataFrame(dict(K_rep=gq.K_rep.median(), queries=gq.queries.median())).reset_index()
    qv.to_csv(OUT / "summary_queries_vs_npv.csv", index=False)
    lines += ["", "Toy: shots and queries against PVs (median):",
              qv.pivot_table(index=["T", "kind"], columns="npv_bin", values="queries", observed=True).to_string(float_format=lambda v: f"{v:.3g}")]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ── figures ──────────────────────────────────────────────────────────────
    order = ["sampler_t_exact", "sampler_d_exact", "scan", "scan_refit", "trk_classical_fit",
             "trk_1bqf_fit", "trk_qsvt_fit", "trk_truth_fit"]
    # I01: toy PV efficiency, fakes and residual vs n_pv
    tv = toy.groupby(["T", "n_pv", "method"]).apply(agg_pv, include_groups=False).reset_index()
    fig, ax = plt.subplots(3, 3, figsize=(WIDTH, 6.6), sharex=True)
    for j, T in enumerate((100, 400, 1000)):
        for mth in order:
            d = tv[(tv["T"] == T) & (tv.method == mth)]
            lab, c, mk, ls = M[mth]
            for i, col in enumerate(("eff", "fakes", "rms_mm")):
                ax[i, j].plot(d.n_pv, d[col] * (1000 if col == "rms_mm" else 1), color=c, marker=mk, ls=ls, ms=3,
                              lw=1, label=lab if (i == 0 and j == 0) else None)
        ax[0, j].set_title(f"T = {T} tracks")
        ax[0, j].set_ylim(0, 1.05)
        ax[1, j].set_yscale("symlog", linthresh=0.1)
        ax[2, j].set_yscale("log")
        ax[2, j].set_xlabel("primary vertices")
    ax[0, 0].set_ylabel("PV efficiency (≥ 4 tracks)")
    ax[1, 0].set_ylabel("fakes per event")
    ax[2, 0].set_ylabel("z residual rms [µm]")
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.tight_layout()
    save(fig, "I01_toy_pv_vs_npv")

    # I02: Run 3 bars
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.6))
    trr = tr.set_index("method").loc[[m for m in order if m in set(tr.method)]]
    y = np.arange(len(trr))
    for i, (col, lab) in enumerate((("eff", "PV efficiency (≥ 4 tracks)"), ("fakes", "fakes per event"),
                                    ("rms_mm", "z residual rms [mm]"))):
        ax[i].barh(y, trr[col], color=[M[m][1] for m in trr.index])
        ax[i].set_xlabel(lab)
        ax[i].set_yticks(y, [M[m][0] for m in trr.index] if i == 0 else [])
        ax[i].invert_yaxis()
        if col == "fakes":
            ax[i].set_xscale("log")
    ax[0].set_title("Run 3 Sim10, 79 hold-out events", loc="left")
    fig.tight_layout()
    save(fig, "I02_run3_pv")

    # I03: efficiency and fakes against shots
    fig, ax = plt.subplots(2, 3, figsize=(WIDTH, 4.2), sharex=True)
    for j, (st, T, title) in enumerate((("toy", 100, "toy, T = 100"), ("toy", 1000, "toy, T = 1000"),
                                       ("real", -1, "Run 3, hold-out"))):
        for kind in ("t", "d"):
            d = sh[(sh.set == st) & (sh["T"] == T) & (sh.kind == kind)].groupby("K").agg(
                me=("matched_elig", "sum"), nel=("n_elig", "sum"), f=("fakes", "mean"))
            e = pv[(pv.set == st) & (pv["T"] == T) & (pv.method == f"sampler_{kind}_exact")]
            lab, c, mk = KIND[kind]
            ax[0, j].plot(d.index, d.me / d.nel, color=c, marker=mk, ms=3, lw=1, label=lab)
            ax[0, j].axhline(e.matched_elig.sum() / e.n_elig.sum(), color=c, ls="--", lw=0.8)
            ax[1, j].plot(d.index, d.f, color=c, marker=mk, ms=3, lw=1)
            ax[1, j].axhline(e.fakes.mean(), color=c, ls="--", lw=0.8)
        ax[0, j].set_title(title)
        ax[1, j].set_xscale("log", base=2)
        ax[1, j].set_yscale("symlog", linthresh=0.1)
        ax[1, j].set_xlabel("accepted shots K")
    ax[0, 0].set_ylabel("PV efficiency")
    ax[1, 0].set_ylabel("fakes per event")
    ax[0, 0].legend(title="dashed: K = ∞ (exact)")
    fig.tight_layout()
    save(fig, "I03_shots")

    # I04: qubits against tracks and PVs
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    for kind in ("t", "d"):
        lab, c, mk = KIND[kind]
        d = res[res.kind == kind]
        ax[0].scatter(d.n_reco, d.q_data, s=6, color=c, marker=mk, label=f"vertex sampler, {lab} (data registers)")
        dt = d[d.set == "toy"].groupby(["T", "n_pv"]).q_data.median().reset_index()
        for T, ls in ((100, ":"), (400, "--"), (1000, "-")):
            x = dt[dt["T"] == T]
            ax[1].plot(x.n_pv, x.q_data, color=c, ls=ls, lw=1, label=f"{lab}, T = {T}" if kind == "t" else None)
    d = res[res.kind == "d"]
    ax[0].scatter(d.n_reco, d.q_seg_1bqf, s=6, color="#1f77b4", marker="s", label="segment register, one-bit filter")
    ax[0].scatter(d.n_reco, d.q_seg_qsvt, s=6, color="#d95f02", marker="^", label="segment register, QSVT")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("reconstructible tracks")
    ax[0].set_ylabel("logical qubits")
    ax[0].legend(fontsize=6)
    ax[1].set_xlabel("primary vertices (toy)")
    ax[1].set_ylabel("logical qubits, vertex sampler")
    ax[1].legend(fontsize=6, ncol=1)
    fig.tight_layout()
    save(fig, "I04_qubits")

    # I05: queries against tracks and PVs
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    for kind in ("t", "d"):
        lab, c, mk = KIND[kind]
        d = fin[fin.kind == kind]
        ax[0].scatter(d.n_reco, d.queries, s=6, color=c, marker=mk, label=f"quantum sampler, {lab}")
        ax[0].scatter(d.n_reco, d.classical_sampling_same_oracle, s=4, color=c, marker=".", alpha=0.4,
                      label=f"classical sampling, same oracle, {lab}")
        if kind == "t":
            ax[0].scatter(d.n_reco, d.n_items, s=4, color=c, marker="+", alpha=0.6,
                          label="coupled-pair list read once (needs A built)")
        dt = d[d.set == "toy"].groupby(["T", "n_pv"]).queries.median().reset_index()
        for T, ls in ((100, ":"), (400, "--"), (1000, "-")):
            x = dt[dt["T"] == T]
            ax[1].plot(x.n_pv, x.queries, color=c, ls=ls, lw=1, label=f"{lab}, T = {T}")
    d = res[res.kind == "d"]
    ax[0].scatter(d.n_reco, d.n_seg, s=4, color="0.3", marker="_", label="every segment read once")
    ax[0].scatter(d.n_reco, d.scan_evals, s=4, color="0.6", marker="x", label="focusing scan, coarse hit evaluations")
    for a in ax:
        a.set_yscale("log")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("reconstructible tracks")
    ax[0].set_ylabel("oracle queries per event")
    ax[1].set_xlabel("primary vertices (toy)")
    ax[1].set_ylabel("quantum queries per event (median)")
    ax[0].legend(fontsize=5.5)
    ax[1].legend(fontsize=5.5, ncol=2)
    fig.tight_layout()
    save(fig, "I05_queries")


if __name__ == "__main__":
    main()
