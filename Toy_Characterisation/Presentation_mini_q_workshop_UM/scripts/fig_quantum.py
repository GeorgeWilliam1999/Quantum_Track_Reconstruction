"""
Quantum (1BQF) vs classical comparison figures.

Now sourced from the FULL store metrics view (qtrk_store/manifest/metrics.csv),
which after the campaign carries quantum solves across T=10..1000 for both
kernels — so the vs-T curves are no longer the thin CPU-sweep stub.  We select
the clean physics cell (sigma_scatt=1e-4, sigma_res=0, no drop, gamma=3) and
pool it across studies (same deterministic events) to maximise T coverage.

Produces (figures/):
  cos_QC_vs_T              quantum-classical fidelity vs T (step vs erf)
  P_anc_vs_T               ancilla success probability vs T
  quantum_seg_metrics_vs_T quantum segment efficiency & purity vs T (+ classical)
  solution_hist_CQ         classical vs rescaled-quantum activation histograms
  quantum_cost_vs_T        n_qubits and statevector solve time vs T
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import _common as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SWEEP = cm.ASSETS / "quantum_sweep.csv"
_M = None


def M():
    global _M
    if _M is None:
        _M = pd.read_csv(cm.METRICS_CSV)
    return _M


def _agg(df, by, col):
    g = df.groupby(by)[col].agg(["mean", "count", "std"]).reset_index()
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    return g


# Per-kernel "clean" cell chosen to reach the highest available T with a single
# consistent physics point.  cos_QC is the CANONICAL signal-support cosine
# (qtrk_pipeline.metrics: cos on sol_C>0.35, false-bulk-free); the full-vector
# cosine is ~0.1 and understates the solve (it is dominated by the 0.25 bulk).
KERNEL_CELL = {                              # (study) reaching highest T
    "step": "Epsilon_study_2",               # formula eps, quantum to T=1000
    "erf":  "ERF",                           # formula eps, quantum to T=400
}


def cell(solver, kernel):
    d = M()
    m = (d.solver == solver) & (d.study == KERNEL_CELL[kernel]) & (d.kernel == kernel) & \
        (np.isclose(d.sigma_scatt, 1e-4)) & (d.sigma_res == 0.0) & (d.hit_ineff == 0.0)
    if "gamma" in d.columns:
        m &= np.isclose(d.gamma, 3.0)
    return d[m].copy()


def _series(df, col):
    """median + (mad-ish) spread per n_trk — robust to the rare failed solve."""
    g = df.groupby("n_trk")[col]
    out = g.median().reset_index().rename(columns={col: "med"})
    out["lo"] = g.quantile(0.16).values; out["hi"] = g.quantile(0.84).values
    out["n"] = g.size().values
    return out.sort_values("n_trk")


def _tmax_note(ax, *dfs):
    tmax = max(int(d.n_trk.max()) for d in dfs if len(d))
    ax.text(0.98, 0.03, f"store metrics view; quantum T up to {tmax}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.45")


# ---------------------------------------------------------------------------
def fig_cos_QC_vs_T():
    qs, qe = cell("quantum", "step"), cell("quantum", "erf")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for df, c, mk, lab in [(qs, cm.C["step"], "o", "step kernel"),
                           (qe, cm.C["erf"], "s", "erf kernel")]:
        a = _series(df, "cos_QC")
        if a.empty:
            continue
        ax.fill_between(a["n_trk"], a["lo"], a["hi"], color=c, alpha=0.15)
        ax.plot(a["n_trk"], a["med"], marker=mk, color=c, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("Track multiplicity  $T$")
    ax.set_ylabel(r"fidelity  $\cos\theta_{QC}$ (signal support)"); ax.set_ylim(0.5, 1.01)
    ax.axhline(1.0, ls=":", c="grey", lw=1)
    ax.set_title("1BQF$-$classical fidelity is high & flat across multiplicity\n"
                 "(measured on the classical signal support, $x_C>0.35$)")
    ax.legend(loc="lower left"); _tmax_note(ax, qs, qe)
    cm.savefig(fig, "cos_QC_vs_T")


def fig_P_anc_vs_T():
    qs, qe = cell("quantum", "step"), cell("quantum", "erf")
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for df, c, mk, lab in [(qs, cm.C["step"], "o", "step kernel"),
                           (qe, cm.C["erf"], "s", "erf kernel")]:
        a = _series(df, "P_anc")
        if a.empty:
            continue
        ax.plot(a["n_trk"], a["med"], marker=mk, color=c, label=lab)
    # 1/T guide
    T = np.array(sorted(qs.n_trk.unique()), float)
    if len(T):
        ref = _series(qs, "P_anc")["med"].iloc[0] * (T[0] / T)
        ax.plot(T, ref, "k--", lw=1.2, alpha=0.6, label=r"$\propto 1/T$ guide")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Track multiplicity  $T$")
    ax.set_ylabel(r"ancilla success $P_{\rm anc}$")
    ax.set_title("Ancilla post-selection probability shrinks $\\sim1/T$")
    ax.legend(); _tmax_note(ax, qs, qe); cm.savefig(fig, "P_anc_vs_T")


def fig_quantum_seg_metrics_vs_T():
    qs, qe = cell("quantum", "step"), cell("quantum", "erf")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for df, col, lab, color, mk in [
        (qs, "segment_efficiency", "step: efficiency", cm.C["step"], "o"),
        (qs, "segment_purity", "step: purity", cm.C["classical"], "o"),
        (qe, "segment_efficiency", "erf: efficiency", cm.C["erf"], "s"),
        (qe, "segment_purity", "erf: purity", cm.C["quantum"], "s")]:
        a = _series(df, col)
        if a.empty:
            continue
        ls = "-" if "efficiency" in lab else "--"
        ax.plot(a["n_trk"], a["med"], marker=mk, color=color, ls=ls, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("Track multiplicity  $T$")
    ax.set_ylabel("quantum segment metric (at $x>0.35$)"); ax.set_ylim(-0.03, 1.05)
    ax.set_title("1BQF segment metrics: erf keeps all true (eff=1);\n"
                 "step drops the outer segment (eff$\\approx$0.75); purity falls at high $T$")
    ax.legend(loc="lower left", fontsize=9, ncol=2); _tmax_note(ax, qs, qe)
    cm.savefig(fig, "quantum_seg_metrics_vs_T")


def fig_quantum_cost_vs_T():
    qs = cell("quantum", "step")
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    if "n_qubits" in qs.columns and qs["n_qubits"].notna().any():
        aq = _agg(qs, "n_trk", "n_qubits").sort_values("n_trk")
    else:  # derive from n_seg: n_sys = ceil(log2 n_seg), +2 (time+ancilla)
        qs = qs.assign(n_qubits=np.ceil(np.log2(qs.n_seg.clip(lower=2))) + 2)
        aq = _agg(qs, "n_trk", "n_qubits").sort_values("n_trk")
    tcol = "t_solve" if "t_solve" in qs.columns else None
    l1, = ax.plot(aq["n_trk"], aq["mean"], "o-", color=cm.C["accent"], label="qubits (sys+2)")
    ax.set_xscale("log"); ax.set_xlabel("Track multiplicity  $T$")
    ax.set_ylabel("number of qubits", color=cm.C["accent"])
    handles = [l1]
    if tcol and qs[tcol].notna().any() and (qs[tcol] > 0).any():
        at = _agg(qs[qs[tcol] > 0], "n_trk", tcol).sort_values("n_trk")
        ax2 = ax.twinx(); ax2.set_xscale("log"); ax2.set_yscale("log")
        l2, = ax2.plot(at["n_trk"], at["mean"], "s--", color=cm.C["quantum"],
                       label="solve time (s)")
        ax2.set_ylabel("solve time (s)", color=cm.C["quantum"]); handles.append(l2)
    ax.set_title("1BQF cost: qubits $\\sim\\log_2 n_{\\rm seg}$;\n"
                 "simulation time $\\sim O(A_{\\rm nnz}\\,2^{n_{\\rm sys}})$")
    ax.legend(handles=handles, loc="upper left", fontsize=10)
    _tmax_note(ax, qs); cm.savefig(fig, "quantum_cost_vs_T")


# --- per-event histogram (unchanged: representative stored vectors) --------
def fig_solution_hist_CQ(T=20, kernel="step"):
    p = cm.ASSETS / f"solvecs_T{T}_{kernel}.npz"
    if not p.exists():
        print(f"  [skip] solution_hist_CQ: {p.name} not present"); return
    d = np.load(p)
    solC = d["solC"].astype(float); solQ = d["solQ"].astype(float)
    tr = d["truth"].astype(bool)
    fig, (axC, axQ) = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    for ax, sol, name in [(axC, solC, "Classical $x_C$"), (axQ, solQ, r"1BQF $x_Q$ (rescaled)")]:
        mx = max(0.5, sol.max() * 1.05); bins = np.linspace(0, mx, 55)
        ax.hist(sol[~tr], bins=bins, color=cm.C["false"], alpha=0.8, label="false")
        ax.hist(sol[tr], bins=bins, color=cm.C["true"], alpha=0.85, label="true")
        ax.axvline(0.35, color=cm.C["quantum"], lw=2, label=r"$\tau=0.35$")
        ax.set_yscale("log"); ax.set_xlabel("activation"); ax.set_title(name); ax.legend()
    axC.set_ylabel("count")
    cosqc = float(d["cos_QC"])
    fig.suptitle(fr"Classical vs 1BQF activations on the same event ($T={T}$, "
                 fr"$\cos\theta_{{QC}}={cosqc:.2f}$): both separate true from false at 0.35",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    cm.savefig(fig, "solution_hist_CQ")


if __name__ == "__main__":
    print("== quantum comparison figures (store-driven) ==")
    fig_cos_QC_vs_T()
    fig_P_anc_vs_T()
    fig_quantum_seg_metrics_vs_T()
    fig_solution_hist_CQ()
    fig_quantum_cost_vs_T()
    print("done.")
