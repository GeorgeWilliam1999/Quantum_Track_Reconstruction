"""
Quantum (1BQF) vs classical comparison figures.

Data sources (all fresh / new):
  * assets/quantum_sweep.csv  — fresh CPU statevector sweep (T=10/20/50, step+erf, 3 reps)
  * qtrk_store metrics.csv     — the campaign's T=200 quantum (step) as a high-T anchor
  * assets/solvecs_T*.npz      — representative classical + rescaled-quantum vectors

Produces (figures/):
  cos_QC_vs_T              quantum-classical fidelity vs T (step vs erf), + T=200 anchor
  P_anc_vs_T               ancilla success probability vs T
  quantum_seg_metrics_vs_T quantum segment efficiency & purity vs T (vs classical)
  solution_hist_CQ         classical vs rescaled-quantum activation histograms (true/false)
  quantum_cost_vs_T        n_qubits and statevector solve time vs T
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import _common as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SWEEP = cm.ASSETS / "quantum_sweep.csv"


def _agg(df, by, col):
    g = df.groupby(by)[col].agg(["mean", "count", "std"]).reset_index()
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    return g


def load_sweep():
    if not SWEEP.exists():
        raise SystemExit("quantum_sweep.csv not present yet — wait for gen_quantum.py")
    return pd.read_csv(SWEEP)


def store_quantum_anchor():
    """Clean-cell T=200 quantum (step) cos_QC from the store, if present."""
    M = pd.read_csv(cm.METRICS_CSV)
    q = M[(M.solver == "quantum") & (M.n_trk == 200) & (M.kernel == "step")]
    clean = q[(q.sigma_res == 0.0) & (np.isclose(q.sigma_scatt, 1e-4)) &
              (q.hit_ineff == 0.0)]
    use = clean if len(clean) else q
    if not len(use):
        return None
    return dict(T=200, cos_mean=use.cos_QC.mean(), cos_sem=use.cos_QC.std()/np.sqrt(len(use)),
                P_anc=use.P_anc.mean(), n=len(use), clean=bool(len(clean)))


# ---------------------------------------------------------------------------
def fig_cos_QC_vs_T():
    s = load_sweep()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for kern, c, mk in [("step", cm.C["step"], "o"), ("erf", cm.C["erf"], "s")]:
        a = _agg(s[s.kernel == kern], "T", "cos_QC").sort_values("T")
        ax.errorbar(a["T"], a["mean"], yerr=a["sem"], marker=mk, capsize=3,
                    color=c, label=f"{kern} kernel")
    anc = store_quantum_anchor()
    if anc:
        ax.errorbar([anc["T"]], [anc["cos_mean"]], yerr=[anc["cos_sem"]], marker="*",
                    ms=15, color=cm.C["step"], mec="k",
                    label=f"step, $T=200$ (store, n={anc['n']})")
    ax.set_xscale("log"); ax.set_xlabel("Track multiplicity  $T$")
    ax.set_ylabel(r"quantum fidelity  $\cos\theta_{QC}$"); ax.set_ylim(0, 1)
    ax.set_title("1BQF$-$classical fidelity vs multiplicity\n(erf $>$ step at every $T$; both fall as $T$ grows)")
    ax.legend(); cm.savefig(fig, "cos_QC_vs_T")


def fig_P_anc_vs_T():
    s = load_sweep()
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for kern, c, mk in [("step", cm.C["step"], "o"), ("erf", cm.C["erf"], "s")]:
        a = _agg(s[s.kernel == kern], "T", "P_anc").sort_values("T")
        ax.errorbar(a["T"], a["mean"], yerr=a["sem"], marker=mk, capsize=3,
                    color=c, label=f"{kern} kernel")
    anc = store_quantum_anchor()
    if anc:
        ax.scatter([anc["T"]], [anc["P_anc"]], marker="*", s=200,
                   color=cm.C["step"], edgecolor="k", zorder=5, label="step, $T=200$ (store)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Track multiplicity  $T$")
    ax.set_ylabel(r"ancilla success $P_{\rm anc}$")
    ax.set_title("Ancilla post-selection probability shrinks with $T$\n(more segments $\\Rightarrow$ smaller true-band weight)")
    ax.legend(); cm.savefig(fig, "P_anc_vs_T")


def fig_quantum_seg_metrics_vs_T():
    s = load_sweep()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sub = s[s.kernel == "step"]
    for col, lab, c in [("effQ", "quantum efficiency", cm.C["true"]),
                        ("purQ", "quantum purity", cm.C["quantum"]),
                        ("effC", "classical efficiency", cm.C["classical"])]:
        a = _agg(sub, "T", col).sort_values("T")
        ls = "-" if col != "effC" else "--"
        ax.errorbar(a["T"], a["mean"], yerr=a["sem"], marker="o", capsize=3,
                    color=c, ls=ls, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("Track multiplicity  $T$")
    ax.set_ylabel("segment metric (after $x>0.35$)"); ax.set_ylim(0.4, 1.05)
    ax.set_title("After threshold, 1BQF recovers the true segments (clean, step)")
    ax.legend(loc="lower left"); cm.savefig(fig, "quantum_seg_metrics_vs_T")


def fig_solution_hist_CQ(T=20, kernel="step"):
    p = cm.ASSETS / f"solvecs_T{T}_{kernel}.npz"
    if not p.exists():
        print(f"  [skip] solution_hist_CQ: {p.name} not present yet"); return
    d = np.load(p)
    solC = d["solC"].astype(float); solQ = d["solQ"].astype(float)
    tr = d["truth"].astype(bool)
    fig, (axC, axQ) = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    for ax, sol, name in [(axC, solC, "Classical $x_C$"), (axQ, solQ, r"1BQF $x_Q$ (rescaled)")]:
        m = max(0.5, sol.max() * 1.05)
        bins = np.linspace(0, m, 55)
        ax.hist(sol[~tr], bins=bins, color=cm.C["false"], alpha=0.8, label="false")
        ax.hist(sol[tr], bins=bins, color=cm.C["true"], alpha=0.85, label="true")
        ax.axvline(0.35, color=cm.C["quantum"], lw=2, label=r"$\tau=0.35$")
        ax.set_yscale("log"); ax.set_xlabel("activation"); ax.set_title(name)
        ax.legend()
    axC.set_ylabel("count")
    cosqc = float(d["cos_QC"])
    fig.suptitle(fr"Classical vs 1BQF activations on the same event ($T={T}$, "
                 fr"$\cos\theta_{{QC}}={cosqc:.2f}$): both separate true from false at 0.35",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    cm.savefig(fig, "solution_hist_CQ")


def fig_quantum_cost_vs_T():
    s = load_sweep()
    sub = s[s.kernel == "step"]
    a_q = _agg(sub, "T", "n_qubits").sort_values("T")
    a_t = _agg(sub, "T", "t_q").sort_values("T")
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    l1, = ax.plot(a_q["T"], a_q["mean"], "o-", color=cm.C["accent"], label="qubits (sys+2)")
    ax.set_xscale("log"); ax.set_xlabel("Track multiplicity  $T$")
    ax.set_ylabel("number of qubits", color=cm.C["accent"])
    ax2 = ax.twinx(); ax2.set_xscale("log"); ax2.set_yscale("log")
    l2, = ax2.plot(a_t["T"], a_t["mean"], "s--", color=cm.C["quantum"], label="statevector solve time (s)")
    ax2.set_ylabel("CPU statevector time (s)", color=cm.C["quantum"])
    ax.set_title("1BQF simulation cost: qubits $\\sim\\log_2 n_{\\rm seg}$,\n"
                 "statevector time $\\sim O(A_{\\rm nnz}\\,2^{n_{\\rm sys}})$")
    ax.legend(handles=[l1, l2], loc="upper left", fontsize=10)
    cm.savefig(fig, "quantum_cost_vs_T")


if __name__ == "__main__":
    print("== quantum comparison figures ==")
    fig_cos_QC_vs_T()
    fig_P_anc_vs_T()
    fig_quantum_seg_metrics_vs_T()
    fig_solution_hist_CQ()
    fig_quantum_cost_vs_T()
    print("done.")
