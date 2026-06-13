#!/usr/bin/env python3
"""
T = 400 evaluation of the segment metrics — the production-multiplicity
companion to the T = 30 epsilon-sensitivity scan, and the efficiency-first
working-point (tau) view the user asked for.

Part A  classical direct epsilon-sweep at T=400 (fresh, paired events): the
        eff/false-rate knee + c*eps^2 law shown to hold at production T.
        Quantum at T=400 over a swept epsilon is a GPU campaign (one 1BQF
        solve is ~360 s, 22 qubits), so the quantum point is taken from the
        store at the formula epsilon and overlaid.

Part B  store-backed working points at T=400 (formula epsilon): BOTH solvers
        recomputed from the stored solution vectors at
          (i)  the fixed absolute tau = 0.35, and
          (ii) the efficiency-first tau — placed just below the 1% quantile
               of that solver's TRUE amplitudes so >= 99% of true segments
               survive, with the false rate paid quoted.
        This recovers the 1BQF's real efficiency from the tau=0.35 ~0.75
        plateau (the plateau is the cut, not lost physics).

Figures -> figures/epsilon_sensitivity/T400_*.png
Numbers -> outputs/T400_eval.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

import helpers                                              # noqa: E402
import pandas as pd                                         # noqa: E402
import qtrk_pipeline as qp                                  # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon          # noqa: E402
from lhcb_velo_toy.analysis.segment_metrics import (        # noqa: E402
    DEFAULT_DZ, DEFAULT_THETA_MIN, segment_truth_mask, solver_segment_metrics)
from lhcb_velo_toy.solvers import SimpleHamiltonianFast     # noqa: E402

FIGDIR = HERE / "figures" / "epsilon_sensitivity"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR = HERE / "outputs"
STORE_METRICS = "/data/bfys/gscriven/qtrk_store/manifest/metrics.csv"

DZ, TH_MIN = DEFAULT_DZ, DEFAULT_THETA_MIN
GEOM = helpers.make_geometry()
TAU = 0.35
TARGET_EFF = 0.99
T = 400
NREP_A = 2                     # paired classical eps-sweep reps (T=400)
# Per-cell RELATIVE eps grid: at T=400, eps >~ 2x the formula value makes A
# dense (A_nnz explodes, spsolve intractable, >145 s measured). The formula
# eps keeps A sparse (A_nnz ~ n_seg, <1 s). So sweep eps as a multiple of each
# cell's formula eps, spanning the knee (0.3x) to moderately above (1.5x),
# with an absolute clip and an A_nnz guard so no single solve runs away.
EPS_MULT = np.array([0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5])
EPS_ABS_MAX = 1.8e-2           # absolute clip (worst-cell safety)
A_NNZ_GUARD = 3_000_000        # skip solve above this (record NaN)
SR_FIX, SS_FIX = 0.01, 1e-4
FAMILY_A = [(SR_FIX, ss) for ss in (1e-4, 5e-4, 2e-3)]
FAMILY_B = [(sr, SS_FIX) for sr in (0.0, 0.02, 0.05)]
CMAP = plt.get_cmap("viridis")
P_DEFAULT = float(np.exp(-9.0))


def sigma_p(sr, ss):
    return float(np.sqrt(ss**2 + 6.0 * np.arctan(sr / DZ) ** 2))


def p_kink(eps, sr, ss):
    x = eps**2 - 2.0 * TH_MIN**2
    return 1.0 if x <= 0 else float(np.exp(-x / (2.0 * sigma_p(sr, ss) ** 2)))


def eff_analytic(p):
    q = 1.0 - np.asarray(p, float)
    return q**2 * (1.0 + 0.5 * (1.0 - q))


def wp_tau(sol, truth, target=TARGET_EFF):
    """Efficiency-first tau: the LARGEST absolute tau on a fine grid whose
    segment efficiency is still >= target (keep purity as high as possible
    while holding efficiency). Robust to the discrete Hopfield levels (a
    quantile lands exactly on a degenerate level and the float32 compare
    drops the whole cluster). If even tau=0 cannot reach the target — the
    indefinite/hub regime where true amplitudes go negative — return 0.0,
    which honestly reports eff<target rather than a nonsense negative tau."""
    ta = np.asarray(sol)[truth]
    if ta.size == 0:
        return TAU
    grid = np.linspace(0.0, 0.50, 1001)
    effs = np.array([(ta > t).mean() for t in grid])
    ok = np.where(effs >= target)[0]
    return float(grid[ok.max()]) if ok.size else 0.0


def agg(rows, xkey, ykey):
    xs = sorted(set(r[xkey] for r in rows))
    m = [np.mean([r[ykey] for r in rows if r[xkey] == x]) for x in xs]
    e = [np.std([r[ykey] for r in rows if r[xkey] == x]) /
         np.sqrt(max(1, sum(r[xkey] == x for r in rows))) for x in xs]
    return np.array(xs), np.array(m), np.array(e)


def fit_quadratic(eps, far):
    keep = (far > 0) & (far < 0.5)
    if keep.sum() < 2:
        return float("nan")
    return float(np.sum(far[keep] * eps[keep] ** 2) / np.sum(eps[keep] ** 4))


# ===========================================================================
# PART A — classical direct eps-sweep at T=400
# ===========================================================================
print("=== PART A: classical eps-sweep T=400 ===", flush=True)
direct: dict[str, list] = {}
fits: dict[str, float] = {}
for sr, ss in FAMILY_A + [(SR_FIX, SS_FIX)] + FAMILY_B:
    key = f"sr{sr:g}_ss{ss:g}"
    if key in direct:
        continue
    eps_f = float(compute_epsilon(sr, ss))
    eps_grid = np.unique(np.clip(eps_f * EPS_MULT, 1e-4, EPS_ABS_MAX))
    rows = []
    for rep in range(NREP_A):
        ev = helpers.safe_generate(T, seed=84_000 + rep, geom=GEOM,
                                   measurement_error=sr, collision_noise=ss)
        for eps in eps_grid:
            t0 = time.time()
            ham = SimpleHamiltonianFast(epsilon=float(eps), gamma=helpers.GAMMA,
                                        delta=helpers.DELTA, theta_d=1e-4)
            ham.construct_segments(ev)
            ham.construct_hamiltonian(ev, convolution=False)
            truth = segment_truth_mask(ham)
            if ham.A.nnz > A_NNZ_GUARD:
                print(f"  {key} rep{rep} eps={eps:.2e} A_nnz={ham.A.nnz} "
                      f">guard, skip", flush=True)
                continue
            solC = np.asarray(ham.solve_classicaly()).ravel()
            m = solver_segment_metrics(truth, solC, threshold=TAU)
            te = wp_tau(solC, truth)
            mw = solver_segment_metrics(truth, solC, threshold=te)
            rows.append(dict(eps=float(eps), eps_mult=float(eps / eps_f),
                             sr=sr, ss=ss,
                             eff_C=m["segment_efficiency"],
                             far_C=m["segment_false_rate"],
                             eff_Cw=mw["segment_efficiency"],
                             far_Cw=mw["segment_false_rate"],
                             tau_w=te, A_nnz=int(ham.A.nnz),
                             p_kink=p_kink(float(eps), sr, ss)))
        print(f"  {key} rep{rep} done ({time.time()-t0:.0f}s last solve)",
              flush=True)
    direct[key] = rows

# store quantum at the formula eps (overlay points), T=400
dfm = pd.read_csv(STORE_METRICS)
dfm = dfm[(dfm["study"].str.contains("Epsilon_study_2")) & (dfm["n_trk"] == T)]
qpt = {}
for sr, ss in FAMILY_A + [(SR_FIX, SS_FIX)] + FAMILY_B:
    sub = dfm[(dfm.solver == "quantum") & (np.isclose(dfm.sigma_res, sr)) &
              (np.isclose(dfm.sigma_scatt, ss))]
    if len(sub):
        qpt[(sr, ss)] = (float(compute_epsilon(sr, ss)),
                         sub.segment_efficiency.mean(),
                         sub.segment_false_rate.mean())

for figname, family, var_lab, fixed_lab in (
    ("eps_scan_T400_fixed_sres", FAMILY_A, "ss",
     rf"$\sigma_{{\rm res}}={SR_FIX}$ mm fixed"),
    ("eps_scan_T400_fixed_sscatt", [(SR_FIX, SS_FIX)] + FAMILY_B, "sr",
     rf"$\sigma_{{\rm scatt}}={SS_FIX:g}$ fixed"),
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    cols = [CMAP(x) for x in np.linspace(0.0, 0.85, len(family))]
    for (sr, ss), col in zip(family, cols):
        rows = direct[f"sr{sr:g}_ss{ss:g}"]
        lab = (rf"$\sigma_s={ss:g}$" if var_lab == "ss"
               else rf"$\sigma_r={sr:g}$ mm")
        eps_f = float(compute_epsilon(sr, ss))
        # efficiency panel: classical tau=0.35 + analytic
        x, m, e = agg(rows, "eps", "eff_C")
        axes[0].errorbar(x, m, yerr=e, fmt="o", ls="-", ms=4.5, capsize=2.5,
                         color=col, label=f"{lab}")
        xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 300)
        axes[0].plot(xx, eff_analytic([p_kink(v, sr, ss) for v in xx]),
                     color=col, lw=1.0, ls=":", alpha=0.8)
        axes[0].axvline(eps_f, color=col, lw=1.0, alpha=0.5)
        # false-rate panel: tau=0.35 (solid) + c eps^2
        x, mC, _ = agg(rows, "eps", "far_C")
        axes[1].errorbar(x, mC, fmt="o", ls="-", ms=4.5, color=col, label=lab)
        c = fit_quadratic(x, mC)
        fits[f"sr{sr:g}_ss{ss:g}"] = c
        if np.isfinite(c):
            axes[1].plot(xx, np.clip(c * xx**2, 0, 1.05), color=col, lw=1.0,
                         ls=":", alpha=0.8)
        axes[1].axvline(eps_f, color=col, lw=1.0, alpha=0.5)
        if (sr, ss) in qpt:
            ef, eq, fq = qpt[(sr, ss)]
            axes[0].plot(ef, eq, "*", ms=13, color=col, mec="k", mew=0.6,
                         zorder=5)
            axes[1].plot(ef, fq, "*", ms=13, color=col, mec="k", mew=0.6,
                         zorder=5)
    axes[0].set_ylabel("segment efficiency")
    axes[1].set_ylabel("segment false rate")
    axes[0].set_ylim(-0.03, 1.06)
    axes[1].set_ylim(-0.03, 1.06)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\varepsilon$ [rad]")
        ax.grid(alpha=0.25)
    axes[0].set_title(r"circles+dotted: classical & analytic $(1-p)^2(1+p/2)$;"
                      r"  $\star$ = store 1BQF at formula $\varepsilon$",
                      fontsize=9)
    axes[1].set_title(r"dotted: $c\,\varepsilon^2$;  vertical: formula "
                      r"$\varepsilon$;  $\star$ = store 1BQF", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(rf"Segment metrics vs $\varepsilon$ at $T={T}$ — {fixed_lab} "
                 rf" (classical, {NREP_A} reps, $\tau={TAU}$)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGDIR / f"{figname}.png", dpi=160)
    plt.close(fig)
    print(f"[A] {figname}.png", flush=True)


# ===========================================================================
# PART B — store working points at T=400, tau=0.35 vs efficiency-first tau
# ===========================================================================
print("=== PART B: store working points T=400 ===", flush=True)

# build classical lookup by (event_key, ham_key) so quantum can be rescaled
cl = dfm[dfm.solver == "classical"].copy()
qu = dfm[dfm.solver == "quantum"].copy()
cl_by_ev = {}
for _, r in cl.iterrows():
    cl_by_ev.setdefault((r.event_key, r.ham_key), r.sol_key)

truth_cache: dict[str, np.ndarray] = {}


def get_truth(event_key):
    if event_key not in truth_cache:
        ev = qp.load_event(qp.event_path(event_key))
        truth_cache[event_key] = qp.truth_from_event(ev)
    return truth_cache[event_key]


def eval_cell(sr, ss, max_cl=12):
    """recompute classical + quantum metrics at tau=0.35 and wp tau."""
    out = {"classical": [], "quantum": []}
    csub = cl[(np.isclose(cl.sigma_res, sr)) & (np.isclose(cl.sigma_scatt, ss))]
    for _, r in csub.head(max_cl).iterrows():
        try:
            truth = get_truth(r.event_key)
            sol = qp.load_solution(r.sol_key)["sol"]
        except Exception as ex:
            print("   skip classical", ex); continue
        m = solver_segment_metrics(truth, sol, threshold=TAU)
        te = wp_tau(sol, truth)
        mw = solver_segment_metrics(truth, sol, threshold=te)
        out["classical"].append(dict(eff=m["segment_efficiency"],
                                     far=m["segment_false_rate"],
                                     eff_w=mw["segment_efficiency"],
                                     far_w=mw["segment_false_rate"], tau_w=te))
    qsub = qu[(np.isclose(qu.sigma_res, sr)) & (np.isclose(qu.sigma_scatt, ss))]
    for _, r in qsub.iterrows():
        ck = cl_by_ev.get((r.event_key, r.ham_key))
        if ck is None:
            continue
        try:
            truth = get_truth(r.event_key)
            solQraw = qp.load_solution(r.sol_key)["sol"]
            solC = qp.load_solution(ck)["sol"]
        except Exception as ex:
            print("   skip quantum", ex); continue
        solQ = qp.rescale_to_signal(solQraw, solC, threshold=TAU)
        m = solver_segment_metrics(truth, solQ, threshold=TAU)
        te = wp_tau(solQ, truth)
        mw = solver_segment_metrics(truth, solQ, threshold=te)
        out["quantum"].append(dict(eff=m["segment_efficiency"],
                                   far=m["segment_false_rate"],
                                   eff_w=mw["segment_efficiency"],
                                   far_w=mw["segment_false_rate"], tau_w=te))
    return out


SR_GRID = [0.0, 0.01, 0.02, 0.05]          # at ss = 1e-4
SS_GRID = [1e-4, 3e-4, 5e-4]               # at sr = 0.01
cells = {("sr", v): (v, SS_FIX) for v in SR_GRID}
cells.update({("ss", v): (SR_FIX, v) for v in SS_GRID})
results = {}
for tag, (sr, ss) in cells.items():
    results[f"{tag[0]}{tag[1]:g}"] = dict(sr=sr, ss=ss, **eval_cell(sr, ss))
    n_c = len(results[f"{tag[0]}{tag[1]:g}"]["classical"])
    n_q = len(results[f"{tag[0]}{tag[1]:g}"]["quantum"])
    print(f"  cell sr{sr:g}_ss{ss:g}: {n_c} classical, {n_q} quantum", flush=True)


def cell_means(rec, solver):
    rows = rec[solver]
    if not rows:
        return None
    f = lambda k: (np.mean([r[k] for r in rows]),
                   np.std([r[k] for r in rows]) / np.sqrt(len(rows)))
    return {k: f(k) for k in ("eff", "far", "eff_w", "far_w", "tau_w")}


# ---- Figure: noise-grid working points at T=400 -------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharey="row")
for j, (axis, grid, xkey, xlab, fixed) in enumerate((
    ("sr", SR_GRID, "sr", r"$\sigma_{\rm res}$ [mm]",
     rf"$\sigma_{{\rm scatt}}={SS_FIX:g}$ fixed"),
    ("ss", SS_GRID, "ss", r"$\sigma_{\rm scatt}$ [rad]",
     rf"$\sigma_{{\rm res}}={SR_FIX}$ mm fixed"),
)):
    xs = grid
    for i, met in enumerate(("eff", "far")):
        ax = axes[i, j]
        for solver, base_col in (("classical", "tab:blue"),
                                 ("quantum", "tab:red")):
            x035, y035, e035, xwp, ywp, ewp = [], [], [], [], [], []
            for v in xs:
                rec = results[f"{axis}{v:g}"]
                cm = cell_means(rec, solver)
                if cm is None:
                    continue
                x035.append(v); y035.append(cm[met][0]); e035.append(cm[met][1])
                xwp.append(v); ywp.append(cm[met + "_w"][0]); ewp.append(cm[met + "_w"][1])
            if x035:
                ax.errorbar(x035, y035, yerr=e035, fmt="o", ls="-", ms=5,
                            capsize=3, color=base_col,
                            label=f"{'classical' if solver=='classical' else '1BQF'} ($\\tau$=0.35)")
                ax.errorbar(xwp, ywp, yerr=ewp, fmt="s", ls="--", ms=5,
                            capsize=3, color=base_col, mfc="none",
                            label=f"{'classical' if solver=='classical' else '1BQF'} (eff-first $\\tau$)")
        if axis == "ss":
            ax.set_xscale("log")
        ax.set_xlabel(xlab); ax.grid(alpha=0.25)
        if i == 0:
            ax.set_title(fixed, fontsize=10.5)
            ax.axhline(TARGET_EFF, color="k", ls=":", lw=1.0, alpha=0.6)
axes[0, 0].set_ylabel("segment efficiency")
axes[1, 0].set_ylabel("segment false rate")
axes[0, 0].set_ylim(0.55, 1.03)
axes[0, 0].legend(fontsize=7.5, loc="lower left", ncol=1)
axes[1, 0].legend(fontsize=7.5, loc="upper left", ncol=1)
fig.suptitle(rf"$T={T}$ working points: fixed $\tau=0.35$ (solid) vs "
             rf"efficiency-first $\tau$ (dashed, $\geq{int(TARGET_EFF*100)}\%$ "
             rf"eff target) — formula $\varepsilon$, qtrk store", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGDIR / "T400_workingpoints_grid.png", dpi=160)
plt.close(fig)
print("[B] T400_workingpoints_grid.png", flush=True)

# ---- Figure: ROC at T=400, representative cell --------------------------
roc_cell = ("sr0.01", (SR_FIX, SS_FIX))
rec = results["sr0.01"]
fig, ax = plt.subplots(figsize=(7.4, 5.6))
taus = np.linspace(0.02, 0.95, 160)
for solver, col in (("classical", "tab:blue"), ("quantum", "tab:red")):
    rows = rec[solver]
    if not rows:
        continue
    # need vectors again for a tau sweep: reuse stored, recompute curve
    # (use the first up-to-3 reps to keep it light)
    sub_keys = []
    if solver == "classical":
        csub = cl[(np.isclose(cl.sigma_res, SR_FIX)) &
                  (np.isclose(cl.sigma_scatt, SS_FIX))].head(3)
        for _, r in csub.iterrows():
            sub_keys.append((r.event_key, r.sol_key, None))
    else:
        qsub = qu[(np.isclose(qu.sigma_res, SR_FIX)) &
                  (np.isclose(qu.sigma_scatt, SS_FIX))]
        for _, r in qsub.iterrows():
            ck = cl_by_ev.get((r.event_key, r.ham_key))
            if ck:
                sub_keys.append((r.event_key, r.sol_key, ck))
    effs, fars = [], []
    for tau in taus:
        ev_, fv_ = [], []
        for ekey, skey, ckey in sub_keys:
            truth = get_truth(ekey)
            sol = qp.load_solution(skey)["sol"]
            if ckey is not None:
                sol = qp.rescale_to_signal(sol, qp.load_solution(ckey)["sol"],
                                           threshold=TAU)
            m = solver_segment_metrics(truth, sol, threshold=float(tau))
            ev_.append(m["segment_efficiency"]); fv_.append(m["segment_false_rate"])
        effs.append(np.mean(ev_)); fars.append(np.mean(fv_))
    effs, fars = np.array(effs), np.array(fars)
    lab = "classical" if solver == "classical" else "1BQF"
    ax.plot(fars, effs, "-", color=col, lw=1.8, label=lab)
    it = int(np.argmin(np.abs(taus - TAU)))
    ax.plot(fars[it], effs[it], "o", ms=9, color=col, label=f"{lab} $\\tau$=0.35")
    cm = cell_means(rec, solver)
    ax.plot(cm["far_w"][0], cm["eff_w"][0], "s", ms=9, color=col, mfc="none",
            mew=2, label=f"{lab} eff-first $\\tau$")
ax.axhline(TARGET_EFF, color="k", ls=":", lw=1.0, alpha=0.6)
ax.set_xlabel("segment false rate"); ax.set_ylabel("segment efficiency")
ax.set_title(rf"ROC at $T={T}$, $\sigma_r={SR_FIX}$, $\sigma_s={SS_FIX:g}$ "
             rf"(formula $\varepsilon$): $\tau$ swept", fontsize=10.5)
ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIGDIR / "T400_roc.png", dpi=160)
plt.close(fig)
print("[B] T400_roc.png", flush=True)

# ---- JSON ----------------------------------------------------------------
summary = {}
for key, rec in results.items():
    summary[key] = dict(sr=rec["sr"], ss=rec["ss"])
    for solver in ("classical", "quantum"):
        cm = cell_means(rec, solver)
        if cm:
            summary[key][solver] = {k: [round(cm[k][0], 4), round(cm[k][1], 4)]
                                    for k in cm}
            summary[key][solver]["n"] = len(rec[solver])
with (OUTDIR / "T400_eval.json").open("w") as f:
    json.dump(dict(T=T, tau=TAU, target_eff=TARGET_EFF,
                   nrep_A=NREP_A, quadratic_fits=fits,
                   direct={k: v for k, v in direct.items()},
                   workingpoints=summary), f, indent=2)
print("[done] T400 figures + JSON", flush=True)
