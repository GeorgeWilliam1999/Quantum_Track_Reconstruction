#!/usr/bin/env python3
"""
Effectiveness scan of the p_miss-parameterised epsilon formula.

    sigma_p^2 = sigma_scatt^2 + 6 arctan^2(sigma_res/dz)
    eps(p_miss) = sqrt( 2 ln(1/p_miss) sigma_p^2 + 2 theta_min^2 )

Three demonstrations for the Notion report's comparison section (S8):

  A1 calibration : requested p_miss vs MEASURED true-kink miss fraction
                   (the dial's central claim), T=100, 300 events/cell.
  A2 working pts : per-track survival + accepted false couplings/event as
                   a continuous function of p_miss, both grid cells.
  B  solvers     : downstream classical + 1BQF (statevector) segment
                   efficiency / false rate vs p_miss on PAIRED events
                   (same seeds across p, only eps changes), T=50.

Figures -> figures/epsilon_derivation/pmiss_*.png
Numbers -> outputs/epsilon_pmiss_scan.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

import helpers  # noqa: E402
from helpers import solve_quantum_statevector, rescale_quantum  # noqa: E402
from qtrk_pipeline.metrics import quantum_metrics_wp  # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402
from lhcb_velo_toy.analysis.segment_metrics import (  # noqa: E402
    DEFAULT_DZ, DEFAULT_THETA_MIN, segment_truth_mask, solver_segment_metrics,
)
from lhcb_velo_toy.solvers import SimpleHamiltonianFast  # noqa: E402

FIGDIR = HERE / "figures" / "epsilon_derivation"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR = HERE / "outputs"

DZ = DEFAULT_DZ
TH_MIN = DEFAULT_THETA_MIN
GEOM = helpers.make_geometry()
TAU = 0.35

CELLS = {
    "grid_mid":   dict(sr=0.02, ss=3e-4),
    "grid_worst": dict(sr=0.05, ss=5e-4),
}

# the two named options sit on this ladder exactly
P_DEFAULT = float(np.exp(-9.0))       # s=3            -> 1.234e-4
P_NO_SQRT2 = float(np.exp(-4.5))      # sqrt(2) removed -> 1.111e-2
P_GRID = [1e-5, P_DEFAULT, 1e-3, P_NO_SQRT2, 5e-2, 1.5e-1]


def sigma_p(sr: float, ss: float) -> float:
    return float(np.sqrt(ss**2 + 6.0 * np.arctan(sr / DZ) ** 2))


def eps_from_pmiss(p: float, sr: float, ss: float) -> float:
    return float(np.sqrt(2.0 * np.log(1.0 / p) * sigma_p(sr, ss) ** 2
                         + 2.0 * TH_MIN**2))


# consistency with the legacy formula at the default point (s inside arctan
# there, outside here — must agree to <0.1% in our regime)
for name, c in CELLS.items():
    legacy = compute_epsilon(c["sr"], c["ss"])
    restated = eps_from_pmiss(P_DEFAULT, c["sr"], c["ss"])
    assert abs(legacy / restated - 1) < 1e-3, (name, legacy, restated)


# ---------------------------------------------------------------------------
# kink collectors (as in gen_epsilon_derivation_figs.py)
# ---------------------------------------------------------------------------

def true_kinks(event):
    pos = {h.hit_id: np.array([h.x, h.y, h.z]) for h in event.hits}
    angs = []
    for trk in event.tracks:
        ids = trk.hit_ids
        for k in range(1, len(ids) - 1):
            v1 = pos[ids[k]] - pos[ids[k - 1]]
            v2 = pos[ids[k + 1]] - pos[ids[k]]
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angs.append(np.arccos(np.clip(c, -1.0, 1.0)))
    return np.array(angs)


def false_kink_hist(event, edges):
    mods: dict[int, list] = {}
    for h in event.hits:
        mods.setdefault(h.module_id, []).append(h)
    packed = {}
    for m, hs in mods.items():
        packed[m] = (np.array([[h.x, h.y, h.z] for h in hs]),
                     np.array([h.track_id for h in hs]))
    mids = sorted(packed.keys())
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for mi in range(1, len(mids) - 1):
        P0, t0 = packed[mids[mi - 1]]
        P1, t1 = packed[mids[mi]]
        P2, t2 = packed[mids[mi + 1]]
        for j in range(len(P1)):
            v1 = P1[j] - P0
            v2 = P2 - P1[j]
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            cos = (v1 @ v2.T) / np.outer(n1, n2)
            ang = np.arccos(np.clip(cos, -1.0, 1.0))
            is_true = np.outer(t0 == t1[j], t2 == t1[j])
            counts += np.histogram(ang[~is_true], bins=edges)[0]
    return counts


# ---------------------------------------------------------------------------
# Part A — kink-level calibration + working points (T=100)
# ---------------------------------------------------------------------------

T_A, NREP_TRUE, NREP_FALSE = 100, 300, 6
EDGES = np.linspace(0.0, 0.4, 4001)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])

partA: dict[str, dict] = {}
for name, c in CELLS.items():
    t0 = time.time()
    angs = []
    for rep in range(NREP_TRUE):
        ev = helpers.safe_generate(T_A, seed=40_000 + rep, geom=GEOM,
                                   measurement_error=c["sr"],
                                   collision_noise=c["ss"])
        angs.append(true_kinks(ev))
    a = np.concatenate(angs)

    fcounts = np.zeros(len(EDGES) - 1, dtype=np.int64)
    for rep in range(NREP_FALSE):
        ev = helpers.safe_generate(T_A, seed=41_000 + rep, geom=GEOM,
                                   measurement_error=c["sr"],
                                   collision_noise=c["ss"])
        fcounts += false_kink_hist(ev, EDGES)
    cum_false = np.cumsum(fcounts) / NREP_FALSE   # accepted false / event

    rows = []
    for p in P_GRID:
        eps = eps_from_pmiss(p, c["sr"], c["ss"])
        miss = float(np.mean(a >= eps))
        rows.append(dict(
            p_miss=p, eps=eps,
            miss_measured=miss,
            miss_err=float(np.sqrt(max(miss * (1 - miss), 1e-12) / a.size)),
            false_per_event=float(np.interp(eps, CENTERS, cum_false)),
        ))
    partA[name] = dict(n_true_kinks=int(a.size), rows=rows,
                       kinks=a, cum_false=cum_false)
    print(f"[A] {name}: {a.size} true kinks, "
          f"{int(fcounts.sum())} false hist entries, "
          f"{time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Part B — downstream solvers on paired events (T=50)
# ---------------------------------------------------------------------------

T_B, NREP_B = 30, 3   # T=50 quantum transpile >3.5 min/solve; T=30 keeps the
                      # paired sweep tractable (~36 statevector solves)
partB: dict[str, list] = {}
for name, c in CELLS.items():
    rows = []
    for rep in range(NREP_B):
        ev = helpers.safe_generate(T_B, seed=50_000 + rep, geom=GEOM,
                                   measurement_error=c["sr"],
                                   collision_noise=c["ss"])
        for p in P_GRID:
            eps = eps_from_pmiss(p, c["sr"], c["ss"])
            t0 = time.time()
            ham = SimpleHamiltonianFast(epsilon=eps, gamma=helpers.GAMMA,
                                        delta=helpers.DELTA, theta_d=1e-4)
            ham.construct_segments(ev)
            ham.construct_hamiltonian(ev, convolution=False)
            sol_C = np.asarray(ham.solve_classicaly()).ravel()
            truth = segment_truth_mask(ham)
            mC = solver_segment_metrics(truth, sol_C, threshold=TAU)
            t_c = time.time() - t0

            t0 = time.time()
            sol_Q_raw, p_anc, n_sys = solve_quantum_statevector(
                ham.A, ham.b, device='CPU')
            sol_Q = rescale_quantum(sol_Q_raw, sol_C)
            mQ = solver_segment_metrics(truth, sol_Q, threshold=TAU)
            # 1BQF HEADLINE = wp99 (efficiency-first working point; the fixed-tau
            # eff_Q/far_Q stay as the labelled cut-artefact record)
            mQw = quantum_metrics_wp(sol_Q_raw, sol_C, truth, signal_threshold=TAU)
            t_q = time.time() - t0

            rows.append(dict(
                rep=rep, p_miss=p, eps=eps,
                A_nnz=int(ham.A.nnz), n_sys=int(n_sys),
                eff_C=mC["segment_efficiency"], far_C=mC["segment_false_rate"],
                eff_Q=mQ["segment_efficiency"], far_Q=mQ["segment_false_rate"],
                eff_Qw=mQw["segment_efficiency"], far_Qw=mQw["segment_false_rate"],
                tau_Q_wp=mQw["tau_wp"],
                n_true_all=mC["n_true_all"],
                t_classical=t_c, t_quantum=t_q,
            ))
            print(f"[B] {name} rep{rep} p={p:.3e} eps={eps:.3e} "
                  f"effC={mC['segment_efficiency']:.3f} "
                  f"farC={mC['segment_false_rate']:.3f} "
                  f"effQ={mQ['segment_efficiency']:.3f} "
                  f"farQ={mQ['segment_false_rate']:.3f} "
                  f"nnz={ham.A.nnz} tq={t_q:.0f}s", flush=True)
    partB[name] = rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

CELL_COL = {"grid_mid": "tab:blue", "grid_worst": "tab:red"}
CELL_LAB = {
    "grid_mid":   r"$\sigma_r=0.02$ mm, $\sigma_s=3\times10^{-4}$",
    "grid_worst": r"$\sigma_r=0.05$ mm, $\sigma_s=5\times10^{-4}$",
}


def mark_options(ax):
    ax.axvline(P_DEFAULT, color="tab:green", lw=2)
    ax.axvline(P_NO_SQRT2, color="tab:orange", ls="--", lw=2)
    ax.text(P_DEFAULT, 1.01, "default $s{=}3$", color="tab:green",
            ha="center", va="bottom", fontsize=8,
            transform=ax.get_xaxis_transform())
    ax.text(P_NO_SQRT2, 1.01, "$\\sqrt{2}$ removed", color="tab:orange",
            ha="center", va="bottom", fontsize=8,
            transform=ax.get_xaxis_transform())


# Fig 7 — calibration
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
for name in CELLS:
    r = partA[name]["rows"]
    pm = np.array([x["p_miss"] for x in r])
    mm = np.array([x["miss_measured"] for x in r])
    me = np.array([x["miss_err"] for x in r])
    n = partA[name]["n_true_kinks"]
    det = mm > 0
    ax.errorbar(pm[det], mm[det], yerr=me[det], fmt="o",
                color=CELL_COL[name], label=CELL_LAB[name], capsize=3)
    if (~det).any():   # 90% CL upper limits where zero kinks were rejected
        ax.plot(pm[~det], np.full((~det).sum(), 2.3 / n), "v",
                color=CELL_COL[name], mfc="none")
x = np.logspace(-5.2, -0.6, 50)
ax.plot(x, x, "k--", lw=1, label="requested = measured")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"requested $p_{\rm miss}$")
ax.set_ylabel("measured true-kink miss fraction")
ax.set_title("the dial is calibrated (open triangles = 90% CL upper limits)",
             fontsize=10)
ax.legend(fontsize=8.5, loc="upper left")

ax = axes[1]
pcont = np.logspace(-5.2, -0.6, 200)
ax.plot(pcont, 100 * (1 - pcont) ** 3, "k-", lw=2,
        label=r"analytic $(1-p_{\rm miss})^3$")
for name in CELLS:
    a = partA[name]["kinks"]
    surv = []
    for p in pcont[::10]:
        eps = eps_from_pmiss(p, CELLS[name]["sr"], CELLS[name]["ss"])
        m = float(np.mean(a >= eps))
        surv.append(100 * (1 - m) ** 3)
    ax.plot(pcont[::10], surv, "o", ms=4, color=CELL_COL[name],
            label=f"MC, {CELL_LAB[name]}")
mark_options(ax)
ax.set_xscale("log")
ax.set_xlabel(r"$p_{\rm miss}$")
ax.set_ylabel("tracks keeping all 3 couplings [%]")
ax.set_title("what a miss budget costs at track level", fontsize=10, pad=22)
ax.legend(fontsize=8.5, loc="lower left")
fig.tight_layout()
fig.savefig(FIGDIR / "pmiss_calibration.png", dpi=160)
plt.close(fig)

# Fig 8 — working points: track survival vs false load, continuous in p_miss
fig, ax = plt.subplots(figsize=(8.5, 5))
ax2 = ax.twinx()
ax.plot(pcont, 100 * (1 - pcont) ** 3, "k-", lw=2,
        label=r"track survival $(1-p_{\rm miss})^3$")
for name in CELLS:
    cum_false = partA[name]["cum_false"]
    fl = [np.interp(eps_from_pmiss(p, CELLS[name]["sr"], CELLS[name]["ss"]),
                    CENTERS, cum_false) for p in pcont]
    ax2.plot(pcont, fl, lw=2, color=CELL_COL[name],
             label=f"false couplings/event, {CELL_LAB[name]}")
mark_options(ax)
ax.set_xscale("log")
ax.set_xlabel(r"$p_{\rm miss}$")
ax.set_ylabel("tracks keeping all 3 couplings [%]", color="k")
ax2.set_ylabel("accepted false couplings / event (T=100)")
ax.set_ylim(55, 101)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="lower left")
ax.set_title(r"the whole working-point line in the $p_{\rm miss}$ language", pad=22)
fig.tight_layout()
fig.savefig(FIGDIR / "pmiss_working_points.png", dpi=160)
plt.close(fig)

# Fig 9 — downstream solver effectiveness
# convention (wp99 log 2026-06-14): classical at the fixed tau=0.35; the 1BQF
# HEADLINE at its efficiency-first wp99 working point (the fixed-tau 1BQF curve
# was the artefactual ~75% plateau)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
for ax, key, ttl in zip(axes, ["eff", "far"],
                        ["segment efficiency (classical $\\tau$=0.35 · 1BQF wp99)",
                         "segment false rate (classical $\\tau$=0.35 · 1BQF wp99)"]):
    for col, ls, mk in (("C", "-", "o"), ("Qw", "--", "s")):
        for name in CELLS:
            rows = partB[name]
            pm = sorted(set(r["p_miss"] for r in rows))
            ys = [np.mean([r[f"{key}_{col}"] for r in rows
                           if r["p_miss"] == p]) for p in pm]
            es = [np.std([r[f"{key}_{col}"] for r in rows
                          if r["p_miss"] == p]) / np.sqrt(NREP_B) for p in pm]
            ax.errorbar(pm, ys, yerr=es, fmt=mk, ls=ls, ms=5, capsize=3,
                        color=CELL_COL[name],
                        label=f"{'classical' if col=='C' else '1BQF (wp99)'}, "
                              f"{CELL_LAB[name]}")
    mark_options(ax)
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_{\rm miss}$")
    ax.set_title(ttl, fontsize=10.5, pad=22)
axes[0].set_ylabel("value")
axes[0].legend(fontsize=7.5, loc="lower left")
fig.suptitle(f"Downstream solver response to the miss budget "
             f"(paired events, T={T_B}, {NREP_B} reps)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIGDIR / "pmiss_solver_effectiveness.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

out = dict(
    p_grid=P_GRID, p_default=P_DEFAULT, p_no_sqrt2=P_NO_SQRT2,
    T_kinks=T_A, nrep_true=NREP_TRUE, nrep_false=NREP_FALSE,
    T_solver=T_B, nrep_solver=NREP_B, tau=TAU,
    partA={k: dict(n_true_kinks=v["n_true_kinks"], rows=v["rows"])
           for k, v in partA.items()},
    partB=partB,
)
with (OUTDIR / "epsilon_pmiss_scan.json").open("w") as f:
    json.dump(out, f, indent=2)
print("[done] figures + JSON written", flush=True)
