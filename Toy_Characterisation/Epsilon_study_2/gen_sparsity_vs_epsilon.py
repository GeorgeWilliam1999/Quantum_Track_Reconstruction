#!/usr/bin/env python3
"""
Sparsity of the segment Hamiltonian A against the acceptance epsilon —
empirically (build A, count nonzeros) and analytically, contrasted.

The matrix only needs to be BUILT, not solved, so this is cheap and the
sweep reaches the dense regime.  A is symmetric with a full diagonal
(A_ii = gamma+delta != 0), so

    A_nnz(eps) = n_seg + 2 * n_coupling(eps),
    n_coupling = n_true_coupling(eps) + n_false_coupling(eps).

ANALYTIC model (both terms are phase-space, zero solver involved):
  * true couplings: each 5-plane track has exactly 3 interior shared-hit
    kinks, so N_true = 3T, and a true coupling survives iff its Rayleigh
    kink is below eps:
        n_true(eps) = 3T * (1 - exp(-(eps^2 - 2 theta_min^2) / (2 sigma_p^2)))
    -> an S-curve saturating at 3T by ~the formula eps.
  * false couplings: a combinatorial segment pair sharing a hit has a kink
    drawn from the near-uniform background; near zero its density is linear
    in theta (2D measure), so the accepted fraction is eps^2/(2 theta0^2).
    The candidate pool is ~3 T^3 (3 interior planes x T middle hits x T^2
    in/out segment pairs per hit), hence
        n_false(eps) = kappa * eps^2,   kappa = 3 T^3 / (2 theta0^2)
    -> grows without saturation, prefactor scales as T^3, theta0 a fixed
    geometric angular spread (T-independent).

Sparse while n_false < n_seg, i.e. eps < eps_dense ~ theta0 * sqrt(8 / (3T));
the dense wall moves to smaller eps as T grows (the T=400 blow-up).

A_nnz is also the 1BQF QRAM gate count (O(A_nnz)), so this is the circuit
cost vs eps as much as a sparsity plot.

Figures -> figures/epsilon_sensitivity/sparsity_*.png
Numbers -> outputs/sparsity_vs_epsilon.json
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
from lhcb_velo_toy.analysis import compute_epsilon          # noqa: E402
from lhcb_velo_toy.analysis.segment_metrics import (        # noqa: E402
    DEFAULT_DZ, DEFAULT_THETA_MIN, segment_truth_mask)
from lhcb_velo_toy.solvers import SimpleHamiltonianFast     # noqa: E402

FIGDIR = HERE / "figures" / "epsilon_sensitivity"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR = HERE / "outputs"
DZ, TH_MIN = DEFAULT_DZ, DEFAULT_THETA_MIN
GEOM = helpers.make_geometry()

T_LIST = [30, 60, 120]
EPS_GRID = np.logspace(np.log10(2e-4), np.log10(5e-2), 24)
CELL = (0.01, 1e-4)                       # primary cell (sigma_p set here)
CELL2 = (0.05, 1e-4)                      # higher-sigma_p cell for components
A_NNZ_GUARD = 6_000_000
BUILD_TIME_GUARD = 30.0                   # s; stop raising eps for a T beyond this
CMAP = plt.get_cmap("viridis")


def sigma_p(sr, ss):
    return float(np.sqrt(ss**2 + 6.0 * np.arctan(sr / DZ) ** 2))


def p_surv(eps, sp):
    x = eps**2 - 2.0 * TH_MIN**2
    return 0.0 if x <= 0 else float(1.0 - np.exp(-x / (2.0 * sp**2)))


def measure(ev, eps):
    ham = SimpleHamiltonianFast(epsilon=float(eps), gamma=3, delta=1,
                                theta_d=1e-4)
    ham.construct_segments(ev)
    ham.construct_hamiltonian(ev, convolution=False)
    truth = segment_truth_mask(ham)
    coo = ham.A.tocoo()
    up = coo.row < coo.col
    r, c = coo.row[up], coo.col[up]
    n_coup = int(len(r))
    n_true = int(np.sum(truth[r] & truth[c]))
    return dict(n_seg=int(ham.A.shape[0]), A_nnz=int(ham.A.nnz),
                n_coup=n_coup, n_true=n_true, n_false=n_coup - n_true)


def sweep(T, sr, ss):
    ev = helpers.safe_generate(T, seed=4242, geom=GEOM,
                               measurement_error=sr, collision_noise=ss)
    rows = []
    for eps in EPS_GRID:
        t0 = time.time()
        m = measure(ev, eps)
        dt = time.time() - t0
        m["eps"] = float(eps)
        rows.append(m)
        if m["A_nnz"] > A_NNZ_GUARD or dt > BUILD_TIME_GUARD:
            print(f"  T={T} eps={eps:.2e} A_nnz={m['A_nnz']} dt={dt:.0f}s "
                  f"-> stop sweep", flush=True)
            break
    print(f"  T={T} sr{sr:g} ss{ss:g}: {len(rows)} eps points", flush=True)
    return rows


def fit_kappa(rows, n_seg):
    eps = np.array([r["eps"] for r in rows])
    nf = np.array([r["n_false"] for r in rows], float)
    keep = (nf >= 3) & (nf < n_seg / 4)        # dilute regime only
    if keep.sum() < 2:
        return float("nan")
    return float(np.sum(nf[keep] * eps[keep] ** 2) / np.sum(eps[keep] ** 4))


# ---------------------------------------------------------------------------
print("=== sparsity vs epsilon ===", flush=True)
data = {}
for T in T_LIST:
    data[T] = dict(primary=sweep(T, *CELL))
data_cell2 = sweep(100, *CELL2)          # one extra cell for the components fig
T_COMP = 120                              # components shown at this T (primary)

sp1 = sigma_p(*CELL)
sp2 = sigma_p(*CELL2)
kappa = {T: fit_kappa(data[T]["primary"],
                      data[T]["primary"][0]["n_seg"]) for T in T_LIST}
theta0 = {T: float(np.sqrt(3.0 * T**3 / (2.0 * kappa[T]))) for T in T_LIST
          if kappa[T] == kappa[T]}
print("kappa(T):", {k: round(v, 1) for k, v in kappa.items()})
print("theta0(T) [rad]:", {k: round(v, 4) for k, v in theta0.items()})


# ===========================================================================
# Fig 1 — components: true (saturating) vs false (eps^2) vs A_nnz, one T
# ===========================================================================
rows = data[T_COMP]["primary"]
ns = rows[0]["n_seg"]
eps = np.array([r["eps"] for r in rows])
A_nnz = np.array([r["A_nnz"] for r in rows])
n_true = np.array([r["n_true"] for r in rows])
n_false = np.array([r["n_false"] for r in rows])
xx = np.logspace(np.log10(eps.min()), np.log10(eps.max()), 400)
true_an = 3 * T_COMP * np.array([p_surv(x, sp1) for x in xx])
kap = kappa[T_COMP]
false_an = kap * xx**2
tot_an = ns + 2 * (true_an + false_an)
eps_f = float(compute_epsilon(*CELL))
eps_dense = float(theta0[T_COMP] * np.sqrt(8.0 / (3.0 * T_COMP)))

fig, ax = plt.subplots(figsize=(8.4, 6.0))
ax.loglog(eps, A_nnz, "ko", ms=5, label=r"$A_{\rm nnz}$ (measured)")
ax.loglog(xx, tot_an, "k-", lw=1.4,
          label=r"$A_{\rm nnz}=n_{\rm seg}+2(n_{\rm true}+n_{\rm false})$ (analytic)")
ax.loglog(eps, np.clip(2 * n_true, 1, None), "s", color="tab:green", ms=5,
          label=r"$2\,n_{\rm true}$ (measured)")
ax.loglog(xx, np.clip(2 * true_an, 1, None), "--", color="tab:green", lw=1.3,
          label=r"$2\cdot 3T\,(1-e^{-\varepsilon^2/2\sigma_p^2})$")
ax.loglog(eps, np.clip(2 * n_false, 1, None), "^", color="tab:red", ms=5,
          label=r"$2\,n_{\rm false}$ (measured)")
ax.loglog(xx, np.clip(2 * false_an, 1, None), ":", color="tab:red", lw=1.6,
          label=r"$2\,\kappa\,\varepsilon^2$  ($\kappa=3T^3/2\theta_0^2$)")
ax.axhline(ns, color="gray", lw=1.0, ls="-", alpha=0.7)
ax.text(eps.min(), ns * 1.08, r"$n_{\rm seg}=4T^2$ (diagonal floor)",
        fontsize=8, color="gray")
ax.axvline(eps_f, color="tab:blue", lw=1.2, alpha=0.7)
ax.text(eps_f * 1.05, ns * 3, "formula $\\varepsilon$", rotation=90,
        fontsize=8, color="tab:blue")
ax.axvline(eps_dense, color="tab:purple", lw=1.2, ls="--", alpha=0.7)
ax.text(eps_dense * 1.05, ns * 3,
        r"$\varepsilon_{\rm dense}=\theta_0\sqrt{8/3T}$", rotation=90,
        fontsize=8, color="tab:purple")
ax.set_xlabel(r"$\varepsilon$ [rad]")
ax.set_ylabel(r"nonzeros / couplings ($\times 2$)")
ax.set_title(rf"Sparsity decomposition vs $\varepsilon$ — $T={T_COMP}$, "
             rf"$\sigma_r={CELL[0]}$, $\sigma_s={CELL[1]:g}$ "
             rf"($\sigma_p={sp1:.2e}$)", fontsize=11)
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIGDIR / "sparsity_components_vs_epsilon.png", dpi=160)
plt.close(fig)
print("[fig] sparsity_components_vs_epsilon.png", flush=True)


# ===========================================================================
# Fig 2 — A_nnz/n_seg vs eps for several T: the dense wall moves left
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
cols = [CMAP(x) for x in np.linspace(0.0, 0.8, len(T_LIST))]
for T, col in zip(T_LIST, cols):
    rows = data[T]["primary"]
    ns = rows[0]["n_seg"]
    eps = np.array([r["eps"] for r in rows])
    A_nnz = np.array([r["A_nnz"] for r in rows])
    fillmult = A_nnz / ns
    xx = np.logspace(np.log10(eps.min()), np.log10(eps.max()), 400)
    true_an = 3 * T * np.array([p_surv(x, sp1) for x in xx])
    false_an = kappa[T] * xx**2
    tot_mult = (ns + 2 * (true_an + false_an)) / ns
    axes[0].loglog(eps, A_nnz, "o", ms=4.5, color=col, label=f"T={T} meas")
    axes[0].loglog(xx, ns + 2 * (true_an + false_an), "-", color=col, lw=1.1)
    axes[1].loglog(eps, fillmult, "o", ms=4.5, color=col, label=f"T={T}")
    axes[1].loglog(xx, tot_mult, "-", color=col, lw=1.1)
    ed = float(theta0[T] * np.sqrt(8.0 / (3.0 * T)))
    axes[1].axvline(ed, color=col, ls="--", lw=1.0, alpha=0.6)
axes[1].axhline(2.0, color="gray", lw=1.0, ls=":", alpha=0.8)
axes[1].text(EPS_GRID[0], 2.05, r"$A_{\rm nnz}=2\,n_{\rm seg}$ (dense onset)",
             fontsize=8, color="gray")
axes[0].set_ylabel(r"$A_{\rm nnz}$ (= 1BQF QRAM gate count)")
axes[1].set_ylabel(r"$A_{\rm nnz}/n_{\rm seg}$ (fill multiplier)")
for ax in axes:
    ax.set_xlabel(r"$\varepsilon$ [rad]")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_title("absolute nonzeros (lines: analytic total)", fontsize=10)
axes[1].set_title(r"fill multiplier; dashed = $\varepsilon_{\rm dense}(T)$ "
                  r"moves left as $T\uparrow$", fontsize=10)
fig.suptitle(rf"Sparsity vs $\varepsilon$ across $T$ — $\sigma_r={CELL[0]}$, "
             rf"$\sigma_s={CELL[1]:g}$", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIGDIR / "sparsity_vs_epsilon_Tscan.png", dpi=160)
plt.close(fig)
print("[fig] sparsity_vs_epsilon_Tscan.png", flush=True)


# ===========================================================================
# Fig 3 — kappa vs T (empirical) against the analytic 3T^3/2theta0^2 scaling
# ===========================================================================
Ts = np.array([T for T in T_LIST if kappa[T] == kappa[T]], float)
ks = np.array([kappa[T] for T in T_LIST if kappa[T] == kappa[T]])
th0_mean = float(np.mean([theta0[T] for T in T_LIST if kappa[T] == kappa[T]]))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
axes[0].loglog(Ts, ks, "ko", ms=8, label=r"fitted $\kappa$ (empirical)")
tt = np.logspace(np.log10(Ts.min() * 0.8), np.log10(Ts.max() * 1.2), 100)
axes[0].loglog(tt, 3.0 * tt**3 / (2.0 * th0_mean**2), "r-", lw=1.5,
               label=rf"$3T^3/2\theta_0^2$, $\theta_0={th0_mean:.3f}$ rad")
# pure T^3 guide through the first point
axes[0].loglog(tt, ks[0] * (tt / Ts[0]) ** 3, "b:", lw=1.2,
               label=r"$\propto T^3$ guide")
axes[0].set_xlabel("T"); axes[0].set_ylabel(r"$\kappa$ [rad$^{-2}$]")
axes[0].set_title(r"false-coupling prefactor $\kappa$ vs $T$", fontsize=10.5)
axes[0].grid(alpha=0.25, which="both"); axes[0].legend(fontsize=8.5)
# theta0(T): should be ~constant (geometry, T-independent)
th = np.array([theta0[T] for T in T_LIST if kappa[T] == kappa[T]])
axes[1].semilogx(Ts, th, "ko-", ms=7)
axes[1].axhline(th0_mean, color="r", ls="--", lw=1.2,
                label=rf"mean $\theta_0={th0_mean:.3f}$ rad")
axes[1].set_xlabel("T")
axes[1].set_ylabel(r"implied $\theta_0=\sqrt{3T^3/2\kappa}$ [rad]")
axes[1].set_ylim(0, max(0.8, 1.3 * th.max()))
axes[1].set_title(r"implied $\theta_0$ is $T$-independent (geometry)",
                  fontsize=10.5)
axes[1].grid(alpha=0.25); axes[1].legend(fontsize=8.5)
fig.suptitle(r"Analytic vs empirical: false couplings $=\kappa\varepsilon^2$ "
             r"with $\kappa\propto T^3$", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGDIR / "sparsity_kappa_vs_T.png", dpi=160)
plt.close(fig)
print("[fig] sparsity_kappa_vs_T.png", flush=True)

# ---- JSON ----------------------------------------------------------------
with (OUTDIR / "sparsity_vs_epsilon.json").open("w") as f:
    json.dump(dict(T_list=T_LIST, eps_grid=EPS_GRID.tolist(),
                   cell=CELL, cell2=CELL2, sigma_p_cell=sp1,
                   N_true_per_T={T: 3 * T for T in T_LIST},
                   kappa=kappa, theta0=theta0, theta0_mean=th0_mean,
                   data={str(T): data[T]["primary"] for T in T_LIST},
                   data_cell2=data_cell2), f, indent=2)
print("[done] sparsity figures + JSON", flush=True)
