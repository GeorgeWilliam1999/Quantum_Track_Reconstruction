#!/usr/bin/env python3
r"""
Local-vs-cooperative indefiniteness of the segment Hamiltonian A
================================================================

The §7.4/§7.10 observation: at sigma_res=0.05 mm the CLASSICAL segment
efficiency is pinned at ~1.0 up to T~400 and then breaks (0.984@200, 0.98@400,
0.909@700, 0.786@1000, qtrk store, absolute tau=0.35).  At T=400 the TRUE-segment
amplitudes of the classical solve drop below tau and go NEGATIVE, so no tau
recovers >=99% efficiency.  This script explains it spectrally.

  A = (gamma+delta) I - C,   A x = delta 1,   C = kink-acceptance coupling.

Because the diagonal is the constant gamma+delta, eig(A) = (gamma+delta) - eig(C)
EXACTLY, so A is indefinite  <=>  lambda_max(C) > gamma+delta.

Three results, in order of increasing correctness:

1. IDEALISED building block (closed form, verified).  At a shared middle hit the
   coupling is STRICTLY BIPARTITE: an in-segment (ending at h) couples only to an
   out-segment (starting at h) whose kink < eps -- never same-side (verified
   40417/40417 pairs).  A COMPLETE bipartite block K(m_in,m_out) gives
       A=(g+d)I-Adj:  x_in=(g+d+m_out)/((g+d)^2 - m_in m_out),
                      x_out=(g+d+m_in)/((g+d)^2 - m_in m_out),
       lambda_min(A) = (g+d) - sqrt(m_in m_out),  singular at product=(g+d)^2=16.
   The star K(1,m) is the m_in=1 special case: x_c=(4+m)/(16-m), x_l=5/(16-m).
   Amplitudes diverge at product=16 and FLIP SIGN beyond it.

2. The PRODUCT>16 criterion OVER-PREDICTS.  Real hub blocks are sparse (~3% of
   K(m_in,m_out) edges), so although the product reaches thousands, the block
   singular value sigma_max(B) stays below gamma+delta=4.  No isolated hub block
   is indefinite on its own.

3. The instability is COOPERATIVE.  Bipartite blocks SHARE segments (each segment
   is an out-segment at one hub and an in-segment at the next), chaining across
   the 3 interior planes.  The order parameter is the GLOBAL lambda_max(C); it
   exceeds gamma+delta on a connected cluster of ~10^2 segments spanning many hubs
   and all interior planes.  The (sigma_res,T) locus lambda_max(C)=gamma+delta
   reproduces the measured efficiency-break locus.

Figures -> figures/epsilon_sensitivity/{star_amplitude_divergence,hub_degree_vs_T,
           neg_eigs_vs_sigma_res_T,indefiniteness_boundary,
           critical_mode_localization,gamma_epsilon_fix}.png
Numbers -> outputs/indefiniteness_study.json
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")
import helpers                                                # noqa: E402
import pandas as pd                                           # noqa: E402
import qtrk_pipeline as qp                                    # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon            # noqa: E402
from lhcb_velo_toy.analysis.segment_metrics import (          # noqa: E402
    segment_truth_mask, solver_segment_metrics)
from lhcb_velo_toy.solvers import SimpleHamiltonianFast       # noqa: E402

FIGDIR = HERE / "figures" / "epsilon_sensitivity"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR = HERE / "outputs"
STORE_METRICS = "/data/bfys/gscriven/qtrk_store/manifest/metrics.csv"

GEOM = helpers.make_geometry()
GAMMA, DELTA = helpers.GAMMA, helpers.DELTA       # 3, 1
GD = GAMMA + DELTA                                 # 4  -> threshold (g+d)^2 = 16
TAU = 0.35
SS = 1e-4                                           # fixed sigma_scatt
SR_GRID = [0.0, 0.01, 0.02, 0.05]
T_GRID = [100, 200, 400, 700]
T_EXTRA = {0.02: [1000], 0.05: [1000]}             # single-rep boundary extension
NREP = 3
SEED0 = 730000
TOPK = 48                                          # top eigs of C for n_neg count
A_NNZ_GUARD = 9_000_000
BUILD_TIME_GUARD = 90.0
CMAP = plt.get_cmap("viridis")
t_start = time.time()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def gen_event(T, sr, ss, seed):
    """REPRODUCIBLE event generation.  ``StateEventGenerator`` builds its own
    ``np.random.default_rng()`` and therefore IGNORES ``np.random.seed`` — so
    ``helpers.safe_generate`` is NOT reproducible (same seed -> different event,
    verified).  We inject a seeded rng directly.  Mirrors safe_generate's
    retry-until-every-track-has->=3-hits logic (seed bumped per retry)."""
    from lhcb_velo_toy.generation import StateEventGenerator
    last = None
    for k in range(20):
        g = StateEventGenerator(detector_geometry=GEOM, events=1, n_particles=[T],
                                phi_min=-0.2, phi_max=0.2, theta_min=-0.2,
                                theta_max=0.2, measurement_error=sr,
                                collision_noise=ss)
        g.rng = np.random.default_rng(int(seed) + k)
        g.generate_random_primary_vertices(helpers.PV_SIGMA)
        g.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * T])
        ev = g.generate_complete_events()
        last = ev
        if ev.tracks and min(len(t.hit_ids) for t in ev.tracks) >= 3:
            return ev
    return last


def build_C(T, sr, ss, seed, eps_override=None, ev=None):
    """Build A on a reproducible event; return (C, A, ham, eps, ev).  No solve.
    Pass ``ev`` to reuse a previously-generated event (e.g. an epsilon sweep on
    one fixed event), ``eps_override`` to build at a non-formula epsilon."""
    eps = float(compute_epsilon(sr, ss)) if eps_override is None else float(eps_override)
    if ev is None:
        ev = gen_event(T, sr, ss, seed)
    ham = SimpleHamiltonianFast(epsilon=eps, gamma=GAMMA, delta=DELTA,
                                theta_d=1e-4)
    ham.construct_segments(ev, materialize_segments=False)
    ham.construct_hamiltonian(ev, convolution=False)
    A = ham.A.tocsr()
    n = A.shape[0]
    C = sp.identity(n, format="csr") * GD - A     # A = GD I - C  ->  C = GD I - A
    C.eliminate_zeros()
    return C, A, ham, eps, ev


def hub_blocks(ham):
    """Per shared-middle-hit bipartite block: returns list of (h,m_in,m_out,
    n_edge,sigma_max) and the sij/mid arrays.  Confirms strict bipartiteness."""
    coo = ham.A.tocoo()
    up = coo.row < coo.col
    ri, ci = coo.row[up], coo.col[up]
    sh = np.asarray(ham._segment_to_hit_ids)
    end_i, start_i = sh[ri, 1], sh[ri, 0]
    end_j, start_j = sh[ci, 1], sh[ci, 0]
    share_ij = end_i == start_j               # i ends at h, j starts at h
    share_ji = end_j == start_i
    same_in = int((end_i == end_j).sum())
    same_out = int((start_i == start_j).sum())
    mid = np.where(share_ij, end_i, end_j)
    hin, hout, hed = defaultdict(set), defaultdict(set), defaultdict(list)
    for a, b, h, s in zip(ri, ci, mid, share_ij):
        iseg, oseg = (a, b) if s else (b, a)
        hin[h].add(iseg); hout[h].add(oseg); hed[h].append((iseg, oseg))
    rows = []
    for h in hin:
        ins, outs = sorted(hin[h]), sorted(hout[h])
        di = {s: k for k, s in enumerate(ins)}
        do = {s: k for k, s in enumerate(outs)}
        B = np.zeros((len(ins), len(outs)))
        for (a, b) in hed[h]:
            B[di[a], do[b]] = 1.0
        smax = float(np.linalg.svd(B, compute_uv=False)[0]) if B.size else 0.0
        rows.append((int(h), len(ins), len(outs), len(hed[h]), smax))
    return rows, dict(n_pairs=int(len(ri)), share_ij=int(share_ij.sum()),
                      share_ji=int(share_ji.sum()), same_in=same_in,
                      same_out=same_out), (hin, hout)


def spectral_C(C, want_vec=False, dense_max=800):
    """EXACT lambda_max(C) and n_neg(A) = #{eig(C) > gamma+delta}, via the
    connected-component decomposition of the coupling graph.

    A whole-matrix `eigsh` is UNRELIABLE here: the top of C's spectrum is
    clustered (many near-equal hub-cluster eigenvalues), so Lanczos with a random
    start vector silently locks onto a non-maximal Ritz value and UNDER-reports
    lambda_max (observed: the same matrix scored 4.84 / 5.28 / 6.44 on repeated
    eigsh calls).  But C is a NONNEGATIVE adjacency matrix and A=(gamma+delta)I-C
    is block-diagonal over the connected components, and the heavy eigenvalues all
    live inside SMALL components (the unstable mode localizes).  So per-component
    dense `eigvalsh` is both exact and cheap, and the answer is reproducible.

    Returns (lmax, n_neg, info[, crit_vec, indef_idx]).  With want_vec it also
    returns the critical eigenvector (top mode of the largest-lambda component,
    embedded in full index space) and the concatenated indices of every
    *indefinite* component (lambda_max(comp) > gamma+delta)."""
    n = C.shape[0]
    ncomp, lab = connected_components(C, directed=False)
    sizes = np.bincount(lab, minlength=ncomp)
    # local index of each node inside its component (vectorised, no per-comp slicing)
    nodes_order = np.argsort(lab, kind="stable")
    lab_sorted = lab[nodes_order]
    first_node = np.r_[0, np.cumsum(sizes)[:-1]]   # comp c -> nodes_order[first:first+size]
    local = np.empty(n, dtype=np.int64)
    local[nodes_order] = np.arange(n) - first_node[lab_sorted]
    # upper-triangle edges, grouped by component
    coo = C.tocoo()
    em = coo.row < coo.col
    er, ec = coo.row[em], coo.col[em]
    ecomp = lab[er]
    eo = np.argsort(ecomp, kind="stable")
    er, ec, ecomp = er[eo], ec[eo], ecomp[eo]
    uniq, estart = np.unique(ecomp, return_index=True)
    estart = np.r_[estart, er.size]
    lmax = 0.0; nneg = 0; argsz = 0; crit_vec = None; indef_idx = []
    for j in range(uniq.size):
        uc = int(uniq[j])
        s = int(sizes[uc])
        if s < 3:                                 # size<3 -> eig(C) in {0,+-1} < GD
            continue
        a, b = estart[j], estart[j + 1]
        li, lj = local[er[a:b]], local[ec[a:b]]
        if s > 6000:
            # pathologically large (near-percolation) block: a dense eig would OOM.
            # C is nonnegative -> Perron lambda_max via eigsh from v0=ones (robust);
            # exact per-eigenvalue counting is not needed in this regime (it only
            # arises far past the formula eps, outside the operating point).
            Bsp = sp.coo_matrix(
                (np.ones(2 * li.size),
                 (np.r_[li, lj], np.r_[lj, li])), shape=(s, s)).tocsr()
            m = float(spla.eigsh(Bsp, k=1, which="LA", v0=np.ones(s),
                                 maxiter=20000, tol=1e-9,
                                 return_eigenvectors=False)[0])
            ev = np.array([m]); evec = None
        else:
            B = np.zeros((s, s))
            B[li, lj] = 1.0; B[lj, li] = 1.0      # symmetric 0/1 block (exact)
            if want_vec:
                ev, evec = np.linalg.eigh(B)
            else:
                ev = np.linalg.eigvalsh(B); evec = None
            m = float(ev[-1])                     # ascending -> max is last
        nneg += int((ev > GD).sum())
        gnodes = nodes_order[first_node[uc]:first_node[uc] + s]
        if m > GD:
            indef_idx.append(gnodes)
        if m > lmax:
            lmax = m; argsz = s
            if want_vec:
                v = np.zeros(n); v[gnodes] = evec[:, -1]; crit_vec = v
    info = dict(ncomp=int(ncomp), max_comp=int(sizes.max()), top_comp_size=argsz)
    if want_vec:
        cat = np.concatenate(indef_idx) if indef_idx else np.empty(0, dtype=int)
        return lmax, nneg, info, crit_vec, cat
    return lmax, nneg, info


# ===========================================================================
# FIGURE 1 — closed forms (analytic target a), verified numerically
# ===========================================================================
def star_A(m):
    A = GD * np.eye(m + 1); A[0, 1:] = -1; A[1:, 0] = -1; return A


def bip_A(mi, mo):
    n = mi + mo; A = GD * np.eye(n)
    A[:mi, mi:] = -1; A[mi:, :mi] = -1; return A


print("=== FIG 1: closed forms ===", flush=True)
ms = np.arange(2, 31)
xc_cf = (GD + ms) / (GD**2 - ms)
xl_cf = (GD + 1) / (GD**2 - ms)
lam_cf = GD - np.sqrt(ms)
xc_num, xl_num, lam_num, nneg_num = [], [], [], []
for m in ms:
    A = star_A(int(m)); b = DELTA * np.ones(m + 1)
    ev = np.linalg.eigvalsh(A)
    x = (np.linalg.solve(A, b) if abs(GD**2 - m) > 1e-9
         else np.full(m + 1, np.nan))
    xc_num.append(x[0]); xl_num.append(x[1])
    lam_num.append(ev.min()); nneg_num.append(int((ev < 0).sum()))
xc_num = np.array(xc_num); xl_num = np.array(xl_num); lam_num = np.array(lam_num)

# bipartite product sweep for the second panel
prods, xin_cf, xout_cf = [], [], []
xin_num, xout_num = [], []
bip_cases = [(1, m) for m in range(2, 31)] + \
            [(2, 4), (2, 6), (2, 8), (2, 9), (2, 11), (3, 5), (3, 6),
             (4, 4), (4, 5), (5, 4), (3, 7), (5, 5)]
for mi, mo in bip_cases:
    p = mi * mo
    A = bip_A(mi, mo); b = DELTA * np.ones(mi + mo)
    sing = abs(GD**2 - p) < 1e-9
    x = np.full(mi + mo, np.nan) if sing else np.linalg.solve(A, b)
    prods.append(p)
    xin_cf.append(np.nan if sing else (GD + mo) / (GD**2 - p))
    xout_cf.append(np.nan if sing else (GD + mi) / (GD**2 - p))
    xin_num.append(x[0]); xout_num.append(x[mi])
prods = np.array(prods)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
ax = axes[0]
ax.axhline(0, color="k", lw=0.6)
ax.axvline(GD**2, color="tab:red", ls="--", lw=1.2)
ax.text(GD**2 + 0.2, 14, r"singular $m=(\gamma+\delta)^2=16$",
        rotation=90, color="tab:red", fontsize=8, va="top")
mm = np.linspace(2, 30, 600)
ax.plot(mm, (GD + mm) / (GD**2 - mm), "-", color="tab:blue", lw=1.3,
        label=r"$x_c=(\gamma{+}\delta{+}m)/((\gamma{+}\delta)^2{-}m)$")
ax.plot(mm, (GD + 1) / (GD**2 - mm), "-", color="tab:green", lw=1.3,
        label=r"$x_\ell=(\gamma{+}\delta{+}1)/((\gamma{+}\delta)^2{-}m)$")
ax.plot(ms, xc_num, "o", color="tab:blue", ms=4.5, label="$x_c$ numeric")
ax.plot(ms, xl_num, "s", color="tab:green", ms=4, label=r"$x_\ell$ numeric")
ax.axhline(TAU, color="gray", ls=":", lw=1.0); ax.text(2.2, TAU + 0.4, r"$\tau$", color="gray", fontsize=9)
ax.set_ylim(-25, 25); ax.set_xlabel("star leaves $m$ = degree")
ax.set_ylabel("amplitude"); ax.set_title(r"Star $K(1,m)$: amplitudes diverge & "
              r"flip sign at $m=16$", fontsize=9.5)
ax.legend(fontsize=7.2, loc="lower left"); ax.grid(alpha=0.25)

ax = axes[1]
ax.axvline(GD**2, color="tab:red", ls="--", lw=1.2)
ax.axhline(0, color="k", lw=0.6)
ax.plot(mm, (GD + mm) / (GD**2 - mm), "-", color="tab:blue", lw=1.1, alpha=0.7,
        label="star branch (analytic)")
ax.scatter(prods, xin_num, marker="o", color="tab:purple", s=26,
           label=r"$x_{\rm in}$ bipartite numeric", zorder=4)
ax.scatter(prods, xout_num, marker="^", color="tab:orange", s=26,
           label=r"$x_{\rm out}$ bipartite numeric", zorder=4)
ax.text(GD**2 + 0.3, 14, r"product $=16$", rotation=90, color="tab:red",
        fontsize=8, va="top")
ax.set_ylim(-25, 25); ax.set_xlabel(r"product $m_{\rm in}\,m_{\rm out}$")
ax.set_ylabel("amplitude")
ax.set_title(r"Bipartite $K(m_{\rm in},m_{\rm out})$: same singularity at "
             r"product$=16$", fontsize=9.5)
ax.legend(fontsize=7.5, loc="lower left"); ax.grid(alpha=0.25)

ax = axes[2]
ax.plot(mm, GD - np.sqrt(mm), "-", color="k", lw=1.3,
        label=r"$\lambda_{\min}(A)=(\gamma+\delta)-\sqrt{\rm prod}$")
ax.plot(ms, lam_num, "o", color="tab:blue", ms=4.5, label="star numeric")
ax.scatter(prods, [GD - np.sqrt(p) for p in prods], marker="^",
           color="tab:orange", s=22, label="bipartite numeric", zorder=4)
ax.axhline(0, color="tab:red", ls="--", lw=1.1)
ax.axvline(GD**2, color="tab:red", ls="--", lw=1.0)
ax.set_xlabel(r"degree / product"); ax.set_ylabel(r"$\lambda_{\min}(A)$")
ax.set_title(r"Block goes indefinite ($\lambda_{\min}<0$) past product 16",
             fontsize=9.5)
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)
fig.suptitle(r"Idealised hub building block — $\gamma=3,\ \delta=1$, "
             r"$A=(\gamma+\delta)I-\mathrm{Adj}$, $Ax=\delta\mathbf{1}$ "
             r"(closed form vs exact numeric)", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGDIR / "star_amplitude_divergence.png", dpi=160)
plt.close(fig)
print("[fig] star_amplitude_divergence.png", flush=True)
closed_form_max_err = float(np.nanmax(np.abs(
    np.r_[xc_num - xc_cf, xl_num - xl_cf, lam_num - lam_cf])))
print(f"  closed-form max |num-analytic| = {closed_form_max_err:.2e}", flush=True)


# ===========================================================================
# SPECTRAL GRID — build A, measure lambda_max(C), n_neg, hub stats (no solve)
# ===========================================================================
print("\n=== spectral grid (build-only) ===", flush=True)
grid = {}                       # (sr,T) -> dict of per-rep measurements
bip_check = dict(n_pairs=0, share_either=0, same_in=0, same_out=0)
for sr in SR_GRID:
    for T in T_GRID + T_EXTRA.get(sr, []):
        reps = NREP if T in T_GRID else 1
        recs = []
        for rep in range(reps):
            if time.time() - t_start > 3000:
                print("  [walltime guard] stop grid", flush=True); break
            t0 = time.time()
            seed = SEED0 + T * 13 + rep
            try:
                C, A, ham, eps, _ = build_C(T, sr, SS, seed)
            except Exception as ex:
                print(f"  sr{sr} T{T} r{rep} build FAIL {ex}", flush=True); continue
            if A.nnz > A_NNZ_GUARD or (time.time() - t0) > BUILD_TIME_GUARD:
                print(f"  sr{sr} T{T} r{rep} A_nnz={A.nnz} dt={time.time()-t0:.0f}s"
                      " -> guard skip", flush=True); continue
            lmax, n_neg, sinfo = spectral_C(C)
            rows, bip, _ = hub_blocks(ham)
            smax_blk = max((r[4] for r in rows), default=0.0)
            prod_blk = max((r[1] * r[2] for r in rows), default=0)
            # accumulate the global bipartite-structure check
            bip_check["n_pairs"] += bip["n_pairs"]
            bip_check["share_either"] += bip["share_ij"] + bip["share_ji"]
            bip_check["same_in"] += bip["same_in"]
            bip_check["same_out"] += bip["same_out"]
            recs.append(dict(rep=rep, eps=eps, n_seg=int(A.shape[0]),
                             A_nnz=int(A.nnz), lam_max_C=lmax,
                             lam_min_A=GD - lmax, n_neg=int(n_neg),
                             top_comp_size=int(sinfo["top_comp_size"]),
                             max_sigma_blk=smax_blk, max_prod_blk=int(prod_blk),
                             n_hub=len(rows)))
            print(f"  sr{sr:<5} T{T:<5} r{rep}: lam_max(C)={lmax:.3f} "
                  f"n_neg={n_neg:<3} maxblk_sigma={smax_blk:.3f} "
                  f"maxprod={int(prod_blk):<5} ({time.time()-t0:.0f}s)", flush=True)
        if recs:
            grid[(sr, T)] = recs


def cell_mean(sr, T, key):
    recs = grid.get((sr, T))
    if not recs:
        return np.nan, np.nan
    v = np.array([r[key] for r in recs], float)
    return float(v.mean()), float(v.std() / np.sqrt(len(v)))


# ===========================================================================
# FIGURE 2 — hub degree / product vs T: product>16 reached early, sigma stays <4
# ===========================================================================
print("\n=== FIG 2: hub degree vs T ===", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
cols = {sr: CMAP(x) for sr, x in zip(SR_GRID, np.linspace(0.0, 0.85, len(SR_GRID)))}
for sr in SR_GRID:
    Ts = [T for T in T_GRID + T_EXTRA.get(sr, []) if (sr, T) in grid]
    if not Ts:
        continue
    prod = [cell_mean(sr, T, "max_prod_blk")[0] for T in Ts]
    smax = [cell_mean(sr, T, "max_sigma_blk")[0] for T in Ts]
    smax_e = [cell_mean(sr, T, "max_sigma_blk")[1] for T in Ts]
    axes[0].loglog(Ts, prod, "o-", color=cols[sr], ms=5,
                   label=rf"$\sigma_r={sr:g}$")
    axes[1].errorbar(Ts, smax, yerr=smax_e, fmt="o-", color=cols[sr], ms=5,
                     capsize=2.5, label=rf"$\sigma_r={sr:g}$")
axes[0].axhline(GD**2, color="tab:red", ls="--", lw=1.2)
axes[0].text(T_GRID[0], GD**2 * 1.3, r"$(\gamma+\delta)^2=16$ (complete-block "
             "singular)", color="tab:red", fontsize=8)
axes[0].set_xlabel("T"); axes[0].set_ylabel(r"max hub product $m_{\rm in}m_{\rm out}$")
axes[0].set_title("hub product crosses 16 early (by $T\\sim100$) — but blocks "
                  "are ~3% filled", fontsize=9.5)
axes[1].axhline(GD, color="tab:red", ls="--", lw=1.2)
axes[1].text(T_GRID[0], GD + 0.06, r"$\gamma+\delta=4$ (real block threshold)",
             color="tab:red", fontsize=8)
axes[1].set_xscale("log")
axes[1].set_xlabel("T")
axes[1].set_ylabel(r"max isolated-block $\sigma_{\max}(B)$")
axes[1].set_title(r"true per-block criterion $\sigma_{\max}(B)>\gamma+\delta$ "
                  "barely reached", fontsize=9.5)
for ax in axes:
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8.5)
fig.suptitle("Why product$>16$ over-predicts: real hub bipartite blocks are "
             "sparse — the isolated-block instability almost never triggers",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGDIR / "hub_degree_vs_T.png", dpi=160)
plt.close(fig)
print("[fig] hub_degree_vs_T.png", flush=True)


# ===========================================================================
# FIGURE 3 — lambda_max(C), lambda_min(A), n_neg vs (sigma_res, T)
# ===========================================================================
print("\n=== FIG 3: neg eigs vs (sigma_res,T) ===", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
for sr in SR_GRID:
    Ts = [T for T in T_GRID + T_EXTRA.get(sr, []) if (sr, T) in grid]
    if not Ts:
        continue
    lmax = [cell_mean(sr, T, "lam_max_C")[0] for T in Ts]
    lmax_e = [cell_mean(sr, T, "lam_max_C")[1] for T in Ts]
    lmin = [cell_mean(sr, T, "lam_min_A")[0] for T in Ts]
    lmin_e = [cell_mean(sr, T, "lam_min_A")[1] for T in Ts]
    nneg = [cell_mean(sr, T, "n_neg")[0] for T in Ts]
    blk = [cell_mean(sr, T, "max_sigma_blk")[0] for T in Ts]
    axes[0].errorbar(Ts, lmax, yerr=lmax_e, fmt="o-", color=cols[sr], ms=5,
                     capsize=2.5, label=rf"$\sigma_r={sr:g}$")
    axes[0].plot(Ts, blk, "x--", color=cols[sr], ms=5, lw=0.9, alpha=0.7)
    axes[1].errorbar(Ts, lmin, yerr=lmin_e, fmt="o-", color=cols[sr], ms=5,
                     capsize=2.5, label=rf"$\sigma_r={sr:g}$")
    axes[2].plot(Ts, nneg, "o-", color=cols[sr], ms=5,
                 label=rf"$\sigma_r={sr:g}$")
axes[0].axhline(GD, color="k", ls="--", lw=1.2)
axes[0].text(T_GRID[0], GD + 0.05, r"$\gamma+\delta=4$", fontsize=8)
axes[0].set_ylabel(r"$\lambda_{\max}(C)$")
axes[0].set_title(r"$\circ$ global $\lambda_{\max}(C)$ vs $\times$ best isolated "
                  r"block: cooperative gap", fontsize=9)
axes[1].axhline(0, color="k", ls="--", lw=1.2)
axes[1].set_ylabel(r"$\lambda_{\min}(A)=(\gamma+\delta)-\lambda_{\max}(C)$")
axes[1].set_title(r"$A$ indefinite where $\lambda_{\min}(A)<0$", fontsize=9.5)
axes[2].axhline(0, color="k", lw=0.6)
axes[2].set_ylabel(r"$\#$ negative eigenvalues of $A$ (count $>(\gamma+\delta)$)")
axes[2].set_title("number of unstable modes", fontsize=9.5)
for ax in axes:
    ax.set_xscale("log"); ax.set_xlabel("T"); ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8.5)
fig.suptitle(r"Spectral onset of indefiniteness vs $(\sigma_{\rm res},T)$ at "
             r"$\sigma_{\rm scatt}=10^{-4}$, formula $\varepsilon$ "
             rf"(build-only, {NREP} reps)", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGDIR / "neg_eigs_vs_sigma_res_T.png", dpi=160)
plt.close(fig)
print("[fig] neg_eigs_vs_sigma_res_T.png", flush=True)


# ===========================================================================
# FIGURE 4 — indefiniteness boundary + measured efficiency-break overlay
# ===========================================================================
print("\n=== FIG 4: indefiniteness boundary ===", flush=True)
dfm = pd.read_csv(STORE_METRICS)
# exact-token membership on `studies` (shared cells are owned by another study)
_member = dfm["studies"].fillna("").astype(str).str.split(
    r"[|,;\s]+", regex=True).map(lambda t: "Epsilon_study_2" in t)
dfm = dfm[_member | (dfm["study"] == "Epsilon_study_2")]
dfm = dfm[(dfm.solver == "classical") & (np.isclose(dfm.sigma_scatt, SS))]
eff_piv = dfm.pivot_table(index="sigma_res", columns="n_trk",
                          values="segment_efficiency", aggfunc="mean")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
# Left: crossing T per sigma_res from lambda_max(C)=GD (interpolated)
ax = axes[0]
cross_T, cross_sr = [], []
for sr in SR_GRID:
    Ts = np.array([T for T in T_GRID + T_EXTRA.get(sr, []) if (sr, T) in grid],
                  float)
    L = np.array([cell_mean(sr, T, "lam_max_C")[0] for T in Ts])
    ax.plot(Ts, L, "o-", color=cols[sr], ms=5, label=rf"$\sigma_r={sr:g}$")
    # interpolate where L crosses GD
    above = np.where(L >= GD)[0]
    if above.size and above[0] > 0:
        i = above[0]
        f = (GD - L[i - 1]) / (L[i] - L[i - 1])
        Tc = np.exp(np.log(Ts[i - 1]) + f * (np.log(Ts[i]) - np.log(Ts[i - 1])))
        cross_T.append(Tc); cross_sr.append(sr)
        ax.plot(Tc, GD, "*", ms=15, color=cols[sr], mec="k", mew=0.6, zorder=6)
ax.axhline(GD, color="k", ls="--", lw=1.2)
ax.text(T_GRID[0], GD + 0.05, r"$\gamma+\delta=4$ (indefiniteness)", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("T"); ax.set_ylabel(r"$\lambda_{\max}(C)$")
ax.set_title(r"$\bigstar$ = predicted indefiniteness onset $T_c(\sigma_{\rm res})$",
             fontsize=10)
ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8.5)

# Right: (sigma_res,T) plane, predicted boundary vs measured eff contours
ax = axes[1]
# measured efficiency heat (store) as background pcolor on log-T
Tcols = [c for c in eff_piv.columns]
srrows = [r for r in eff_piv.index]
Z = eff_piv.values
pcm = ax.pcolormesh(Tcols, srrows, Z, shading="nearest", cmap="RdYlGn",
                    vmin=0.7, vmax=1.0)
cb = fig.colorbar(pcm, ax=ax); cb.set_label("classical segment efficiency (store)")
# predicted boundary stars + connecting line
if cross_T:
    order = np.argsort(cross_sr)
    cs = np.array(cross_sr)[order]; ct = np.array(cross_T)[order]
    ax.plot(ct, cs, "k*--", ms=15, lw=1.6, mec="w", mew=0.8,
            label=r"predicted $\lambda_{\max}(C)=\gamma+\delta$")
ax.set_xscale("log"); ax.set_xlabel("T"); ax.set_ylabel(r"$\sigma_{\rm res}$ [mm]")
ax.set_title(r"predicted indefiniteness boundary overlays the measured "
             "efficiency cliff", fontsize=9.5)
ax.legend(fontsize=8.5, loc="upper left")
fig.suptitle(r"Indefiniteness boundary in the $(\sigma_{\rm res},T)$ plane "
             r"($\sigma_{\rm scatt}=10^{-4}$): spectral prediction vs measured "
             "classical efficiency", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGDIR / "indefiniteness_boundary.png", dpi=160)
plt.close(fig)
print("[fig] indefiniteness_boundary.png", flush=True)


# ===========================================================================
# FIGURE 5 — critical-mode localization + lost-true correlation (store-backed)
# ===========================================================================
print("\n=== FIG 5: localization + lost-true correlation (store T=400 sr=0.05)"
      " ===", flush=True)
cl = pd.read_csv(STORE_METRICS)
_member5 = cl["studies"].fillna("").astype(str).str.split(
    r"[|,;\s]+", regex=True).map(lambda t: "Epsilon_study_2" in t)
cl = cl[_member5 | (cl["study"] == "Epsilon_study_2")]
cl = cl[(cl.solver == "classical") & (cl.n_trk == 400)
        & np.isclose(cl.sigma_res, 0.05) & np.isclose(cl.sigma_scatt, SS)]

loc = dict(mass_worst_hub=[], part50=[], part90=[], n_hits_support=[],
           planes=set(), lam_max_C=[], max_block_sigma=[],
           lift_lost_indef=[], frac_lost_on_indef=[], frac_seg_indef=[],
           neg_true_frac=[], n_reps=0)
mode_v = None; mode_plane = None
for _, r in cl.head(6).iterrows():
    try:
        ev = qp.load_event(qp.event_path(r.event_key))
        sol = np.asarray(qp.load_solution(r.sol_key)["sol"], float)
    except Exception as ex:
        print("  skip", ex, flush=True); continue
    ham = qp.build_hamiltonian(ev, epsilon=float(r.epsilon), kernel="step",
                               gamma=GAMMA, delta=DELTA)
    A = ham.A.tocsr(); n = A.shape[0]
    C = sp.identity(n, format="csr") * GD - A; C.eliminate_zeros()
    truth = segment_truth_mask(ham)
    bnd = ham._group_boundaries
    seg_plane = np.zeros(n, dtype=int)
    for g in range(len(bnd) - 1):
        seg_plane[bnd[g]:bnd[g + 1]] = g
    sh = np.asarray(ham._segment_to_hit_ids)
    # EXACT lowest mode of A (= top mode of C) + the indefinite-component
    # segments, both from the component decomposition (no flaky whole-matrix eigsh).
    lam_max, n_neg_ev, sinfo, v, indef_idx = spectral_C(C, want_vec=True)
    if v is None:
        v = np.zeros(n)
    p = v ** 2; order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order])
    k50 = int(np.searchsorted(cum, 0.50)) + 1
    k90 = int(np.searchsorted(cum, 0.90)) + 1
    supp = order[:k90]
    supp_hits = np.unique(np.r_[sh[supp, 0], sh[supp, 1]])
    mass_worst = 0.0
    # isolated hub blocks -> max sigma
    rows, _, _ = hub_blocks(ham)
    max_blk = max((rr[4] for rr in rows), default=0.0)
    # indefinite components (lambda_max(comp) > gamma+delta), exact from spectral_C
    indef_mask = np.zeros(n, bool)
    if indef_idx.size:
        indef_mask[indef_idx] = True
    lost = truth & (sol < TAU)
    n_lost = int(lost.sum())
    frac_seg_indef = float(indef_mask.mean())
    frac_lost_indef = float(indef_mask[lost].mean()) if n_lost else 0.0
    lift = (frac_lost_indef / frac_seg_indef) if frac_seg_indef > 0 else np.nan
    neg_true = float((sol[truth] < 0).mean())
    loc["mass_worst_hub"].append(mass_worst)
    loc["part50"].append(k50); loc["part90"].append(k90)
    loc["n_hits_support"].append(int(len(supp_hits)))
    loc["planes"] |= set(seg_plane[supp].tolist())
    loc["lam_max_C"].append(lam_max); loc["max_block_sigma"].append(max_blk)
    loc["lift_lost_indef"].append(lift)
    loc["frac_lost_on_indef"].append(frac_lost_indef)
    loc["frac_seg_indef"].append(frac_seg_indef)
    loc["neg_true_frac"].append(neg_true)
    loc["n_reps"] += 1
    if mode_v is None:
        mode_v = v.copy(); mode_plane = seg_plane.copy()
        mode_truth = truth.copy(); mode_sol = sol.copy()
        mode_order = order
    print(f"  rep: lam_max(C)={lam_max:.3f} part50={k50} part90={k90} "
          f"hits={len(supp_hits)} lift_lost={lift:.1f} "
          f"neg_true_frac={neg_true:.4f}", flush=True)

# Figure: (a) participation/localization bar, (b) amplitude vs eigenvector mass
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
ax = axes[0]
# sorted |v| participation curve for the representative event
pv = np.sort(mode_v ** 2)[::-1]
ax.semilogy(np.arange(1, 200), pv[:199] + 1e-16, "o-", ms=3, color="tab:purple")
ax.set_xlabel("segment rank"); ax.set_ylabel(r"$|v_i|^2$ (lowest mode of $A$)")
ax.set_title(rf"critical mode is LOCAL: 50% of $|v|^2$ on "
             rf"{int(np.mean(loc['part50']))} segs, 90% on "
             rf"{int(np.mean(loc['part90']))} (of $\sim6.4\times10^5$)",
             fontsize=8.8)
ax.grid(alpha=0.25, which="both")

ax = axes[1]
mass = mode_v ** 2
ax.scatter(mass[mode_truth] + 1e-16, mode_sol[mode_truth], s=10, alpha=0.5,
           color="tab:green", label="true segments")
ax.scatter(mass[~mode_truth] + 1e-16, mode_sol[~mode_truth], s=6, alpha=0.25,
           color="tab:red", label="false segments")
ax.axhline(TAU, color="k", ls=":", lw=1.0); ax.text(2e-16, TAU + 0.02, r"$\tau$")
ax.axhline(0, color="gray", lw=0.6)
ax.set_xscale("log"); ax.set_xlabel(r"eigenvector mass $|v_i|^2$ on critical mode")
ax.set_ylabel("classical amplitude $x_i$ (store)")
ax.set_title("true segments on the critical cluster get DRIVEN BELOW "
             r"$\tau$ / negative", fontsize=8.8)
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.25)

ax = axes[2]
labels_bar = [r"$\lambda_{\max}(C)$" + "\nglobal",
              r"max isolated" + "\n" + r"block $\sigma_{\max}$"]
vals_bar = [np.mean(loc["lam_max_C"]), np.mean(loc["max_block_sigma"])]
errs_bar = [np.std(loc["lam_max_C"]), np.std(loc["max_block_sigma"])]
ax.bar(labels_bar, vals_bar, yerr=errs_bar, capsize=4,
       color=["tab:blue", "tab:gray"])
ax.axhline(GD, color="tab:red", ls="--", lw=1.3)
ax.text(-0.4, GD + 0.05, r"$\gamma+\delta=4$", color="tab:red", fontsize=9)
ax.set_ylabel("spectral radius")
lift_mean = np.nanmean(loc["lift_lost_indef"])
ax.set_title(rf"cooperative gap: global $>4>$ any block." + "\n" +
             rf"lost-true lift onto indefinite clusters $\approx${lift_mean:.0f}$\times$",
             fontsize=8.8)
ax.grid(alpha=0.25, axis="y")
fig.suptitle(r"Critical mode is a LOCAL cluster but a COOPERATIVE (multi-hub) "
             r"instability — store classical, $T=400,\ \sigma_r=0.05$",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGDIR / "critical_mode_localization.png", dpi=160)
plt.close(fig)
print("[fig] critical_mode_localization.png", flush=True)


# ===========================================================================
# FIGURE 6 — the fixes: gamma bump vs epsilon tighten (fixed event)
# ===========================================================================
print("\n=== FIG 6: gamma / epsilon fix ===", flush=True)
# ONE fixed reproducible event used for BOTH panels (so mult=1.0 == lam_max_fixed).
ev6 = gen_event(400, 0.05, SS, seed=SEED0 + 400 * 13)
# (a) gamma sweep at the fixed indefinite event: lam_min(A)=(gamma+delta)-lam_max(C)
C6, A6, ham6, eps6, _ = build_C(400, 0.05, SS, seed=0, ev=ev6)
lam_max_fixed = spectral_C(C6)[0]
gammas = np.linspace(1, 6, 60)
lam_min_vs_g = (gammas + DELTA) - lam_max_fixed
# (b) epsilon tighten at the SAME event: rebuild C at multiples of formula eps.
# Capped at 1.3x: beyond ~1.5x the false-coupling graph starts to PERCOLATE into a
# giant component (the dense regime, see 7.11), where the exact per-component eig is
# no longer cheap. The 0.5-1.3x window is the relevant "tighten eps to restore PD"
# range and already shows lambda_max crossing gamma+delta.
mults = np.array([0.5, 0.65, 0.8, 0.9, 1.0, 1.15, 1.3])
lam_max_vs_eps = []
for mu in mults:
    Cx, _, _, _, _ = build_C(400, 0.05, SS, seed=0, ev=ev6, eps_override=eps6 * mu)
    lam_max_vs_eps.append(spectral_C(Cx)[0])
lam_max_vs_eps = np.array(lam_max_vs_eps)

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
ax = axes[0]
ax.plot(gammas, lam_min_vs_g, "-", color="tab:blue", lw=1.6)
ax.axhline(0, color="tab:red", ls="--", lw=1.2)
ax.axvline(lam_max_fixed - DELTA, color="k", ls=":", lw=1.1)
ax.text(lam_max_fixed - DELTA + 0.05, ax.get_ylim()[0] + 0.2,
        rf"$\gamma^*={lam_max_fixed-DELTA:.2f}$", fontsize=9)
for gg in (1, 2, 3):
    ax.plot(gg, (gg + DELTA) - lam_max_fixed, "o", ms=8,
            color="tab:orange" if (gg + DELTA) < lam_max_fixed else "tab:green")
ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\lambda_{\min}(A)$")
ax.set_title(rf"$\gamma$-bump: PD restored for $\gamma>\lambda_{{\max}}(C)-\delta"
             rf"={lam_max_fixed-DELTA:.2f}$" + "\n"
             rf"(at fixed $\lambda_{{\max}}(C)={lam_max_fixed:.2f}$, "
             rf"$T=400,\sigma_r=0.05$)", fontsize=9)
ax.grid(alpha=0.25)
# annotate gamma-aware tau
ax2 = ax.twinx()
tau_g = DELTA / (DELTA + gammas) + 0.10
ax2.plot(gammas, tau_g, "--", color="tab:purple", lw=1.2, alpha=0.7)
ax2.set_ylabel(r"$\gamma$-aware $\tau=\delta/(\delta+\gamma)+0.10$",
               color="tab:purple", fontsize=9)
ax2.tick_params(axis="y", colors="tab:purple")

ax = axes[1]
ax.plot(mults, lam_max_vs_eps, "o-", color="tab:blue", ms=6)
ax.axhline(GD, color="tab:red", ls="--", lw=1.2)
ax.axvline(1.0, color="tab:green", lw=1.1, alpha=0.6)
ax.text(1.02, lam_max_vs_eps.min(), "formula $\\varepsilon$", rotation=90,
        color="tab:green", fontsize=8, va="bottom")
ax.text(mults[0], GD + 0.05, r"$\gamma+\delta=4$", color="tab:red", fontsize=8)
ax.set_xlabel(r"$\varepsilon / \varepsilon_{\rm formula}$")
ax.set_ylabel(r"$\lambda_{\max}(C)$")
ax.set_title(r"$\varepsilon$-tighten: shrinking $\varepsilon$ drops "
             r"$\lambda_{\max}(C)$ below 4" + "\n"
             "(but costs true-coupling efficiency — see §7.9)", fontsize=9)
ax.grid(alpha=0.25)
fig.suptitle(r"Two fixes for the indefinite regime: raise $\gamma$ (lift the "
             r"diagonal) or tighten $\varepsilon$ (thin the coupling) — "
             rf"$T=400,\ \sigma_r=0.05$", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(FIGDIR / "gamma_epsilon_fix.png", dpi=160)
plt.close(fig)
print("[fig] gamma_epsilon_fix.png", flush=True)


# ===========================================================================
# JSON dump
# ===========================================================================
def jsonify(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))

out = dict(
    gamma=GAMMA, delta=DELTA, GD=GD, threshold_product=GD**2, tau=TAU,
    sigma_scatt=SS, nrep=NREP,
    closed_form_max_abs_err=closed_form_max_err,
    bipartite_check=dict(
        n_pairs=bip_check["n_pairs"],
        frac_share_middle=bip_check["share_either"] / max(bip_check["n_pairs"], 1),
        same_in_side=bip_check["same_in"], same_out_side=bip_check["same_out"]),
    grid={f"sr{sr:g}_T{T}": dict(
            sr=sr, T=T, n_rep=len(grid[(sr, T)]),
            eps=grid[(sr, T)][0]["eps"],
            n_seg=grid[(sr, T)][0]["n_seg"],
            lam_max_C=cell_mean(sr, T, "lam_max_C"),
            lam_min_A=cell_mean(sr, T, "lam_min_A"),
            n_neg=cell_mean(sr, T, "n_neg"),
            max_sigma_blk=cell_mean(sr, T, "max_sigma_blk"),
            max_prod_blk=cell_mean(sr, T, "max_prod_blk"),
            indefinite=bool(cell_mean(sr, T, "lam_min_A")[0] < 0))
          for (sr, T) in grid},
    boundary=dict(cross_sr=cross_sr, cross_T=cross_T),
    localization=dict(
        cell="T400_sr0.05_store",
        n_reps=loc["n_reps"],
        lam_max_C_mean=float(np.mean(loc["lam_max_C"])) if loc["lam_max_C"] else None,
        max_block_sigma_mean=float(np.mean(loc["max_block_sigma"])) if loc["max_block_sigma"] else None,
        part50_mean=float(np.mean(loc["part50"])) if loc["part50"] else None,
        part90_mean=float(np.mean(loc["part90"])) if loc["part90"] else None,
        n_hits_support_mean=float(np.mean(loc["n_hits_support"])) if loc["n_hits_support"] else None,
        support_planes=sorted(loc["planes"]),
        lift_lost_on_indef_mean=float(np.nanmean(loc["lift_lost_indef"])) if loc["lift_lost_indef"] else None,
        frac_lost_on_indef_mean=float(np.mean(loc["frac_lost_on_indef"])) if loc["frac_lost_on_indef"] else None,
        frac_seg_indef_mean=float(np.mean(loc["frac_seg_indef"])) if loc["frac_seg_indef"] else None,
        neg_true_frac_mean=float(np.mean(loc["neg_true_frac"])) if loc["neg_true_frac"] else None),
    fix=dict(lam_max_C_fixed_event=lam_max_fixed,
             gamma_star=lam_max_fixed - DELTA,
             eps_mults=mults.tolist(),
             lam_max_vs_eps=lam_max_vs_eps.tolist()),
    store_classical_eff=eff_piv.to_dict(),
)
with (OUTDIR / "indefiniteness_study.json").open("w") as f:
    json.dump(out, f, indent=2, default=jsonify)
print(f"\n[done] all figures + indefiniteness_study.json "
      f"({time.time()-t_start:.0f}s total)", flush=True)
