"""bif.py — build and solve the segment Hamiltonian WITH a Denby-Peterson
bifurcation (fork) term, and recompute segment metrics on the new solutions.

Two forms (see bifurcation_hamiltonian.md):
  off-diag :  A' = (gamma+delta) I - C + beta B ,  b = delta 1     (notch at gamma+delta)
  full     :  A''= (gamma+delta+2 beta) I - C + beta B , b=(delta+beta)1  (notch at gamma+delta+2beta)

C = continuation adjacency (the base Hamiltonian's off-diagonal, opposite-side
hit sharing, angle<eps). B = fork adjacency (same-side hit sharing: shared start
OR shared end hit, ALL co-hit segments regardless of angle -> dense, O(T^3)).
Both forms keep a CONSTANT diagonal (every segment touches exactly 2 hits), so
both are 1BQF-compatible.

Events are reused from the qtrk_store generator; only the solution vectors change
(A changed), so metrics are recomputed here, never read from the base store.
"""
from __future__ import annotations
import sys
import numpy as np
import scipy.sparse as sp

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import os
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp  # noqa: E402

GAMMA, DELTA, EPS = 3.0, 1.0, 0.002
STD = dict(sigma_scatt=1e-4, sigma_res=0.0, phi_max=0.2, hit_ineff=0.0)


def event(T, rep=0, **kw):
    p = dict(STD, **kw)
    return qp.ensure_event(n_trk=int(T), rep=int(rep), **p)[0]


def base_hamiltonian(ev, eps=EPS, gamma=GAMMA, delta=DELTA):
    """The base (continuation-only) Hamiltonian ham, with .A and segment->hit ids."""
    return qp.build_hamiltonian(ev, epsilon=float(eps), gamma=float(gamma), delta=float(delta))


def fork_graph(seg_hits) -> sp.csr_matrix:
    """0/1 fork adjacency B: segments sharing a start-hit OR an end-hit (i!=j)."""
    seg_hits = np.asarray(seg_hits)
    n = len(seg_hits)
    rows, cols = [], []
    for col in (0, 1):                                   # 0 = start hit, 1 = end hit
        order = np.argsort(seg_hits[:, col], kind="stable")
        key = seg_hits[order, col]
        bnd = np.r_[0, np.where(np.diff(key))[0] + 1, n]
        for a, c in zip(bnd[:-1], bnd[1:]):
            grp = order[a:c]
            if len(grp) > 1:
                ii, jj = np.meshgrid(grp, grp)
                m = ii < jj
                rows.append(ii[m]); cols.append(jj[m])
    if not rows:
        return sp.csr_matrix((n, n))
    r = np.concatenate(rows); c = np.concatenate(cols)
    B = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return (B + B.T).tocsr()


def fork_graph_eps(seg_hits, seg_vecs, epsilon=EPS) -> sp.csr_matrix:
    """ε-WINDOWED fork adjacency: co-hit segment pairs whose mutual angle < ε.

    The acceptance-window version of the bifurcation term — only segments that
    share a hit AND fall inside the angular window ε (so they are genuinely
    competing continuations) are penalised. This is SPARSE (nnz ≈ O(n_seg) or
    less, vs the dense `fork_graph`'s O(T³)), so it preserves the sparse-A
    invariant and keeps the 1BQF tractable. Idea: of all the segments at a hit
    that are within ε, you are punished for choosing more than one.
    """
    seg_hits = np.asarray(seg_hits); seg_vecs = np.asarray(seg_vecs)
    n = len(seg_hits); cos_eps = float(np.cos(epsilon))
    rows, cols = [], []
    for col in (0, 1):                                   # 0 = start hit, 1 = end hit
        order = np.argsort(seg_hits[:, col], kind="stable")
        key = seg_hits[order, col]
        bnd = np.r_[0, np.where(np.diff(key))[0] + 1, n]
        for a, c in zip(bnd[:-1], bnd[1:]):
            grp = order[a:c]
            if len(grp) < 2:
                continue
            V = seg_vecs[grp]
            cosM = np.clip(V @ V.T, -1.0, 1.0)
            ii, jj = np.triu_indices(len(grp), 1)
            keep = cosM[ii, jj] > cos_eps                # angle < ε
            if keep.any():
                rows.append(grp[ii[keep]]); cols.append(grp[jj[keep]])
    if not rows:
        return sp.csr_matrix((n, n))
    r = np.concatenate(rows); c = np.concatenate(cols)
    B = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return (B + B.T).tocsr()


def threshold(beta, mode, gamma=GAMMA, delta=DELTA, margin=0.10):
    """beta-aware activation threshold = (Hopfield attractor) + margin."""
    if mode == "full":
        attr = (delta + beta) / (gamma + delta + 2 * beta)
    else:                                                # off-diag: attractor unchanged
        attr = delta / (gamma + delta)
    return attr + margin


def bif_system(A0, B, beta, mode, gamma=GAMMA, delta=DELTA):
    """Return (A, b, diag, tau) for the chosen form."""
    n = A0.shape[0]
    if mode == "full":
        A = (A0 + 2 * beta * sp.identity(n) + beta * B).tocsc()
        b = (delta + beta) * np.ones(n)
        diag = gamma + delta + 2 * beta
    elif mode == "off":
        A = (A0 + beta * B).tocsc()
        b = delta * np.ones(n)
        diag = gamma + delta
    else:
        raise ValueError(mode)
    return A, b, diag, threshold(beta, mode, gamma, delta)


def solve_classical(A, b):
    n = A.shape[0]
    if n < 5000:
        return np.asarray(sp.linalg.spsolve(A, b)).ravel()
    sol, _ = sp.linalg.minres(A, b, rtol=1e-8, maxiter=5000)
    return np.asarray(sol).ravel()


def solve_quantum(A, b, sol_C, tau, device="CPU"):
    """1BQF statevector solve; returns the solution rescaled on the classical
    signal support (so the same tau applies), and the raw success probability."""
    from helpers import solve_quantum_statevector
    sol_raw, p_anc, n_sys = solve_quantum_statevector(A, b, device=device)
    sol_raw = np.asarray(sol_raw[: A.shape[0]], float)
    m = sol_C > tau
    denom = np.linalg.norm(sol_raw[m]) if m.any() else np.linalg.norm(sol_raw)
    scale = (np.linalg.norm(sol_C[m]) / denom) if (m.any() and denom > 0) else 1.0
    return np.abs(sol_raw) * scale, float(p_anc), int(n_sys)


def metrics(sol, truth, tau):
    return qp.metrics_at(np.asarray(sol, float), np.asarray(truth, bool), float(tau))


def truth_mask(ev):
    return np.asarray(qp.truth_from_event(ev), bool)


def auc(score, truth):
    """Rank-AUC of true vs false (1 = perfect separation, scale-free)."""
    score = np.asarray(score, float); truth = np.asarray(truth, bool)
    order = np.argsort(score); rank = np.empty(len(score)); rank[order] = np.arange(len(score))
    nt, nf = truth.sum(), (~truth).sum()
    return float((rank[truth].sum() - nt * (nt - 1) / 2) / (nt * nf)) if nt and nf else np.nan
