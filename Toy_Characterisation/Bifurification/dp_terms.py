"""Denby–Peterson penalty terms on the segment Hamiltonian (v2).

Two penalties from the original Denby track-finding Hamiltonian, implemented
as sparse modifications of the base system A0 x = b (A0 = (γ+δ)I − C,
b = δ·1) so both MINRES and the 1BQF can solve them:

* **Bifurcation (fork)** — penalise co-hit segment pairs within a mutual-angle
  window ε_B (the ε-windowed form; the dense all-pair fork uniformly rescales
  and breaks the 1BQF — Bifurification v1):
      A_fork = A0 + β·B_fork ,  b unchanged (off-diagonal mode, attractor kept)

* **Occupancy (per-hit)** — Denby's hit-uniqueness constraint
      E_occ = α Σ_h (o_h − 1)² ,  o_h = Σ_{s∋h} x_s
  Expanding: quadratic matrix M = 2I + B_all (B_all = ALL co-hit pairs, any
  angle; every segment has exactly 2 hits) and a linear pull toward o_h = 1:
      A_occ = A0 + 2α·M = A0 + 4α·I + 2α·B_all ,   b_occ = b + 4α·1
  The isolated-segment attractor shifts to (δ+4α)/(γ+δ+4α); the β-aware
  threshold below tracks it.  NOTE: Denby's GLOBAL count term α(Σx − N)² is a
  dense rank-one coupling — classical-only (LinearOperator); NOT implementable
  as 1BQF pair gates.  The per-hit form is the quantum-compatible reading;
  flagged for George on the DP to-do.

Both compose:  A = A0 + β·B_fork + 4α·I + 2α·B_all ,  b = δ·1 + 4α·1.
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.sparse as sp

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification_initial"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bif  # v1 machinery: fork_graph (all co-hit), fork_graph_eps (windowed)


def cohit_graph(ham) -> sp.csr_matrix:
    """B_all: every co-hit segment pair (any mutual angle). Symmetric 0/1."""
    return bif.fork_graph(ham._segment_to_hit_ids)


def fork_graph_eps(ham, eps_B: float) -> sp.csr_matrix:
    """B_fork: co-hit pairs with mutual angle < eps_B (the viable v1 form)."""
    return bif.fork_graph_eps(ham._segment_to_hit_ids, ham._segment_vectors, eps_B)


def dp_system(ham, beta: float = 0.0, eps_B: float | None = None,
              alpha: float = 0.0, gamma: float = 3.0, delta: float = 1.0):
    """Compose fork (β, ε_B) and per-hit occupancy (α) onto the base system.

    Returns (A, b, tau, info): the modified sparse system, the attractor-aware
    absolute threshold (isolated attractor + 0.10), and nnz bookkeeping.
    """
    A = ham.A.tocsr().copy()
    n = A.shape[0]
    b = float(delta) * np.ones(n)
    info = dict(A0_nnz=int(A.nnz), Bfork_nnz=0, Ball_nnz=0)
    if beta:
        Bf = fork_graph_eps(ham, float(eps_B if eps_B is not None else ham.epsilon))
        A = (A + float(beta) * Bf).tocsr()
        info["Bfork_nnz"] = int(Bf.nnz)
    if alpha:
        Ba = cohit_graph(ham)
        A = (A + 4.0 * float(alpha) * sp.identity(n, format="csr")
             + 2.0 * float(alpha) * Ba).tocsr()
        b = b + 4.0 * float(alpha)
        info["Ball_nnz"] = int(Ba.nnz)
    attractor = (float(delta) + 4.0 * float(alpha)) / \
                (float(gamma) + float(delta) + 4.0 * float(alpha))
    tau = attractor + 0.10
    info["A_nnz"] = int(A.nnz)
    return A.tocsc(), b, float(tau), info
