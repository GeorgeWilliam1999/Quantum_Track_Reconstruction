#!/usr/bin/env python3
"""Data for the Gershgorin figure of paper section 2.2.

Section 2.2 states the sufficient condition gamma + delta > d_max and says where
it lapses, but the paper never draws the discs, never shows what drives d_max,
and never demonstrates the claim that the regularisation can always restore
positive definiteness.  This produces all three from real events.

  (a) the degree distribution -- the bound is set by a thin tail, while the
      bulk of segments have no compatible neighbour at all;
  (b) the Gershgorin floor s - d_max against the measured lambda_min vs T;
  (c) lambda_min against gamma on the worst component of a heavy event: the
      control knob.

Output: outputs/03_gershgorin.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp                             # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon     # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)

GAMMA, DELTA = 3.0, 1.0
S = GAMMA + DELTA
CONFIGS = {
    "clean":    dict(sigma_scatt=1e-4, sigma_res=0.0,  phi_max=0.2, hit_ineff=0.0),
    "moderate": dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01),
    "heavy":    dict(sigma_scatt=1e-4, sigma_res=0.02, phi_max=0.2, hit_ineff=0.01),
}
TGRID = (100, 200, 400, 700)
GAMMAS = np.round(np.arange(1.0, 8.01, 0.25), 3)


def offdiag(A):
    O = A.copy().tolil(); O.setdiag(0); O = O.tocsr(); O.eliminate_zeros()
    O.data[:] = 1.0
    return O


def lam_min_exact(Off, gamma, delta):
    """Exact lambda_min of A = (gamma+delta)I - C, component by component."""
    n = Off.shape[0]
    nc, lab = connected_components(Off, directed=False)
    sizes = np.bincount(lab)
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(nc + 1))
    lo = gamma + delta                       # isolated segments sit at s
    for c in np.flatnonzero(sizes > 1):
        idx = order[bounds[c]:bounds[c + 1]]
        sub = Off[np.ix_(idx, idx)].toarray()
        mu = eigh((gamma + delta) * np.eye(len(idx)) - sub,
                  eigvals_only=True)
        lo = min(lo, float(mu[0]))
    return lo


def main():
    out = {"gamma": GAMMA, "delta": DELTA, "s": S,
           "degree_hist": {}, "vs_T": {}, "gamma_scan": {}}

    # (a) degree distribution at a fixed T, all three configs
    for cfg, nz in CONFIGS.items():
        eps = compute_epsilon(sigma_scatt=nz["sigma_scatt"], sigma_res=nz["sigma_res"])
        ev, _ = qp.ensure_event(n_trk=200, rep=0, **nz)
        ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                                   gamma=GAMMA, delta=DELTA)
        Off = offdiag(sp.csr_matrix(ham.A))
        deg = np.asarray(Off.sum(axis=1)).ravel().astype(int)
        cnt = np.bincount(deg)
        out["degree_hist"][cfg] = dict(
            T=200, eps_mrad=float(eps * 1e3), counts=cnt.tolist(),
            n_seg=int(len(deg)), d_max=int(deg.max()))
        print(f"[a] {cfg:8s} eps={eps*1e3:5.2f} mrad  n_seg={len(deg)}  "
              f"d_max={deg.max()}  frac(deg=0)={cnt[0]/len(deg):.4f}", flush=True)

    # (b) floor s - d_max and measured lambda_min vs T
    for cfg, nz in CONFIGS.items():
        eps = compute_epsilon(sigma_scatt=nz["sigma_scatt"], sigma_res=nz["sigma_res"])
        rows = []
        for T in TGRID:
            ev, _ = qp.ensure_event(n_trk=T, rep=0, **nz)
            ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                                       gamma=GAMMA, delta=DELTA)
            Off = offdiag(sp.csr_matrix(ham.A))
            deg = np.asarray(Off.sum(axis=1)).ravel().astype(int)
            lm = lam_min_exact(Off, GAMMA, DELTA)
            rows.append(dict(T=T, d_max=int(deg.max()),
                             floor=float(S - deg.max()), lam_min=lm,
                             n_seg=int(Off.shape[0])))
            print(f"[b] {cfg:8s} T={T:5d}  d_max={deg.max():2d}  "
                  f"floor={S-deg.max():6.2f}  lam_min={lm:.4f}", flush=True)
        out["vs_T"][cfg] = rows

    # (c) the control knob.  lambda_min(A) = (gamma+delta) - lambda_max(C)
    # exactly, and lambda_max(C) does not depend on gamma at all, so the scan
    # is one eigen-solve and a straight line -- not one solve per gamma.  The
    # crossing is at gamma = lambda_max(C) - delta.
    for cfg in ("moderate", "heavy"):
        nz = CONFIGS[cfg]
        eps = compute_epsilon(sigma_scatt=nz["sigma_scatt"], sigma_res=nz["sigma_res"])
        for T in (400, 700):
            ev, _ = qp.ensure_event(n_trk=T, rep=0, **nz)
            ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                                       gamma=GAMMA, delta=DELTA)
            Off = offdiag(sp.csr_matrix(ham.A))
            deg = np.asarray(Off.sum(axis=1)).ravel().astype(int)
            lam_max_C = float(S - lam_min_exact(Off, GAMMA, DELTA))
            out["gamma_scan"][f"{cfg}_T{T}"] = dict(
                T=T, config=cfg, d_max=int(deg.max()), eps_mrad=float(eps * 1e3),
                lam_max_C=lam_max_C, gamma_crit=lam_max_C - DELTA,
                gammas=GAMMAS.tolist(),
                lam_min=[float(g + DELTA - lam_max_C) for g in GAMMAS])
            print(f"[c] {cfg:8s} T={T}  d_max={deg.max():2d}  "
                  f"lam_max(C)={lam_max_C:.4f}  gamma needed for positive "
                  f"definiteness: {lam_max_C - DELTA:.4f}", flush=True)

    (OUT / "03_gershgorin.json").write_text(json.dumps(out, indent=1))
    print("[saved] outputs/03_gershgorin.json")


if __name__ == "__main__":
    main()
