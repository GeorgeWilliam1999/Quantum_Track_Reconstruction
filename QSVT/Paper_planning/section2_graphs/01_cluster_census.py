#!/usr/bin/env python3
"""Cluster-topology census for paper section 2.

Section 2.5 names three cluster families (isolated segment P1, chain P_m, star
K(1,m)) and asserts that these are the motifs that matter.  Nothing in the paper
measures how often each actually occurs.  This script counts them.

For every connected component of the compatibility graph it records the number
of segments, the number of compatibility edges, a topology class, the true/false
composition, and the number of *bright* modes -- eigenvectors with non-zero
overlap with the uniform bias b = delta*1.  Dark modes carry an eigenvalue but no
amplitude, which is the distinction the paper currently misses.

Topology classes (a component with n vertices, e edges):
  isolated   n = 1
  path       tree, max degree <= 2          (P_m; note P_2 = K(1,1), P_3 = K(1,2)
                                             are counted as paths, matching the
                                             paper's 'pair'/'triple' naming)
  star       tree, some vertex of degree n-1, n >= 4
  tree       any other tree (e = n-1)
  cyclic     e > n-1

Outputs
  outputs/01_cluster_census.csv        one row per (config, T, rep, class, n)
  outputs/01_cluster_summary.csv       fractions by class, per (config, T)
  outputs/01_bright_dark.csv           bright/total mode counts by class and n
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import os
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp                              # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon      # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)

GAMMA, DELTA = 3.0, 1.0
S = GAMMA + DELTA
CONFIGS = {
    "clean":    dict(sigma_scatt=1e-4, sigma_res=0.0,   phi_max=0.2, hit_ineff=0.0),
    "moderate": dict(sigma_scatt=1e-4, sigma_res=0.01,  phi_max=0.2, hit_ineff=0.01),
    "heavy":    dict(sigma_scatt=1e-4, sigma_res=0.02,  phi_max=0.2, hit_ineff=0.01),
}
TGRID = (100, 200, 400, 700, 1000)
REPS = (0, 1, 2)
MAX_EIGH_N = 16          # bright/dark decomposition only for components this small
OVERLAP_TOL = 1e-9

LOG = open(OUT / "01_census.log", "w")
def say(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


def classify(n, e, degs):
    if n == 1:
        return "isolated"
    if e == n - 1:                       # tree
        if degs.max() <= 2:
            return "path"
        if degs.max() == n - 1:
            return "star"
        return "tree"
    return "cyclic"


def census_one(config, T, rep):
    nz = CONFIGS[config]
    eps = compute_epsilon(sigma_scatt=nz["sigma_scatt"], sigma_res=nz["sigma_res"])
    ev, _ = qp.ensure_event(n_trk=T, rep=rep, **nz)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                               gamma=GAMMA, delta=DELTA)
    A = sp.csr_matrix(ham.A)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    n_seg = A.shape[0]

    Off = A.copy().tolil(); Off.setdiag(0); Off = Off.tocsr(); Off.eliminate_zeros()
    Off.data[:] = 1.0
    ncomp, lab = connected_components(Off, directed=False)
    deg = np.asarray(Off.sum(axis=1)).ravel().astype(int)

    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(ncomp + 1))
    b = DELTA * np.ones(n_seg)

    # Vectorised per-component quantities.  A Python loop over every component
    # is hopeless here -- at T=1000 there are four million of them, almost all
    # single isolated segments.  Only the handful with two or more segments
    # needs an eigendecomposition.
    csize = np.bincount(lab, minlength=ncomp)
    cedges = (np.bincount(lab, weights=deg, minlength=ncomp) / 2).astype(int)
    cdmax = np.zeros(ncomp, int)
    np.maximum.at(cdmax, lab, deg)
    ctrue = np.bincount(lab, weights=truth.astype(float),
                        minlength=ncomp).astype(int)

    cls = np.empty(ncomp, dtype=object)
    is_tree = cedges == csize - 1
    cls[:] = "cyclic"
    cls[is_tree] = "tree"
    cls[is_tree & (cdmax <= 2)] = "path"
    cls[is_tree & (cdmax == csize - 1) & (csize >= 4)] = "star"
    cls[csize == 1] = "isolated"

    n_bright = np.full(ncomp, -1)
    n_dark = np.full(ncomp, -1)
    n_bright[csize == 1] = 1
    n_dark[csize == 1] = 0
    for c in np.flatnonzero((csize > 1) & (csize <= MAX_EIGH_N)):
        idx = order[bounds[c]:bounds[c + 1]]
        lam, V = eigh(A[np.ix_(idx, idx)].toarray())
        nb = int((np.abs(V.T @ b[idx]) > OVERLAP_TOL).sum())
        n_bright[c], n_dark[c] = nb, len(idx) - nb

    rows = list(zip(cls, csize, cedges, ctrue, csize - ctrue, n_bright, n_dark))
    df = pd.DataFrame(rows, columns=["cls", "n", "e", "n_true", "n_false",
                                     "n_bright", "n_dark"])
    # Aggregate here, not at the end.  Holding every component of every event
    # and rewriting the raw CSV each iteration costs far more than the census
    # itself -- at heavy T=1000 that is ~12M rows written 45 times over.
    agg = (df.groupby(["cls", "n"], as_index=False)
             .agg(n_comp=("n", "size"), n_true=("n_true", "sum"),
                  n_false=("n_false", "sum"),
                  n_bright=("n_bright", "sum"), n_dark=("n_dark", "sum"),
                  n_eigh=("n_bright", lambda v: int((v >= 0).sum()))))
    agg["config"], agg["T"], agg["rep"] = config, T, rep
    agg["n_seg_event"], agg["n_true_event"] = n_seg, int(truth.sum())
    return agg


def main():
    only = sys.argv[1:] or list(CONFIGS)
    parts = []
    for config in only:
        for T in TGRID:
            for rep in REPS:
                t0 = time.time()
                try:
                    parts.append(census_one(config, T, rep))
                except Exception as exc:                       # noqa: BLE001
                    say(f"[skip] {config} T={T} rep={rep}: {type(exc).__name__}: {exc}")
                    continue
                d = parts[-1]
                say(f"[ok] {config:8s} T={T:5d} rep={rep}  "
                    f"n_seg={d['n_seg_event'].iloc[0]:>8d}  comps={len(d):>7d}  "
                    f"{time.time()-t0:6.1f}s")
                pd.concat(parts).to_csv(OUT / "01_cluster_census.csv", index=False)

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT / "01_cluster_census.csv", index=False)

    # summary: fraction of components and of segments by class
    s = (df.assign(segs=df["n"] * df["n_comp"])
           .groupby(["config", "T", "cls"])
           .agg(comps=("n_comp", "sum"), segs=("segs", "sum"),
                true_segs=("n_true", "sum"))
           .reset_index())
    tot = s.groupby(["config", "T"])[["comps", "segs", "true_segs"]].transform("sum")
    s["frac_comps"] = s["comps"] / tot["comps"]
    s["frac_segs"] = s["segs"] / tot["segs"]
    s["frac_true_segs"] = s["true_segs"] / tot["true_segs"]
    s.to_csv(OUT / "01_cluster_summary.csv", index=False)
    say("\n" + s.to_string(index=False))

    e = df[df["n_eigh"] > 0]
    bd = (e.groupby(["cls", "n"])
            .agg(n_comp=("n_eigh", "sum"), bright_tot=("n_bright", "sum"),
                 dark_tot=("n_dark", "sum")).reset_index())
    bd["bright"] = bd["bright_tot"] / bd["n_comp"]
    bd["dark"] = bd["dark_tot"] / bd["n_comp"]
    bd.to_csv(OUT / "01_bright_dark.csv", index=False)
    say("\n" + bd.to_string(index=False))


if __name__ == "__main__":
    main()
