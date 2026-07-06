#!/usr/bin/env python3
"""Build the false-positive-TYPE census across the Epsilon_study_2 noise grid.

For every stored event pickle (n_trk in TVALS) we regenerate the *exact* event
from its deterministic seed, rebuild the sparse Hamiltonian A, and classify every
segment by its topology in the compatibility graph C = sI - A (off-diagonal):

    TRUE       -- a real track segment
    isolated   -- false, no coupling (component size 1)        [Hopfield attractor]
    pair       -- false, in a 2-node component
    chain>=3   -- false, component >=3 but max degree < 3 (a path)
    hub        -- false, component with a node of degree >= 3 (a star centre)

This mirrors Run3_Verification/Run3_segment_characterisation.ipynb (cluster_classes).
Per (cell, T, rep) we then count, for each type, how many segments exist and how
many are ACTIVE (a false positive) under the classical (sol_C) and quantum (sol_Q)
solvers at the absolute threshold tau = 0.35.

Output: outputs/false_positive_types/fp_census.csv  (long format, one row per
(event, type)).  The notebook reads this CSV and makes the noise-grid figures.

Event reproduction is verified identical to the stored truth mask before counting.
"""
from __future__ import annotations
import os, sys, glob, pickle, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

from helpers import make_geometry, safe_generate, segment_truth_mask  # noqa: E402
from lhcb_velo_toy.solvers import SimpleHamiltonianFast                # noqa: E402
from qtrk_pipeline.metrics import working_point_threshold              # noqa: E402

TVALS = (50, 100, 200, 400)          # density range with full noise-grid coverage
TYPES = ["isolated", "pair", "chain>=3", "hub", "TRUE"]
GEOM = make_geometry()


def cluster_classes(A, n, truth, S):
    """Per-segment topology class from the compatibility graph C = sI - A."""
    C = (S * sp.identity(n, format="csr") - A)
    C.setdiag(0); C.eliminate_zeros()
    C = (abs(C) > 1e-9).astype(np.int8)
    deg = np.asarray(C.sum(1)).ravel()
    _, lab = connected_components(C, directed=False)
    csz = np.bincount(lab)[lab]
    maxdeg = pd.Series(deg).groupby(lab).transform("max").values
    return np.where(truth, "TRUE",
           np.where(csz == 1, "isolated",
           np.where(csz == 2, "pair",
           np.where(maxdeg >= 3, "hub", "chain>=3"))))


def process(pf: str):
    d = pickle.load(open(pf, "rb"))
    if d.get("status") != "ok" or int(d["n_trk"]) not in TVALS:
        return []
    seed = int(d["rep"]) + int(d["n_trk"]) * 100 + 200_000
    ev = safe_generate(int(d["n_trk"]), seed=seed, geom=GEOM,
                       measurement_error=float(d["sigma_res"]),
                       collision_noise=float(d["sigma_scatt"]),
                       phi_max=float(d["phi_max"]), theta_max=float(d["phi_max"]))
    ham = SimpleHamiltonianFast(epsilon=float(d["epsilon"]), gamma=float(d["gamma"]),
                                delta=float(d["delta"]), theta_d=float(d["erf_sigma"]))
    ham.construct_segments(ev)
    ham.construct_hamiltonian(ev, convolution=bool(d["convolution"]))
    n = ham.n_segments
    truth = np.asarray(segment_truth_mask(ham), bool)
    if n != int(d["n_seg"]) or not np.array_equal(truth, d["truth"]):
        return [dict(tag=d["tag"], n_trk=int(d["n_trk"]), rep=int(d["rep"]),
                     repro_ok=False, type="ERR", count=0)]
    S = float(d["gamma"]) + float(d["delta"])
    cls = cluster_classes(ham.A.tocsr(), n, truth, S)
    tau = float(d["tau_default"])
    solC = np.asarray(d["sol_C"]).ravel()
    actC = solC > tau
    solQ = d.get("sol_Q")
    have_q = solQ is not None
    # 1BQF HEADLINE activation = wp99 (efficiency-first, tau at the 1% quantile of
    # the TRUE quantum amplitudes; wp99 log 2026-06-14). Scale-invariant, so it is
    # valid even on these legacy-rescale pickles. The fixed-tau act_Q stays as the
    # labelled cut-artefact record (it under-counts true actives by ~25%).
    if have_q:
        solQa = np.abs(np.asarray(solQ).ravel())
        actQ = solQa > tau                                   # fixed-tau record
        tauQw = float(working_point_threshold(solQa, truth)) # wp99 headline
        actQw = solQa > tauQw
    else:
        actQ = actQw = None
        tauQw = np.nan
    rows = []
    for t in TYPES:
        sel = cls == t
        rows.append(dict(
            tag=d["tag"], sigma_res=float(d["sigma_res"]), sigma_scatt=float(d["sigma_scatt"]),
            n_trk=int(d["n_trk"]), rep=int(d["rep"]), n_seg=n, repro_ok=True,
            type=t, count=int(sel.sum()),
            act_C=int((sel & actC).sum()),
            act_Q=(int((sel & actQ).sum()) if have_q else np.nan),
            act_Q_wp=(int((sel & actQw).sum()) if have_q else np.nan),
            tau_Q_wp=tauQw,
            have_q=have_q,
        ))
    return rows


def main():
    files = sorted(glob.glob(str(HERE / "results" / "**" / "*.pkl"), recursive=True))
    files = [f for f in files if "seg14e" not in f]   # the seg14e sub-study has its own grid
    out_rows, t0, done = [], time.time(), 0
    for pf in files:
        try:
            r = process(pf)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {os.path.basename(pf)}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if r:
            out_rows.extend(r); done += 1
            if done % 25 == 0:
                print(f"  {done} events  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(out_rows)
    outdir = HERE / "outputs" / "false_positive_types"; outdir.mkdir(parents=True, exist_ok=True)
    csv = outdir / "fp_census.csv"
    df.to_csv(csv, index=False)
    print(f"\n[done] {done} events -> {len(df)} rows -> {csv}  ({time.time()-t0:.0f}s)")
    bad = df[df.get("repro_ok") == False] if "repro_ok" in df else df.iloc[0:0]  # noqa: E712
    if len(bad):
        print(f"[WARN] {bad['tag'].nunique()} events failed reproduction")


if __name__ == "__main__":
    main()
