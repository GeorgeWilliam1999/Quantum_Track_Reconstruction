#!/usr/bin/env python3
"""Direct (heuristic-free) floor twin count — settles the rep-2 outlier (George, 2026-07-18).

The paper's floor census (04_fit_comb_to_measured_spectrum.py) counted floor trues
via an amplitude leak-tier heuristic that is scale-sensitive: rep 2's fit selected a
tenfold-larger regulariser and the heuristic read 52, vs directly-counted 15 and 22
on reps 0/1.  This script does the DIRECT count on all three events, from the floor
theorem itself, with no reference to any fitted response:

  A true segment i is ON THE FLOOR iff its *formation* — the multiset of
  (eigenvalue, |uniform-b excitation coefficient|) pairs
  {(lam_k, |beta_k V[i,k]|) : |beta_k V[i,k]| > tol}, beta_k = sum_j V[j,k] —
  is identical to the formation of at least one FALSE segment in the same event.
  Identical formation => identical amplitude under EVERY spectral response f
  (the floor theorem), so such trues are unrecoverable by any filter.

Isolated segments have formation {(s, 1)} and are twins of the isolated-false
grass by construction (the P1-stub floor).  Counts are reported at three match
tolerances (1e-6 / 1e-8 / 1e-10 rounding) to show the count is not a tolerance
artefact.  Config = the fit study's: heavy noise, T=200, beta=0, gmode='ref'.

Validation: reps 0/1 must reproduce the paper's directly-counted 15 and 22.
Output: outputs/fork_noisy/twin_count.json (+ stdout log).
"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
          "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
          "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if p not in sys.path:
        sys.path.insert(0, p)

spec = importlib.util.spec_from_file_location(
    "fitmod", HERE / "04_fit_comb_to_measured_spectrum.py")
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)

from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402

NZ = fit.NOISES["heavy"]
EPS = float(compute_epsilon(NZ["sigma_res"], NZ["sigma_scatt"]))
TOLS = (6, 8, 10)          # rounding decimals for the formation match
WTOL = 1e-10               # excitation coefficient cutoff

out = {"config": dict(T=fit.T, noise="heavy", eps=EPS, beta=0.0, gmode="ref"),
       "reps": {}}

for rep in (0, 1, 2):
    sysd = fit.build_system(rep, 0.0, "ref", EPS, NZ)
    truth = sysd["truth"]
    n = sysd["n"]
    s_iso = sysd["s_p"] - 0.0  # isolated diagonal = g_use + delta = 4.0 at ref
    rep_out = {"n_true": int(truth.sum()), "n_seg": int(n)}
    for dec in TOLS:
        # formation signature per segment
        sig = [None] * n
        # isolated segments: formation {(s, 1)}
        iso = sysd["iso"]
        iso_sig = ((round(float(s_iso), dec), 1.0),)
        for i in np.flatnonzero(iso):
            sig[i] = iso_sig
        for c, (idx, lam, V) in sysd["comps"].items():
            beta_k = V.T @ np.ones(len(idx))
            W = np.abs(V * beta_k[None, :])          # |beta_k V[i,k]|
            for li, gi in enumerate(idx):
                nzk = np.flatnonzero(W[li] > WTOL)
                sig[gi] = tuple(sorted(
                    (round(float(lam[k]), dec), round(float(W[li, k]), dec))
                    for k in nzk))
        false_sigs = set(sig[i] for i in np.flatnonzero(~truth))
        floor_true = [int(i) for i in np.flatnonzero(truth)
                      if sig[i] in false_sigs]
        n_iso_true = int(sum(1 for i in floor_true if iso[i]))
        rep_out[f"tol_1e-{dec}"] = dict(
            n_floor_true=len(floor_true),
            n_isolated_stub=n_iso_true,
            n_multi_seg=len(floor_true) - n_iso_true)
        print(f"[rep {rep} tol=1e-{dec}] floor trues = {len(floor_true)} "
              f"of {int(truth.sum())} (isolated stubs {n_iso_true}, "
              f"multi-segment {len(floor_true) - n_iso_true})", flush=True)
    out["reps"][rep] = rep_out

OUT = HERE / "outputs" / "fork_noisy"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "twin_count.json").write_text(json.dumps(out, indent=2))
print("saved", OUT / "twin_count.json")
