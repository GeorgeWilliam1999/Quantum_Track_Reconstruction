#!/usr/bin/env python3
"""Degree-1 version of the fit: move the 1BQF's single root onto the MEASURED
false pile-up line (George's review flow applied to the notch itself).

The standard 1BQF places its one root at lam = s (the isolated-false line) via
t = pi/s.  At heavy noise the measured false-ACTIVE pile-up is instead at the
pair line lam = s-1 (and s-sqrt2).  A retuned evolution time t = pi/(s-1)
moves the root there; the isolated false then pass at |cos(s*t/2)|*delta —
e.g. t=pi/3, s=4: |cos(2pi/3)|*0.25 = 0.125, still under any working tau.

Evaluated exactly per-component (same machinery as study 04), heavy T=200
reps 0-2, beta=0: far at matched eff for t in {pi/s (standard), pi/(s-1)
(pair-tuned), pi/lam* (argmax measured false formation)}, each alone and
composed with the hit-uniqueness gate.  Outputs appended to fit_comb.log +
tuned_notch.csv.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/QSVT/Codesign")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fm", "/data/bfys/gscriven/Quantum_Track_Reconstruction/QSVT/Codesign/"
          "04_fit_comb_to_measured_spectrum.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)
qp = fm.qp
from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402

OUT = Path(fm.OUT)


def main():
    t0 = time.time()
    nz = fm.NOISES["heavy"]
    eps = float(compute_epsilon(nz["sigma_res"], nz["sigma_scatt"]))
    rows = []
    for rep in (0, 1, 2):
        sysd = fm.build_system(rep, 0.0, "ref", eps, nz)
        truth, s_p = sysd["truth"], sysd["s_p"]
        seg_hits = sysd["seg_hits"]
        x_cls = fm.amplitudes(sysd, lambda l: 1.0 / l)

        # measured argmax false-formation line (excluding the isolated line)
        edges = np.arange(sysd["lam_min"] - fm.BIN_W,
                          sysd["lam_max"] + 2 * fm.BIN_W, fm.BIN_W)
        D, seg_rows, tru_rows = fm.design_matrix(sysd, edges)
        tau_c = qp.threshold_for(sysd["g_use"], fm.DELTA)
        fa_rows = (x_cls[seg_rows] > tau_c) & ~tru_rows
        F_act = (D[fa_rows] ** 2).sum(0)
        lam_star = 0.5 * (edges[np.argmax(F_act)] + edges[np.argmax(F_act) + 1])

        cands = {"t=pi/s (standard)": s_p,
                 "t=pi/(s-1) (pair-tuned)": s_p - 1.0,
                 f"t=pi/lam* (lam*={lam_star:.2f})": lam_star}
        for name, root in cands.items():
            t_evo = np.pi / root
            x = fm.amplitudes(sysd, lambda l: np.abs(np.cos(l * t_evo / 2)))
            fe = fm.far_at_eff(x, truth)
            ge, gf = fm.gate_frontier(x, truth, seg_hits)
            row = dict(rep=rep, notch=name, root=root,
                       iso_level=float(abs(np.cos(s_p * t_evo / 2)) * fm.DELTA
                                       / s_p))
            for tgt, (e_, f_, _) in fe.items():
                row[f"far_e{int(tgt*1000)}"] = f_
            for tgt in fm.EFF_TARGETS:
                ok = ge >= tgt - 1e-9
                row[f"far_gate_e{int(tgt*1000)}"] = (float(gf[ok].min())
                                                     if ok.any() else np.nan)
            rows.append(row)
            print(f"[rep{rep}] {name:28s} far@.99/.985/.98/.97 = "
                  f"{row['far_e990']:.3f}/{row['far_e985']:.3f}/"
                  f"{row['far_e980']:.3f}/{row['far_e970']:.3f}  "
                  f"gate@.985 {row['far_gate_e985']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tuned_notch.csv", index=False)
    print(f"[done] {time.time()-t0:.0f}s -> tuned_notch.csv", flush=True)


if __name__ == "__main__":
    main()
