#!/usr/bin/env python3
"""Extend the QSVT third-solver coverage to the full study suite.

Mirrors every 1BQF (quantum, statevector) solve in the store as a
solver='qsvt' row — same event_key/ham_key, the d=40 gamma-aware line-comb
inverse — across ERF (erf-kernel spectra: a direct comb-robustness test),
Epsilon_study_2, Larger_Scatter, Larger_Scatter_Density and the realistic-
noise pair (Noisy_Ladder / Noisy_Density).  Verify_new_results already has
qsvt coverage (qsvt_store_campaign.py).

Rows are APPENDED to manifest/solutions.csv (dedup on sol_key; build_manifest
is additive since 2026-07-03 so they persist).  Solving happens on the Condor
'qsvt' lane:  build_submission.py --lanes qsvt --only-missing --suffix _qsvt.

Usage: qsvt_extend_campaign.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp  # noqa: E402

STUDIES = ["ERF", "Epsilon_study_2", "Larger_Scatter", "Larger_Scatter_Density",
           "Noisy_Ladder", "Noisy_Density"]


def main(dry_run: bool = False) -> int:
    md = qp.manifest_dir()
    sols = pd.read_csv(md / "solutions.csv")
    have = set(sols.sol_key)
    q = sols[(sols.solver == "quantum") & (sols.readout == "statevector") &
             (sols.study.isin(STUDIES))]
    q = q.drop_duplicates(subset=["event_key", "ham_key"])
    new_rows = []
    for r in q.itertuples(index=False):
        key = qp.sol_key(r.event_key, r.ham_key, "qsvt", readout="statevector")
        if key in have:
            continue
        row = r._asdict()
        row.update(sol_key=key, solver="qsvt", device="na",
                   readout="statevector", shots=0)
        new_rows.append(row)
    nd = pd.DataFrame(new_rows)
    print(f"[plan] {len(nd)} new qsvt rows")
    if len(nd):
        print(nd.groupby(["study", "n_trk"]).size().unstack(fill_value=0).to_string())
    if dry_run or not len(nd):
        return 0
    merged = pd.concat([sols, nd[sols.columns]], ignore_index=True)
    merged = merged.drop_duplicates(subset="sol_key", keep="first")
    import shutil
    shutil.copy(md / "solutions.csv", md / "solutions.csv.prev")
    merged.to_csv(md / "solutions.csv", index=False)
    print(f"[plan] solutions.csv: {len(sols)} -> {len(merged)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(ap.parse_args().dry_run))
