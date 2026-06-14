#!/usr/bin/env python3
"""
Build the metrics VIEW from the solution store (regenerable, never canonical).

Walks ``manifest/solutions.csv``, processes every solution that exists on disk,
recomputes segment metrics with ONE consistent definition, and writes
``manifest/metrics.csv`` (one tidy row per solved point).  Re-run any time as
jobs finish — it only processes what is present and is fully idempotent.

Definitions (the single source of truth — see the Notion page):
  active_i           = x_i > 0.35  (ABSOLUTE; quantum rescaled to ||x_C|| first)
  segment_efficiency = n_true_active / n_true_all
  segment_purity     = n_true_active / n_active
  segment_false_rate = n_false_active / n_active        (denominator = n_active,
                       for BOTH classical and quantum; = 1 - purity).
                       NOT the leakage rate n_false_active / n_false_all.
  cos_QC             = <x_Q, x_C> / (||x_Q|| ||x_C||)   (quantum only)

Truth is recomputed from the stored event (no A build); cached per event_key.
Quantum metrics need the matching classical solution (same event+ham) to set the
norm — quantum rows whose classical partner is not yet on disk are skipped.

Usage:  build_metrics.py [--out PATH] [--study NAME]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
_SHARED = _PKG.parent
for p in (str(_SHARED), "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if p not in sys.path:
        sys.path.insert(0, p)

import qtrk_pipeline as qp  # noqa: E402

# physics/config columns carried straight through from the solutions manifest
PASS_COLS = [
    "study", "studies", "n_trk", "rep", "sigma_scatt", "sigma_res", "phi_max",
    "hit_ineff", "ghost_rate", "epsilon", "eps_provenance", "kernel",
    "erf_sigma", "gamma", "delta", "fork_beta", "fork_eps", "solver", "device", "readout", "shots",
    "event_key", "ham_key",
]


def _metrics_for_group(rows: pd.DataFrame, truth: np.ndarray) -> list[dict]:
    """Recompute metrics for all solves of one event (rows share event_key)."""
    out: list[dict] = []
    # classical solutions for this event, keyed by ham_key (norm reference)
    classical = {}
    for r in rows[rows.solver == "classical"].itertuples(index=False):
        if qp.solution_exists(r.sol_key):
            classical[r.ham_key] = qp.load_solution(r.sol_key)["sol"]

    for r in rows.itertuples(index=False):
        if not qp.solution_exists(r.sol_key):
            continue
        d = qp.load_solution(r.sol_key)
        sol = d["sol"]
        row = {c: getattr(r, c, None) for c in PASS_COLS if hasattr(r, c)}
        row["sol_key"] = r.sol_key

        tau = qp.threshold_for(getattr(r, "gamma", 3.0), getattr(r, "delta", 1.0))
        row["threshold"] = tau
        if str(r.solver) == "classical":
            m = qp.metrics_at(sol, truth, tau)
            mwp = qp.metrics_at_wp(sol, truth, qp.WP_TARGET_EFF)
            row["cos_QC"] = None
            row["P_anc"] = None
        else:
            sol_C = classical.get(r.ham_key)
            if sol_C is None:
                continue  # need the classical partner to set the norm
            m = qp.quantum_metrics(sol, sol_C, truth, tau)
            mwp = qp.quantum_metrics_wp(sol, sol_C, truth, qp.WP_TARGET_EFF,
                                        signal_threshold=tau)
            row["cos_QC"] = m.pop("cos_QC")
            mwp.pop("cos_QC", None)
            row["P_anc"] = d.get("P_anc")
        row.update(
            n_seg=d.get("n_seg"), n_true=int(np.asarray(truth, bool).sum()),
            n_active=m["n_active"], n_true_active=m["n_true_active"],
            n_false_active=m["n_false_active"],
            segment_efficiency=m["segment_efficiency"],
            segment_purity=m["segment_purity"],
            segment_false_rate=m["segment_false_rate"],
            A_nnz=d.get("A_nnz"), t_solve=d.get("t_solve"),
        )
        # wp99 high-efficiency working point (HEADLINE; see metrics.working_point_threshold)
        row.update(
            target_eff=qp.WP_TARGET_EFF, tau_wp=mwp["tau_wp"],
            n_active_wp=mwp["n_active"], n_true_active_wp=mwp["n_true_active"],
            n_false_active_wp=mwp["n_false_active"],
            segment_efficiency_wp=mwp["segment_efficiency"],
            segment_purity_wp=mwp["segment_purity"],
            segment_false_rate_wp=mwp["segment_false_rate"],
        )
        out.append(row)
    return out


def build(study: str | None = None) -> pd.DataFrame:
    md = qp.manifest_dir()
    solves = pd.read_csv(md / "solutions.csv")
    if study:
        solves = solves[solves.study == study]

    rows: list[dict] = []
    t0 = time.time()
    groups = list(solves.groupby("event_key"))
    for i, (ekey, g) in enumerate(groups):
        # skip rows whose solution npz is absent (planned-but-unsolved entries)
        g = g[g.sol_key.map(qp.solution_exists)]
        if g.empty:
            continue
        ev = qp.load_event(qp.event_path(ekey))
        truth = qp.truth_from_event(ev)
        rows.extend(_metrics_for_group(g, truth))
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(groups)} events, {len(rows)} rows, "
                  f"{time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV (default $QTRK_STORE/manifest/metrics.csv)")
    ap.add_argument("--study", type=str, default=None)
    args = ap.parse_args()

    df = build(args.study)
    out = args.out or (qp.manifest_dir() / "metrics.csv")
    df.to_csv(out, index=False)
    n_q = int((df.solver == "quantum").sum()) if len(df) else 0
    print(f"[build_metrics] {len(df)} rows ({len(df)-n_q} classical, {n_q} quantum)"
          f" -> {out}")
    if len(df):
        print(df.groupby("study")[["segment_efficiency", "segment_false_rate"]]
              .mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
