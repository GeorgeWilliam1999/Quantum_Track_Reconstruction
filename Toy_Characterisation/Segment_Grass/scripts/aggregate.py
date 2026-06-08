#!/usr/bin/env python3
"""
Aggregate results from all segment-grass sweep jobs.

Merges per-job results.json and arrays.npz into a single aggregated
dataset for the analysis notebook.

Usage
-----
    python aggregate.py --results-dir /path/to/results
"""

import argparse
import json
from pathlib import Path

import numpy as np


def aggregate(results_dir: str):
    results_dir = Path(results_dir)
    agg_dir = results_dir / "aggregated"
    agg_dir.mkdir(exist_ok=True)

    # Load run summary
    with open(results_dir / "run_summary.json") as f:
        summary = json.load(f)

    total_jobs = summary["total_jobs"]
    all_results = []
    n_done = 0
    n_failed = 0

    for job_id in range(total_jobs):
        job_dir = results_dir / f"job_{job_id:05d}"
        if (job_dir / "DONE").exists():
            n_done += 1
            with open(job_dir / "results.json") as f:
                all_results.append(json.load(f))
        elif (job_dir / "FAILED").exists():
            n_failed += 1
        # else: still running or missing

    print(f"Status: {n_done} done, {n_failed} failed, "
          f"{total_jobs - n_done - n_failed} pending")

    if n_done == 0:
        print("No completed jobs to aggregate.")
        return

    # Build structured result tables
    # Key: (angle, n_tracks, ghost_rate, drop_rate)
    table = {}
    for r in all_results:
        p = r["params"]
        key = (p["angle"], p["n_tracks"], p["ghost_rate"], p["drop_rate"])
        table[key] = {
            "occ_mean": r["occ_mean"],
            "occ_std": r["occ_std"],
            "n_anomalous_occ": r["n_anomalous_occ"],
            "frac_anomalous_occ": r["frac_anomalous_occ"],
            "mean_n_clipped": r["mean_n_clipped"],
            "mean_n_ghost_hits": r["mean_n_ghost_hits"],
            "mean_n_dropped": r["mean_n_dropped"],
            "mean_n_segments": r["mean_n_segments"],
            "efficiency": r["reco_metrics"]["efficiency"]["mean"],
            "efficiency_std": r["reco_metrics"]["efficiency"]["std"],
            "reco_ghost_rate": r["reco_metrics"]["ghost_rate"]["mean"],
            "reco_ghost_rate_std": r["reco_metrics"]["ghost_rate"]["std"],
            "clone_fraction": r["reco_metrics"]["clone_fraction"]["mean"],
            "mean_true_activation": r.get("mean_true_activation", 0.0),
            "mean_false_activation": r.get("mean_false_activation", 0.0),
            "hits_per_mod_mean": r["hits_per_mod_mean"],
            # Selected-segment corpus
            "occ_sel_mean": r.get("occ_sel_mean", 0.0),
            "occ_sel_std": r.get("occ_sel_std", 0.0),
            "n_anomalous_occ_sel": r.get("n_anomalous_occ_sel", 0),
            "frac_anomalous_occ_sel": r.get("frac_anomalous_occ_sel", 0.0),
            "mean_n_segments_sel": r.get("mean_n_segments_sel", 0.0),
            "mean_n_true_sel": r.get("mean_n_true_sel", 0.0),
            "mean_n_false_sel": r.get("mean_n_false_sel", 0.0),
            "mean_true_activation_sel": r.get("mean_true_activation_sel", 0.0),
            "mean_false_activation_sel": r.get("mean_false_activation_sel", 0.0),
            "hits_sel_per_mod_mean": r.get("hits_sel_per_mod_mean", {}),
        }

    # Save aggregated
    with open(agg_dir / "sweep_results.json", "w") as f:
        json.dump({
            "summary": summary,
            "status": {"done": n_done, "failed": n_failed,
                       "total": total_jobs},
            "results": {str(k): v for k, v in table.items()},
        }, f, indent=2)

    # Also save as a flat CSV-like JSON for easy pandas loading
    rows = []
    for r in all_results:
        p = r["params"]
        row = {
            "angle": p["angle"],
            "n_tracks": p["n_tracks"],
            "ghost_rate": p["ghost_rate"],
            "drop_rate": p["drop_rate"],
            "occ_mean": r["occ_mean"],
            "occ_std": r["occ_std"],
            "n_anomalous_occ": r["n_anomalous_occ"],
            "frac_anomalous_occ": r["frac_anomalous_occ"],
            "mean_n_clipped": r["mean_n_clipped"],
            "mean_n_ghost_hits": r["mean_n_ghost_hits"],
            "mean_n_dropped": r["mean_n_dropped"],
            "mean_n_segments": r["mean_n_segments"],
            "efficiency": r["reco_metrics"]["efficiency"]["mean"],
            "efficiency_std": r["reco_metrics"]["efficiency"]["std"],
            "reco_ghost_rate": r["reco_metrics"]["ghost_rate"]["mean"],
            "reco_ghost_rate_std": r["reco_metrics"]["ghost_rate"]["std"],
            "clone_fraction": r["reco_metrics"]["clone_fraction"]["mean"],
            "mean_true_activation": r.get("mean_true_activation", 0.0),
            "mean_false_activation": r.get("mean_false_activation", 0.0),
        }
        for m in range(5):
            row[f"hits_mod{m}_mean"] = r["hits_per_mod_mean"].get(str(m), 0)
        # Selected-segment corpus
        row.update({
            "occ_sel_mean": r.get("occ_sel_mean", 0.0),
            "occ_sel_std": r.get("occ_sel_std", 0.0),
            "n_anomalous_occ_sel": r.get("n_anomalous_occ_sel", 0),
            "frac_anomalous_occ_sel": r.get("frac_anomalous_occ_sel", 0.0),
            "mean_n_segments_sel": r.get("mean_n_segments_sel", 0.0),
            "mean_n_true_sel": r.get("mean_n_true_sel", 0.0),
            "mean_n_false_sel": r.get("mean_n_false_sel", 0.0),
            "mean_true_activation_sel": r.get("mean_true_activation_sel", 0.0),
            "mean_false_activation_sel": r.get("mean_false_activation_sel", 0.0),
        })
        for m in range(5):
            row[f"hits_sel_mod{m}_mean"] = r.get(
                "hits_sel_per_mod_mean", {}).get(str(m), 0)
        rows.append(row)

    with open(agg_dir / "sweep_flat.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Aggregated {n_done} jobs → {agg_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    aggregate(args.results_dir)
