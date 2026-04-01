#!/usr/bin/env python3
"""
Aggregate results from all Condor pipeline jobs into unified data files.

Reads per-job output directories (job_XXXXX/) and merges results into:
  - aggregated/char_scan.json         (characterisation parametric scan)
  - aggregated/char_hist.npz          (characterisation acceptance histograms)
  - aggregated/char_bulk_angles.npz   (100×50 angle distributions)
  - aggregated/hc_density.json        (hit-competition Steps 1-4 metrics)
  - aggregated/hc_density_arrays.npz  (occupancy/activation/competition arrays)
  - aggregated/hc_roc.json            (hit-competition Step 5 ROC data)
  - aggregated/hc_scatt.json          (hit-competition Step 6 scattering scan)
  - aggregated/hc_scatt_hist.npz      (hit-competition Step 6 histograms)
  - aggregated/run_status.json        (completion status of all jobs)

Usage
-----
    python aggregate.py --results-dir /path/to/results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)


def aggregate(results_dir):
    results_dir = Path(results_dir)
    agg_dir = results_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Load run summary for parameters reference
    summary_path = results_dir / "run_summary.json"
    if summary_path.exists():
        run_summary = load_json(summary_path)
    else:
        run_summary = {}

    # Discover all job directories
    job_dirs = sorted(results_dir.glob("job_*/"))
    print(f"Found {len(job_dirs)} job directories")

    # Categorise by task
    tasks = {}  # {task_type: [list of (job_dir, results_dict)]}
    done_count = 0
    fail_count = 0
    missing_count = 0

    for jd in job_dirs:
        done_file = jd / "DONE"
        fail_file = jd / "FAILED"
        results_file = jd / "results.json"

        if fail_file.exists():
            fail_count += 1
            continue
        if not done_file.exists():
            missing_count += 1
            continue
        if not results_file.exists():
            fail_count += 1
            continue

        done_count += 1
        data = load_json(results_file)
        task = data["task"]
        tasks.setdefault(task, []).append((jd, data))

    status = {
        "total_jobs": len(job_dirs),
        "completed": done_count,
        "failed": fail_count,
        "missing": missing_count,
        "tasks_found": {k: len(v) for k, v in tasks.items()},
    }
    save_json(status, agg_dir / "run_status.json")
    print(f"Status: {done_count} done, {fail_count} failed, {missing_count} pending")
    for t, jobs in tasks.items():
        print(f"  {t:20s}: {len(jobs)} jobs")

    # ─── Aggregate each task type ─────────────────────────────────
    if "char_scan" in tasks:
        _agg_char_scan(tasks["char_scan"], agg_dir)

    if "char_hist" in tasks:
        _agg_char_hist(tasks["char_hist"], agg_dir)

    if "char_bulk_angles" in tasks:
        _agg_char_bulk_angles(tasks["char_bulk_angles"], agg_dir)

    if "hc_density" in tasks:
        _agg_hc_density(tasks["hc_density"], agg_dir)

    if "hc_roc" in tasks:
        _agg_hc_roc(tasks["hc_roc"], agg_dir)

    if "hc_scatt" in tasks:
        _agg_hc_scatt(tasks["hc_scatt"], agg_dir)

    if "hc_scatt_hist" in tasks:
        _agg_hc_scatt_hist(tasks["hc_scatt_hist"], agg_dir)

    # Save the run summary alongside aggregated data
    save_json(run_summary, agg_dir / "run_summary.json")

    print(f"\nAggregation complete. Results in: {agg_dir}")
    return status


# ═══════════════════════════════════════════════════════════════════
#  char_scan: merge by (angle, scatt_mult)
# ═══════════════════════════════════════════════════════════════════
def _agg_char_scan(jobs, agg_dir):
    # {angle_str: {mult_str: results_list}}
    merged = {}
    epsilons = {}
    for jd, data in jobs:
        angle = data["angle"]
        mult  = data["scatt_mult"]
        key_a = str(angle)
        key_m = str(mult)
        merged.setdefault(key_a, {})[key_m] = data["results"]
        epsilons[str(mult)] = data["epsilon"]

    save_json({
        "scan_data": merged,
        "epsilons": epsilons,
    }, agg_dir / "char_scan.json")
    print(f"  char_scan: aggregated {len(jobs)} jobs")


# ═══════════════════════════════════════════════════════════════════
#  char_hist: merge angle arrays by (angle, scatt_mult)
# ═══════════════════════════════════════════════════════════════════
def _agg_char_hist(jobs, agg_dir):
    arrays = {}
    for jd, data in jobs:
        angle = data["angle"]
        mult  = data["scatt_mult"]
        key = f"a{angle}_m{mult}"
        npz = np.load(str(jd / "angles.npz"))
        arrays[f"{key}_true"] = npz["true_angles"]
        arrays[f"{key}_false"] = npz["false_angles"]

    np.savez_compressed(str(agg_dir / "char_hist.npz"), **arrays)
    print(f"  char_hist: aggregated {len(jobs)} jobs")


# ═══════════════════════════════════════════════════════════════════
#  char_bulk_angles: concatenate chunks
# ═══════════════════════════════════════════════════════════════════
def _agg_char_bulk_angles(jobs, agg_dir):
    all_true = []
    all_false = []
    # Sort by event_start to maintain order
    sorted_jobs = sorted(jobs, key=lambda j: j[1]["event_start"])
    for jd, data in sorted_jobs:
        npz = np.load(str(jd / "angles.npz"))
        all_true.append(npz["true_angles"])
        all_false.append(npz["false_angles"])

    np.savez_compressed(
        str(agg_dir / "char_bulk_angles.npz"),
        all_true_angles=np.concatenate(all_true),
        all_false_angles=np.concatenate(all_false),
    )
    print(f"  char_bulk_angles: aggregated {len(jobs)} jobs "
          f"({sum(len(a) for a in all_true):,} true, "
          f"{sum(len(a) for a in all_false):,} false)")


# ═══════════════════════════════════════════════════════════════════
#  hc_density: merge Steps 1-4 by (angle, n_tracks)
# ═══════════════════════════════════════════════════════════════════
def _agg_hc_density(jobs, agg_dir):
    # JSON metrics
    all_reco = []
    occ_summary = {}  # {angle_str: {n_tracks_str: {mean, median, max, std}}}

    # NPZ arrays keyed by (angle, n_tracks)
    occ_arrays = {}
    act_arrays = {}
    comp_arrays = {}

    for jd, data in jobs:
        angle = data["angle"]
        n_trk = data["n_tracks"]
        key = f"a{angle}_n{n_trk}"

        all_reco.extend(data["reco_metrics"])

        occ_summary.setdefault(str(angle), {})[str(n_trk)] = {
            "mean": data["occ_mean"],
            "median": data["occ_median"],
            "max": data["occ_max"],
            "std": data["occ_std"],
        }

        # Load per-step arrays
        occ_npz = np.load(str(jd / "step1_occupancy.npz"))
        occ_arrays[f"{key}_occ"] = occ_npz["occupancy"]

        act_npz = np.load(str(jd / "step2_activation.npz"))
        act_arrays[f"{key}_true_x"] = act_npz["true_x"]
        act_arrays[f"{key}_false_x"] = act_npz["false_x"]

        comp_npz = np.load(str(jd / "step3_competition.npz"))
        comp_arrays[f"{key}_true_act"] = comp_npz["true_act"]
        comp_arrays[f"{key}_false_sum"] = comp_npz["false_sum"]
        comp_arrays[f"{key}_n_comp"] = comp_npz["n_competitors"]

    save_json({
        "reco_metrics": all_reco,
        "occupancy_summary": occ_summary,
    }, agg_dir / "hc_density.json")

    np.savez_compressed(str(agg_dir / "hc_density_step1.npz"), **occ_arrays)
    np.savez_compressed(str(agg_dir / "hc_density_step2.npz"), **act_arrays)
    np.savez_compressed(str(agg_dir / "hc_density_step3.npz"), **comp_arrays)

    print(f"  hc_density: aggregated {len(jobs)} jobs ({len(all_reco)} reco entries)")


# ═══════════════════════════════════════════════════════════════════
#  hc_roc: merge by (angle, n_tracks)
# ═══════════════════════════════════════════════════════════════════
def _agg_hc_roc(jobs, agg_dir):
    roc_data = {}  # {angle_str: {n_tracks_str: {thresholds, eff, ghost}}}

    for jd, data in jobs:
        angle = data["angle"]
        n_trk = data["n_tracks"]
        roc_data.setdefault(str(angle), {})[str(n_trk)] = {
            "thresholds": data["thresholds"],
            "eff": data["eff"],
            "ghost": data["ghost"],
        }

    save_json(roc_data, agg_dir / "hc_roc.json")
    print(f"  hc_roc: aggregated {len(jobs)} jobs")


# ═══════════════════════════════════════════════════════════════════
#  hc_scatt: merge by scatt_mult
# ═══════════════════════════════════════════════════════════════════
def _agg_hc_scatt(jobs, agg_dir):
    merged = {}
    epsilons = {}

    for jd, data in jobs:
        mult = data["scatt_mult"]
        merged[str(mult)] = data["results"]
        epsilons[str(mult)] = data["epsilon"]

    save_json({
        "scan_data": merged,
        "epsilons": epsilons,
    }, agg_dir / "hc_scatt.json")
    print(f"  hc_scatt: aggregated {len(jobs)} jobs")


# ═══════════════════════════════════════════════════════════════════
#  hc_scatt_hist: merge angle arrays by scatt_mult
# ═══════════════════════════════════════════════════════════════════
def _agg_hc_scatt_hist(jobs, agg_dir):
    arrays = {}
    for jd, data in jobs:
        mult = data["scatt_mult"]
        key = f"m{mult}"
        npz = np.load(str(jd / "angles.npz"))
        arrays[f"{key}_true"] = npz["true_angles"]
        arrays[f"{key}_false"] = npz["false_angles"]

    np.savez_compressed(str(agg_dir / "hc_scatt_hist.npz"), **arrays)
    print(f"  hc_scatt_hist: aggregated {len(jobs)} jobs")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Condor pipeline results")
    parser.add_argument("--results-dir", required=True,
                        help="Root results directory containing job_*/ subdirectories")
    args = parser.parse_args()

    status = aggregate(args.results_dir)

    if status["failed"] > 0:
        print(f"\nWARNING: {status['failed']} jobs failed!")
    if status["missing"] > 0:
        print(f"\nWARNING: {status['missing']} jobs still running/pending!")


if __name__ == "__main__":
    main()
