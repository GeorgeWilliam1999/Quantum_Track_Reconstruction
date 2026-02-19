#!/usr/bin/env python3
"""
aggregate.py – Merge per-batch metrics.csv files into a single all_metrics.csv.

Also computes summary statistics grouped by parameter configuration
(mean ± std over repeats) and writes summary_metrics.csv.

Usage:
    python aggregate.py --runs-dir /path/to/results
    python aggregate.py --runs-dir /path/to/results --batch 42   # one batch only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def collect_metrics(runs_dir: Path, batch: int | None = None) -> pd.DataFrame:
    """
    Scan batch_*/metrics.csv under runs_dir and concatenate into one DataFrame.
    If batch is given, only that batch is loaded.
    """
    frames = []

    if batch is not None:
        pattern = [runs_dir / f"batch_{batch}" / "metrics.csv"]
    else:
        pattern = sorted(runs_dir.glob("batch_*/metrics.csv"))

    for csv_path in pattern:
        if not csv_path.exists():
            print(f"[WARN] Missing: {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path)
            # Store which batch this came from
            batch_id = csv_path.parent.name.split("_", 1)[1]
            df["batch"] = int(batch_id)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {csv_path}: {e}")

    if not frames:
        print("[ERROR] No metrics files found.", file=sys.stderr)
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by physics parameters (excluding repeat) and compute
    mean, std, min, max for every metric column.
    """
    group_cols = [
        "hit_res", "multi_scatter", "ghost_rate", "drop_rate",
        "scale", "n_tracks",
    ]
    # Only keep columns that actually exist
    group_cols = [c for c in group_cols if c in df.columns]

    metric_cols = [c for c in df.columns if c.startswith("m_")]

    agg_dict = {}
    for mc in metric_cols:
        agg_dict[mc] = ["mean", "std", "min", "max", "count"]

    # Also aggregate count-like columns
    for extra in ["n_reco_tracks", "n_truth_tracks", "n_segments",
                  "n_hits_truth", "n_hits_noisy"]:
        if extra in df.columns:
            agg_dict[extra] = ["mean", "std"]

    summary = df.groupby(group_cols).agg(agg_dict)
    # Flatten multi-level columns
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Velo scan metrics.")
    ap.add_argument("--runs-dir", type=Path, required=True,
                    help="Root results directory containing batch_* folders")
    ap.add_argument("--batch", type=int, default=None,
                    help="Optional: aggregate only this batch")
    ap.add_argument("--output", type=Path, default=None,
                    help="Override output directory (default: runs-dir)")
    args = ap.parse_args()

    runs_dir = args.runs_dir.resolve()
    out_dir = (args.output or runs_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Collecting metrics from {runs_dir} ...")
    df = collect_metrics(runs_dir, batch=args.batch)
    print(f"[INFO] Loaded {len(df)} metric rows")

    # ── all_metrics.csv ─────────────────────────────────────────
    all_path = out_dir / "all_metrics.csv"
    df.to_csv(all_path, index=False)
    print(f"[OK] {all_path}  ({len(df)} rows)")

    # ── summary_metrics.csv ─────────────────────────────────────
    summary = summarise(df)
    sum_path = out_dir / "summary_metrics.csv"
    summary.to_csv(sum_path, index=False)
    print(f"[OK] {sum_path}  ({len(summary)} configs)")


if __name__ == "__main__":
    main()
