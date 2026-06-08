#!/usr/bin/env python3
"""
Universal aggregator for the segment-level experiments.

Reads all ``*.pkl`` files under ``--indir`` (recursively) and emits:

- ``<outprefix>_events.csv`` : one row per pickle, all scalar fields
  including angle/solver metrics flattened with ``angle_`` and
  ``cls_default_`` / ``q_default_`` prefixes.
- ``<outprefix>_summary.csv`` : per (tag, n_trk, *grid_keys) aggregate
  (mean / stderr / count). Grid keys are inferred from the columns the
  caller passes via ``--group-keys`` (CSV, e.g. ``sigma_scatt,hit_ineff``).
- Optionally ``<outprefix>_angles.parquet`` (if pyarrow available): raw
  per-pair true/false angle samples for histogram-based threshold
  studies, with one row per pair (long format).

Designed to be re-runnable: existing outputs are overwritten.
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _flatten_metrics(prefix: str, d):
    if d is None:
        return {}
    return {f"{prefix}_{k}": v for k, v in d.items()}


def load_records(indir: Path) -> pd.DataFrame:
    rows = []
    for pkl in sorted(indir.rglob("*.pkl")):
        try:
            with open(pkl, "rb") as f:
                r = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to load {pkl}: {exc}")
            continue
        scalar = {k: v for k, v in r.items()
                  if not isinstance(v, (np.ndarray, list, dict))}
        scalar.update(_flatten_metrics("angle", r.get("angle_metrics")))
        scalar.update(_flatten_metrics("cls_default", r.get("classical_metrics_default")))
        scalar.update(_flatten_metrics("q_default", r.get("quantum_metrics_default")))
        scalar["file"] = str(pkl)
        rows.append(scalar)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, group_keys, tag_col="tag"):
    cols = [tag_col, "n_trk"] + [k for k in group_keys if k in df.columns]
    cols = [c for c in cols if c in df.columns]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in cols]
    if not cols:
        cols = ["n_trk"]
    g = df.groupby(cols, dropna=False)
    mean = g[numeric].mean().add_suffix("_mean")
    std = g[numeric].std(ddof=1).add_suffix("_std")
    count = g[numeric].count().add_suffix("_n")
    out = pd.concat([mean, std, count], axis=1).reset_index()
    # stderr = std / sqrt(n)
    for c in numeric:
        n = out[f"{c}_n"].clip(lower=1)
        out[f"{c}_sem"] = out[f"{c}_std"] / np.sqrt(n)
    return out


def angle_long(indir: Path) -> pd.DataFrame:
    chunks = []
    for pkl in sorted(indir.rglob("*.pkl")):
        try:
            with open(pkl, "rb") as f:
                r = pickle.load(f)
        except Exception:
            continue
        ta = r.get("true_angles")
        fa = r.get("false_angles")
        if ta is None and fa is None:
            continue
        meta = {k: r.get(k) for k in ("tag", "n_trk", "rep",
                                       "sigma_res", "sigma_scatt",
                                       "phi_max", "hit_ineff",
                                       "convolution", "erf_sigma",
                                       "epsilon")}
        if ta is not None and len(ta):
            d = pd.DataFrame({"angle": np.asarray(ta), "label": 1})
            for k, v in meta.items():
                d[k] = v
            chunks.append(d)
        if fa is not None and len(fa):
            d = pd.DataFrame({"angle": np.asarray(fa), "label": 0})
            for k, v in meta.items():
                d[k] = v
            chunks.append(d)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, type=Path)
    ap.add_argument("--outprefix", required=True, type=Path,
                    help="prefix path (e.g. /.../results/eps2)")
    ap.add_argument("--group-keys", default="",
                    help="comma-separated grid columns (e.g. sigma_scatt,hit_ineff)")
    ap.add_argument("--emit-angles", action="store_true",
                    help="also write a long-format parquet of pair angles")
    args = ap.parse_args()

    args.outprefix.parent.mkdir(parents=True, exist_ok=True)
    df = load_records(args.indir)
    print(f"[agg] loaded {len(df)} records from {args.indir}")
    if df.empty:
        return 1
    events_csv = args.outprefix.with_name(args.outprefix.name + "_events.csv")
    df.to_csv(events_csv, index=False)
    print(f"[agg] wrote {events_csv}")

    keys = [k.strip() for k in args.group_keys.split(",") if k.strip()]
    summary = summarise(df, keys)
    summary_csv = args.outprefix.with_name(args.outprefix.name + "_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"[agg] wrote {summary_csv}")

    if args.emit_angles:
        ang = angle_long(args.indir)
        if ang.empty:
            print("[agg] no angle arrays in pickles; skipping parquet")
        else:
            ang_pq = args.outprefix.with_name(args.outprefix.name + "_angles.parquet")
            try:
                ang.to_parquet(ang_pq, index=False)
                print(f"[agg] wrote {ang_pq} ({len(ang)} rows)")
            except Exception as exc:
                print(f"[warn] parquet failed ({exc}); writing CSV instead")
                ang_csv = ang_pq.with_suffix(".csv")
                ang.to_csv(ang_csv, index=False)
                print(f"[agg] wrote {ang_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
