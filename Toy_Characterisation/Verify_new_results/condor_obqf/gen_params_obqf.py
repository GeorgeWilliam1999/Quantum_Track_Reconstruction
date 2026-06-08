#!/usr/bin/env python3
"""
Emit a CSV of (n_trk, rep, shots, device) rows for the §18b/c 1BQF condor sweep.

Usage
-----
    python gen_params_obqf.py --device CPU --out params_cpu.csv \\
        --n-trk-min 8 --n-trk-max 40 --n-trk-step 4 --reps 3

    python gen_params_obqf.py --device GPU --out params_gpu.csv \\
        --n-trk-min 40 --n-trk-max 100 --n-trk-step 10 --reps 3

One row per job.  Columns: n_trk, rep, shots, device.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', required=True, choices=['CPU', 'GPU'])
    p.add_argument('--shots', type=int, default=8192)
    p.add_argument('--n-trk-min',  type=int, default=8)
    p.add_argument('--n-trk-max',  type=int, default=40)
    p.add_argument('--n-trk-step', type=int, default=4)
    p.add_argument('--reps',       type=int, default=3)
    args = p.parse_args()

    rows = []
    for n in range(args.n_trk_min, args.n_trk_max + 1, args.n_trk_step):
        for rep in range(args.reps):
            rows.append((n, rep, args.shots, args.device))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        w = csv.writer(f)
        # NOTE: condor `queue from` does NOT want a header row.
        for row in rows:
            w.writerow(row)

    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"  n_trk: {args.n_trk_min}..{args.n_trk_max} step {args.n_trk_step}")
    print(f"  reps:  {args.reps}  shots: {args.shots}  device: {args.device}")


if __name__ == '__main__':
    main()
