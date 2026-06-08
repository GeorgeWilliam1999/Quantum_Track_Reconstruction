#!/usr/bin/env python3
"""Generate the params CSV for the segment-level condor scan up to n_trk = 500.

Convention (per user request):
  * n_trk in [2, 100]   → up to 5 reps, each row is still its own condor job.
  * n_trk in (100, 500] → 1 job per event (single rep).

Columns (no header — condor `queue` reads positionally):
  n_trk, rep, gamma, delta, eps, tau, save_vec
"""
from __future__ import annotations
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'params_segment_full.csv'

# Track-multiplicity grid
N_TRK_LOW  = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100]
N_TRK_HIGH = [125, 150, 200, 250, 300, 400, 500]

# Reps per multiplicity (low end already covered by the in-notebook sweep,
# but condor is cheap so we re-run for full provenance).
REPS_LOW = {n: 5 for n in N_TRK_LOW if n <= 10}
REPS_LOW.update({n: 3 for n in [15, 20, 25]})
REPS_LOW.update({n: 2 for n in [30, 40, 50]})
REPS_LOW.update({n: 1 for n in [75, 100]})

GAMMAS  = [1.0, 2.0]
DELTA   = 1.0
EPS     = 0.002
TAU     = 0.35

rows = []
for g in GAMMAS:
    for n in N_TRK_LOW:
        for r in range(REPS_LOW[n]):
            rows.append([n, r, g, DELTA, EPS, TAU, 1])
    # Above 100 tracks: one event = one job, single rep.
    for n in N_TRK_HIGH:
        rows.append([n, 0, g, DELTA, EPS, TAU, 1])

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    for row in rows:
        w.writerow(row)

print(f"wrote {len(rows)} rows → {OUT}")

# Quick summary
from collections import defaultdict
per_g = defaultdict(int)
for n, r, g, *_ in rows:
    per_g[g] += 1
for g, c in sorted(per_g.items()):
    print(f"  γ={g}: {c} jobs")
