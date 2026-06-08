#!/usr/bin/env python3
"""
Generate a higher-shot params CSV for the §14e sampling sweep so the
sampling-based quantum efficiency approaches the statevector limit (E1).

Decision (May 2026, user-confirmed):
  * T in [50, 100, 200, 500]: shots x16 vs current tier, 30 reps each.
  * T = 1000                : shots x8 (not x16) and 10 reps, to keep
                              per-job wallclock under the typical "long"
                              category runtime cap.

Columns (no header — condor `queue` reads positionally):
  n_trk, rep, shots, device, drop_rate, gamma, readout
"""
from __future__ import annotations
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent / "params_seg14e_T1000_sampling_hires.csv"

# (n_trk, shots_old, scale, n_reps)
TIERS = [
    (  50,    250_000, 16, 30),
    ( 100,  1_000_000, 16, 30),
    ( 200,  4_000_000, 16, 30),
    ( 500, 25_000_000, 16, 30),
    (1000,100_000_000,  8, 10),
]

DEVICE    = "GPU"
DROP_RATE = 0.01
GAMMA     = 3.0
READOUT   = "sampling"

rows = []
for n_trk, shots_old, scale, n_reps in TIERS:
    shots = int(shots_old * scale)
    for rep in range(n_reps):
        rows.append([n_trk, rep, shots, DEVICE, DROP_RATE, GAMMA, READOUT])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    for row in rows:
        w.writerow(row)

print(f"wrote {len(rows)} rows -> {OUT}")
from collections import defaultdict
per_T = defaultdict(int)
for r in rows:
    per_T[(r[0], r[2])] += 1
for (T, s), c in sorted(per_T.items()):
    print(f"  T={T:4d}  shots={s:>11d}  reps={c}")
