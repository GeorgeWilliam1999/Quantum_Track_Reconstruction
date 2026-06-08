#!/usr/bin/env python3
"""Generate top-up params CSVs for Epsilon_study_2.

Two categories of output:

1. ``eps2_topup_mem32.csv``  — T=200 statevector fill-in for (sigma_res=0.05)
   cells missing reps on disk.  CPU, run_quantum=1, 32 GB.
   Uses _shared/submit_base.sub (15 columns, no device column).

2. ``eps2_topup_gpu_mem64.csv``  — T ∈ {400, 700, 1000} using AerSimulator
   GPU (CUDA) statevector — same approach as Verify_new_results submit_gpu.sub.
   Empirical timing from that study: T=1000 → ~5400 s on GPU (fits in 24 h).
   Uses submit_gpu.sub (16 columns, device column appended).
   64 GB memory follows the Verify_new_results tier for T=500–1499.

All jobs still use run_quantum=1 so the quantum solution is computed.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "Toy_Characterisation" / "_shared"))
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

from lhcb_velo_toy.analysis import compute_epsilon

# ── fixed physics / solver params ────────────────────────────────────────────
PHI_MAX     = 0.2
HIT_INEFF   = 0.0
CONVOLUTION = 0
ERF_SIGMA   = 1e-4
GAMMA       = 3.0
DELTA       = 1.0
TAU_DEFAULT = 0.35

VAL_POINT        = (0.0, 1e-4)
SENSITIVITY_GRID = [(sr, ss) for sr in (0.0, 0.01, 0.02, 0.05)
                              for ss in (1e-4, 3e-4, 5e-4)]

OUTROOT   = HERE / "results"
VAL_DIR   = OUTROOT / "validation"
SENS_ROOT = OUTROOT / "sensitivity"

FIELDS_CPU = ["n_trk", "rep", "sigma_res", "sigma_scatt", "phi_max",
              "hit_ineff", "convolution", "erf_sigma", "epsilon",
              "gamma", "delta", "tau_default", "run_quantum", "tag", "outdir"]
FIELDS_GPU = FIELDS_CPU + ["device"]


def pkl_name(tag: str, T: int, rep: int) -> str:
    return f"{tag}_n{T:04d}_r{rep:03d}.pkl"


def cells():
    """Yield (tag, sr, ss, outdir) for all 13 output locations."""
    yield ("eps2val", VAL_POINT[0], VAL_POINT[1], VAL_DIR)
    for sr, ss in SENSITIVITY_GRID:
        gtag = f"sr{sr:g}_ss{ss:g}"
        yield (f"eps2_{gtag}", sr, ss, SENS_ROOT / gtag)


def make_row(T, rep, sr, ss, tag, outdir, device=None):
    row = dict(
        n_trk=T, rep=rep,
        sigma_res=sr, sigma_scatt=ss,
        phi_max=PHI_MAX, hit_ineff=HIT_INEFF,
        convolution=CONVOLUTION, erf_sigma=ERF_SIGMA,
        epsilon=compute_epsilon(sr, ss),
        gamma=GAMMA, delta=DELTA, tau_default=TAU_DEFAULT,
        run_quantum=1,
        tag=tag, outdir=str(outdir),
    )
    if device is not None:
        row["device"] = device
    return row


# ── category 1: T=200 CPU statevector top-up ─────────────────────────────────
rows_200 = []
for tag, sr, ss, outdir in cells():
    outdir.mkdir(parents=True, exist_ok=True)
    for rep in range(5):
        if not (outdir / pkl_name(tag, 200, rep)).exists():
            rows_200.append(make_row(200, rep, sr, ss, tag, outdir))

# ── category 2: T=400/700/1000 GPU quantum jobs ───────────────────────────────
# Empirical from Verify_new_results (wn-lot-008/009, CUDA):
#   T=1000, n_seg≈3.9 M → t_q ≈ 5384 s (statevector) < 24 h MaxRuntime
# remove the currently queued classical-only cluster first (4789479).
T_GPU   = [400, 700, 1000]
N_REPS  = 3  # reps_for_T(T≥400) per experiment plan

rows_gpu = []
for T in T_GPU:
    for tag, sr, ss, outdir in cells():
        outdir.mkdir(parents=True, exist_ok=True)
        for rep in range(N_REPS):
            if not (outdir / pkl_name(tag, T, rep)).exists():
                rows_gpu.append(make_row(T, rep, sr, ss, tag, outdir, device='GPU'))

# ── write CSVs ────────────────────────────────────────────────────────────────
PARAMS_DIR = HERE / "params"
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

p200 = PARAMS_DIR / "eps2_topup_mem32.csv"
with p200.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS_CPU)
    for r in rows_200:
        w.writerow(r)
print(f"[topup] T=200 CPU statevector  -> {len(rows_200):4d} rows -> {p200}")

p_gpu = PARAMS_DIR / "eps2_topup_gpu_mem128.csv"
with p_gpu.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS_GPU)
    for r in rows_gpu:
        w.writerow(r)
print(f"[topup] T=400/700/1000 GPU     -> {len(rows_gpu):4d} rows -> {p_gpu}")
by_T: dict[int, int] = {}
for r in rows_gpu:
    by_T[r["n_trk"]] = by_T.get(r["n_trk"], 0) + 1
for T, n in sorted(by_T.items()):
    print(f"         T={T:4d}  jobs={n}")
print(f"[topup] total = {len(rows_200) + len(rows_gpu)} jobs")
