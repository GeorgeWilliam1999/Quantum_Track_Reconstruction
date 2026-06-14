#!/usr/bin/env python3
"""
Shard the qtrk manifest into Condor jobs and (optionally) submit.

Lanes (independent, different resources):
    events        event generation                         (CPU, 16 GB)
    classical     classical sparse solves                  (CPU, 16 GB)
    quantum_cpu   1BQF statevector, T<=100                 (CPU, 16 GB)
    quantum_gpu   1BQF statevector, T>=200                 (GPU, 16-64 GB)

Each lane is split into homogeneous-T shards (so memory and runtime are uniform),
a per-lane Condor submit file is written, and `--submit` fires them.  Workers are
idempotent, so re-running skips finished work.

Usage:
    build_submission.py --lanes events,classical,quantum_cpu            # write only
    build_submission.py --lanes events,classical,quantum_cpu --submit   # + condor_submit
    build_submission.py --lanes quantum_gpu --max-shards 2 --submit     # GPU pilot
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
_SHARED = _PKG.parent
sys.path.insert(0, str(_SHARED))
import qtrk_pipeline as qp  # noqa: E402

CONDOR_DIR = Path(__file__).resolve().parent
STORE = qp.store_root()
SHARD_DIR = STORE / "manifest" / "shards"
LOG_DIR = STORE / "logs"

# rows per job, by track count, per lane (keeps per-job wallclock bounded)
def _chunk(lane: str, T: int) -> int:
    if lane == "events":
        return 500 if T <= 200 else 300 if T <= 400 else 200 if T <= 700 else 120
    if lane == "classical":
        # High-T A-build is ~2/7/19 s per event at T=400/700/1000, so cap each
        # shard near ~10-15 min of work and let the cluster parallelise the rest.
        return 300 if T <= 200 else 60 if T <= 400 else 25 if T <= 700 else 15
    if lane in ("quantum_cpu", "quantum_gpu"):
        # The matrix-free OneBQF engine (default) solves in seconds even at T=1000
        # (A-build ~5 s + solve ~12 s), not the Aer hours, so pack many per job.
        # (If you force engine=aer at high T, drop these back to ~1-2/job.)
        return 200 if T <= 200 else 100 if T <= 400 else 50
    return 100


def _mem_gb(lane: str, T: int, eps: float = 0.0) -> int:
    if lane in ("events", "classical"):
        return 16                      # sparse A -> small even at T=1000
    # Quantum now uses the matrix-free OneBQF engine (helpers default), whose host
    # memory is just the 2^n statevector + the build: <=256 MB at T=1000, <1.5 GB
    # incl. A. The old eps-aware 16/24/48/96 GB tiers existed only because the Aer
    # path made the host assemble ~millions of transpiled gates (13-119 GB OOMs).
    # A flat 16 GB has ~10x headroom and schedules fast. (If you force engine=aer
    # at high T/eps you must restore the big eps-aware tiers — see SCALING_DEEP_DIVE.md.)
    return 16


def _lane_rows(lane: str, events: pd.DataFrame, solves: pd.DataFrame) -> pd.DataFrame:
    if lane == "events":
        return events
    q = solves
    if lane == "classical":
        return q[q.solver == "classical"]
    if lane == "quantum_cpu":
        return q[(q.solver == "quantum") & (q.device == "CPU")]
    if lane == "quantum_gpu":
        return q[(q.solver == "quantum") & (q.device == "GPU")]
    raise ValueError(f"unknown lane {lane}")


def shard_lane(lane: str, df: pd.DataFrame, max_shards: int | None,
               name: str) -> list[tuple[Path, int]]:
    out_dir = SHARD_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear stale shards for THIS named shard set only (never another lane's
    # live shards — a running job reads its shard CSV at runtime).
    for old in out_dir.glob("*.csv"):
        old.unlink()
    # Per-row memory: quantum scales with epsilon (circuit size), so group shards
    # by (T, mem) to keep each shard homogeneous; other lanes are flat by T.
    df = df.copy()
    if lane in ("quantum_cpu", "quantum_gpu"):
        df["_mem"] = [_mem_gb(lane, int(t), float(e))
                      for t, e in zip(df["n_trk"], df["epsilon"])]
    else:
        df["_mem"] = _mem_gb(lane, 0)
    shards: list[tuple[Path, int]] = []
    for (T, mem), g in df.groupby(["n_trk", "_mem"], sort=True):
        cs = _chunk(lane, int(T))
        g = g.reset_index(drop=True)
        for k in range(0, len(g), cs):
            chunk = g.iloc[k:k + cs].drop(columns="_mem")
            p = out_dir / f"{name}_T{int(T):04d}_m{int(mem):03d}_{k // cs:04d}.csv"
            chunk.to_csv(p, index=False)
            shards.append((p, int(mem)))
            if max_shards and len(shards) >= max_shards:
                return shards
    return shards


def write_submit(lane: str, shards: list[tuple[Path, int]],
                 name: str) -> tuple[Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    quantum = lane in ("quantum_cpu", "quantum_gpu")
    listfile = SHARD_DIR / f"{name}.list"
    with listfile.open("w") as f:
        for p, mem in shards:
            if quantum:
                # Aer memory budget = request minus 8 GB headroom. The TOTAL process
                # is Aer's budget PLUS the transpiled Qiskit circuit (~1.3 GB at
                # T=400) + Aer's internal circuit copy + numpy/qiskit baseline
                # (~4 GB overshoot observed), so 4 GB headroom OOM'd a 16 GB job at
                # 16.3 GB; 8 GB is safe.
                aer_mb = max(4096, (int(mem) - 8) * 1024)
                f.write(f"{p}, {mem}, {aer_mb}\n")
            else:
                f.write(f"{p}, {mem}\n")
    gpu = (lane == "quantum_gpu")
    qcpu = (lane == "quantum_cpu")
    ncpu = 8 if qcpu else (1 if gpu else 2)   # AerSimulator CPU is multi-threaded
    sub = SHARD_DIR / f"{name}.sub"
    lines = [
        "universe        = vanilla",
        '+UseOS          = "el9"',
        '+JobCategory    = "long"',
        f"executable      = {CONDOR_DIR/'worker.sh'}",
        'arguments       = "$(shardfile)"',
        f"request_cpus    = {ncpu}",
        "request_memory  = $(mem_gb) GB",
        "request_disk    = 4 GB",
    ]
    if gpu:
        lines += ["request_gpus    = 1",
                  "requirements    = (GPUs_Capability >= 7.0)"]
    # OMP_NUM_THREADS drives AerSimulator's CPU thread count; AER_MAX_MEM_MB caps
    # Aer's memory so it doesn't over-allocate past the cgroup (the 64 GB OOM fix).
    env = f"QTRK_STORE={STORE}"
    if qcpu:
        env += f" OMP_NUM_THREADS={ncpu}"
    if quantum:
        env += " AER_MAX_MEM_MB=$(aer_mem)"
    qvars = "shardfile, mem_gb, aer_mem" if quantum else "shardfile, mem_gb"
    lines += [
        "should_transfer_files = NO",
        "getenv = True",
        f'environment = "{env}"',
        f"output = {LOG_DIR}/{name}_$(Cluster)_$(Process).out",
        f"error  = {LOG_DIR}/{name}_$(Cluster)_$(Process).err",
        f"log    = {LOG_DIR}/{name}_$(Cluster).log",
        "+MaxRuntime = 259200",
        f"queue {qvars} from {listfile}",
        "",
    ]
    sub.write_text("\n".join(lines))
    return sub, listfile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", required=True,
                    help="comma list of: events,classical,quantum_cpu,quantum_gpu")
    ap.add_argument("--submit", action="store_true", help="condor_submit each lane")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="cap shards per lane (for pilots/testing)")
    ap.add_argument("--where", type=str, default=None,
                    help="pandas query to filter solves, e.g. \"epsilon==0.002\" "
                         "(submit only a subset, e.g. newly-added rows)")
    ap.add_argument("--suffix", type=str, default="",
                    help="suffix for shard/list/sub/log names, e.g. '_eps2'. Use "
                         "for incremental submits so live shards are NOT overwritten.")
    ap.add_argument("--only-missing", action="store_true",
                    help="shard ONLY solves whose solution is not yet on disk "
                         "(avoids a pile of idempotent no-op jobs near completion)")
    args = ap.parse_args()

    md = STORE / "manifest"
    events = pd.read_csv(md / "events.csv")
    solves = pd.read_csv(md / "solutions.csv")
    if args.where:
        solves = solves.query(args.where).reset_index(drop=True)
    if args.only_missing:
        solves = solves[~solves.sol_key.map(qp.solution_exists)].reset_index(drop=True)

    print(f"store={STORE}  events={len(events)}  solves={len(solves)}"
          + (f"  (filtered: {args.where})" if args.where else ""))
    for lane in [s.strip() for s in args.lanes.split(",") if s.strip()]:
        name = lane + args.suffix
        df = _lane_rows(lane, events, solves)
        shards = shard_lane(lane, df, args.max_shards, name)
        sub, listfile = write_submit(lane, shards, name)
        njobs = len(shards)
        print(f"[{name}] {len(df)} rows -> {njobs} shards -> {sub}")
        if args.submit:
            r = subprocess.run(["condor_submit", str(sub)],
                               capture_output=True, text=True)
            print(r.stdout.strip() or r.stderr.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
