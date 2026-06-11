"""qsvt_finish_B_point.py — solve ONE remaining phase-B 1BQF point in a fresh
process and flush immediately, so an OOM kill costs exactly one solve.
Usage: Q_env python qsvt_finish_B_point.py T DROP REP
"""
import sys

from qsvt_densify_campaign import BASE_NOISE, flush, solve_point

T, drop, rep = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
rows = []
did, dt = solve_point(rows, study="Verify_new_results", T=T, rep=rep, gamma=3.0,
                      hit_ineff=drop, do_classical=True, do_qsvt=True,
                      do_quantum=True, **BASE_NOISE)
print(f"[finish-B] 1BQF T={T} drop={drop} rep={rep}: solved={did} ({dt:.0f}s)",
      flush=True)
flush(rows, ["Verify_new_results"])
