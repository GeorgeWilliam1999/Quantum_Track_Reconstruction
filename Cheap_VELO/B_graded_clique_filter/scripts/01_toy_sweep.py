#!/usr/bin/env python3
"""Experiment B on the toy: graded clique filters on the per-vertex hit operator.

Events: the section 4.5 / 8 events of the direct-vertexing write-up, regenerated with the
same generator and seeds as Vertexing/Focusing_scan/scripts/07_full_sweep.py
(fs_lib.generate_multi_pv, seed 30000 + 1000 n_pv + 10 rep + T): T = 100, 400, 1000 with
1, 5, 10, 20 vertices, 5 events per point (3 at T = 1000).

Per event, the per-vertex hit operator at 4w is built at the focusing-scan vertices ("scan")
and at the truth vertices ("truth"), and every filter of b_lib.FILTERS is applied exactly
(b_lib.evaluate).

Reproduction check (stops the script if it fails): at the scan vertices the {3, 4} projector P,
the coupled pairs and the Givens price per step must equal hitV_P, hitV_k and hitV_cx_givens of
Vertexing/Focusing_scan/outputs/full_sweep_events.csv event by event (published means at
T = 1000: P = 0.97 / 0.99 / 0.96 / 0.77 for 1 / 5 / 10 / 20 vertices).

Baselines (classical, one-bit filter and QSVT on the segment Hamiltonian, and the focused-cell
hit register) are the per-event values of full_sweep_events.csv / full_sweep_tracks.csv on the
same events; they are not rerun.

Outputs: outputs/toy_events.csv (one row per event), outputs/toy_tracks.csv (one row per
reconstructible truth track, found flag per filter and vertex set),
outputs/toy_reproduction.txt.
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b_lib as B  # noqa: E402

L = B.L
REF = B.FS.parent / "outputs" / "full_sweep_events.csv"


def jobs():
    return [(T, npv, rep) for T in (100, 400, 1000) for npv in (1, 5, 10, 20)
            for rep in range(5 if T < 700 else 3)]


def run(job):
    T, npv, rep = job
    t0 = time.perf_counter()
    ev = L.generate_multi_pv(npv, T, seed=30000 + 1000 * npv + 10 * rep + T)
    vs, _ = L.find_vertices(ev)
    fz = [v[0] for v in vs]
    tz = list(ev.pv_xyz[:, 2])
    row = dict(T=T, n_pv=npv, rep=rep, n_found=len(fz))
    found = {}
    for tag, vz in (("scan", fz), ("truth", tz)):
        r, mt = B.evaluate(ev, vz, tag)
        row.update(r)
        found.update({f"{tag}_{k}": v for k, v in mt.items()})
    trows = []
    for tr in ev.pkg_event.tracks:
        if len(tr.hit_ids) < 3:
            continue
        trows.append(dict(T=T, n_pv=npv, rep=rep, t=float(np.hypot(*ev.t_of_track[tr.track_id])),
                          n_hits=len(tr.hit_ids), **{f"found_{k}": tr.track_id in v for k, v in found.items()}))
    row["t_event"] = time.perf_counter() - t0
    return row, trows


if __name__ == "__main__":
    J = jobs()
    print(len(J), "events", flush=True)
    t0 = time.time()
    R, TR = [], []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as pool:
        for i, (r, t) in enumerate(pool.imap_unordered(run, J, chunksize=1)):
            R.append(r); TR += t
            print(f"{i + 1}/{len(J)} T={r['T']} n_pv={r['n_pv']} rep={r['rep']}: "
                  f"P34 {r['scan_ideal34_P']:.3f} step {r['scan_step2_P']:.3f} {r['t_event']:.0f}s", flush=True)
    d = pd.DataFrame(R).sort_values(["T", "n_pv", "rep"]).reset_index(drop=True)
    # ── reproduction check against the published per-event values ──
    ref = pd.read_csv(REF).set_index(["T", "n_pv", "rep"])
    m = d.set_index(["T", "n_pv", "rep"]).join(ref[["hitV_P", "hitV_k", "hitV_cx_givens"]], how="left")
    ok_P = np.allclose(m.scan_ideal34_P, m.hitV_P, rtol=0, atol=1e-9)
    ok_k = np.array_equal(m.scan_k.values, m.hitV_k.values)
    ok_cx = np.allclose(m.scan_cx_step, m.hitV_cx_givens, rtol=1e-9)
    means = m.loc[1000].groupby("n_pv")[["scan_ideal34_P", "hitV_P"]].mean()
    lines = [f"events: {len(m)}; per-event equality with full_sweep_events.csv: "
             f"P{{3,4}} {ok_P} (max |diff| {np.max(np.abs(m.scan_ideal34_P - m.hitV_P)):.2e}), "
             f"pairs {ok_k}, Givens price per step {ok_cx}",
             "T = 1000 mean P{3,4} (this run / published):"]
    lines += [f"  n_pv = {k:2d}: {r.scan_ideal34_P:.4f} / {r.hitV_P:.4f}" for k, r in means.iterrows()]
    lines.append("REPRODUCTION " + ("PASSED" if ok_P and ok_k and ok_cx else "FAILED"))
    txt = "\n".join(lines)
    print(txt)
    (B.OUT / "toy_reproduction.txt").write_text(txt + "\n")
    d.to_csv(B.OUT / "toy_events.csv", index=False)
    pd.DataFrame(TR).to_csv(B.OUT / "toy_tracks.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
    if not (ok_P and ok_k and ok_cx):
        sys.exit("reproduction failed: stop and report")
