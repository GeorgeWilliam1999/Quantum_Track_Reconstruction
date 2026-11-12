#!/usr/bin/env python3
"""Experiment B on the 103 Run 3 events (Gauss Sim10 minimum bias, 2024, nu = 7.6).

Events: Vertexing/Real_events (real_lib.load_real, same per-event smearing seed as run_real.py:
crc32 of "<job>/<dump>"). Vertices: the focusing-scan peaks with the coarse range widened to
[-250, 250) mm (set by importing run_real.py), and the truth vertices. The per-vertex hit
operator at 4w and every filter of b_lib.FILTERS, applied exactly (b_lib.evaluate).

Reproduction check (stops the script if it fails): at the scan peaks the {3, 4} projector P and
the ideal step on every line >= 2 must equal hitV_P and hitV_P_all of
Vertexing/Real_events/outputs/real_events.csv event by event (published medians 0.18 and 0.36).

Categories per reconstructible truth track (Run 3 README): 3 hits, |t| > 0.25, p < 1 GeV,
not prompt (secondaries), electrons, and toy-like (prompt, not an electron, >= 4 hits,
p > 2 GeV).

Baselines (classical, one-bit filter, QSVT, focused cells) are the per-event and per-track
values of real_events.csv / real_tracks.csv on the same events; they are not rerun.

Outputs: outputs/run3_events.csv, outputs/run3_tracks.csv, outputs/run3_reproduction.txt.
"""
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b_lib as B  # noqa: E402

RR = B.load("run_real", B.RE / "run_real.py")     # widens L.COARSE to [-250, 250) mm on import
R = RR.R
L = B.L
assert L.COARSE["lo"] == -250.0 and L.COARSE["hi"] == 250.0
REF = B.RE.parent / "outputs" / "real_events.csv"


def run(path):
    t0 = time.perf_counter()
    name = f"{Path(path).parent.name}/{Path(path).stem}"
    ev, ex = R.load_real(path, seed=zlib.crc32((Path(path).parent.name + Path(path).stem).encode()))
    vs, _ = L.find_vertices(ev)
    fz = [v[0] for v in vs]
    tz = list(ev.pv_xyz[:, 2])
    reco = ev.n_hits_of_track >= 3
    row = dict(event=name, n_hits=len(ev.x), n_reco=int(reco.sum()), n_pv=len(tz), n_found=len(fz))
    found = {}
    for tag, vz in (("scan", fz), ("truth", tz)):
        r, mt = B.evaluate(ev, vz, tag)
        row.update(r)
        found.update({f"{tag}_{k}": v for k, v in mt.items()})
    trows = []
    for t in np.nonzero(reco)[0]:
        trows.append(dict(event=name, t=float(np.hypot(*ev.t_of_track[t])), n_hits=int(ev.n_hits_of_track[t]),
                          p=float(ex.p[t]), prompt=bool(ex.prompt[t]), electron=bool(ex.electron[t]),
                          **{f"found_{k}": int(t) in v for k, v in found.items()}))
    row["t_event"] = time.perf_counter() - t0
    return row, trows


if __name__ == "__main__":
    files = [str(f) for f in R.dump_files()]
    print(len(files), "events", flush=True)
    t0 = time.time()
    Rr, TR = [], []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as pool:
        for i, (r, t) in enumerate(pool.imap_unordered(run, files, chunksize=1)):
            Rr.append(r); TR += t
            print(f"{i + 1}/{len(files)} {r['event']}: P34 {r['scan_ideal34_P']:.3f} step {r['scan_step2_P']:.3f} "
                  f"{r['t_event']:.0f}s", flush=True)
    d = pd.DataFrame(Rr).sort_values("event").reset_index(drop=True)
    ref = pd.read_csv(REF).set_index("event")
    m = d.set_index("event").join(ref[["hitV_P", "hitV_P_all"]], how="left")
    ok34 = np.allclose(m.scan_ideal34_P, m.hitV_P, rtol=0, atol=1e-9)
    okall = np.allclose(m.scan_step2_P, m.hitV_P_all, rtol=0, atol=1e-9)
    lines = [f"events: {len(m)}",
             f"P{{3,4}} at the scan peaks, per-event equality with real_events.csv hitV_P: {ok34} "
             f"(max |diff| {np.max(np.abs(m.scan_ideal34_P - m.hitV_P)):.2e}); median "
             f"{m.scan_ideal34_P.median():.4f} / published {m.hitV_P.median():.4f}",
             f"P on every line >= 2, per-event equality with hitV_P_all: {okall} "
             f"(max |diff| {np.max(np.abs(m.scan_step2_P - m.hitV_P_all)):.2e}); median "
             f"{m.scan_step2_P.median():.4f} / published {m.hitV_P_all.median():.4f}",
             "REPRODUCTION " + ("PASSED" if ok34 and okall else "FAILED")]
    txt = "\n".join(lines)
    print(txt)
    (B.OUT / "run3_reproduction.txt").write_text(txt + "\n")
    d.to_csv(B.OUT / "run3_events.csv", index=False)
    pd.DataFrame(TR).to_csv(B.OUT / "run3_tracks.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
    if not (ok34 and okall):
        sys.exit("reproduction failed: stop and report")
