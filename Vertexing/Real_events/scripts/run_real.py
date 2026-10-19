#!/usr/bin/env python3
"""The section 8 methods on Run 3 simulated minimum-bias events (Gauss Sim10, 2024, nu = 7.6).

Per event (one Gauss main crossing, all its VELO hits):
  vertices   focusing scan (fs_lib.find_vertices) with the coarse range widened to the
             luminous region, [-250, 250) mm; everything else as in the toy. Matched to
             the truth PVs within 1 mm (fs_lib.match_vertices). A truth PV is counted for
             efficiency when it has >= 4 reconstructible tracks.
             Track-based baseline: 08_pv_baseline.vertices_from_tracks on the classical
             tracks at <= 1 % segment false rate.
  tracks     classical, one-bit filter, QSVT comb (07_full_sweep.segment_solutions, the
             package segment Hamiltonian, step kernel, gamma 3, delta 1) with
             epsilon = EPS_REAL; read with both rules of section 8 ("paper", "far1").
             Hit register: focused cells at the found vertices (512 grid, four shifted
             grids), and the same at the truth vertices.
             All scored by the package EventValidator (purity >= 0.7, reconstructible =
             >= 3 hits), as in section 8.
  resources  segment register (from segment_solutions) and the per-vertex hit operator
             at 4w (07_full_sweep.per_vertex_operator, lines {3, 4}); plus P on all the
             integer track lines m - 1 >= 2, since real tracks have 3 ... 13 hits.

Output: outputs/real_events.csv, outputs/real_vertices.csv, outputs/real_tracks.csv
"""
import importlib.util
import math
import zlib
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_lib as R  # noqa: E402
import fs_lib as L  # noqa: E402

FSD = Path(R.FS)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, FSD / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F7 = _load("fs7", "07_full_sweep.py")
F8 = _load("fs8", "08_pv_baseline.py")
MPV = F7.MPV
OUT = HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)

EPS_REAL = 0.005          # rad; the Run3_Verification choice on real VELO events
L.COARSE.update(lo=-250.0, hi=250.0)
F7.EPS = EPS_REAL
# scan-refit clustering, chosen on the first 24 events (job_5001 ... job_5008); the later
# runs (job_501x) are the hold-out
REFIT = dict(gap=0.5, min_trk=4, tmin=0.05)
METRICS = ("efficiency", "ghost_rate", "clone_fraction", "mean_purity", "hit_efficiency",
           "n_candidates", "n_ghosts", "n_clones", "n_reconstructible", "n_matched")


def p_all_lines(ev, fz, width):
    """P of the per-vertex operator on every integer line >= 2 (cliques K_m, m >= 3)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    n = len(ev.x)
    owner = -np.ones(n, int)
    for vi, grp in L.tracks_at_vertices(ev, fz, offsets=L.OFFSETS4, ngrid=512):
        owner[grp] = vi
    if len(fz):
        best = np.zeros((n, len(fz)), int)
        for vi, zv in enumerate(fz):
            dx, dy = L.directions(ev, zv)
            ok = np.nonzero((np.abs(dx) < 1) & (np.abs(dy) < 1))[0]
            pr = cKDTree(np.c_[dx[ok], dy[ok]]).query_pairs(width, output_type="ndarray")
            i, j = ok[pr[:, 0]], ok[pr[:, 1]]
            keep = ev.module[i] != ev.module[j]
            np.add.at(best[:, vi], i[keep], 1)
            np.add.at(best[:, vi], j[keep], 1)
        free = owner < 0
        owner[free] = np.where(best[free].max(1) > 0, best[free].argmax(1), -1)
    pi, pj = [], []
    for vi, zv in enumerate(fz):
        mem = np.nonzero(owner == vi)[0]
        if len(mem) < 2:
            continue
        dx, dy = L.directions(ev, zv)
        pr = cKDTree(np.c_[dx[mem], dy[mem]]).query_pairs(width, output_type="ndarray")
        i, j = mem[pr[:, 0]], mem[pr[:, 1]]
        keep = ev.module[i] != ev.module[j]
        pi.append(i[keep]); pj.append(j[keep])
    i = np.concatenate(pi) if pi else np.zeros(0, int)
    j = np.concatenate(pj) if pj else np.zeros(0, int)
    k = len(i)
    Cm = coo_matrix((np.ones(2 * k), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    nc, lab = connected_components(Cm, directed=False)
    sizes = np.bincount(lab, minlength=nc)
    P = 0.0
    for c in np.nonzero(sizes >= 3)[0]:
        nodes = np.nonzero(lab == c)[0]
        lam, V = np.linalg.eigh(Cm[nodes][:, nodes].toarray())
        r = np.round(lam)
        on = (np.abs(lam - r) < 1e-6) & (r >= 2)
        P += float(((V[:, on].T @ np.ones(len(nodes))) ** 2).sum()) / n
    return P


def run(path):
    t0 = time.perf_counter()
    ev, ex = R.load_real(path, seed=zlib.crc32((Path(path).parent.name + Path(path).stem).encode()))
    hid = np.array([h.hit_id for h in ev.pkg_event.hits])
    hitpos = -np.ones(hid.max() + 1, int)
    hitpos[hid] = np.arange(len(hid))
    F7.hitpos_inv = hid
    reco = ev.n_hits_of_track >= 3
    T = int(reco.sum())
    row = dict(event=f"{Path(path).parent.name}/{Path(path).stem}", n_hits=len(ev.x),
               n_particles=len(ev.n_hits_of_track), n_reco=T, n_pv=len(ev.pv_xyz),
               n_pv_reco=int((ex.pv_n_reco >= 4).sum()),
               frac_prompt=float(ex.prompt[reco].mean()), frac_electron=float(ex.electron[reco].mean()),
               med_hits_per_track=float(np.median(ev.n_hits_of_track[reco])))
    # ── vertices: focusing scan ─────────────────────────────────────────────
    tv = time.perf_counter()
    vs, _ = L.find_vertices(ev)
    row["t_scan"] = time.perf_counter() - tv
    fz = [v[0] for v in vs]
    tz = ev.pv_xyz[:, 2]
    m, res, merges, fakes = L.match_vertices(fz, tz, tol=1.0)
    rec_pv = ex.pv_n_reco >= 4
    vrows = []
    for zf_, h_ in [(v[0], v[1]) for v in vs]:
        d = np.abs(tz - zf_)
        vrows.append(dict(event=row["event"], method="focusing_peak", z_found=zf_, height=h_,
                          nearest_true=float(d.min()), nearest_true_reco=int(ex.pv_n_reco[np.argmin(d)])))
    row.update(n_found=len(fz), matched_reco_pv=int(((m >= 0) & rec_pv).sum()), merges=merges, fakes=fakes)
    for i in range(len(tz)):
        other = np.delete(tz, i)
        vrows.append(dict(event=row["event"], method="focusing", z_true=tz[i], n_reco_trk=int(ex.pv_n_reco[i]),
                          matched=bool(m[i] >= 0), dz=(fz[m[i]] - tz[i]) if m[i] >= 0 else np.nan,
                          nearest_other=float(np.min(np.abs(other - tz[i]))) if len(other) else np.nan))
    # ── segment methods ──────────────────────────────────────────────────────
    ts = time.perf_counter()
    h, truth, sols, sres = F7.segment_solutions(ev, T)
    row["t_segments"] = time.perf_counter() - ts
    row.update(sres)
    row["seg_n_true"] = int(truth.sum())
    seg_hits = np.asarray(h._segment_to_hit_ids)
    Ac = h.A.tocsr()
    iso_seg = np.diff(Ac.indptr) == 1          # only the diagonal entry: no coupling
    ISO = {k: (float(v[iso_seg].max()) if iso_seg.any() else 0.0) for k, v in sols.items()}
    matched = {}
    groups_cf = None
    for name, x in sols.items():
        for reading in ("paper", "far1", "iso"):
            if reading == "iso":
                # truth-free: every segment above the isolated-segment value (1/s for the
                # classical solver, 0 for the one-bit filter, |comb(s)| for the QSVT)
                tau = ISO[name] * (1 + 1e-6) + 1e-12
            else:
                rule = "far1" if (name == "qsvt" or reading == "far1") else "eff99"
                tau = F7.threshold(x, truth, rule)
            act = x >= tau
            key = {"paper": name, "far1": f"{name}F", "iso": f"{name}I"}[reading]
            row[f"{key}_seg_eff"] = float((act & truth).sum() / max(truth.sum(), 1))
            row[f"{key}_seg_far"] = float((act & ~truth).sum() / max(act.sum(), 1))
            groups = F7.tracks_from_segments(seg_hits, act, hitpos)
            if key == "classicalF":
                groups_cf = groups
            if key == "qsvtI":
                groups_q = groups
            met, mt = F7.validate(ev, F7.make_tracks(groups, ev))
            matched[key] = mt
            for k in METRICS:
                row[f"{key}_{k}"] = met[k]
    # track-based PV baseline on the classical <= 1 % false tracks
    bz = F8.vertices_from_tracks(ev, groups_cf)
    mb, resb, mergb, fakb = L.match_vertices(bz, tz, tol=1.0)
    row.update(base_n_found=len(bz), base_matched_reco_pv=int(((mb >= 0) & rec_pv).sum()),
               base_merges=mergb, base_fakes=fakb)
    for i in range(len(tz)):
        vrows.append(dict(event=row["event"], method="track_baseline", z_true=tz[i],
                          n_reco_trk=int(ex.pv_n_reco[i]), matched=bool(mb[i] >= 0),
                          dz=(bz[mb[i]] - tz[i]) if mb[i] >= 0 else np.nan,
                          nearest_other=float(np.min(np.abs(np.delete(tz, i) - tz[i]))) if len(tz) > 1 else np.nan))
    # ── hit register ─────────────────────────────────────────────────────────
    for key, vz, dmax, ngrid in (("hit", fz, 0.25, 512), ("hitT", list(tz), 0.25, 512),
                                 ("hitW", fz, 0.5, 1024), ("hitTW", list(tz), 0.5, 1024)):
        # W: direction window widened from 0.25 to 0.5 rad at the same cell width
        L.DMAX = dmax
        cands = L.tracks_at_vertices(ev, vz, offsets=L.OFFSETS4, ngrid=ngrid)
        L.DMAX = 0.25
        met, mt = F7.validate(ev, F7.make_tracks([list(g) for _, g in cands], ev))
        matched[key] = mt
        for k in METRICS:
            row[f"{key}_{k}"] = met[k]
    r = F7.per_vertex_operator(ev, fz, 4 * L.W)
    row.update({f"hitV_{k}": v for k, v in r.items()})
    row["hitV_P_all"] = p_all_lines(ev, fz, 4 * L.W)
    # ── adapted vertexing: scan peaks seed track candidates, vertices from their fits ──
    # (every scan peak seeds candidates; each candidate's z of closest approach to the
    # beamline is clustered as in 08_pv_baseline; the hit register is rebuilt there)
    seeds = L.tracks_at_vertices(ev, fz, offsets=L.OFFSETS4, ngrid=512)
    rz = F8.vertices_from_tracks(ev, [g for _, g in seeds], gap=REFIT["gap"], min_trk=REFIT["min_trk"],
                                 tmin=REFIT["tmin"])
    for tag, vz_list, src in (("refit", rz, "scan_refit"),
                              ("baseQ", F8.vertices_from_tracks(ev, groups_q), "track_baseline_qsvtI")):
        mr, _, mergr, fakr = L.match_vertices(vz_list, tz, tol=1.0)
        row.update({f"{tag}_n_found": len(vz_list), f"{tag}_matched_reco_pv": int(((mr >= 0) & rec_pv).sum()),
                    f"{tag}_merges": mergr, f"{tag}_fakes": fakr})
        for i in range(len(tz)):
            vrows.append(dict(event=row["event"], method=src, z_true=tz[i], n_reco_trk=int(ex.pv_n_reco[i]),
                              matched=bool(mr[i] >= 0), dz=(vz_list[mr[i]] - tz[i]) if mr[i] >= 0 else np.nan,
                              nearest_other=float(np.min(np.abs(np.delete(tz, i) - tz[i]))) if len(tz) > 1 else np.nan))
    cands = L.tracks_at_vertices(ev, rz, offsets=L.OFFSETS4, ngrid=512)
    met, mt = F7.validate(ev, F7.make_tracks([list(g) for _, g in cands], ev))
    matched["hitR"] = mt
    for k in METRICS:
        row[f"hitR_{k}"] = met[k]
    r = F7.per_vertex_operator(ev, rz, 4 * L.W)
    row.update({f"hitVR_{k}": v for k, v in r.items()})
    row["hitVR_P_all"] = p_all_lines(ev, rz, 4 * L.W)
    r = F7.per_vertex_operator(ev, list(tz), 4 * L.W)
    row.update({f"hitVT_{k}": v for k, v in r.items()})
    row["hitVT_P_all"] = p_all_lines(ev, list(tz), 4 * L.W)
    # ── per truth track ──────────────────────────────────────────────────────
    trows = []
    for t in np.nonzero(reco)[0]:
        trows.append(dict(event=row["event"], t=float(np.hypot(*ev.t_of_track[t])),
                          n_hits=int(ev.n_hits_of_track[t]), n_modules=int(ex.n_modules[t]),
                          p=float(ex.p[t]), prompt=bool(ex.prompt[t]), electron=bool(ex.electron[t]),
                          pv=int(ev.pv_of_track[t]),
                          **{f"found_{k}": int(t) in v for k, v in matched.items()}))
    row["t_event"] = time.perf_counter() - t0
    return row, vrows, trows


if __name__ == "__main__":
    files = R.dump_files()
    print(len(files), "events", flush=True)
    t0 = time.time()
    Rr, VR, TR = [], [], []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as pool:
        for i, (r, v, t) in enumerate(pool.imap_unordered(run, [str(f) for f in files], chunksize=1)):
            Rr.append(r); VR += v; TR += t
            print(f"{i + 1}/{len(files)} {r['event']}: {r['n_hits']} hits, {r['seg_n']} segments, "
                  f"{r['t_event']:.0f}s", flush=True)
    pd.DataFrame(Rr).sort_values("event").to_csv(OUT / "real_events.csv", index=False)
    pd.DataFrame(VR).to_csv(OUT / "real_vertices.csv", index=False)
    pd.DataFrame(TR).to_csv(OUT / "real_tracks.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
