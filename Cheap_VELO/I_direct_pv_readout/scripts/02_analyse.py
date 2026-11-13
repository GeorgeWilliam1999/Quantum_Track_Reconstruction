#!/usr/bin/env python3
"""Item I/J, step 2: tune the vertex-sampler peak finder, score every PV method, shot sweep.

Reads cache/{tune,toy,real}/*.npz from 01_events.py.

Scoring (all methods): found PV z matched greedily to all truth PVs within 1 mm
(package solvers.vertexing.match_vertices, the focusing-scan rule). A PV counts for the
efficiency if it has >= 4 reconstructible tracks (the Run 3 README rule); the toy also
gets the all-PV efficiency (the section 8 rule). Fakes: found vertices matched to no truth
PV. Merges: unmatched truth PVs within 1 mm of a used found vertex.

Tuning (sampler only; the baselines keep their published rules): grid over item kind,
dmax, smoothing sigma, minimum peak content and shoulder ratio at K = infinity (the
exhaustive histogram). Objective: matched eligible PVs minus fakes, summed over the
tuning events (toy: the 18 tune events; Run 3: job_5001 ... job_5008, 24 events). The best
setting per (event set, kind) is then frozen and evaluated on the other events.

Shots: K = 2^4 ... 2^18 accepted shots per event, drawn from the exact distribution
(pv_sampler.sample_counts, seed fixed per event), same peak finder with scale
n_flagged / K. Queries = K x queries_per_shot (amplitude amplification at the padded P).

Outputs (outputs/): tune_grid.csv, tuned.json, pv_events.csv, shots_events.csv,
resources_events.csv
"""
import itertools
import json
import sys
import zlib
from pathlib import Path

PKG = "/data/bfys/gscriven/worktrees/velo-toy-pv-readout/src"
sys.path.insert(0, PKG)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lhcb_velo_toy.solvers.vertexing import (  # noqa: E402
    VertexOracleTable, find_vertex_peaks, match_vertices, output_distribution, sample_counts,
    sampler_resources, zeta_bins)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
TOL = 1.0
MIN_SEP = 1.0
BIN = dict(toy=0.05, real=0.1)                 # vertex-register bin width [mm]
GRID = dict(
    toy=dict(dmax=(0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5), sigma=(0.05, 0.1, 0.2, 0.4), cmin=(3, 5, 8, 12, 20, 30), shoulder=(0.0, 0.1)),
    real=dict(dmax=(0.1, 0.2, 0.5, 1.0, 2.0, 3.0), sigma=(0.2, 0.4, 0.8, 1.2, 1.6), cmin=(5, 10, 20, 30, 45, 60, 90), shoulder=(0.0, 0.1)),
)
KS = [2 ** k for k in range(4, 19)]
BASELINES = ("scan", "scan_refit", "trk_classical_median", "trk_classical_fit", "trk_1bqf_fit",
             "trk_qsvt_fit", "trk_truth_fit")
SEG_ANCILLA = {"1bqf": 2, "qsvt": 4}          # paper 1 ancillas (Focusing_scan README section 6)


def load(mode):
    evs = []
    for f in sorted((ROOT / "cache" / mode).glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        e = {k: d[k] for k in d.files}
        e["meta"] = json.loads(str(e["meta"]))
        e["found"] = json.loads(str(e["found"]))
        e["set"] = "real" if mode == "real" else "toy"
        e["holdout"] = (mode == "toy") or (mode == "real" and not e["meta"]["event"].startswith("job_500"))
        evs.append(e)
    return evs


def score(e, fz):
    tz = e["pv_xyz"][:, 2]
    elig = e["pv_nreco"] >= 4
    m, _, merges, fakes = match_vertices(fz, tz, TOL)
    fz = np.asarray(fz, float)
    res = np.array([fz[m[i]] - tz[i] for i in range(len(tz)) if m[i] >= 0 and elig[i]])
    return dict(n_true=len(tz), n_elig=int(elig.sum()), matched=int((m >= 0).sum()),
                matched_elig=int(((m >= 0) & elig).sum()), merges=merges, fakes=fakes,
                n_found=len(fz), res_sq=float((res ** 2).sum()), n_res=len(res),
                res_abs_med=float(np.median(np.abs(res))) if len(res) else np.nan,
                res_list=json.dumps([round(float(r), 5) for r in res]))


def table(e, kind, dmax):
    z0, doca = e[f"{kind}_z0"], e[f"{kind}_doca"]
    keep = doca <= dmax
    return VertexOracleTable("doublet" if kind == "d" else "triplet", int(e[f"{kind}_n_items"]),
                             int(e[f"{kind}_nq"]), z0[keep], np.zeros((int(keep.sum()), 2), int), doca[keep])


def sampler_peaks(e, kind, p, K=None, rng=None):
    tab = table(e, kind, p["dmax"])
    edges, nqz = zeta_bins(e["meta"]["cuts"]["zwin"], BIN[e["set"]])
    if K is None:
        c = np.histogram(tab.z0, edges)[0]
        scale = 1.0
    else:
        c = sample_counts(output_distribution(tab, edges), K, rng)
        scale = tab.n_flagged / K
    pk = find_vertex_peaks(c, edges, scale, p["sigma"], p["cmin"], MIN_SEP, shoulder=p["shoulder"])
    return [z for z, _ in pk], tab, nqz


def tune(evs, setname):
    rows = []
    g = GRID[setname]
    for kind in ("d", "t"):
        for dmax, sigma, cmin, sh in itertools.product(g["dmax"], g["sigma"], g["cmin"], g["shoulder"]):
            p = dict(dmax=dmax, sigma=sigma, cmin=cmin, shoulder=sh)
            tot = dict(matched_elig=0, fakes=0, n_elig=0, merges=0)
            for e in evs:
                s = score(e, sampler_peaks(e, kind, p)[0])
                for k in tot:
                    tot[k] += s[k]
            rows.append(dict(set=setname, kind=kind, **p, **tot, objective=tot["matched_elig"] - tot["fakes"]))
    df = pd.DataFrame(rows)
    best = {}
    for kind, d in df.groupby("kind"):
        b = d.sort_values(["objective", "fakes", "dmax"], ascending=[False, True, True]).iloc[0]
        best[kind] = dict(dmax=float(b.dmax), sigma=float(b.sigma), cmin=float(b.cmin), shoulder=float(b.shoulder))
    return df, best


def main():
    tune_toy = load("tune")
    toy = load("toy")
    real = load("real")
    print(len(tune_toy), len(toy), len(real), "events", flush=True)
    real_tune = [e for e in real if not e["holdout"]]
    g1, b_toy = tune(tune_toy, "toy")
    g2, b_real = tune(real_tune, "real")
    pd.concat([g1, g2]).to_csv(OUT / "tune_grid.csv", index=False)
    tuned = dict(toy=b_toy, real=b_real)
    json.dump(tuned, open(OUT / "tuned.json", "w"), indent=1)
    print("tuned", tuned, flush=True)

    pv_rows, shot_rows, res_rows = [], [], []
    for e in toy + real:
        meta = e["meta"]
        base = dict(set=e["set"], event=meta["event"], T=meta["T"], n_pv=meta["n_pv"], holdout=e["holdout"],
                    n_hits=meta["n_hits"], n_reco=meta["n_reco"], n_seg=meta["n_seg"])
        for b in BASELINES:
            pv_rows.append(dict(base, method=b, **score(e, e["found"][b])))
        for kind in ("d", "t"):
            p = tuned[e["set"]][kind]
            fz, tab, nqz = sampler_peaks(e, kind, p)
            pv_rows.append(dict(base, method=f"sampler_{kind}_exact", **score(e, fz)))
            R = sampler_resources(tab, nqz, 1)
            flagged_pv = e[f"{kind}_pv"][e[f"{kind}_doca"] <= p["dmax"]]
            res_rows.append(dict(base, kind=kind, **{k: v for k, v in R.items() if k not in ("K", "kind", "queries", "classical_sampling")},
                                 frac_flag_true=float((flagged_pv >= 0).mean()) if len(flagged_pv) else np.nan,
                                 seg_ns=meta["sres"]["seg_ns"],
                                 q_seg_1bqf=meta["sres"]["seg_ns"] + SEG_ANCILLA["1bqf"],
                                 q_seg_qsvt=meta["sres"]["seg_ns"] + SEG_ANCILLA["qsvt"]))
            rng = np.random.default_rng(zlib.crc32(f"{meta['event']}{kind}".encode()))
            for K in KS:
                fzK, _, _ = sampler_peaks(e, kind, p, K=K, rng=rng)
                s = score(e, fzK)
                s.pop("res_list")
                shot_rows.append(dict(base, kind=kind, K=K, queries=K * R["queries_per_shot"],
                                      classical_sampling=K / R["P"] if R["P"] > 0 else np.inf, **s))
    pd.DataFrame(pv_rows).to_csv(OUT / "pv_events.csv", index=False)
    pd.DataFrame(shot_rows).to_csv(OUT / "shots_events.csv", index=False)
    pd.DataFrame(res_rows).to_csv(OUT / "resources_events.csv", index=False)
    print("done", len(pv_rows), len(shot_rows), len(res_rows))


if __name__ == "__main__":
    main()
