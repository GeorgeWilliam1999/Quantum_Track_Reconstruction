#!/usr/bin/env python3
"""Paper 1 corpus: one noise configuration, T = 10 ... 1000, three solvers, exact.

Configuration: set by CORPUS_SIGMA_RES / CORPUS_SIGMA_SCATT / CORPUS_HIT_INEFF.
    First run (outputs/): 15 um, 0.45 mrad, 1 % hit loss.
    Paper corpus (outputs_5um/, George 16 Sep 2026 after the configuration scan):
    sigma_res = 5 um (0.005 mm), sigma_scatt = 0.15 mrad, hit_ineff = 1 %
    phi_max     = 0.2 rad
    acceptance  = the formula epsilon (kappa = 3), step window
    gamma = 3, delta = 1, s = 4
Every solver sees the identical events and the identical matrix A.

Solvers (all exact; A is block-diagonal over connected groups of compatible
segments, the largest of which is a few tens of segments, so each group is
diagonalised exactly):
    classical  x = A^{-1} b,  b = delta * 1
    1BQF       exact readout of the correctly wired one-bit circuit, t = pi / s:
               p_i = [cos^2(At/2) u]_i^2 + [1/2 sin(At) u]_i^2,  amplitude sqrt(p_i)
    QSVT       the production comb: degree 40, four passes, half-width 0.18,
               on base A, domain = exact spectral bounds padded as in the
               efficiency-frontier campaign; amplitude |p(A) b|

Independent checks on every event (all logged to checks.csv, and the run stops on
a failure):
    C1 classical: MINRES on the full sparse A against the exact per-group solve
    C2 1BQF: sparse expm_multiply of e^{-iAt} on the full A against the per-group
       eigen formula, for both cos^2(At/2)u and sin(At)u
    C3 QSVT: Chebyshev three-term recursion of p on the full sparse A against the
       polynomial evaluated on the exact eigenvalues
    C4 Hamiltonian: every head-to-tail hit-sharing segment pair is recomputed from the
       stored segment directions; C_ij = 1 exactly when the kink is below epsilon
    C5 closed forms: every complete isolated true track solves to 4/11, 5/11
       classically; every isolated segment is 1/4 classically and 0 after the 1BQF
    C6 noise: measured hit-loss fraction; kink width of true continuations against the
       Rayleigh scale sigma_p; fraction of true continuations outside the window

Outputs (QSVT/Paper_planning/1BQF_to_spectral_filter_corpus/outputs/):
    metrics.csv      one row per (T, rep, solver): efficiency-first working point and
                     matched-efficiency points, efficiency at fixed false rates, AUC
    error_types.csv  one row per (T, rep, error type): segment counts (true/false) and,
                     per solver, false segments kept at that solver's working point
    checks.csv       one row per event: every check value
    amplitudes/T{T}_rep{rep}.npz   per-segment amplitudes for every segment in a
                     group of two or more, plus labels; isolated segments are stored
                     as one value per solver (they are identical by construction)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm_multiply, minres

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp                                          # noqa: E402
import dp_terms                                                     # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon                  # noqa: E402
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("CORPUS_OUT", "outputs")
AMP = OUT / "amplitudes"
AMP.mkdir(parents=True, exist_ok=True)

NOISE = dict(sigma_scatt=float(os.environ.get("CORPUS_SIGMA_SCATT", "4.5e-4")),
             sigma_res=float(os.environ.get("CORPUS_SIGMA_RES", "0.015")), phi_max=0.2,
             hit_ineff=float(os.environ.get("CORPUS_HIT_INEFF", "0.01")))
GAMMA, DELTA = 3.0, 1.0
S = GAMMA + DELTA
T_ONEBIT = np.pi / S
COMB = dict(degree=40, hw=0.18)
TS = [int(t) for t in os.environ.get("CORPUS_TS", "10,20,50,100,200,400,700,1000").split(",")]
REPS = range(int(os.environ.get("CORPUS_REPS", "10")))
EFF_TARGETS = (0.95, 0.97, 0.98, 0.99, 0.995)
FAR_TARGETS = (0.001, 0.01, 0.10)
TOL = dict(C1=1e-6, C2=1e-8, C3=1e-8, C5=1e-9)

LOG = open(OUT / "build.log", "a")


def say(*a):
    print(*a, flush=True)
    print(*a, file=LOG, flush=True)


# ── metrics ──────────────────────────────────────────────────────────────────
def working_points(x, truth):
    """Efficiency-first thresholds: tau just below the k-th largest true amplitude."""
    x = np.abs(np.asarray(x, float).ravel())
    n_true = int(truth.sum())
    ts = np.sort(x[truth])[::-1]
    out = {}
    for e in EFF_TARGETS:
        k = min(n_true, max(1, int(np.ceil(e * n_true))))
        tau = ts[k - 1]
        act = x >= tau
        na = int(act.sum())
        tag = f"e{int(round(e * 1000)):03d}"
        out[f"tau_{tag}"] = float(tau)
        out[f"far_{tag}"] = int((act & ~truth).sum()) / na if na else np.nan
        out[f"eff_{tag}"] = float((act & truth).sum()) / n_true
        out[f"nfalse_{tag}"] = int((act & ~truth).sum())
    taus = ts
    xs, xt = np.sort(x), np.sort(x[truth])
    nact = len(x) - np.searchsorted(xs, taus, side="left")
    ntru = n_true - np.searchsorted(xt, taus, side="left")
    far = np.where(nact > 0, (nact - ntru) / np.maximum(nact, 1), np.nan)
    eff = ntru / n_true
    for ft in FAR_TARGETS:
        ok = far <= ft
        out[f"eff_f{int(round(ft * 1000)):03d}"] = float(eff[ok].max()) if ok.any() else 0.0
    # AUC of true-vs-false ordering (Mann-Whitney, ties counted half)
    xf = np.sort(x[~truth])
    lo = np.searchsorted(xf, x[truth], side="left")
    hi = np.searchsorted(xf, x[truth], side="right")
    out["auc"] = float((lo + 0.5 * (hi - lo)).sum() / (n_true * len(xf)))
    return out


# ── error types ──────────────────────────────────────────────────────────────
TYPES = ["complete track", "broken track", "contaminated track",
         "isolated false segment", "false pair", "false triple or bridge", "hub"]


def classify_group(n, maxdeg, n_true):
    if n_true == n:
        return "complete track" if (n == 4 and maxdeg <= 2) else "broken track"
    if n_true > 0:
        return "contaminated track"
    if n == 1:
        return "isolated false segment"
    if n == 2:
        return "false pair"
    return "false triple or bridge" if maxdeg <= 2 else "hub"


# ── one event ────────────────────────────────────────────────────────────────
def run_event(T, rep, eps):
    t0 = time.time()
    ev, ekey = qp.ensure_event(n_trk=T, rep=rep, **NOISE)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step", gamma=GAMMA, delta=DELTA)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    A, b, _, _ = dp_terms.dp_system(ham, beta=0.0, eps_B=None, alpha=0.0,
                                    gamma=GAMMA, delta=DELTA)
    A = A.tocsr()
    n = A.shape[0]
    assert len(truth) == n
    diag = A.diagonal()
    assert np.allclose(diag, S), "diagonal is not s"
    assert np.allclose(b, DELTA), "bias is not delta * 1"
    C = (-A).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    assert np.allclose(C.data, 1.0), "off-diagonal entries are not -1"

    ncomp, lab = connected_components(C, directed=False)
    sizes = np.bincount(lab, minlength=ncomp)
    deg = np.asarray(C.sum(axis=1)).ravel()

    x_cls = np.full(n, DELTA / S)
    y_cos2 = np.zeros(n)          # [cos^2(At/2) u]_i
    y_sin = np.zeros(n)           # [sin(At) u]_i
    lam_all = [np.array([S])]     # isolated line always present
    groups = []                   # (idx, evals, evecs)
    order = np.argsort(lab, kind="stable")
    starts = np.r_[0, np.cumsum(sizes)]
    # isolated segments: lambda = s -> cos^2(pi/2) = 0, sin(pi) = 0 (set exactly)
    for c in np.nonzero(sizes > 1)[0]:
        idx = order[starts[c]:starts[c + 1]]
        blk = A[idx][:, idx].toarray()
        w, V = np.linalg.eigh(blk)
        proj = V.T @ np.ones(len(idx))
        x_cls[idx] = V @ (DELTA * proj / w)
        y_cos2[idx] = V @ (np.cos(w * T_ONEBIT / 2) ** 2 * proj)
        y_sin[idx] = V @ (np.sin(w * T_ONEBIT) * proj)
        lam_all.append(w)
        groups.append((idx, w, V, proj))
    lam = np.concatenate(lam_all)
    lo, hi = float(lam.min()), float(lam.max())
    x_1b = np.sqrt(y_cos2 ** 2 + (0.5 * y_sin) ** 2)

    pad = 0.05 * (hi - lo) + 0.2
    dom = (lo - pad, hi + pad)
    comb = design_line_comb_inverse(degree=COMB["degree"], s=S, hw=COMB["hw"], domain=dom)
    p_iso = float(comb(S))
    x_q = np.full(n, abs(DELTA * p_iso))
    for idx, w, V, proj in groups:
        x_q[idx] = np.abs(V @ (DELTA * comb(w) * proj))

    chk = dict(T=T, rep=rep, event_key=ekey, n_seg=n, n_true=int(truth.sum()),
               n_groups_multi=len(groups), max_group=int(sizes.max()),
               lam_min=lo, lam_max=hi, eps=eps)

    # C1 classical against MINRES
    xm, info = minres(A.tocsc(), b, rtol=1e-12, maxiter=50000)
    chk["C1_minres_maxabs"] = float(np.max(np.abs(xm - x_cls)))
    chk["C1_minres_info"] = int(info)

    # C2 1BQF against sparse matrix exponential on the full A
    u = np.ones(n)
    z = expm_multiply(-1j * T_ONEBIT * A.astype(np.complex128), u.astype(np.complex128))
    cosAt = z.real                      # cos(At) u
    sinAt = -z.imag                     # sin(At) u
    chk["C2_cos2_maxabs"] = float(np.max(np.abs(0.5 * (u + cosAt) - y_cos2)))
    chk["C2_sin_maxabs"] = float(np.max(np.abs(sinAt - y_sin)))

    # C3 QSVT recursion on the full sparse A
    lo_d, hi_d = comb.domain
    sc, sh = 2.0 / (hi_d - lo_d), (lo_d + hi_d) / (hi_d - lo_d)
    c = comb.coef
    t_prev = b.astype(float)
    y = c[0] * t_prev
    t_cur = sc * (A @ t_prev) - sh * t_prev
    y = y + c[1] * t_cur
    for k in range(2, len(c)):
        t_next = 2.0 * (sc * (A @ t_cur) - sh * t_cur) - t_prev
        t_prev, t_cur = t_cur, t_next
        y = y + c[k] * t_cur
    chk["C3_recursion_maxabs"] = float(np.max(np.abs(np.abs(y) - x_q)))
    grid = np.linspace(lo_d, hi_d, 20001)
    chk["C3_max_abs_p_on_domain"] = float(np.max(np.abs(comb(grid))))

    # C4 the compatibility rule, recomputed from segment directions and hit ids
    hid = np.asarray(ham._segment_to_hit_ids).astype(np.int64)
    vec = np.asarray(ham._segment_vectors, float)
    tid = np.asarray(ham._segment_track_ids)

    def kink_of(i, j):
        return np.arccos(np.clip(np.einsum("ij,ij->i", vec[i], vec[j]), -1, 1))

    # (a) every edge of C joins a head-to-tail pair whose kink is inside the window
    Cu = C.tocoo()
    up = Cu.row < Cu.col
    ei, ej = Cu.row[up], Cu.col[up]
    fwd = hid[ei, 1] == hid[ej, 0]
    bwd = hid[ej, 1] == hid[ei, 0]
    k_edge = kink_of(ei, ej)
    chk["C4_edges_total"] = int(len(ei))
    chk["C4_edges_not_head_to_tail"] = int(np.sum(~(fwd | bwd)))
    chk["C4_edges_outside_window"] = int(np.sum(k_edge >= eps))
    # (b) candidate head-to-tail pairs: exhaustive for n_seg <= 2e5, else a random sample
    srt = np.argsort(hid[:, 0], kind="stable")
    start_sorted = hid[srt, 0]
    lo_i = np.searchsorted(start_sorted, hid[:, 1], side="left")
    hi_i = np.searchsorted(start_sorted, hid[:, 1], side="right")
    nout = hi_i - lo_i
    if n <= 200_000:
        ii = np.repeat(np.arange(n), nout)
        offs = np.arange(len(ii)) - np.repeat(np.cumsum(nout) - nout, nout)
        jj = srt[np.repeat(lo_i, nout) + offs]
        chk["C4_mode"] = "exhaustive"
    else:
        rng = np.random.default_rng(1000 * T + rep)
        cand = np.nonzero(nout > 0)[0]
        ii = rng.choice(cand, size=500_000, replace=True)
        jj = srt[lo_i[ii] + (rng.random(len(ii)) * nout[ii]).astype(np.int64)]
        chk["C4_mode"] = "sample 500000"
    in_C = np.asarray(C[ii, jj]).ravel() > 0
    k_pair = kink_of(ii, jj)
    chk["C4_pairs_checked"] = int(len(ii))
    chk["C4_mismatch"] = int(np.sum(in_C != (k_pair < eps)))
    # (c) every true continuation (consecutive segments of one track)
    tr = np.nonzero(truth)[0]
    t_start = {int(hid[i, 0]): i for i in tr}
    ti, tj = [], []
    for i in tr:
        j = t_start.get(int(hid[i, 1]))
        if j is not None and tid[j, 0] == tid[i, 0]:
            ti.append(i)
            tj.append(j)
    ti, tj = np.asarray(ti, dtype=np.int64), np.asarray(tj, dtype=np.int64)
    kt = kink_of(ti, tj) if len(ti) else np.array([])
    in_Ct = np.asarray(C[ti, tj]).ravel() > 0 if len(ti) else np.array([], bool)
    chk["C4_true_continuation_mismatch"] = int(np.sum(in_Ct != (kt < eps)))

    # C5 closed forms
    iso = sizes[lab] == 1
    chk["C5_iso_classical_maxdev"] = float(np.max(np.abs(x_cls[iso] - 0.25))) if iso.any() else 0.0
    chk["C5_iso_onebit_max"] = float(np.max(x_1b[iso])) if iso.any() else 0.0
    dev = 0.0
    n_clean_tracks = 0
    for idx, w, V, proj in groups:
        if len(idx) == 4 and truth[idx].all() and deg[idx].max() <= 2:
            vals = np.sort(x_cls[idx])
            dev = max(dev, abs(vals[0] - 4 / 11), abs(vals[1] - 4 / 11),
                      abs(vals[2] - 5 / 11), abs(vals[3] - 5 / 11))
            n_clean_tracks += 1
    chk["C5_track_closedform_maxdev"] = float(dev)
    chk["C5_clean_tracks"] = n_clean_tracks

    # C6 noise
    n_hits_expected = 5 * T
    chk["C6_hit_loss_fraction"] = 1.0 - ev.n_hits / n_hits_expected
    chk["C6_true_continuations"] = int(len(kt))
    sp_meas = float(np.sqrt(np.mean(kt ** 2) / 2)) if len(kt) else np.nan
    chk["C6_sigma_p_measured"] = sp_meas
    chk["C6_sigma_p_expected"] = float(np.sqrt(NOISE["sigma_scatt"] ** 2
                                               + 6 * np.arctan(NOISE["sigma_res"] / 33.0) ** 2))
    chk["C6_true_outside_window"] = float(np.mean(kt >= eps)) if len(kt) else np.nan

    # failures stop the run
    fails = []
    if chk["C1_minres_maxabs"] > TOL["C1"]:
        fails.append("C1")
    if max(chk["C2_cos2_maxabs"], chk["C2_sin_maxabs"]) > TOL["C2"]:
        fails.append("C2")
    if chk["C3_recursion_maxabs"] > TOL["C3"]:
        fails.append("C3")
    if (chk["C4_mismatch"] != 0 or chk["C4_edges_not_head_to_tail"] != 0
            or chk["C4_edges_outside_window"] != 0 or chk["C4_true_continuation_mismatch"] != 0):
        fails.append("C4")
    if max(chk["C5_iso_classical_maxdev"], chk["C5_iso_onebit_max"],
           chk["C5_track_closedform_maxdev"]) > TOL["C5"]:
        fails.append("C5")
    chk["failed"] = ",".join(fails)

    # metrics, two conventions:
    #   "all true"  every true segment counts towards efficiency
    #   "findable"  true segments of tracks that lost a hit are removed from the
    #               evaluation (neither required nor counted as false)
    hit_tid = np.array([h.track_id if h.track_id is not None else -1 for h in ev.hits])
    hits_per_track = np.bincount(hit_tid[hit_tid >= 0], minlength=T + 1)
    seg_track = tid[:, 0]
    unbroken = np.zeros(n, bool)
    unbroken[truth] = hits_per_track[seg_track[truth]] == 5
    keep_findable = ~truth | unbroken
    chk["n_true_findable"] = int(unbroken.sum())
    sols = {"classical": x_cls, "1BQF": x_1b, "QSVT": x_q}
    mrows = []
    wp = {}
    for name, x in sols.items():
        m = working_points(x, truth)
        wp[name] = m["tau_e990"]
        mrows.append(dict(T=T, rep=rep, solver=name, convention="all true", n_seg=n,
                          n_true=int(truth.sum()), **m))
        mf = working_points(x[keep_findable], truth[keep_findable])
        mrows.append(dict(T=T, rep=rep, solver=name, convention="findable",
                          n_seg=int(keep_findable.sum()), n_true=int(unbroken.sum()), **mf))

    # error types
    gtype = np.empty(ncomp, dtype=object)
    gmaxdeg = np.zeros(ncomp)
    np.maximum.at(gmaxdeg, lab, deg)
    gtrue = np.bincount(lab, weights=truth.astype(float), minlength=ncomp)
    for cidx in range(ncomp):
        gtype[cidx] = classify_group(int(sizes[cidx]), gmaxdeg[cidx], int(gtrue[cidx]))
    stype = gtype[lab]
    trows = []
    for ty in TYPES:
        sel = stype == ty
        r = dict(T=T, rep=rep, error_type=ty, n_segments=int(sel.sum()),
                 n_true=int((sel & truth).sum()), n_false=int((sel & ~truth).sum()),
                 n_groups=int(np.sum(gtype == ty)))
        for name, x in sols.items():
            act = x >= wp[name]
            r[f"false_kept_{name}"] = int((sel & ~truth & act).sum())
            r[f"true_kept_{name}"] = int((sel & truth & act).sum())
        trows.append(r)

    multi = ~iso
    np.savez_compressed(
        AMP / f"T{T}_rep{rep}.npz",
        seg_index=np.nonzero(multi)[0].astype(np.int64),
        group=lab[multi].astype(np.int64), truth=truth[multi],
        error_type=np.array([TYPES.index(t) for t in stype[multi]], dtype=np.int8),
        x_classical=x_cls[multi].astype(np.float64), x_1bqf=x_1b[multi].astype(np.float64),
        x_qsvt=x_q[multi].astype(np.float64),
        iso_classical=np.float64(DELTA / S), iso_1bqf=np.float64(0.0),
        iso_qsvt=np.float64(abs(DELTA * p_iso)),
        n_isolated_true=np.int64((iso & truth).sum()), n_isolated_false=np.int64((iso & ~truth).sum()),
        eigenvalues=lam, comb_domain=np.array(dom), comb_coef=comb.coef,
        tau_e990=np.array([wp["classical"], wp["1BQF"], wp["QSVT"]]),
        types=np.array(TYPES), event_key=ekey)
    chk["seconds"] = time.time() - t0
    return mrows, trows, chk


def main():
    eps = float(compute_epsilon(NOISE["sigma_res"], NOISE["sigma_scatt"]))
    say(f"=== corpus build {time.strftime('%Y-%m-%d %H:%M')} eps={eps * 1e3:.4f} mrad "
        f"T={TS} reps={list(REPS)}")
    files = {k: OUT / f"{k}.csv" for k in ("metrics", "error_types", "checks")}
    done = set()
    if files["checks"].exists():
        prev = pd.read_csv(files["checks"])
        done = set(zip(prev["T"], prev["rep"]))
    for T in TS:
        for rep in REPS:
            if (T, rep) in done:
                continue
            mrows, trows, chk = run_event(T, rep, eps)
            for k, rows in (("metrics", mrows), ("error_types", trows), ("checks", [chk])):
                pd.DataFrame(rows).to_csv(files[k], mode="a", index=False,
                                          header=not files[k].exists())
            q = {r["solver"]: r for r in mrows if r["convention"] == "all true"}
            say(f"[T{T} rep{rep}] n={chk['n_seg']:,} groups>1={chk['n_groups_multi']:,} "
                f"max={chk['max_group']} | far@99%: classical {q['classical']['far_e990']:.4f} "
                f"1BQF {q['1BQF']['far_e990']:.4f} QSVT {q['QSVT']['far_e990']:.4f} | "
                f"checks C1 {chk['C1_minres_maxabs']:.1e} C2 {max(chk['C2_cos2_maxabs'], chk['C2_sin_maxabs']):.1e} "
                f"C3 {chk['C3_recursion_maxabs']:.1e} C4 {chk['C4_mismatch']} "
                f"sigma_p {chk['C6_sigma_p_measured'] * 1e3:.3f}/{chk['C6_sigma_p_expected'] * 1e3:.3f} mrad "
                f"| {chk['seconds']:.0f}s {'FAILED ' + chk['failed'] if chk['failed'] else 'ok'}")
            if chk["failed"]:
                say("STOPPING on failed checks")
                sys.exit(2)
    say("done.")


if __name__ == "__main__":
    main()
