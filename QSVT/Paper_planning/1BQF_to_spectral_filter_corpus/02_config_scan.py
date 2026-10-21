#!/usr/bin/env python3
"""Configuration scan for the paper corpus (16 Sep 2026).

Same Hamiltonian, the same exact per-group solvers as 01_build_corpus.py (whose
exactness checks passed on every event), no checks repeated. For each candidate
configuration and T: classical and 1BQF efficiency and false rate at the 99 % point;
QSVT efficiency at fixed false rates and false rate at fixed efficiencies.
Output: outputs_scan/scan.csv. Events are cached in the shared store.
"""
import os, sys, time
import numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp, dp_terms
from lhcb_velo_toy.analysis import compute_epsilon
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_scan"); os.makedirs(OUT, exist_ok=True)
S, DELTA, TT = 4.0, 1.0, np.pi / 4.0
CFGS = [("5 um, 0.15 mrad", 0.005, 1.5e-4, 0.2), ("10 um, 0.30 mrad", 0.010, 3.0e-4, 0.2),
        ("3 um, 0.10 mrad", 0.003, 1.0e-4, 0.2), ("15 um, 0.45 mrad, cone 0.4", 0.015, 4.5e-4, 0.4)]
TS, REPS = (200, 400, 700, 1000), (0, 1)

def solve(T, rep, sr, ss, phi):
    eps = float(compute_epsilon(sr, ss))
    ev, _ = qp.ensure_event(n_trk=T, rep=rep, sigma_scatt=ss, sigma_res=sr, phi_max=phi, hit_ineff=0.01)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step", gamma=3.0, delta=1.0)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    A, b, _, _ = dp_terms.dp_system(ham, beta=0.0, eps_B=None, alpha=0.0, gamma=3.0, delta=1.0)
    A = A.tocsr(); C = (-A).tocsr(); C.setdiag(0); C.eliminate_zeros()
    nc, lab = connected_components(C, directed=False); sizes = np.bincount(lab)
    n = A.shape[0]
    xc = np.full(n, DELTA / S); c2 = np.zeros(n); sn = np.zeros(n)
    order = np.argsort(lab, kind="stable"); st = np.r_[0, np.cumsum(sizes)]
    groups, lam = [], [np.array([S])]
    for c in np.nonzero(sizes > 1)[0]:
        idx = order[st[c]:st[c + 1]]
        w, V = np.linalg.eigh(A[idx][:, idx].toarray()); pr = V.T @ np.ones(len(idx))
        xc[idx] = V @ (DELTA * pr / w); c2[idx] = V @ (np.cos(w * TT / 2) ** 2 * pr); sn[idx] = V @ (np.sin(w * TT) * pr)
        groups.append((idx, w, V, pr)); lam.append(w)
    lam = np.concatenate(lam); lo, hi = lam.min(), lam.max(); pad = 0.05 * (hi - lo) + 0.2
    comb = design_line_comb_inverse(degree=40, s=S, hw=0.18, domain=(lo - pad, hi + pad))
    xq = np.full(n, abs(DELTA * comb(S)))
    for idx, w, V, pr in groups:
        xq[idx] = np.abs(V @ (DELTA * comb(w) * pr))
    x1 = np.sqrt(c2 ** 2 + (0.5 * sn) ** 2)
    gt = np.bincount(lab, weights=truth.astype(float)); contam = (gt > 0) & (gt < sizes)
    return eps, truth, {"classical": xc, "1BQF": x1, "QSVT": xq}, float(truth[contam[lab]].sum() / truth.sum() if truth.any() else 0)  # share of TRUE segments in contaminated groups (fixed 16 Sep: the running scan printed truth[contam].mean(), the true share of contaminated-group segments)

def curve(x, t):
    o = np.argsort(-x, kind="stable"); xs, ts = x[o], t[o]
    tp, fp = np.cumsum(ts), np.cumsum(~ts); last = np.r_[xs[1:] != xs[:-1], True]
    return tp[last] / t.sum(), fp[last] / (tp[last] + fp[last])

rows = []
for name, sr, ss, phi in CFGS:
    for T in TS:
        for rep in REPS:
            t0 = time.time()
            eps, truth, sols, contam = solve(T, rep, sr, ss, phi)
            r = dict(config=name, eps_mrad=eps * 1e3, T=T, rep=rep, contaminated_true_share=contam)
            for s, x in sols.items():
                eff, far = curve(x, truth)
                for e in (0.95, 0.97, 0.99):
                    k = min(np.searchsorted(eff, e), len(eff) - 1); r[f"{s} far@{e}"] = far[k]
                for fm in (0.01, 0.05):
                    ok = far <= fm; r[f"{s} eff@far{fm}"] = eff[ok].max() if ok.any() else 0.0
            rows.append(r); pd.DataFrame(rows).to_csv(os.path.join(OUT, "scan.csv"), index=False)
            print(f"{name} T{T} rep{rep} contam {contam:.3f} | cls far@.99 {r['classical far@0.99']:.3f} "
                  f"1BQF far@.99 {r['1BQF far@0.99']:.3f} | QSVT eff@1% {r['QSVT eff@far0.01']:.3f} "
                  f"far@.97 {r['QSVT far@0.97']:.3f} ({time.time() - t0:.0f}s)", flush=True)
print("done.")
