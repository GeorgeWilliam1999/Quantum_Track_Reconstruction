#!/usr/bin/env python3
"""Where the surviving errors live in the spectrum (paper section 3).

The kept probability of the one-bit filter decomposes exactly over the
eigenmodes of A, P_anc = sum_j beta_j^2 cos^2(lambda_j t/2), and each mode's
weight splits into a true and a false share by the squared eigenvector
components on true and false segments.  Histogramming the FALSE share against
lambda shows what the filter does: before it (classical inversion, weight
beta_j^2 / lambda_j^2) almost all false weight is one spike at lambda = s, the
isolated line; after it that spike is exactly zero and what survives sits on
the coupled-false lines (pairs at s +- 1, triples at s +- sqrt 2, hubs at
s +- sqrt m) and, under noise, on a continuum between them.

Two events: the clean fixed-acceptance T=400 event of figure 12 (rep 0) and the
heavy-noise T=200 event of figure 18 (rep 0).

Output: outputs/02_surviving_false_spectrum.json
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path: sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp
from lhcb_velo_toy.analysis import compute_epsilon

HERE = Path(__file__).resolve().parent; OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)
GAMMA, DELTA, S = 3.0, 1.0, 4.0
T_FILT = np.pi / S
EVENTS = {
    "clean_T400": dict(n_trk=400, rep=0, eps=0.002,
                       nz=dict(sigma_scatt=1e-4, sigma_res=0.0, phi_max=0.2, hit_ineff=0.0)),
    "heavy_T200": dict(n_trk=200, rep=0, eps=None,
                       nz=dict(sigma_scatt=1e-4, sigma_res=0.02, phi_max=0.2, hit_ineff=0.01)),
}
LINES = {"isolated": [S], "pair": [S - 1, S + 1], "triple": [S - np.sqrt(2), S + np.sqrt(2)],
         "hub": [S - np.sqrt(3), S + np.sqrt(3), S - 2, S + 2, S - np.sqrt(5), S + np.sqrt(5)],
         "true": list(S - 2 * np.cos(np.arange(1, 5) * np.pi / 5))}
TOL = 2e-3


def modes(A, truth):
    """Per mode: lambda, beta^2 (uniform input), true share, false share."""
    n = A.shape[0]; u = np.ones(n) / np.sqrt(n)
    Off = A.copy().tolil(); Off.setdiag(0); Off = Off.tocsr(); Off.eliminate_zeros()
    ncomp, lab = connected_components(Off, directed=False)
    sizes = np.bincount(lab); order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(ncomp + 1))
    iso = sizes[lab] == 1
    lam = [np.full(int(iso.sum()), S)]; b2 = [u[iso] ** 2]
    tsh = [truth[iso].astype(float)]; fsh = [(~truth[iso]).astype(float)]
    for c in np.flatnonzero(sizes > 1):
        idx = order[bounds[c]:bounds[c + 1]]
        w, V = eigh(A[np.ix_(idx, idx)].toarray())
        beta = V.T @ u[idx]
        t_i = truth[idx]
        lam.append(w); b2.append(beta ** 2)
        tsh.append((V[t_i] ** 2).sum(axis=0)); fsh.append((V[~t_i] ** 2).sum(axis=0))
    return (np.concatenate(lam), np.concatenate(b2), np.concatenate(tsh), np.concatenate(fsh))


def main():
    out = {"s": S, "lines": {k: [float(x) for x in v] for k, v in LINES.items()}, "events": {}}
    edges = np.round(np.arange(0.0, 8.0 + 1e-9, 0.02), 6)
    for key, cfg in EVENTS.items():
        nz = cfg["nz"]
        eps = cfg["eps"] if cfg["eps"] else compute_epsilon(sigma_scatt=nz["sigma_scatt"], sigma_res=nz["sigma_res"])
        ev, _ = qp.ensure_event(n_trk=cfg["n_trk"], rep=cfg["rep"], **nz)
        ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step", gamma=GAMMA, delta=DELTA)
        truth = np.asarray(qp.truth_from_event(ev), bool)
        A = sp.csr_matrix(ham.A)
        lam, b2, tsh, fsh = modes(A, truth)
        w_before = b2 / lam ** 2                       # classical inversion, per-mode weight
        w_after = b2 * np.cos(lam * T_FILT / 2) ** 2   # one-bit filter, per-mode kept weight
        rec = dict(n_trk=cfg["n_trk"], rep=cfg["rep"], eps_mrad=float(eps * 1e3), n_seg=int(A.shape[0]),
                   n_true=int(truth.sum()), n_modes=int(len(lam)), edges=edges.tolist())
        for name, w in (("before", w_before), ("after", w_after)):
            wf, wt = w * fsh, w * tsh
            hf, _ = np.histogram(lam, bins=edges, weights=wf)
            ht, _ = np.histogram(lam, bins=edges, weights=wt)
            on_s = np.abs(lam - S) < TOL
            fam = {}
            for fname, ls in LINES.items():
                m = np.zeros(len(lam), bool)
                for l in ls: m |= np.abs(lam - l) < TOL
                fam[fname] = float(wf[m].sum())
            rec[name] = dict(false_hist=hf.tolist(), true_hist=ht.tolist(),
                             false_total=float(wf.sum()), true_total=float(wt.sum()),
                             false_at_s=float(wf[on_s].sum()), false_by_family=fam,
                             false_elsewhere=float(wf.sum() - sum(fam.values())))
            print(f"[{key}] {name:6s}: false weight total {wf.sum():.4e}  at lambda=s {wf[on_s].sum():.4e} "
                  f"({100*wf[on_s].sum()/max(wf.sum(),1e-300):.2f}%)  true weight {wt.sum():.4e}  "
                  f"families {{ {', '.join(f'{k}: {100*v/max(wf.sum(),1e-300):.1f}%' for k,v in fam.items() if k!='true')} }}"
                  f"  on true lines {100*fam['true']/max(wf.sum(),1e-300):.1f}%")
        out["events"][key] = rec
    (OUT / "02_surviving_false_spectrum.json").write_text(json.dumps(out))
    print("[saved] outputs/02_surviving_false_spectrum.json")

if __name__ == "__main__":
    main()
