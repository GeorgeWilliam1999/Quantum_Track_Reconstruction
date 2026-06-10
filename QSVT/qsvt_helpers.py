"""qsvt_helpers.py — shared utilities for the QSVT failure-mode-filter study.

Events are REUSED from the qtrk_store generator; only the eigen-analysis / filter
application is new here. Mirrors the conventions of
``../Toy_Characterisation/Bifurification/bif.py``:

  base segment Hamiltonian  A0 = (gamma+delta) I - C ,  b = delta * 1
  active iff x_i > tau = delta/(gamma+delta) + 0.10 = 0.35   (gamma=3, delta=1)
  the 1BQF returns the mode-filtered solve  x_i = sum_k beta_k g(lambda_k) u_k(i),
  where g replaces 1/lambda: classical g=1/lambda, 1-bit g=cos(lambda t/2),
  QSVT g=p(lambda).
"""
import os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
os.environ.setdefault("MPLBACKEND", "Agg")
import qtrk_pipeline as qp  # noqa: E402

GAMMA, DELTA, EPS = 3.0, 1.0, 0.002
S = GAMMA + DELTA                       # = 4 ; the diagonal / notch eigenvalue
TAU = qp.threshold_for(GAMMA, DELTA)    # = 0.35
T1 = np.pi / S                          # base 1BQF evolution time -> notch at lambda=S

# the clean true-track (P4) path-graph spectrum, lambda_k = S - 2 cos(k pi / 5)
P4 = np.array([S - 2 * np.cos(k * np.pi / 5) for k in (1, 2, 3, 4)])  # 2.382,3.382,4.618,5.618


def phi_of(lam):
    """QPE phase varphi = lambda / (2 s)."""
    return lam / (2 * S)


def solve_event(rep, n_trk=200, sigma_scatt=1e-4, sigma_res=0.0, phi_max=0.2, hit_ineff=0.0):
    """Build + classically solve one event. Returns (A_csr, b, truth, sol, lab, deg, n)."""
    ev = qp.ensure_event(n_trk=n_trk, rep=rep, sigma_scatt=sigma_scatt, sigma_res=sigma_res,
                         phi_max=phi_max, hit_ineff=hit_ineff)[0]
    ham = qp.build_hamiltonian(ev, epsilon=EPS, gamma=GAMMA, delta=DELTA)
    A = ham.A.tocsr(); n = ham.n_segments
    truth = np.asarray(qp.truth_from_event(ev), bool)
    b = DELTA * np.ones(n)
    sol = np.asarray(sp.linalg.minres(A.tocsc(), b, rtol=1e-9, maxiter=8000)[0])
    lab, deg = cluster_labels(A, n)
    return A, b, truth, sol, lab, deg, n


def cluster_labels(A_csr, n):
    """Connected components of the continuation graph C = off-diagonal support of sI-A."""
    Cm = (S * sp.identity(n, format="csr") - A_csr)
    Cm.setdiag(0); Cm.eliminate_zeros()
    Cm = (abs(Cm) > 1e-9).astype(np.int8)
    _, lab = connected_components(Cm, directed=False)
    deg = np.asarray(Cm.sum(1)).ravel()
    return lab, deg


def classify(maxdeg):
    """Cluster failure-type label from its max continuation degree."""
    return "hub" if maxdeg >= 3 else "bridge"


def collect_clusters(reps, n_trk=200, max_block=60, sample_true=40, **noise):
    """Accumulate true+FP clusters over events.

    Returns a list of dicts, each:
       w (eigvals), U (eigvecs), beta=U^T b, xC (classical sol on block),
       truth (mask), fp (false-active mask), maxdeg, ftype, members
    plus arrays: true_dom (dominant lambda of sampled true-active segments),
                 fp_rows (list of per-FP dicts), n_fp_per_event.
    """
    clusters, true_dom, fp_rows, n_fp = [], [], [], []
    for rep in reps:
        A, b, truth, sol, lab, deg, n = solve_event(rep, n_trk=n_trk, **noise)
        fa = (~truth) & (sol > TAU)
        n_fp.append(int(fa.sum()))
        want = set(lab[np.where(fa)[0]])
        want |= set(lab[np.where(truth & (sol > TAU))[0]][::sample_true])
        for c in want:
            mem = np.where(lab == c)[0]
            if len(mem) > max_block:
                continue
            sub = A[mem][:, mem].toarray()
            w, U = np.linalg.eigh(sub)
            beta = U.T @ b[mem]
            ck = beta / w
            tr_m = truth[mem]; fp_m = fa[mem]; maxdeg = int(deg[mem].max())
            clusters.append(dict(w=w, U=U, beta=beta, xC=sol[mem], truth=tr_m, fp=fp_m,
                                 maxdeg=maxdeg, ftype=classify(maxdeg), members=mem))
            for i in np.where(fp_m)[0]:
                contrib = ck * U[i]; kdom = int(np.argmax(np.abs(contrib)))
                # spectral support: fraction of |x_i| carried by each mode
                supp = np.abs(contrib); supp = supp / supp.sum()
                fp_rows.append(dict(rep=rep, clu_size=len(mem), maxdeg=maxdeg,
                                    ftype=classify(maxdeg), lam_dom=float(w[kdom]),
                                    amp=float(sol[mem][i]),
                                    lam_support=w[supp > 0.10].tolist()))  # modes with >10% weight
            if (tr_m & (sol[mem] > TAU)).any():
                for i in np.where(tr_m & (sol[mem] > TAU))[0]:
                    contrib = ck * U[i]; true_dom.append(float(w[int(np.argmax(np.abs(contrib)))]))
    return dict(clusters=clusters, true_dom=np.array(true_dom), fp_rows=fp_rows,
                n_fp_per_event=n_fp)


# ----------------------------------------------------------------------------
# mode filters  g(lambda)  (the multiplier that replaces 1/lambda)
# ----------------------------------------------------------------------------
def g_classical(w):
    return 1.0 / w


def g_1bit(w):
    return np.cos(w * T1 / 2)


def g_2notch(w, lam2):
    return np.cos(w * T1 / 2) * np.cos(w * (np.pi / lam2) / 2)


def g_2notch_arr(w, lam2s):
    """1-bit notch at s, plus extra notches at each lambda in lam2s (product of cosines)."""
    out = np.cos(w * T1 / 2)
    for l2 in np.atleast_1d(lam2s):
        out = out * np.cos(w * (np.pi / l2) / 2)
    return out


# ----------------------------------------------------------------------------
# QSVT band-limited-inverse polynomial design (Step B / C)
# ----------------------------------------------------------------------------
# in-band false eigenvalues to notch (P3-bridge low/mid/high; the isolated-false 4.0
# coincides with the P3 mid, so the 4.0 notch handles both)
NOTCHES = (2.586, 4.0, 5.414)


def _tophat(lam, lo, hi, edge):
    """Smooth top-hat ~1 on [lo,hi], ~0 outside, transition width ~edge."""
    return 0.25 * (1 + np.tanh((lam - lo) / edge)) * (1 + np.tanh((hi - lam) / edge))


def keep_target(lam, band=(2.15, 5.85), edge=0.12, notches=NOTCHES, hw=0.15, invert=True):
    """Band-limited-inverse target: ~1/lambda on the *contiguous* true band [band],
    notched at the in-band false eigenvalues, ~0 outside the band (kills the hubs).
    Contiguous (not per-mode bumps) so it does not force spurious zeros between the
    true modes — that was what collapsed the naive design."""
    lam = np.asarray(lam, float)
    g = _tophat(lam, band[0], band[1], edge)
    for nf in notches:
        g = g * (1 - np.exp(-((lam - nf) / hw) ** 2))
    return (g / lam) if invert else g


def design_keep_poly(degree, dom=(0.5, 6.8), ngrid=4000, **kw):
    """Least-squares Chebyshev fit of `keep_target` on `dom` at the given degree.
    Returns a callable p(lambda) (numpy Chebyshev series)."""
    from numpy.polynomial import chebyshev as Cheb
    x = np.linspace(dom[0], dom[1], ngrid)
    y = keep_target(x, **kw)
    return Cheb.Chebyshev.fit(x, y, degree, domain=dom)


def auc(score, truth):
    """Mann-Whitney AUC: P(score(true) > score(false)). Threshold-free separation."""
    t = np.asarray(truth, bool); s = np.asarray(score, float)
    pos, neg = s[t], s[~t]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(len(allv))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def apply_filter(clusters, g):
    """Apply mode-filter g to every cluster; renormalise to the TRUE-active classical
    signal support; threshold at TAU. Returns dict(eff, fp_kept, n_true, n_fp, scale, auc).

    eff      = fraction of classically-true-active segments still active   (fixed-tau, knife-edge)
    fp_kept  = fraction of false-active segments still active               (fixed-tau)
    auc      = threshold-free separation of all-true vs all-false segments  (robust)
    (eff/fp_kept use the exact nb04 two-notch-scan harness; auc is the noise-robust readout.)
    """
    xCT, xQT, xQF, sAll, tAll = [], [], [], [], []
    for cl in clusters:
        xq = np.abs(cl["U"] @ (cl["beta"] * g(cl["w"])))
        xCT.append(cl["xC"][cl["truth"]]); xQT.append(xq[cl["truth"]]); xQF.append(xq[cl["fp"]])
        sAll.append(xq); tAll.append(cl["truth"])
    xCT = np.concatenate(xCT); xQT = np.concatenate(xQT)
    xQF = np.concatenate(xQF) if any(len(a) for a in xQF) else np.array([])
    mT = xCT > TAU
    scale = np.linalg.norm(xCT[mT]) / max(np.linalg.norm(xQT[mT]), 1e-12)
    eff = float((xQT[mT] * scale > TAU).mean())
    fp_kept = float(((xQF * scale) > TAU).mean()) if xQF.size else 0.0
    return dict(eff=eff, fp_kept=fp_kept, n_true=int(mT.sum()), n_fp=int(xQF.size),
                scale=scale, auc=auc(np.concatenate(sAll), np.concatenate(tAll)))
