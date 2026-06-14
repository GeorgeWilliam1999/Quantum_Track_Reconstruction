#!/usr/bin/env python3
"""Validate the IN-PLACE fast.py edit against an independent dense reference.

A_lib   = the (now modified) SimpleHamiltonianFast.construct_hamiltonian.
A_dense = built here from scratch with the ORIGINAL full (m x p) dot logic,
          using only the cached segment arrays (no optimised code path).
Asserts max|A_lib - A_dense| == 0 and identical nnz. Also times both.

Usage: python _validate_p3_inplace.py T [sigma_res] [sigma_scatt] [kernel]
"""
import sys, time
import numpy as np
from scipy.sparse import coo_matrix
from pathlib import Path
_SHARED = Path(__file__).resolve().parent
sys.path.insert(0, str(_SHARED))
import helpers as H
from lhcb_velo_toy.solvers import SimpleHamiltonianFast

T = int(sys.argv[1])
sigma_res = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
sigma_scatt = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
kernel = sys.argv[4] if len(sys.argv) > 4 else "step"
conv = (kernel == "erf")

eps = H.compute_epsilon(sigma_res, sigma_scatt)
ev = H.safe_generate(T, seed=T*100+200000, geom=H.make_geometry(),
                     measurement_error=sigma_res, collision_noise=sigma_scatt)

ham = SimpleHamiltonianFast(epsilon=eps, gamma=3.0, delta=1.0, theta_d=1e-4)
ham.construct_segments(ev, materialize_segments=False)
t0 = time.time(); A_lib, b_lib = ham.construct_hamiltonian(ev, convolution=conv); t_lib = time.time()-t0
A_lib = A_lib.tocsr()


def dense_ref(ham, eps, conv):
    """Original dense per-middle-hit scan (no cKDTree, no block shortcut)."""
    from scipy.special import erf
    n = ham.n_segments
    seg_hit = np.asarray(ham._segment_to_hit_ids, dtype=np.int64)
    start_ids = seg_hit[:, 0]; end_ids = seg_hit[:, 1]
    vecs = ham._segment_vectors
    cos_thresh = float(np.cos(eps)); sqrt2_td = ham.theta_d * np.sqrt(2)
    bnd = ham._group_boundaries
    ii_ch, jj_ch, vv_ch = [], [], []
    for g in range(len(bnd) - 2):
        b1, b2, b3 = bnd[g], bnd[g+1], bnd[g+2]
        if b2 == b1 or b3 == b2:
            continue
        in_idx = np.arange(b1, b2); out_idx = np.arange(b2, b3)
        in_key = end_ids[b1:b2]; out_key = start_ids[b2:b3]
        io = np.argsort(in_key, kind="stable"); oo = np.argsort(out_key, kind="stable")
        in_idx_s = in_idx[io]; in_key_s = in_key[io]
        out_idx_s = out_idx[oo]; out_key_s = out_key[oo]
        iu, ifst = np.unique(in_key_s, return_index=True)
        ou, ofst = np.unique(out_key_s, return_index=True)
        ife = np.append(ifst, in_key_s.size); ofe = np.append(ofst, out_key_s.size)
        common, ci, co = np.intersect1d(iu, ou, assume_unique=True, return_indices=True)
        for k in range(common.size):
            ia, ib = ife[ci[k]], ife[ci[k]+1]; oa, ob = ofe[co[k]], ofe[co[k]+1]
            iic = in_idx_s[ia:ib]; jjc = out_idx_s[oa:ob]
            vi = vecs[iic]; vj = vecs[jjc]
            cos_mat = vi @ vj.T
            if conv:
                np.clip(cos_mat, -1.0, 1.0, out=cos_mat)
                val = 1.0 + erf((eps - np.arccos(cos_mat)) / sqrt2_td)
                mask = val > 1e-9
                if not mask.any():
                    continue
                si, sj = np.nonzero(mask)
                ii_ch.append(iic[si]); jj_ch.append(jjc[sj]); vv_ch.append(val[si, sj])
            else:
                mask = cos_mat > cos_thresh
                if not mask.any():
                    continue
                si, sj = np.nonzero(mask)
                ii_ch.append(iic[si]); jj_ch.append(jjc[sj]); vv_ch.append(np.ones(si.size))
    ii = np.concatenate(ii_ch) if ii_ch else np.empty(0, np.int64)
    jj = np.concatenate(jj_ch) if jj_ch else np.empty(0, np.int64)
    vv = np.concatenate(vv_ch) if vv_ch else np.empty(0)
    diag = np.arange(n)
    rows = np.concatenate([diag, ii, jj]); cols = np.concatenate([diag, jj, ii])
    data = np.concatenate([np.full(n, -(ham.delta+ham.gamma)), vv, vv])
    return (-coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()).tocsr()

t0 = time.time(); A_dense = dense_ref(ham, eps, conv); t_dense = time.time()-t0
dA = (A_lib - A_dense)
maxabs = float(np.max(np.abs(dA.data))) if dA.nnz else 0.0
print(f"T={T} kernel={kernel} eps={eps:.2e} n_seg={ham.n_segments} "
      f"nnz_lib={A_lib.nnz} nnz_dense={A_dense.nnz} "
      f"max|dA|={maxabs:.2e}  t_lib={t_lib:.2f}s t_dense={t_dense:.2f}s "
      f"speedup={t_dense/max(t_lib,1e-9):.1f}x  "
      f"{'OK' if (maxabs==0.0 and A_lib.nnz==A_dense.nnz) else 'MISMATCH!!'}")
