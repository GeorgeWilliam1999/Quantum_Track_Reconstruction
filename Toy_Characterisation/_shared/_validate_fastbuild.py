#!/usr/bin/env python3
"""Validate an exact O(T^2 log T) A-build (cKDTree radius query) against the
current O(T^3) full-block scan in SimpleHamiltonianFast.construct_hamiltonian.

The accelerated build finds candidate (in,out) segment pairs per middle hit via a
fixed-radius neighbour query on the 2D projected directions, then applies the SAME
exact cos>cos(eps) test — so the surviving pair set (and A) is identical.

Usage: python _validate_fastbuild.py T  -> prints max|dA|, nnz match, timings.
"""
import sys, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from pathlib import Path
_SHARED = Path(__file__).resolve().parent
sys.path.insert(0, str(_SHARED))
import helpers as H
from lhcb_velo_toy.solvers import SimpleHamiltonianFast

T = int(sys.argv[1])
eps = H.compute_epsilon(0.0, 1e-4)
ev = H.safe_generate(T, seed=T*100+200000, geom=H.make_geometry(),
                     measurement_error=0.0, collision_noise=1e-4)

# --- reference: the library build ---
ham = SimpleHamiltonianFast(epsilon=eps, gamma=3.0, delta=1.0)
ham.construct_segments(ev, materialize_segments=False)
t0 = time.time(); A_ref, b_ref = ham.construct_hamiltonian(ev, convolution=False); t_ref = time.time()-t0
A_ref = A_ref.tocsr()

# --- accelerated build (step kernel), reusing the SAME cached arrays ---
def fast_build(ham, eps, gamma=3.0, delta=1.0):
    n = ham.n_segments
    seg_hit = np.asarray(ham._segment_to_hit_ids, dtype=np.int64)
    start_ids = seg_hit[:, 0]; end_ids = seg_hit[:, 1]
    vecs = ham._segment_vectors
    cos_thresh = float(np.cos(eps))
    # radius in direction space: |u-v|^2 = 2(1-cos angle); angle<eps <=> |u-v|<r
    r = float(np.sqrt(max(2.0*(1.0 - cos_thresh), 0.0)))
    bnd = ham._group_boundaries
    ii_ch, jj_ch = [], []
    for g in range(len(bnd) - 2):
        b1, b2, b3 = bnd[g], bnd[g+1], bnd[g+2]
        if b2 == b1 or b3 == b2:
            continue
        in_idx = np.arange(b1, b2); out_idx = np.arange(b2, b3)
        in_key = end_ids[b1:b2]; out_key = start_ids[b2:b3]
        # group by shared middle hit (incoming end == outgoing start)
        in_order = np.argsort(in_key, kind="stable"); out_order = np.argsort(out_key, kind="stable")
        in_idx_s = in_idx[in_order]; in_key_s = in_key[in_order]
        out_idx_s = out_idx[out_order]; out_key_s = out_key[out_order]
        in_u, in_first = np.unique(in_key_s, return_index=True)
        out_u, out_first = np.unique(out_key_s, return_index=True)
        in_first_ext = np.append(in_first, in_key_s.size)
        out_first_ext = np.append(out_first, out_key_s.size)
        common, ci, co = np.intersect1d(in_u, out_u, assume_unique=True, return_indices=True)
        for k in range(common.size):
            ia, ib = in_first_ext[ci[k]], in_first_ext[ci[k]+1]
            oa, ob = out_first_ext[co[k]], out_first_ext[co[k]+1]
            ii_chunk = in_idx_s[ia:ib]; jj_chunk = out_idx_s[oa:ob]
            vi = vecs[ii_chunk]; vj = vecs[jj_chunk]
            # radius query: for each outgoing, candidate incomings within r
            tree = cKDTree(vi)
            cand = tree.query_ball_point(vj, r)            # list per outgoing
            for jo, lst in enumerate(cand):
                if not lst:
                    continue
                li = np.asarray(lst, dtype=np.int64)
                # EXACT test, identical to the library
                cosvals = vi[li] @ vj[jo]
                keep = cosvals > cos_thresh
                if keep.any():
                    ii_ch.append(ii_chunk[li[keep]])
                    jj_ch.append(np.full(int(keep.sum()), jj_chunk[jo], dtype=np.int64))
    if ii_ch:
        ii = np.concatenate(ii_ch); jj = np.concatenate(jj_ch)
    else:
        ii = np.empty(0, np.int64); jj = np.empty(0, np.int64)
    vv = np.ones(ii.size)
    diag = np.arange(n)
    rows = np.concatenate([diag, ii, jj]); cols = np.concatenate([diag, jj, ii])
    diag_val = -(delta + gamma)
    data = np.concatenate([np.full(n, diag_val), vv, vv])
    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
    return (-A).tocsr()

t0 = time.time(); A_fast = fast_build(ham, eps); t_fast = time.time()-t0

dA = (A_ref - A_fast)
maxabs = float(np.max(np.abs(dA.data))) if dA.nnz else 0.0
print(f"T={T} n_seg={ham.n_segments} nnz_ref={A_ref.nnz} nnz_fast={A_fast.nnz} "
      f"max|dA|={maxabs:.2e} t_ref={t_ref:.2f}s t_fast={t_fast:.2f}s speedup={t_ref/max(t_fast,1e-9):.1f}x")
