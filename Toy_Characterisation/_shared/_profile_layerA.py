#!/usr/bin/env python3
"""Layer-A profiler: sparse matrix A build / store / solve scaling.

Run ONE T per process so resource.getrusage peak RSS is clean.
Usage: python _profile_layerA.py T [sigma_res] [sigma_scatt]
Emits one JSON line on stdout.
"""
import sys, time, json, resource, gc, tracemalloc
import numpy as np
from pathlib import Path

_SHARED = Path(__file__).resolve().parent
sys.path.insert(0, str(_SHARED))

import helpers as H
from lhcb_velo_toy.solvers import SimpleHamiltonianFast

T = int(sys.argv[1])
sigma_res = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
sigma_scatt = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4

eps = H.compute_epsilon(sigma_res, sigma_scatt)
geom = H.make_geometry()

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB->MB on linux

t0 = time.time()
ev = H.safe_generate(T, seed=T * 100 + 200000, geom=geom,
                     measurement_error=sigma_res, collision_noise=sigma_scatt)
t_gen = time.time() - t0
n_hits = len(ev.hits)
n_tracks = len(ev.tracks)

ham = SimpleHamiltonianFast(epsilon=eps, gamma=3.0, delta=1.0)

# --- segment build (materialize=False, the high-T path) ---
gc.collect()
t0 = time.time()
ham.construct_segments(ev, materialize_segments=False)
t_seg = time.time() - t0
n_seg = ham.n_segments

# --- A build (step kernel; the O(T^3?) candidate scan) ---
tracemalloc.start()
gc.collect()
rss_before = rss_mb()
t0 = time.time()
A, b = ham.construct_hamiltonian(ev, convolution=False)
t_Abuild = time.time() - t0
cur, peak_py = tracemalloc.get_traced_memory()
tracemalloc.stop()
rss_after_build = rss_mb()

nnz = int(A.nnz)
ratio = nnz / max(n_seg, 1)
A_bytes = A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
fmt = type(A).__name__

# --- classical solve (matches pipeline solve_classical) ---
from scipy.sparse.linalg import spsolve, minres
t0 = time.time()
if n_seg < 5000:
    sol = spsolve(A, b); minres_iter = -1; info = 0
else:
    # count iterations via callback
    it = {"n": 0}
    def cb(xk): it["n"] += 1
    sol, info = minres(A, b, rtol=1e-8, maxiter=5000, callback=cb)
    minres_iter = it["n"]
t_solve = time.time() - t0
sol = np.asarray(sol).ravel()
resid = float(np.linalg.norm(A @ sol - b) / (np.linalg.norm(b) + 1e-30))
rss_after_solve = rss_mb()

# --- conditioning: spectral extremes (block-diagonal -> Lanczos OK) ---
lam_lo = lam_hi = None
kappa = None
try:
    from scipy.sparse.linalg import eigsh
    t0 = time.time()
    lam_hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False, maxiter=3000)[0])
    lam_lo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False, maxiter=3000)[0])
    t_eig = time.time() - t0
    if lam_lo > 0:
        kappa = lam_hi / lam_lo
except Exception as e:
    t_eig = -1.0

out = dict(
    T=T, sigma_res=sigma_res, sigma_scatt=sigma_scatt, eps=eps,
    n_hits=n_hits, n_tracks=n_tracks, n_seg=n_seg,
    t_gen=round(t_gen, 3), t_seg=round(t_seg, 4), t_Abuild=round(t_Abuild, 4),
    t_solve=round(t_solve, 4), t_eig=round(t_eig, 3),
    nnz=nnz, nnz_over_nseg=round(ratio, 4), A_MB=round(A_bytes / 1e6, 2),
    fmt=fmt, peak_py_MB=round(peak_py / 1e6, 2),
    rss_before_MB=round(rss_before, 1), rss_after_build_MB=round(rss_after_build, 1),
    rss_after_solve_MB=round(rss_after_solve, 1),
    minres_iter=minres_iter, minres_info=info, resid=resid,
    lam_lo=lam_lo, lam_hi=lam_hi, kappa=kappa,
)
print(json.dumps(out))
