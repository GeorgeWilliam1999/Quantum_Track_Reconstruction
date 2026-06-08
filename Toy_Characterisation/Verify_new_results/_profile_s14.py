"""Quick profile of §14 per-event cost vs n_trk."""
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spsolve
from collections import defaultdict
from lhcb_velo_toy.solvers import SimpleHamiltonianFast
from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator


def build(ham):
    n = ham.n_segments
    if n == 0:
        return
    start_ids = np.array([s.hit_start.hit_id for s in ham.segments], dtype=np.int64)
    end_ids = np.array([s.hit_end.hit_id for s in ham.segments], dtype=np.int64)
    vecs = ham._segment_vectors
    cos_eps = float(np.cos(ham.epsilon))
    bs, be = defaultdict(list), defaultdict(list)
    for i in range(n):
        bs[int(start_ids[i])].append(i)
        be[int(end_ids[i])].append(i)
    r, c, v = [], [], []
    dv = -(ham.delta + ham.gamma)
    r += list(range(n)); c += list(range(n)); v += [dv] * n
    for mid, sj in bs.items():
        si = be.get(mid, [])
        if not si or not sj:
            continue
        sa = np.asarray(si); ja = np.asarray(sj)
        cm = np.clip(vecs[sa] @ vecs[ja].T, -1, 1)
        ii, jj = np.where(cm >= cos_eps)
        if ii.size == 0:
            continue
        ri = sa[ii]; cj = ja[jj]
        r += ri.tolist(); c += cj.tolist(); v += [1.0] * ri.size
        r += cj.tolist(); c += ri.tolist(); v += [1.0] * ri.size
    A = coo_matrix((v, (r, c)), shape=(n, n)).tocsc()
    ham.A = -A
    ham.b = np.ones(n) * ham.delta


GEO = PlaneGeometry(module_id=list(range(5)), lx=[100.0] * 5, ly=[100.0] * 5,
                    z=[33.0 * (i + 1) for i in range(5)])

def gen_evt(n_trk, seed):
    np.random.seed(seed)
    gen = StateEventGenerator(detector_geometry=GEO, events=1, n_particles=[n_trk],
                              phi_min=-0.2, phi_max=0.2,
                              theta_min=-0.2, theta_max=0.2,
                              measurement_error=0.005, collision_noise=1e-4)
    gen.generate_random_primary_vertices({'z': 0.001})
    gen.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * n_trk])
    return gen.generate_complete_events()


print(f"{'n_trk':>5} {'N_seg':>8} {'seg[s]':>8} {'build[s]':>10} {'spsolve[s]':>12} {'cg[s]':>8} {'diff':>10}")
for n_trk in [50, 100, 200, 400]:
    ev = gen_evt(n_trk, seed=42)
    h = SimpleHamiltonianFast(epsilon=0.002, gamma=3.0, delta=1.0)
    t0 = time.time(); h.construct_segments(ev); t_seg = time.time() - t0
    t0 = time.time(); build(h); t_bld = time.time() - t0
    t0 = time.time(); sol_sp = spsolve(h.A, h.b); t_sp = time.time() - t0
    t0 = time.time(); sol_cg, _ = cg(h.A, h.b, atol=1e-10); t_cg = time.time() - t0
    print(f"{n_trk:5d} {h.n_segments:8d} {t_seg:8.3f} {t_bld:10.3f} {t_sp:12.3f} {t_cg:8.3f} {abs(sol_sp-sol_cg).max():10.2e}")
