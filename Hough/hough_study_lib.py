"""
Shared analysis library for the Hough efficiency deep-dive.

First-principles vote model
---------------------------
Hit k of a track with true slopes t=(tx,ty) from a PV at (0,0,z_pv):

    x_k = t_x (z_k - z_pv) + s_k          (s_k = accumulated scattering displ.)
    d_k = x_k / z_k = t_x (1 - z_pv/z_k) + s_k / z_k

Decomposition of one track's vote cloud:
  * centroid  c = t (1 - z_pv * mean(1/z))        -> radially biased by ~1.4%/mm
  * vertex smear (RADIAL, deterministic per track):
        delta_k = -(z_pv/z_k) t ;  rms = |t||z_pv| std(1/z) ; pp = |t||z_pv| (1/z1-1/z5)
  * scattering smear (random):  rms_k = sigma_s sqrt(sum_{j<k}(z_k-z_j)^2)/z_k  (<~1.1e-4)
  * measurement smear:          rms_k = sigma_res/z_k

Loss model
----------
A track is lost by (M1) MERGE: another track within ~r0 in vote space -> one peak;
(M2) SPLIT: own smear spread > claim radius -> <3 planes claimed; (M3) other.
With T tracks uniform on [-0.2,0.2]^2 (area A=0.16), nearest-neighbour distances are
Poisson:  f(r) = 2 pi lam r exp(-pi lam r^2),  lam=(T-1)/A.  Efficiency law:
    eff(T) = 1 - int p_loss(r) f(r;lam) dr
with p_loss(r) the (T-independent) measured conditional loss curve.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "Toy_Characterisation", "_shared"))

from hough_prototype import TX_RANGE, TY_RANGE, MIN_HITS, PURITY, hit_directions  # noqa

Z_PLANES = np.array([33.0, 66.0, 99.0, 132.0, 165.0])
INVZ = 1.0 / Z_PLANES
INVZ_MEAN = INVZ.mean()          # 0.013838 /mm  (centroid radial bias per mm z_pv)
INVZ_STD = INVZ.std()            # 0.008793 /mm  (vote rms per |t| per mm z_pv)
INVZ_PP = INVZ[0] - INVZ[-1]     # 0.024242 /mm  (peak-to-peak spread)
AREA = (TX_RANGE[1] - TX_RANGE[0] - 0.1) * (TY_RANGE[1] - TY_RANGE[0] - 0.1)  # 0.16: tracks live in [-0.2,0.2]^2


# ---------------------------------------------------------------------------
# Accumulator / peak finding / assignment, fully parameterised
# ---------------------------------------------------------------------------

def bin_width(n_bins):
    return (TX_RANGE[1] - TX_RANGE[0]) / n_bins


def accumulate(D, n_bins, weights=None):
    """Point-vote accumulator. Returns 2-D counts."""
    w = bin_width(n_bins)
    ix = np.clip(((D[:, 0] - TX_RANGE[0]) / w).astype(int), 0, n_bins - 1)
    iy = np.clip(((D[:, 1] - TY_RANGE[0]) / w).astype(int), 0, n_bins - 1)
    flat = ix * n_bins + iy
    counts = np.bincount(flat, weights=weights, minlength=n_bins * n_bins)
    return counts.reshape(n_bins, n_bins).astype(float)


def find_peaks(counts, smooth_sigma=1.0, peak_min=0.0):
    """Smooth + local maxima. Returns (centres[N,2], smoothed)."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    n_bins = counts.shape[0]
    sm = gaussian_filter(counts, smooth_sigma, mode="constant")
    mx = maximum_filter(sm, size=3, mode="constant")
    pj, pk = np.where((sm == mx) & (sm > peak_min))
    w = bin_width(n_bins)
    centres = np.column_stack([TX_RANGE[0] + (pj + 0.5) * w,
                               TY_RANGE[0] + (pk + 0.5) * w])
    return centres, sm


def claim_radius(n_bins, smooth_sigma=1.0):
    """The committed prototype's hit-claim radius (smoothing support)."""
    return 3.0 * bin_width(n_bins) * (smooth_sigma + 1)


def assign(D, planes, centres, radius, min_hits=MIN_HITS):
    """Nearest-peak assignment, one hit per plane. Returns candidate list."""
    from scipy.spatial import cKDTree
    if len(centres) == 0:
        return []
    tree = cKDTree(centres)
    dist, idx = tree.query(D, k=1)
    lab = np.where(dist <= radius, idx, -1)
    cands = []
    for p in np.unique(lab[lab >= 0]):
        members = np.where(lab == p)[0]
        keep = []
        for pl in np.unique(planes[members]):
            cm = members[planes[members] == pl]
            keep.append(cm[np.argmin(np.linalg.norm(D[cm] - centres[p], axis=1))])
        keep = np.array(keep)
        if len(keep) < min_hits:
            continue
        cands.append(dict(hit_idx=keep, tx=D[keep, 0].mean(), ty=D[keep, 1].mean(),
                          n_planes=len(keep)))
    return cands


def run_point(D, planes, n_bins=256, smooth_sigma=1.0):
    """Full point-vote pipeline. Returns (cands, centres, smoothed)."""
    counts = accumulate(D, n_bins)
    centres, sm = find_peaks(counts, smooth_sigma)
    cands = assign(D, planes, centres, claim_radius(n_bins, smooth_sigma))
    return cands, centres, sm


# ---------------------------------------------------------------------------
# Locus voting (the exact hit -> curve Hough)
# ---------------------------------------------------------------------------
# A hit at (x,y,z) is consistent with ANY vertex position zeta in the beam line,
# tracing the curve  d(zeta) = (x,y)/(z - zeta).  All 5 curves of one track
# intersect EXACTLY at t when zeta = z_pv  ->  concurrency restored, the vertex
# smear is gone by construction. This is the reference's hit->line mapping; the
# point vote is its zeta=0 collapse.

def locus_curves(xs, ys, zs, zeta_half=3.5, n_zeta=29):
    """Sampled curves d_i(zeta_m), shape (n_hits, n_zeta, 2), + segment endpoints."""
    zet = np.linspace(-zeta_half, zeta_half, n_zeta)
    denom = zs[:, None] - zet[None, :]
    Dz = np.stack([xs[:, None] / denom, ys[:, None] / denom], axis=-1)
    return Dz, Dz[:, 0, :], Dz[:, -1, :]   # curves, endpoint A, endpoint B


def accumulate_locus(Dz, n_bins):
    """Deduplicated locus accumulator: each (hit, cell) pair counts ONCE,
    so a slow curve (small |t|: d(zeta) barely moves) cannot fake a peak."""
    n_hits, n_zeta, _ = Dz.shape
    w = bin_width(n_bins)
    ix = np.clip(((Dz[..., 0] - TX_RANGE[0]) / w).astype(int), 0, n_bins - 1)
    iy = np.clip(((Dz[..., 1] - TY_RANGE[0]) / w).astype(int), 0, n_bins - 1)
    flat = ix * n_bins + iy                      # (n_hits, n_zeta)
    flat.sort(axis=1)
    first = np.ones_like(flat, dtype=bool)
    first[:, 1:] = flat[:, 1:] != flat[:, :-1]   # first occurrence per hit
    counts = np.bincount(flat[first].ravel(), minlength=n_bins * n_bins)
    return counts.reshape(n_bins, n_bins).astype(float)


def _seg_dist(p, a, b):
    """Distance of points p[N,2] to segment a-b (each [2])."""
    ab = b - a
    denom = (ab * ab).sum()
    if denom == 0:
        return np.linalg.norm(p - a, axis=-1)
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=-1)


def assign_locus(D, planes, centres, heights, segA, segB, radius,
                 min_hits=MIN_HITS):
    """Assign each hit to the HIGHEST peak within `radius` of its LOCUS SEGMENT.

    Height priority is essential: a hit's curve passes EXACTLY through every
    accidental 2-curve crossing it participates in (segment distance 0 ties),
    so nearest-peak assignment scatters hits across spurious low crossings.
    The true track crossing is the 5-fold (highest) peak -> it wins the claim.
    This is the classic Hough readout (peaks processed by height)."""
    from scipy.spatial import cKDTree
    if len(centres) == 0:
        return []
    tree = cKDTree(centres)
    mid = 0.5 * (segA + segB)
    halflen = 0.5 * np.linalg.norm(segB - segA, axis=1)
    n = len(D)
    best_peak = np.full(n, -1)
    best_d = np.full(n, np.inf)
    neigh = tree.query_ball_point(mid, halflen + radius)
    for i in range(n):
        cand = neigh[i]
        if not cand:
            continue
        cand = np.asarray(cand)
        dseg = _seg_dist(centres[cand], segA[i], segB[i])
        ok = dseg <= radius
        if not ok.any():
            continue
        j = np.argmax(heights[cand] - 1e-6 * dseg / radius)  # height, dist tiebreak
        if dseg[j] <= radius:
            best_peak[i] = cand[j]
            best_d[i] = dseg[j]
        else:
            jj = np.where(ok)[0][np.argmax(heights[cand[ok]])]
            best_peak[i] = cand[jj]
            best_d[i] = dseg[jj]
    cands = []
    for p in np.unique(best_peak[best_peak >= 0]):
        members = np.where(best_peak == p)[0]
        keep = []
        for pl in np.unique(planes[members]):
            cm = members[planes[members] == pl]
            keep.append(cm[np.argmin(best_d[cm])])
        keep = np.array(keep)
        if len(keep) < min_hits:
            continue
        cands.append(dict(hit_idx=keep, tx=D[keep, 0].mean(), ty=D[keep, 1].mean(),
                          n_planes=len(keep)))
    return cands


def n_zeta_for(n_bins, zeta_half=3.5, d_max=0.25, z_min=33.0):
    """Sample the locus finely enough that consecutive zeta steps move the vote
    by < 1 bin even for the fastest curve (plane 1, |d|=d_max):
        step <= w * z_min / d_max."""
    w = bin_width(n_bins)
    return int(np.ceil(2 * zeta_half * d_max / (w * z_min))) + 3


def run_locus(event_arrays, n_bins=1024, smooth_sigma=1.0, peak_min=0.35,
              zeta_half=3.5, n_zeta=None, radius_bins=2.0):
    """Full locus-vote pipeline. radius is in BINS (segment distance), default 2."""
    xs, ys, zs, planes = event_arrays
    if n_zeta is None:
        n_zeta = n_zeta_for(n_bins, zeta_half)
    Dz, segA, segB = locus_curves(xs, ys, zs, zeta_half, n_zeta)
    counts = accumulate_locus(Dz, n_bins)
    centres, sm = find_peaks(counts, smooth_sigma, peak_min=peak_min)
    n_b = counts.shape[0]
    w = bin_width(n_b)
    pj = np.clip(((centres[:, 0] - TX_RANGE[0]) / w).astype(int), 0, n_b - 1)
    pk = np.clip(((centres[:, 1] - TY_RANGE[0]) / w).astype(int), 0, n_b - 1)
    heights = sm[pj, pk]
    D = np.column_stack([xs / zs, ys / zs])
    cands = assign_locus(D, planes, centres, heights, segA, segB,
                         radius_bins * bin_width(n_bins))
    return cands, centres, sm


# ---------------------------------------------------------------------------
# Truth bookkeeping (per-track) and evaluation
# ---------------------------------------------------------------------------

def truth_tables(ev, D, planes, truth):
    """Per-truth-track table: centroid, smear spread, z_pv, n_hits, NN distance."""
    pvz = {pv.pv_id: pv.z for pv in ev.primary_vertices}
    rows = []
    for t in ev.tracks:
        m = np.where(truth == t.track_id)[0]
        if len(m) < MIN_HITS:
            continue
        c = D[m].mean(axis=0)
        dd = np.linalg.norm(D[m] - c, axis=1)
        rows.append(dict(tid=t.track_id, cx=c[0], cy=c[1],
                         t_mag=float(np.hypot(*c)),
                         z_pv=float(pvz.get(t.pv_id, 0.0)),
                         n_hits=len(m),
                         spread_rms=float(np.sqrt((dd ** 2).mean())),
                         spread_max=float(dd.max())))
    import pandas as pd
    df = pd.DataFrame(rows)
    if len(df) > 1:
        from scipy.spatial import cKDTree
        C = df[["cx", "cy"]].to_numpy()
        d2, _ = cKDTree(C).query(C, k=2)
        df["nn_dist"] = d2[:, 1]
    else:
        df["nn_dist"] = np.inf
    return df


def eval_candidates(cands, truth):
    """Per-candidate purity table + the matched truth-id set (LHCb majority rule)."""
    import pandas as pd
    rows, matched = [], {}
    for c in cands:
        tids = truth[c["hit_idx"]]
        real = tids[tids >= 0]
        if len(real) == 0:
            rows.append(dict(majority=-1, purity=0.0, n_hits=len(tids),
                             second=-1, n_major=0, n_second=0))
            continue
        vals, cnt = np.unique(real, return_counts=True)
        order = np.argsort(cnt)[::-1]
        maj, n_maj = int(vals[order[0]]), int(cnt[order[0]])
        sec, n_sec = (int(vals[order[1]]), int(cnt[order[1]])) if len(vals) > 1 else (-1, 0)
        pur = n_maj / len(tids)
        rows.append(dict(majority=maj, purity=pur, n_hits=len(tids),
                         second=sec, n_major=n_maj, n_second=n_sec))
        if pur >= PURITY:
            matched[maj] = matched.get(maj, 0) + 1
    return pd.DataFrame(rows), matched


def per_track_outcome(df_truth, matched):
    df = df_truth.copy()
    df["matched"] = df["tid"].map(lambda t: matched.get(t, 0) > 0)
    return df


# ---------------------------------------------------------------------------
# Analytic pieces
# ---------------------------------------------------------------------------

def f_nn(r, T):
    """Poisson nearest-neighbour pdf for T uniform tracks on area AREA."""
    lam = max(T - 1, 1) / AREA
    return 2 * np.pi * lam * r * np.exp(-np.pi * lam * r * r)


def logistic_loss(r, b, a, r0, s):
    """Conditional loss model: baseline b (smear losses) + merge sigmoid."""
    return b + a / (1.0 + np.exp((r - r0) / s))


def eff_model(T, popt, rmax=0.06):
    """eff(T) = 1 - integral p_loss(r) f_nn(r;T) dr  (no per-T fitting)."""
    r = np.linspace(1e-6, rmax, 4000)
    p = logistic_loss(r, *popt)
    return 1.0 - np.trapezoid(p * f_nn(r, T), r)


def load_clean_store_events():
    """The 160 clean Verify events shared with the 3-solver benchmark."""
    import pandas as pd
    import qtrk_pipeline as qp
    df = pd.read_csv(qp.store.manifest_dir() / "events.csv")
    sel = df[(df.sigma_scatt == 1e-4) & (df.sigma_res == 0.0) &
             (df.hit_ineff == 0.0) & (df.ghost_rate == 0.0) &
             (df.phi_max == 0.2)]
    return sel, qp


def event_arrays(ev):
    xs = np.array([h.x for h in ev.hits]); ys = np.array([h.y for h in ev.hits])
    zs = np.array([h.z for h in ev.hits])
    planes = np.array([h.module_id for h in ev.hits])
    truth = np.array([h.track_id for h in ev.hits])
    return xs, ys, zs, planes, truth
