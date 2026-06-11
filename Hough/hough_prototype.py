"""
Hough-transform track finding on the LHCb VeLo toy — classical prototype.

Goal
----
Test the idea (cf. https://dnicotra.github.io/hough_tracking/) that tracks can be
built *directly* by a Hough transform instead of solving the segment-segment
linear system that the 1BQF / QSVT solvers use.  This is the classical reference
implementation + a truth-matched performance measurement, used to decide whether a
quantum Hough primitive (amplitude-encoded voting + Grover maximum-finding) is
worth building to compete with QSVT / 1BQF.

Geometry / physics (from Toy_Characterisation/_shared/helpers.py)
-----------------------------------------------------------------
- 5 planes at z = {33, 66, 99, 132, 165} mm, half-width 40 mm.
- Tracks are straight lines  x(z) = x0 + tx*z ,  y(z) = y0 + ty*z .
- The primary vertex is pinned to x=y=0 (PV_SIGMA x:0, y:0; only z spreads, sigma=1mm).
  => x0 = -tx*z_pv, y0 = -ty*z_pv are O(0.2 * 1mm) = O(0.2mm) small.
  => to leading order each hit's direction (x/z, y/z) ~ (tx, ty): the 4-parameter
     line-Hough collapses to a 2-D *directional* Hough in (tx, ty).  Each track is a
     concurrent bundle of its 5 hits at one (tx, ty) point; the accumulator peaks
     there.  (The general 2-parameter (slope, intercept) line-Hough is also provided
     below for illustration / the unconstrained-vertex case.)

Method
------
Directional accumulator over (tx, ty):
  for each hit i:  t_i = (x_i / z_i, y_i / z_i)   -> one vote
  bin votes on a TX x TY grid; a bin with >= MIN_HITS votes from >= MIN_HITS
  *distinct planes* is a track candidate, refined to the vote centroid.
Greedy peak extraction with hit removal (deflation) so each hit is used once.

Truth matching (LHCb-style): a reconstructed candidate is matched to the truth
track contributing >= 70% of its hits ("majority rule"); efficiency, ghost rate,
clone rate reported per the usual tracking definitions.
"""
from __future__ import annotations
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Toy_Characterisation", "_shared"))
from helpers import make_geometry, safe_generate  # noqa: E402

# --- accumulator configuration ------------------------------------------------
TX_RANGE = (-0.25, 0.25)
TY_RANGE = (-0.25, 0.25)
N_BINS = 256          # per axis  -> 256x256 = 65536 accumulator bins
MIN_HITS = 3          # min hits (== min distinct planes) for a track candidate
PURITY = 0.70         # majority-rule truth-match fraction


def hit_directions(event):
    """Return (D, planes, truth) where D[i]=(x/z, y/z), planes=module_id, truth=track_id."""
    xs = np.array([h.x for h in event.hits])
    ys = np.array([h.y for h in event.hits])
    zs = np.array([h.z for h in event.hits])
    planes = np.array([h.module_id for h in event.hits])
    truth = np.array([h.track_id for h in event.hits])
    D = np.column_stack([xs / zs, ys / zs])
    return D, planes, truth


def hough_accumulate(D, n_bins=N_BINS):
    """2-D directional Hough accumulator. Returns (counts, hit_bin_index)."""
    tx_edges = np.linspace(*TX_RANGE, n_bins + 1)
    ty_edges = np.linspace(*TY_RANGE, n_bins + 1)
    ix = np.clip(np.digitize(D[:, 0], tx_edges) - 1, 0, n_bins - 1)
    iy = np.clip(np.digitize(D[:, 1], ty_edges) - 1, 0, n_bins - 1)
    flat = ix * n_bins + iy
    counts = np.bincount(flat, minlength=n_bins * n_bins).reshape(n_bins, n_bins)
    return counts, flat


SMOOTH_SIGMA = 1.0   # Gaussian smoothing of the accumulator (in bins)


def extract_tracks(D, planes, flat, n_bins=N_BINS, min_hits=MIN_HITS,
                   smooth_sigma=SMOOTH_SIGMA):
    """
    Standard Hough peak-finder (robust to the vertex-z / scattering smear):
      1. smooth the accumulator with a small Gaussian so a track's votes, spread
         over a few adjacent bins, reinforce into ONE local maximum;
      2. detect local maxima above threshold (a peak = a track direction);
      3. assign each hit to the nearest peak centre in (tx,ty) within one bin-radius
         of smoothing support, one hit per plane (closest);
      4. accept peaks covering >= min_hits DISTINCT planes.
    Returns list of candidate dicts: {hit_idx, tx, ty, n_planes}.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    counts = np.bincount(flat, minlength=n_bins * n_bins).reshape(n_bins, n_bins).astype(float)
    sm = gaussian_filter(counts, smooth_sigma, mode="constant")
    # local maxima of the smoothed accumulator
    mx = maximum_filter(sm, size=3, mode="constant")
    peaks = (sm == mx) & (sm > 0)
    pj, pk = np.where(peaks)
    if len(pj) == 0:
        return []
    tx_c = 0.5 * (TX_RANGE[0] + TX_RANGE[1])
    tx_w = (TX_RANGE[1] - TX_RANGE[0]) / n_bins
    ty_w = (TY_RANGE[1] - TY_RANGE[0]) / n_bins
    peak_tx = TX_RANGE[0] + (pj + 0.5) * tx_w
    peak_ty = TY_RANGE[0] + (pk + 0.5) * ty_w
    centres = np.column_stack([peak_tx, peak_ty])
    # assign each hit to nearest peak centre
    from scipy.spatial import cKDTree
    tree = cKDTree(centres)
    radius = 3.0 * max(tx_w, ty_w) * (smooth_sigma + 1)   # smoothing support
    dist, idx = tree.query(D, k=1)
    assign = np.where(dist <= radius, idx, -1)
    cands = []
    for p in range(len(centres)):
        members = np.where(assign == p)[0]
        if len(members) < min_hits:
            continue
        keep = []
        for pl in np.unique(planes[members]):
            cm = members[planes[members] == pl]
            keep.append(cm[np.argmin(np.linalg.norm(D[cm] - centres[p], axis=1))])
        keep = np.array(keep)
        if len(keep) < min_hits:
            continue
        tx, ty = D[keep].mean(axis=0)
        cands.append(dict(hit_idx=keep, tx=tx, ty=ty, n_planes=len(keep)))
    return cands


def evaluate(cands, truth, event):
    """LHCb-style efficiency / ghost / clone using majority (>=PURITY) truth match."""
    true_ids = [t.track_id for t in event.tracks if len(t.hit_ids) >= MIN_HITS]
    n_true = len(true_ids)
    matched_truth = {}   # truth_id -> n candidates matched (for clones)
    n_ghost = 0
    for c in cands:
        tids = truth[c["hit_idx"]]
        vals, counts = np.unique(tids[tids >= 0], return_counts=True)
        if len(vals) == 0:
            n_ghost += 1
            continue
        j = np.argmax(counts)
        frac = counts[j] / len(tids)
        if frac >= PURITY:
            matched_truth.setdefault(int(vals[j]), 0)
            matched_truth[int(vals[j])] += 1
        else:
            n_ghost += 1
    n_reco = len(matched_truth)
    n_clone = sum(v - 1 for v in matched_truth.values())
    eff = n_reco / n_true if n_true else 0.0
    ghost_rate = n_ghost / max(1, len(cands))
    clone_rate = n_clone / max(1, n_reco) if n_reco else 0.0
    return dict(n_true=n_true, n_cands=len(cands), eff=eff,
                ghost_rate=ghost_rate, clone_rate=clone_rate, n_ghost=n_ghost)


def run_one(n_trk, seed, geom, sigma_scatt=1e-4, sigma_res=0.0):
    ev = safe_generate(n_trk, seed, geom,
                       measurement_error=sigma_res, collision_noise=sigma_scatt)
    D, planes, truth = hit_directions(ev)
    t0 = time.time()
    counts, flat = hough_accumulate(D)
    cands = extract_tracks(D, planes, flat)
    dt = time.time() - t0
    res = evaluate(cands, truth, ev)
    res.update(n_trk=n_trk, seed=seed, t_solve=dt, n_hits=len(D))
    return res


if __name__ == "__main__":
    geom = make_geometry()
    Ts = [10, 25, 50, 100, 200]
    reps = 5
    print(f"{'T':>5} {'eff':>7} {'ghost':>7} {'clone':>7} {'nhit':>6} {'t_ms':>7}")
    rows = []
    for T in Ts:
        accum = []
        for r in range(reps):
            res = run_one(T, 1000 + r, geom)
            accum.append(res)
        eff = np.mean([a["eff"] for a in accum])
        gh = np.mean([a["ghost_rate"] for a in accum])
        cl = np.mean([a["clone_rate"] for a in accum])
        nh = np.mean([a["n_hits"] for a in accum])
        tms = 1e3 * np.mean([a["t_solve"] for a in accum])
        print(f"{T:>5} {eff:>7.3f} {gh:>7.3f} {cl:>7.3f} {nh:>6.0f} {tms:>7.1f}")
        rows.append(dict(T=T, eff=eff, ghost=gh, clone=cl, nhit=nh, t_ms=tms))
    np.save(os.path.join(os.path.dirname(__file__), "outputs", "hough_sweep.npy"),
            np.array(rows, dtype=object))
    print("saved outputs/hough_sweep.npy")
