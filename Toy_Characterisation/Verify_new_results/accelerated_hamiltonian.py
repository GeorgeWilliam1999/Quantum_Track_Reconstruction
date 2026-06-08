"""
Accelerated Hamiltonian construction for large-event track reconstruction.

Uses vectorised NumPy + Numba-parallel kernels to replace the Python-loop
bottleneck in ``SimpleHamiltonianFast.construct_hamiltonian``.
The API mirrors the original package so it can be used as a drop-in.

Key optimisations
-----------------
1. **Hash-grouped segment pairing** – segments are grouped by their shared
   middle-hit ID so we never iterate over non-matching pairs.
2. **Numba-JIT angle kernel** – pairwise dot-products + arccos are compiled
   to native code and executed in parallel across middle hits.
3. **Vectorised COO assembly** – row/col/data arrays are built without
   Python-level append loops.
4. **Parallel solve** – uses ``scipy.sparse.linalg.spsolve`` for moderate
   sizes and conjugate-gradient for very large matrices.

Requires: numpy, scipy, numba.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import cg, spsolve


# ═══════════════════════════════════════════════════════════════════════════
#  Numba kernels (compiled once, then reused)
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _angles_for_one_hit(vecs_in, vecs_out, epsilon):
    """Compute angles for all (in, out) pairs sharing a middle hit.

    Returns (n_pass, rows_local, cols_local) where rows/cols index into
    vecs_in / vecs_out respectively.  Only pairs with angle < epsilon
    are returned.
    """
    n_in = vecs_in.shape[0]
    n_out = vecs_out.shape[0]
    max_pairs = n_in * n_out
    rows = np.empty(max_pairs, dtype=np.int64)
    cols = np.empty(max_pairs, dtype=np.int64)
    count = 0
    for i in range(n_in):
        vix = vecs_in[i, 0]
        viy = vecs_in[i, 1]
        viz = vecs_in[i, 2]
        for j in range(n_out):
            dot = vix * vecs_out[j, 0] + viy * vecs_out[j, 1] + viz * vecs_out[j, 2]
            # clamp to [-1, 1]
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0
            angle = np.arccos(dot)
            if angle < epsilon:
                rows[count] = i
                cols[count] = j
                count += 1
    return count, rows[:count], cols[:count]


@njit(parallel=True, cache=True)
def _batch_angle_kernel(
    n_middle_hits,
    in_offsets,       # shape (n_middle_hits+1,)
    out_offsets,      # shape (n_middle_hits+1,)
    in_vecs,          # shape (total_in_segs, 3)
    out_vecs,         # shape (total_out_segs, 3)
    epsilon,
):
    """Process all middle hits in parallel.

    Returns (total_pairs, pair_in_local, pair_out_local, pair_mid_idx)
    packed into pre-allocated arrays.  The caller converts local indices
    to global segment IDs.
    """
    # First pass: count pairs per middle hit
    counts = np.zeros(n_middle_hits, dtype=np.int64)
    for m in prange(n_middle_hits):
        n_in = in_offsets[m + 1] - in_offsets[m]
        n_out = out_offsets[m + 1] - out_offsets[m]
        c = 0
        for i in range(n_in):
            vi0 = in_vecs[in_offsets[m] + i, 0]
            vi1 = in_vecs[in_offsets[m] + i, 1]
            vi2 = in_vecs[in_offsets[m] + i, 2]
            for j in range(n_out):
                idx_j = out_offsets[m] + j
                dot = vi0 * out_vecs[idx_j, 0] + vi1 * out_vecs[idx_j, 1] + vi2 * out_vecs[idx_j, 2]
                if dot > 1.0:
                    dot = 1.0
                elif dot < -1.0:
                    dot = -1.0
                if np.arccos(dot) < epsilon:
                    c += 1
        counts[m] = c

    # Compute output offsets
    total = 0
    pair_offsets = np.empty(n_middle_hits + 1, dtype=np.int64)
    pair_offsets[0] = 0
    for m in range(n_middle_hits):
        total += counts[m]
        pair_offsets[m + 1] = total

    # Allocate output arrays
    pair_in_local = np.empty(total, dtype=np.int64)
    pair_out_local = np.empty(total, dtype=np.int64)
    pair_mid = np.empty(total, dtype=np.int64)

    # Second pass: fill output arrays
    for m in prange(n_middle_hits):
        n_in = in_offsets[m + 1] - in_offsets[m]
        n_out = out_offsets[m + 1] - out_offsets[m]
        pos = pair_offsets[m]
        for i in range(n_in):
            vi0 = in_vecs[in_offsets[m] + i, 0]
            vi1 = in_vecs[in_offsets[m] + i, 1]
            vi2 = in_vecs[in_offsets[m] + i, 2]
            for j in range(n_out):
                idx_j = out_offsets[m] + j
                dot = vi0 * out_vecs[idx_j, 0] + vi1 * out_vecs[idx_j, 1] + vi2 * out_vecs[idx_j, 2]
                if dot > 1.0:
                    dot = 1.0
                elif dot < -1.0:
                    dot = -1.0
                if np.arccos(dot) < epsilon:
                    pair_in_local[pos] = i
                    pair_out_local[pos] = j
                    pair_mid[pos] = m
                    pos += 1

    return total, pair_in_local, pair_out_local, pair_mid


# ═══════════════════════════════════════════════════════════════════════════
#  AcceleratedHamiltonian
# ═══════════════════════════════════════════════════════════════════════════

class AcceleratedHamiltonian:
    """Drop-in replacement for ``SimpleHamiltonianFast``.

    Parameters match the original: *epsilon*, *gamma*, *delta*.
    """

    def __init__(self, epsilon: float, gamma: float, delta: float):
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = delta
        self.A: csc_matrix | None = None
        self.b: np.ndarray | None = None

        # Segment bookkeeping (populated by construct_segments)
        self.segments: list = []
        self.n_segments: int = 0
        self._segment_vectors: np.ndarray | None = None
        self._segment_to_hit_ids: list[tuple[int, int]] = []
        self._group_boundaries: list[int] = []
        self.segments_grouped: list[list] = []

    # ─── segment construction ─────────────────────────────────────────
    def construct_segments(self, event) -> None:
        """Build segments from consecutive-module hit pairs.

        Identical semantics to ``SimpleHamiltonianFast.construct_segments``
        but stores data in contiguous NumPy arrays for the JIT kernels.
        """
        from itertools import product as _product
        from lhcb_velo_toy.solvers.reconstruction.segment import Segment

        evt = getattr(event, "true_event", event)

        segments_grouped: list[list] = []
        segments: list = []
        vecs: list[tuple[float, float, float]] = []
        hit_ids: list[tuple[int, int]] = []
        boundaries: list[int] = [0]
        sid = 0

        for idx in range(len(evt.modules) - 1):
            from_hits = evt.get_hits_by_ids(evt.modules[idx].hit_ids)
            to_hits = evt.get_hits_by_ids(evt.modules[idx + 1].hit_ids)
            group: list = []
            for fh, th in _product(from_hits, to_hits):
                seg = Segment(hit_start=fh, hit_end=th, segment_id=sid)
                group.append(seg)
                segments.append(seg)
                dx = th.x - fh.x
                dy = th.y - fh.y
                dz = th.z - fh.z
                norm = np.sqrt(dx * dx + dy * dy + dz * dz)
                if norm > 0:
                    vecs.append((dx / norm, dy / norm, dz / norm))
                else:
                    vecs.append((0.0, 0.0, 1.0))
                hit_ids.append((fh.hit_id, th.hit_id))
                sid += 1
            segments_grouped.append(group)
            boundaries.append(sid)

        self.segments_grouped = segments_grouped
        self.segments = segments
        self.n_segments = sid
        self._segment_vectors = np.ascontiguousarray(vecs, dtype=np.float64)
        self._segment_to_hit_ids = hit_ids
        self._group_boundaries = boundaries

    # ─── Hamiltonian matrix assembly ──────────────────────────────────
    def construct_hamiltonian(self, event, convolution: bool = False,
                              erf_sigma=None):
        """Build the sparse A matrix and bias vector b.

        The heavy inner loops are dispatched to Numba-parallel kernels.
        """
        if not self.segments_grouped:
            self.construct_segments(event)

        n = self.n_segments

        # Start COO lists with diagonal entries
        diag_val = -(self.delta + self.gamma)
        diag_idx = np.arange(n, dtype=np.int64)
        diag_data = np.full(n, diag_val, dtype=np.float64)

        all_rows = [diag_idx]
        all_cols = [diag_idx]
        all_data = [diag_data]

        seg_hit_ids = self._segment_to_hit_ids
        seg_vecs = self._segment_vectors

        for gi in range(len(self.segments_grouped) - 1):
            off_i = self._group_boundaries[gi]
            off_j = self._group_boundaries[gi + 1]
            n_i = self._group_boundaries[gi + 1] - off_i
            n_j = self._group_boundaries[gi + 2] - off_j

            # Group segments by shared middle hit
            # end_hit of group_i segments → list of (local_index_in_group_i)
            end_hit_map: dict[int, list[int]] = {}
            for li in range(n_i):
                end_hid = seg_hit_ids[off_i + li][1]
                end_hit_map.setdefault(end_hid, []).append(li)

            # start_hit of group_j segments → list of (local_index_in_group_j)
            start_hit_map: dict[int, list[int]] = {}
            for lj in range(n_j):
                start_hid = seg_hit_ids[off_j + lj][0]
                start_hit_map.setdefault(start_hid, []).append(lj)

            # Collect shared middle hits
            shared_hits = set(end_hit_map.keys()) & set(start_hit_map.keys())
            if not shared_hits:
                continue

            # Build packed arrays for the batch kernel
            mid_hit_list = sorted(shared_hits)
            n_mid = len(mid_hit_list)

            # Offsets and packed vectors for incoming segments
            in_offsets = np.empty(n_mid + 1, dtype=np.int64)
            out_offsets = np.empty(n_mid + 1, dtype=np.int64)
            in_offsets[0] = 0
            out_offsets[0] = 0

            in_local_indices: list[int] = []
            out_local_indices: list[int] = []

            for k, hid in enumerate(mid_hit_list):
                in_locals = end_hit_map[hid]
                out_locals = start_hit_map[hid]
                in_local_indices.extend(in_locals)
                out_local_indices.extend(out_locals)
                in_offsets[k + 1] = in_offsets[k] + len(in_locals)
                out_offsets[k + 1] = out_offsets[k] + len(out_locals)

            in_global = np.array(in_local_indices, dtype=np.int64) + off_i
            out_global = np.array(out_local_indices, dtype=np.int64) + off_j

            in_vecs = np.ascontiguousarray(seg_vecs[in_global])
            out_vecs = np.ascontiguousarray(seg_vecs[out_global])

            # Dispatch to Numba kernel
            total, p_in, p_out, p_mid = _batch_angle_kernel(
                n_mid, in_offsets, out_offsets, in_vecs, out_vecs, self.epsilon
            )

            if total == 0:
                continue

            # Convert local packed indices back to global segment IDs
            in_global_arr = in_global   # maps packed-in-index → global seg id
            out_global_arr = out_global

            # p_in[k] is the local index within the packed in-array for mid_hit p_mid[k]
            # Actual packed index = in_offsets[p_mid[k]] + p_in[k]
            packed_in_idx = np.empty(total, dtype=np.int64)
            packed_out_idx = np.empty(total, dtype=np.int64)
            for k in range(total):
                packed_in_idx[k] = in_offsets[p_mid[k]] + p_in[k]
                packed_out_idx[k] = out_offsets[p_mid[k]] + p_out[k]

            row_ids = in_global_arr[packed_in_idx]
            col_ids = out_global_arr[packed_out_idx]

            if not convolution:
                vals = np.ones(total, dtype=np.float64)
            else:
                from scipy.special import erf
                # Recompute angles for accepted pairs
                v_i = seg_vecs[row_ids]
                v_j = seg_vecs[col_ids]
                dots = np.sum(v_i * v_j, axis=1)
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.arccos(dots)
                td = erf_sigma if erf_sigma is not None else 1e-4
                vals = 1.0 + erf((self.epsilon - angles) / (td * np.sqrt(2)))

            # Symmetric: add both (i,j) and (j,i)
            all_rows.append(row_ids)
            all_rows.append(col_ids)
            all_cols.append(col_ids)
            all_cols.append(row_ids)
            all_data.append(vals)
            all_data.append(vals)

        rows_cat = np.concatenate(all_rows)
        cols_cat = np.concatenate(all_cols)
        data_cat = np.concatenate(all_data)

        A = coo_matrix((data_cat, (rows_cat, cols_cat)), shape=(n, n)).tocsc()
        b = np.ones(n, dtype=np.float64) * self.delta

        self.A = -A
        self.b = b
        return -A, b

    # ─── solve ────────────────────────────────────────────────────────
    def solve_classicaly(self) -> np.ndarray:
        """Solve A x = b (matches original API including the typo)."""
        if self.A is None or self.b is None:
            raise ValueError("Hamiltonian not constructed.")
        if self.n_segments < 5000:
            try:
                return spsolve(self.A, self.b)
            except Exception:
                x, _ = cg(self.A, self.b, atol=1e-10)
                return x
        else:
            x, _ = cg(self.A, self.b, atol=1e-10)
            return x

    # ─── energy ───────────────────────────────────────────────────────
    def evaluate(self, solution: np.ndarray) -> float:
        sol = np.asarray(solution).ravel()
        return float(-0.5 * sol @ self.A @ sol + self.b @ sol)


# ═══════════════════════════════════════════════════════════════════════════
#  Track extraction (mirrors get_tracks from the package)
# ═══════════════════════════════════════════════════════════════════════════

def get_tracks_accel(hamiltonian, solution, event, threshold=0.0):
    """Extract tracks from solution vector (drop-in for ``get_tracks``)."""
    from lhcb_velo_toy.generation.entities.track import Track

    active_idx = np.where(solution > threshold)[0]
    if len(active_idx) == 0:
        return []

    active_segs = [hamiltonian.segments[i] for i in active_idx]

    # Build adjacency via shared hits
    hit_to_seg: dict[int, list[int]] = {}
    for k, seg in enumerate(active_segs):
        for hid in (seg.hit_start.hit_id, seg.hit_end.hit_id):
            hit_to_seg.setdefault(hid, []).append(k)

    # Connected components via union-find
    parent = list(range(len(active_segs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for indices in hit_to_seg.values():
        for i in range(1, len(indices)):
            union(indices[0], indices[i])

    # Group by root
    groups: dict[int, list[int]] = {}
    for k in range(len(active_segs)):
        groups.setdefault(find(k), []).append(k)

    tracks = []
    for tid, members in enumerate(groups.values()):
        hit_set: set[int] = set()
        hits_list = []
        for k in members:
            seg = active_segs[k]
            for h in (seg.hit_start, seg.hit_end):
                if h.hit_id not in hit_set:
                    hit_set.add(h.hit_id)
                    hits_list.append(h)
        hits_list.sort(key=lambda h: h.z)
        tracks.append(Track(track_id=tid, hit_ids=[h.hit_id for h in hits_list]))

    return tracks
