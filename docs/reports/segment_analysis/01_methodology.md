# Methodology notes

<!-- STATUS: final -->
<!-- SOURCES: segment_level_analysis.ipynb §1, §2, §2b, §10b, §13d -->

## Detector model
Plane geometry, 5 equispaced modules at $z \in \{33, 66, 99, 132, 165\}$ mm.
Default half-width $40 \times 40\,\text{mm}^2$ (§3–§9, §11–§12); the §10b
old-reference reproduction uses $80\times 80\,\text{mm}^2$ for exact parameter matching.
Primary-vertex sampling: Gaussian with $\sigma = (0,0,1)$ mm for the single-event sweeps
and $(1,1,1)$ mm for the multi-event (§10, §10b) reproduction.

## Event generator
`lhcb_velo_toy.generation.StateEventGenerator`. Two particle types:
- "pion" ($m=139.6$ MeV, $q=1$) for the single-event sweeps (safe_generate).
- "MIP" ($m=0.511$, $q=1$) for §10/§10b multi-event runs.

Track direction uniform in $\phi \in [-\phi_{\max}, \phi_{\max}]$, $\theta \in [-\theta_{\max}, \theta_{\max}]$.
Two `ANGLE_SETTINGS` sweep: dense cone ($\phi_{\max}=0.02$) and broad cone ($\phi_{\max}=0.2$).
Measurement error σ_res is applied to hit $x,y$; collision noise σ_scatt is applied to the
propagation direction per module.

## Acceptance threshold (ε)
[source: §2 `compute_epsilon`]
$$\varepsilon = \sqrt{2\theta_s^2 + 12\theta_r^2 + 2\theta_{\min}^2},$$
with $\theta_s = \text{scale}\cdot\sigma_s$, $\theta_r = \arctan(\text{scale}\cdot\sigma_r / \Delta z)$, scale $=3$, $\theta_{\min}=1.5\times10^{-5}$ rad. The $12\theta_r^2$ term comes from propagating the hit-resolution kink error through both segments in a triplet.

## Segment-pair definitions
Two distinct definitions are used in the notebook:

1. **Shared-middle-hit (triplet) method** — `fast_segment_metrics` in §2b.
   Enumerates all ordered triples $(h_{prev}, h_{mid}, h_{next})$ on consecutive modules.
   A pair is "true" iff both segments share the same truth track. Combinatorics: $O(T^3)$ false pairs.

2. **Pairwise truth-segment method** — `compute_segment_angles_old` in §10b, reproduced in §13d as `fast_pairwise_metrics`.
   Extracts one segment per consecutive-hit-pair *only from truth tracks*, then compares all segment pairs $(i<j)$.
   A pair is "true" iff both segments come from the same truth track. Combinatorics: $O(T^2)$ false pairs.

The two methods coincide in "true" population (both enumerate truth-track adjacent-hit pairs) but differ by orders of magnitude in the "false" pool.

## Hamiltonian solver
`SimpleHamiltonianFast` (library implementation, sanity-checked in §14 against a
custom vectorised builder that assembles the same sparse matrix). Parameters:
$\gamma=3$, $\delta=1$, threshold $\tau=0.35$ for §14+. The Hopfield fixed point
for an isolated segment is $s^*=\delta/(\delta+\gamma)=0.25$; on an isolated
5-hit truth chain, the outermost segment fixed point is $s^*_\text{outer}=0.3636\dots$
(invariant used throughout the analysis).

## Validation
`EventValidator.match_tracks` with LHCb-standard cuts:
purity ≥ 0.70, `hit_efficiency_min = 0.0`, `min_rec_hits = 3`.

## Acceleration
Numba `@njit(parallel=True)` kernels (`_segment_counts_module`, `_pairwise_seg_counts`)
validated against pure-Python reference implementations [source: §2b warm-up cell output "Numba JIT compiled & validated"].
The vectorised builder `_build_ham_vectorized` in §14 is numerically identical to
`SimpleHamiltonianFast.construct_hamiltonian` [source: §14 sanity-check: "A matrices equal: True, b vectors equal: True"].

## Event generation for §14+ (1% hit-drop model)
`apply_hit_noise(event, drop_rate, ghost_rate)` [source: §13 cell] deep-copies
the event and removes hits with probability `drop_rate`, updating
`event.hits`, `track.hit_ids`, and critically `module.hit_ids` (the last is
what `construct_segments` reads). For §14+, the 1%-drop case uses the *clean*
event's true-segment count as efficiency denominator so structural losses
(tracks reduced below 3 hits) are counted as missed.
