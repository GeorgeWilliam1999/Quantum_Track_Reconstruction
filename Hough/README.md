# Hough-transform track finding on the LHCb VeLo toy

A *direct* track-building alternative to the segment linear-system solvers
(1BQF / QSVT). Instead of building the segment–segment Hamiltonian `A = sI − C`
and filtering its spectrum, each hit **votes** in a parameter-space accumulator
and tracks appear as **peaks** (cf. https://dnicotra.github.io/hough_tracking/).

## The mapping (why it is clean on this toy)

Toy tracks are straight lines `x(z) = x0 + tx·z`, `y(z) = y0 + ty·z` from a PV
pinned to `x = y = 0` (only `z_pv ~ N(0, 1 mm)`). So `x0 = −tx·z_pv` is O(0.2 mm)
small, and to leading order each hit's direction `(x/z, y/z) ≈ (tx, ty)`. The
general 4-parameter line-Hough therefore **collapses to a 2-D directional Hough**
in `(tx, ty)`: each track is a concurrent bundle of its 5 hits at one point; the
accumulator peaks there.

## Method (`hough_prototype.py`)

1. Each hit → vote at `(x/z, y/z)` on a 256×256 grid over `[−0.25, 0.25]²`.
2. Gaussian-smooth the accumulator (σ=1 bin) so a track's votes — spread over a
   few bins by the vertex-z / scattering smear — reinforce into one local maximum.
3. Local-maxima detection = track directions; assign each hit to its nearest peak
   (one hit per plane); accept peaks covering ≥3 distinct planes.
4. Truth-match by the LHCb majority rule (≥70% of a candidate's hits from one
   truth track) → efficiency / ghost / clone.

## Result (5 planes, σ_scatt = 1e-4, σ_res = 0)

| T   | efficiency | ghost rate | clone rate | solve (1 core) |
|-----|-----------:|-----------:|-----------:|---------------:|
| 10  | 1.000      | 0.000      | 0.000      | ~9 ms          |
| 25  | 1.000      | 0.000      | 0.000      | ~13 ms         |
| 50  | 0.98–1.00  | 0.00       | 0.000      | ~20 ms         |
| 100 | 0.97       | 0.00–0.01  | 0.000      | ~35 ms         |
| 200 | 0.955–0.96 | 0.00–0.01  | 0.000      | ~64 ms         |
| 400 | 0.914      | 0.012      | 0.000      | ~116 ms        |

**Headline:** 95–100 % efficiency with a **~0 % ghost rate** to T=200, 91 % at
T=400. Figures in `outputs/`:
- `hough_accumulator.png` — 50 tracks → 50 clean peaks + the (z,x) event display.
- `hough_efficiency.png` — efficiency / ghost vs T and wall time.

## Why this matters vs 1BQF / QSVT

- **Ghost-free by construction.** The segment solvers' entire false-positive story
  is the segment graph: cross-track "bridge" chains that are spectrally degenerate
  with true tracks (the floor theorem) → far grows to 20 % (classical) / 44 %
  (1BQF) by high T; QSVT's comb pushes it back to ~1 % at degree cost. Hough has
  **no segment graph** — a ghost needs ≥3 hits accidentally collinear *through the
  origin*, which is geometrically rare (measured ~0 %).
- **Parameter space is fixed in T.** The accumulator size is set by angular
  **resolution**, not by occupancy. This is the structural opposite of the segment
  register `⌈log2(4T²)⌉` (16→24 qubits over T=50→1000) — and is exactly the
  constant-width win D3 (cluster decomposition) chases, but Hough gets it natively.

See `PLAN.md` for the quantum design and the honest competitive scoreboard.

## The efficiency, from first principles (deep dive)

Scripts `01..04_*.py` + `hough_study_lib.py`, figures in `outputs/deep_dive/`.
Everything below is derived, then measured, then cross-checked on the 160 shared
store events.

**The vote model.** Hit k of a track with slopes **t** from a PV at (0,0,z_pv):
`d_k = t (1 − z_pv/z_k) + s_k/z_k`. One track's 5 votes are smeared **radially**
by the vertex term `δ_k = −(z_pv/z_k) t`:
- rms = `|t||z_pv|·std(1/z)` = `|t||z_pv|·8.79e-3/mm` — verified point-by-point
  (fig03; controlled experiments fig02 reproduce the sampled closed form);
- scattering contributes only ~5e-5 (**18× below** the vertex term);
- with z_pv = 0 and σ_s = 0 the votes coincide to machine precision.

**The loss mechanism is merging, full stop (figs 04–10).**
- NN distances in vote space are Poisson `2πλr·exp(−πλr²)`, λ=(T−1)/0.16 (fig04).
- `P(lost | NN = r)` is one **universal, T-independent sigmoid**: amplitude
  a = 0.63, midpoint r₀ = 2.5 bins, baseline b = 0.000 (fig05). The peak-finder,
  not the event, owns the curve.
- The parameter-free law `eff(T) = 1 − ∫ p_loss(r) f_NN(r;λ) dr` reproduces every
  measured point to ≤1% (fig06). Census: ~100% of lost tracks are merges; the
  321 isolated tracks have **zero** losses at the 256 grid (fig07/08).
- Ghosts are merged-pair products (impure mixed candidates), not random froth
  (fig09); fig10 shows a merged (1.6-bin) vs resolved (3.8-bin) pair in the
  accumulator.

**The resolution law (figs 11–13).** Across grids 64²–2048²: r₀ = 2.5·w at every
grid (no smear floor — the radial smear never broadens the peak; the far-plane
votes form the core and the near-plane votes *fragment away* instead). The
fragmentation (M2) channel switches on when r₀ undercuts the radial spread:
0% (256), ~1.7% (512), ~8.8% (1024), ~33% (2048) — predicted by a single-linkage
cluster criterion at r₀ with no fitting. The two-mechanism model
`eff = (1 − split_N)(1 − a(1 − e^{−πλr₀²}))` collapses all (grid, T) points.

**Locus voting dismantles the smear (figs 14–15).** The point vote assumes
z_pv = 0; the exact Hough votes along the hit's vertex locus
`d(ζ) = (x,y)/(z−ζ)`, ζ ∈ ±3.5 mm — a short radial segment. All 5 segments of a
track cross **exactly** at t (fig14), so the smear is gone *by construction* and
fine grids stop fragmenting; residual losses are genuine near-collinear pairs
(verified per-track). Implementation notes that matter: the ζ sampling must move
< 1 bin per step (`n_zeta_for`), votes must be deduplicated per (hit, cell), and
hits must be claimed by **peak height** (a hit's locus passes exactly through
every accidental 2-fold crossing it participates in — nearest-peak assignment
scatters hits; height-priority is the classic Hough readout).
