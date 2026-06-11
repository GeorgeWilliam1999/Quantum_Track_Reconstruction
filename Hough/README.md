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
