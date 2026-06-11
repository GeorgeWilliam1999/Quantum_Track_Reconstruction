# Quantum Hough tracking — design & competitive analysis vs 1BQF / QSVT

## 0. The question

The 1BQF and QSVT solvers both attack the **same object**: the segment–segment
operator `A = (γ+δ)I − C`, solved as a linear system and filtered in its spectrum.
Their hard limit is the **coupling floor** — cross-track chains graph-isomorphic to
true tracks (floor theorem). The Hough transform is a *structurally different*
primitive: a **voting accumulator + peak-finding**. There is no `A`, no spectrum,
no coupling graph. The question is whether a *quantum* Hough can compete on the
metrics that matter (efficiency, ghost rate) **and** on resources (qubits, depth).

The classical prototype (`README.md`) already establishes the physics baseline:
95–100 % efficiency, ~0 % ghost to T=200. So the Hough *response* is competitive.
This plan is about the **quantum realisation** and where it could actually win.

## 1. The quantum primitive

A Hough accumulator is `acc[bin] = Σ_hits 1[hit votes for bin]`. Two quantum maps:

**(P1) Amplitude-encoded voting.** Prepare, over a bin register of
`q = ⌈log2 N_bins⌉` qubits, a state `|ψ⟩ = Σ_bin (acc[bin]/Z) |bin⟩` using a
"vote" unitary that, in superposition over bins, accumulates the indicator of each
hit. Sampling `|ψ⟩` returns bins with probability `∝ acc[bin]²` — quadratic
contrast favouring peaks (tracks).

**(P2) Grover / Dürr–Høyer maximum-finding.** With a counting oracle
`O|bin⟩ = |bin⟩|acc[bin]⟩` (quantum counting / amplitude estimation over the hit
set), Dürr–Høyer returns the **fullest bin in O(√N_bins)** oracle calls. Deflate
the claimed hits and repeat per track.

The deployable algorithm is **P1 to concentrate amplitude on the peak band, then
P2 (amplitude amplification) to read out the T peaks** — analogous to the QSVT
"active-set readout with AA" (WP3/WP6), but over bins instead of segments.

## 2. Resource scoreboard (the actual competition)

| primitive | system qubits | scaling in T | false-positive mechanism |
|-----------|---------------|--------------|--------------------------|
| 1BQF      | ⌈log2 4T²⌉ + 2 (16→24) | grows as 2·log T | coupling floor, far→44 % |
| QSVT comb | ⌈log2 4T²⌉ + 1 + ⌈log2(d+1)⌉ (20→29) | grows + degree | floor pushed to ~1 % at degree cost |
| QSVT + D3 | ⌈log2 max_cluster⌉ + reg (3–6) | **const in T** | floor unchanged (per-cluster) |
| **Hough** | ⌈log2 N_bins⌉ (= 16 @ 256², **const in T**) | **set by resolution, not T** | accidental 3-hit collinearity ≈ 0 % |

Two genuine structural wins, both measured/derived:
1. **Width constant in T** — like D3, but native (the accumulator does not grow
   with occupancy, only with angular resolution).
2. **No coupling floor** — the dominant QSVT/1BQF failure mode is simply absent.

## 3. The honest costs (where Hough is worse — do not oversell)

- **Resolution wall.** Bin size must exceed the scattering + vertex smear (~0.005)
  yet resolve adjacent tracks; at fixed resolution efficiency degrades with density
  (94 %→91 %, T=200→400). The segment solvers have no binning wall. Quantum does
  **not** remove this — finer bins = more qubits, not free.
- **Displaced vertices kill the 2-D reduction.** The directional collapse only
  holds because the toy PV is pinned to `x=y=0`. Real LHCb / the Run-3 events have
  displaced vertices → the full **4-parameter** Hough `(tx,ty,x0,y0)` → `N_bins⁴`
  bins, `q ≈ 32` qubits, and a far harder peak-find. This is the Hough analogue of
  the QSVT "real-geometry comb breakdown" (WP5). **Test on Run-3 before any claim.**
- **Quantum vs classical peak-find is not obviously a win.** Classical Hough is
  already O(N_hits + N_bins) and costs ~60 ms at T=200. Grover max-finding is
  `O(√N_bins)` *per track* × T tracks — not asymptotically better than one
  classical O(N_bins) scan in 2-D.

## 4. Where quantum Hough could actually win (the thesis)

Quantum helps Hough **exactly where the classical accumulator is too large to
materialise**: high-resolution, **high-dimensional** parameter spaces. Add the
displaced vertex (4-D) or a magnetic-field curvature term (5-D) and `N_bins`
reaches 10⁹–10¹². There:
- a **~32–40 qubit** register (still **constant in T**) represents the entire
  accumulator that classically cannot be stored;
- amplitude-encoded voting (P1) fills it implicitly from the hit oracle, never
  building the array;
- Grover/AA readout (P2) finds the T peaks in `O(√N_bins) = O(N_bins^{1/2})` vs the
  classical `O(N_bins)` materialise-and-scan — a genuine quadratic edge **in the
  regime where classical Hough breaks**.

Meanwhile the segment approach in that same regime has 4T² segments **and** the
coupling floor. So the paper-shaped claim is:

> **Hough trades the segment coupling floor for a resolution wall. A quantum Hough
> has a register constant in track multiplicity and helps precisely in the
> high-dimension regime (displaced vertex / curvature) where the classical
> accumulator is too large to store — the regime where the segment solvers pay
> 4T² width and an irreducible false-positive floor.**

## 5. Validation ladder (mirrors the QSVT programme)

1. **[DONE] Classical 2-D baseline** — `hough_prototype.py`: eff/ghost vs T. ✅
2. **Resolution & smear law** — eff/ghost vs (N_bins, σ_scatt, σ_res, σ_pv);
   derive optimal bin size ≈ smear; the resolution-wall curve. (numpy only)
3. **Store integration** — run on the **same `qtrk_store` events** as the 3-solver
   benchmark so Hough drops into the Verify_new_results comparison directly
   (eff/ghost/clone vs T alongside classical / 1BQF / QSVT). (`qtrk-data-pipeline`)
4. **4-D displaced-vertex stress** — release the PV pin (PV_SIGMA x,y > 0); measure
   the classical accumulator blow-up + where 2-D fails → motivates quantum.
5. **Run-3 real geometry** — variable track length, real vertices (the WP5 events).
   Does the Hough peak survive? (honest negative result is fine.)
6. **Quantum demonstrator** — small qiskit P2 (Dürr–Høyer) on one event over a
   coarse accumulator: the **8–9 (here ~12) qubit hardware candidate**, analogous
   to the QSVT NISQ demo. Vote oracle from the hit adjacency; count the fullest
   bin; show the peak is found in O(√N_bins). Measure gate count vs the 1BQF.
7. **Resource model** — qubits & walk/oracle calls vs (T, resolution, dimension)
   → the scoreboard row, plotted against 1BQF / QSVT / D3.

Steps 2–5 need **no quantum** and reuse the existing store + helpers; step 6 is the
one new circuit. Recommend doing 2→3 next (cheap, makes Hough a first-class fourth
solver in the benchmark), then 6 for the quantum claim.
