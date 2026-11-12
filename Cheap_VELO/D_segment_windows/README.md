# Experiment D: physics windows on the segment graph, shots counted (12 Nov 2026)

Status: **provisional**. This is plan candidate D ("windowed segment graph") of the Notion research
plan "cheap and effective VELO track finding"
(https://app.notion.com/p/4155d544b9d98186919cd5024e93e8cb).

## Question

Can physics windows applied at segment construction shrink the segment register and its
components before any solver runs, at no cost in efficiency? The windows are a slope cut and a
cut on the segment's pointing to the beamline. Two further questions:

- Can moving the construction in the other direction make the problem bigger and harder, in a
  way where a quantum solver could matter?
- Does the conclusion stay classical?

## What is measured and what is a model

**Measured:**
- Segments kept, true-segment retention, component counts and sizes, and qubit counts.
- Exact outputs of every solver on every component, and the tracks they give (EventValidator,
  purity ≥ 0.7, reconstructible = ≥ 3 hits).
- Efficiency by category.
- Track efficiency against the shot budget. Shots are drawn from the exact per-component output
  distributions (no circuit).
- Classical eigendecomposition time. These times were measured on a node with load 40–50, so they
  are upper bounds.

**Model:**
- Two-qubit gates. This is experiment C's pricing: the Givens price on each component's own
  register, r = 2 product steps, calls 2 (1BQF) / 40 (QSVT), ideal amplitude amplification, and
  at most 79 rounds for components with no kept item.

## Code

| what | where |
|---|---|
| windows | package `LHCb_VeLo_Toy_Model`, branch `claude/cheap-velo-D` (off `claude/coupled-readout-tiebreak`), `src/lhcb_velo_toy/solvers/hamiltonians/windows.py`: `segment_geometry` (tx, ty, transverse impact parameter d0 and z0 of closest approach to the beamline), `segment_window_mask`, `restrict_system`; tests `tests/test_windows.py` |
| per event | `scripts/01_windows.py`: windows line 75, configurations line 76, seeds line 78, mask line 135, restricted system line 138, random-tie target line 189, per-component shots line 201 |
| tables, verdict, figures | `scripts/02_summary.py`: C reference line 42, β90 line 55, tables line 72, kill test line 157 |
| reused | experiment C (`C_small_circuits/scripts/01_components.py`: per-component filter, pricing, MCRX table, sampler); experiment B (`b_lib`, hit-register reference in `B_graded_clique_filter/outputs/run3_tracks.csv`); C outputs for the hit-register gates |

To rerun: `Q_env/bin/python scripts/01_windows.py real 12` (56 min on the loaded node),
`... toy 10` (2 h 4 min; the T = 1000 events dominate), then `scripts/02_summary.py` (tables in
`outputs/02_summary.log`).

## Method

- **Events.** The events of A, B, C and G:
  - toy: T = 100, 400, 1000 × 1, 5, 10, 20 vertices, 52 events;
  - Run 3: 103 events.
- **Construction.** The package builds segments only between consecutive modules (all hit pairs)
  and couples them by the angle ε (step kernel, γ = 3, δ = 1).
  - The diagonal is the constant −(δ + γ) and the couplings are pairwise. So the system built from
    the kept segments alone is exactly A[keep][:, keep] (`restrict_system`).
  - A window can therefore only remove couplings: every windowed component lies inside a full one.
  - There are no layer-skipping segments, so there is no layer-gap window to scan.
- **Configurations:**

  | config | change |
  |---|---|
  | L4 / L2 | coupling angle ε × 4 / × 2, no window. This is the *looser* direction. |
  | W0 | baseline: ε of A / C (toy 1.70 mrad, Run 3 5 mrad). Reproduces C. |
  | W1 | |tx|, |ty| ≤ 0.60, d0 ≤ 10 mm |
  | W2 | ≤ 0.45, d0 ≤ 3 mm |
  | W3 | ≤ 0.35, d0 ≤ 1 mm |
  | W4 | ≤ 0.30, d0 ≤ 0.3 mm |
  | W5 | ≤ 0.25, d0 ≤ 0.1 mm |

  - W1–W5 also require z0 to lie in the luminous region: |z0| ≤ 250 mm for Run 3, and −200 … 50 mm
    for the toy.
  - The ladder was placed from the true-segment distributions of 11 Run 3 events. True segments
    have slope q50/q90/q99 = 0.07/0.25/0.69 and d0 = 0.13/1.25/14.6 mm. Coupled false segments
    have d0 = 6.5/13.5/22.8 mm.
- **Solvers per configuration:**
  - exact outputs per component: classical A⁻¹1, 1BQF, and the QSVT comb (degree 40, with the
    domain from the windowed spectrum);
  - reading I with the coupled readout and the deterministic tie-break;
  - the no-solver control.
- **Shots.** For the 1BQF: C's per-component budget ⌈β m ln m⌉ with the estimate rule and 3 seeds.
  β90 is the smallest β that reaches the target − 0.02, where the target is the exact output with
  random tie-breaks. The QSVT gets exact output and gates per solved shot only, because QSVT shots
  were not affordable at toy T = 1000.
- **Reference.** The hit register (one-bit filter on 8I − C) from experiments C / B on the same
  events.

## Reproduction

- **Run 3, W0 against C:**
  - 1BQF efficiency 0.870 (C 0.872 with C's rounding; 0.870 is the tie-break value of
    `Vertexing/Readout_A/outputs/summary_tiebreak.txt`);
  - gates per event 2.2e7 (C 2.23e7);
  - gates per found track 1.09e5 (C 1.08e5);
  - largest component 81 (C 81).
- **Toy:** W0 matches C within the β grid.
  - The exception is toy T = 1000 with 1 vertex: 6.9e7 against C's 1.4e7 gates per event.
  - There, one of the three events has β90 = 50 instead of 10. The shot curve levels off at
    0.96–0.97, right at the target − 0.02 line.
  - D uses 3 seeds where C used 5, and its random-tie target is 0.982 against C's 0.981.
  - With 3 events per point, the median follows that one event.

## Results

### Run 3, 103 events (`outputs/summary_run3.csv`; pooled efficiency, medians for sizes and gates)

| config | segments kept | true kept | components ≥ 2 | largest component median / max | 1BQF qubits max | eff. classical / 1BQF / QSVT / control | ghosts 1BQF | 1BQF gates per event | per found track | eigh time |
|---|---|---|---|---|---|---|---|---|---|---|
| L4 (ε × 4) | 65 700 | 1.00 | 1170 | 75 / **3253** | 14 | 0.770 / 0.758 / 0.505 / 0.707 | 0.257 | 4.3e8 | 2.7e6 | 2.1 s |
| L2 (ε × 2) | 65 700 | 1.00 | 556 | 39 / 224 | 10 | 0.857 / 0.856 / 0.660 / 0.821 | 0.117 | 4.4e7 | 2.4e5 | 0.55 s |
| W0 baseline | 65 700 | 1.00 | 313 | 30 / 81 | 9 | 0.870 / 0.870 / 0.692 / 0.855 | 0.053 | 2.2e7 | 1.09e5 | 0.31 s |
| W1 0.60 / 10 mm | 15 000 | 0.951 | 251 | 30 / 81 | 9 | 0.853 / 0.854 / 0.682 / 0.840 | 0.035 | 2.2e7 | 1.02e5 | 0.21 s |
| W2 0.45 / 3 mm | 3 080 | 0.900 | 217 | 30 / 68 | 9 | 0.816 / 0.817 / 0.663 / 0.806 | 0.018 | 2.0e7 | 9.6e4 | 0.24 s |
| W3 0.35 / 1 mm | 1 670 | 0.838 | 201 | 27 / 62 | 8 | 0.760 / 0.761 / 0.630 / 0.754 | 0.015 | 1.4e7 | 7.6e4 | 0.12 s |
| W4 0.30 / 0.3 mm | 1 140 | 0.678 | 184 | 17 / 40 | 8 | 0.666 / 0.666 / 0.529 / 0.662 | 0.013 | 7.1e6 | 4.8e4 | 0.14 s |
| W5 0.25 / 0.1 mm | 575 | 0.368 | 125 | 9 / 17 | 7 | 0.417 / 0.417 / 0.305 / 0.414 | 0.013 | 8.4e5 | 7.6e3 | 0.09 s |
| hit register 8I − C (C) | — | — | 258 | 18 hits | 7 | 0.637 | 0.055 | 1.3e8 | 9.2e5 | 0.09 s |

- The 1BQF reaches its target in 0.91–1.00 of the events at every configuration.
- QSVT gates per solved shot (median): 3.9e7 at W0, falling to 5.4e6 at W5.

Run 3 efficiency by category (1BQF, exact output; `outputs/summary_run3_categories.csv`):

| config | toy-like | 3-hit | secondaries | p < 1 GeV |
|---|---|---|---|---|
| L4 | 0.868 | 0.324 | 0.598 | 0.582 |
| L2 | 0.970 | 0.480 | 0.670 | 0.629 |
| W0 | 0.991 | 0.542 | 0.656 | 0.577 |
| W1 | 0.993 | 0.527 | 0.574 | 0.543 |
| W2 | 0.992 | 0.460 | 0.458 | 0.460 |
| W3 | 0.984 | 0.361 | 0.349 | 0.349 |
| W4 | 0.943 | 0.249 | 0.193 | 0.207 |
| W5 | 0.651 | 0.095 | 0.078 | 0.074 |
| hit register 8I − C | 0.922 | 0.116 | 0.242 | 0.201 |

### Toy (`outputs/summary_toy.csv`; 1BQF; medians)

- **The windows cost nothing on the toy.** All tracks are prompt and |t| < 0.2, so W1–W4 keep 100 %
  of the true segments and W5 keeps 98 %.
- At T = 1000 the windows cut the segment register from 3.9 M to 1.0e3–3.6e3 components ≥ 2.
  Efficiency is unchanged (0.983–0.990), and the ghost rate falls from 0.001–0.002 to 0.000–0.001.
  - Gates per found track at 1 vertex: 7.1e4 (W0) → 6.6e3 (W1) → 2.6e3 (W3–W5).
  - At 20 vertices: 1.1e4 → 4.7e3 → 1.9e3.
  - Most of the saving is false components (each a circuit with shots) that the window removes
    altogether.
- At T = 100 and 400 the windows change little: 1.6e3–2.3e3 gates per found track. The largest
  component is ≤ 14 segments (≤ 7 qubits) at every T, vertex count and window.
- **The looser direction grows the problem and breaks it.** At T = 1000, 1 vertex:
  - L2: largest component 30–33, 1BQF efficiency 0.934, control 0.637;
  - L4: largest component 45–56, 1BQF efficiency 0.407, ghost rate 0.18, classical 0.467.
  - At L2 / L4 (T = 1000) and L4 (T = 400), the shot curves do not reach the target within
    β = 100.

## Kill criterion (`outputs/kill_criterion.txt`)

The test: stop D if no window keeps Run 3 efficiency ≥ 0.85 while cutting the gates per found track
by ≥ 2× against C's 1.1e5 (that is, to ≤ 5.5e4).

| config | 1BQF efficiency | gates per found track | factor below C | both? |
|---|---|---|---|---|
| W1 | 0.854 | 1.02e5 | 1.08× | no |
| W2 | 0.817 | 9.6e4 | 1.15× | no |
| W3 | 0.761 | 7.6e4 | 1.45× | no |
| W4 | 0.666 | 4.8e4 | 2.28× | no |
| W5 | 0.417 | 7.6e3 | 14.5× | no |

**Verdict: STOP D.**
- On Run 3, every window that saves 2× in gates loses 0.19 or more in efficiency.
- The efficiency is lost in exactly the categories the plan wanted to recover: secondaries
  0.66 → 0.19, p < 1 GeV 0.58 → 0.21, and 3-hit 0.54 → 0.25 at W4.
- On the toy, the windows would pass easily (up to 27× at no efficiency cost), but toy tracks are
  all prompt.

## Answers to the direction questions

1. **No window makes the problem bigger.** By construction a window is a sub-matrix, so components
   can only shrink. On Run 3 the largest component goes 81 → 81 → 68 → 62 → 40 → 17 from W0 to W5.
2. **The only direction that makes it bigger is a looser coupling angle, and it makes the tracking
   worse, not harder in a useful way.**
   - ε × 2 / × 4 grows the largest Run 3 component from 81 to 224 / 3253 segments (8 / 12 qubits
     system register).
   - Efficiency falls (0.870 → 0.856 → 0.758), ghosts rise (0.05 → 0.12 → 0.26), and the gates per
     found track grow 2.2× / 25×.
   - Even the 3253-segment component is solved exactly by a dense eigendecomposition in about 2 s
     per event (median for L4, on the loaded node).
3. **The conclusion is classical.**
   - From W0 to W5 the classical solver equals the 1BQF to ≤ 0.002 in efficiency, and the
     no-solver control is within 0.015. On the looser L2 / L4 the classical solver is ahead by
     0.001 / 0.012.
   - Every component is at most a few hundred segments, except at L4.
   - The classical eigendecomposition of all components takes 0.09–0.3 s per Run 3 event, and
     2 s at L4.
   - Nothing in the segment-window direction produces a problem that is classically hard.

## Figures (`figures/`, PDF and PNG, paper 1 style)

- `D01_run3_windows`: (a) segments kept and largest component against the configuration;
  (b) efficiency of classical / 1BQF / QSVT / control, the true-segment retention and the
  hit-register line; (c) 1BQF gates per found track with the kill line.
- `D02_run3_tradeoff`: efficiency against gates per found track per configuration, with the hit
  register and the kill lines.
- `D03_run3_categories`: toy-like, 3-hit, secondaries and p < 1 GeV against the configuration, all
  solvers.
- `D04_toy_gates_per_track_vs_T`: 1BQF gates per found track against T per vertex count, for L4,
  W0, W2 and W4.
- `D05_toy_components_and_efficiency`: largest component and 1BQF efficiency against the
  configuration, per T.
