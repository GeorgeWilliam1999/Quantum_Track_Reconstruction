# Experiment C: every connected component as its own small circuit (12 Nov 2026)

Status: **provisional**. This is plan candidate C ("decomposed small circuits") of the Notion
research plan "cheap and effective VELO track finding"
(https://app.notion.com/p/4155d544b9d98186919cd5024e93e8cb).

## Question

A spectral filter on a block-diagonal operator acts block by block. So every connected component
(segment register: the components of A; hit register: the components of the clique graph C) can
be its own circuit on its own register. The questions are:

1. How many circuits does an event need, and how wide are they?
2. Is the result exactly the same?
3. What do the circuits cost in two-qubit gates with shots included, against the whole
   register of experiment G?
4. What changes when small components are solved classically?

## What is measured and what is a model

**Measured:**
- The component sizes and the qubit counts.
- Exactness: the per-component outputs are compared with the whole-register output and with an
  independent evaluation on the full sparse matrix.
- The track efficiencies: shots drawn from the exact per-component output distributions, read
  out and scored with the EventValidator.
- The classical time to solve all components (dense eigendecomposition).

**Model:**
- Gates per solved shot. This is the section 4.7 Givens price on each component's own register:
  pairs × (2(dH − 1) + MCRX CX), with r = 4 product steps for hits and r = 2 for segments,
  calls 2 / 6 / 2 / 40, and rounds = max(1, π/(4√P_c,padded)).
- The MCRX CX counts for 1–14 controls were transpiled for this study, because the 05 fit is
  negative below 3 controls (`01_components.py:81`).
- Amplitude amplification is assumed ideal.
- No circuit is built or simulated.

## Code

| what | where |
|---|---|
| split, per-component price, per-component shots | package `LHCb_VeLo_Toy_Model`, branch `claude/cheap-velo-C` (off `claude/cheap-velo-G`), `src/lhcb_velo_toy/solvers/quantum/components.py`: `split_components`, `component_givens_price`, `popcount_mean`, `shot_allocation`, `ComponentShotSampler`; tests `tests/test_components.py` |
| per event | `scripts/01_components.py`: MCRX table line 81, amplification cap line 102, `account` line 148 (gates line 153), shots line 157 (allocation line 171, estimate rule line 176), independent check lines 259 / 319, random-tie target line 336, empty components lines 264 / 339 |
| tables, verdict, figures | `scripts/02_summary.py`: β90 line 56 (target line 64), tables line 87, kill criterion line 172 |
| reused | `Cheap_VELO/B_graded_clique_filter/scripts/b_lib.py` (hit operator and filters), `07_full_sweep.segment_solutions`, package `coupled_hit_groups`, experiment G per-event N90 / gates (`Cheap_VELO/G_shot_budget/outputs/n90_*_events.csv`) |

To rerun: `Q_env/bin/python scripts/01_components.py real 12` (~8.5 min), `... toy 10` (~17 min),
then `scripts/02_summary.py` (tables in `outputs/02_summary.log`).

## Method

- **Events and solvers.** The events are those of A, B and G:
  - toy: T = 100, 400, 1000 × 1, 5, 10, 20 vertices, 52 events;
  - Run 3: 103 events.

  The solvers are those of G, at the scan vertices:
  - hit register: the one-bit filter on 8I − C, and the degree-6 shifted step polynomial;
  - segment register: the one-bit filter (1BQF) and the QSVT comb, both at the truth-free
    reading I with the coupled readout.
- **Components.** Components are the connected components of the off-diagonal support.
  - A component with one item has no coupling, so no circuit can change it. Its output f(diag)
    is known: 0 for both one-bit filters, |comb(s)| below the reading-I threshold for the QSVT,
    and f(0) for the polynomial.
  - So "components" means every component with at least 2 items is a circuit (t = 1).
  - Hybrids t = 2, 4, 8 solve the components with at most t items classically and run only the
    larger ones as circuits.
- **Shots.**
  - Every circuit gets N_c = ⌈β m_c ln m_c⌉ shots, which is G's coupon-collector law applied
    with the component size.
  - β runs over {0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100}, with 5 seeds and nested
    multinomial draws per component.
  - Items are switched on by G's estimate rule, per component register.
  - β90 = the smallest β with mean efficiency ≥ target − 0.02.
  - Total gates = Σ N_c × gates per solved shot of component c.
- **Empty components.** A component with no item of the exact reading can have P_c ≈ 1e-28,
  because its output cancels. It is run with at most 79 rounds (a P floor of 1e-4) and read as
  empty.
  - Components with kept items always get their full rounds, however small P_c is.
  - Whether a component is empty is taken from the exact output. This is an accounting choice.
  - It affects only empty components. `kept_in_capped_max` = 0 on every event.

## Exactness: passed (`outputs/exactness.txt`)

- **Outputs.** Per-component outputs, rescaled to the whole-register probe, agree with the
  whole-register outputs of B / G to ≤ 7.3e-14. They agree with an independent evaluation on the
  full sparse matrix with no split to ≤ 7.3e-14: `expm_multiply` for the one-bit filters,
  Clenshaw recursion for the step polynomial and the comb. This holds on all 155 events and all
  four solvers.
- **Hit register tracks.** The tracks equal G's / B's on every event (|Δ efficiency| = 0).
- **Segment tracks.** The on-sets are identical (0 differences). With both outputs read the same
  way, the tracks are identical (|Δ efficiency| = 0).
- **A fragility found on the way.** Experiment A's greedy hit uniqueness keeps the larger
  amplitude, and many segments tie exactly. On one Run 3 event, 101 of 131 active segments share
  their value with another segment. So a 1e-15 difference between two exact evaluations changes
  the winner.
  - The raw readout of the same exact output gives 0.742 or 0.806 on that event.
  - This round-off noise is in A's and G's exact segment efficiencies. Rounding the scores to 9
    relative digits (ties broken by index) changes 78 of 103 Run 3 events. The mean change is
    +0.007 for the 1BQF and +0.0075 for the QSVT, with a range of −0.036 to +0.065 per event.
  - On the toy the change is ≤ 0.003.
  - C reads both outputs with rounded scores.
- **The target for the segment shot curves.** Shots break ties at random, so the target is the
  exact output with ties broken at random, as the mean of 5 draws. On Run 3 this gives 0.861 for
  the 1BQF (sd 0.007) and 0.691 for the QSVT, against 0.872 and 0.697 with index ties.

## Results

### Run 3, 103 events (medians; `outputs/summary_run3.csv`)

| solver | circuits per event (t = 1 / t = 8) | largest component, qubits median / max | whole register qubits (G) | shots per event | gates per event: whole (G) | components | hybrid t = 8 | gates per found track: whole / components |
|---|---|---|---|---|---|---|---|---|
| hit, one-bit 8I − C | 258 / 36 | 7 / 7 | 13 | 1.2e4 (G: 3.2e4) | 1.4e11 | **1.3e8** | 5.0e6 | 1.1e9 / 9.2e5 |
| hit, polynomial d = 6 | 258 / 36 | 9 / 9 | 15 | 1.6e4 | 5.1e11 | 3.6e8 | 9.1e6 | 3.5e9 / 2.0e6 |
| segment 1BQF | 313 / 50 | 7 / 9 | 19 | 1.5e4 (G: 3.2e4) | 3.4e11 | **2.2e7** | 2.6e6 | 1.7e9 / 1.1e5 |
| segment QSVT | 313 / 50 | 9 / 11 | 21 | 5.8e3 (G: 1e6) | 8.0e14 | 2.9e9 | 5.0e8 | 4.9e12 / 1.6e7 |

- Hybrids t = 2 / 4 (gates per event):
  - one-bit hit: 1.3e8 / 1.6e7;
  - 1BQF: 1.0e7 / 4.2e6;
  - QSVT: 2.5e9 / 1.0e9.
- The per-event ratio of whole-register gates to component gates has median:
  - 1.1e3 for the hit filters;
  - 1.1e4 for the 1BQF;
  - 3.0e5 for the QSVT.
- Every event reaches its target (`reached_t1` = 1 for all solvers).
- Exact efficiencies are unchanged by the split: 0.641 / 0.644 / 0.872 / 0.697.
- Components per event by size, Run 3, segments: 1.0e5 single segments, 245 of 2, 113 of 3–4,
  71 of 5–8, 50 of 9–16, 4.6 of 17–32, 0.6 of 33–64, and 0.02 of 65–128 (the largest is 81 items).
- For hits: 500 single hits, 61 of 2, 94 of 3–4, 91 of 5–8, 39 of 9–16, and 1.5 of 17–32.
- All values are in `outputs/size_distribution.csv`.

### Toy (medians over events; `outputs/summary_toy.csv`)

T = 1000 (1 / 20 vertices):

| solver | circuits (t = 1) | largest qubits | whole qubits | gates per event: whole (G) | components | hybrid t = 8 | per found track: whole / components |
|---|---|---|---|---|---|---|---|
| hit one-bit 8I − C | 988 / 999 | 6 / 6 | 15 | 7.1e11 / 6.6e12 | 2.3e7 / 1.4e8 | 1.8e5 / 2.5e4 | 7.2e8 / 2.4e4 ; 7.0e9 / 1.5e5 |
| hit polynomial d = 6 | 988 / 999 | 8 / 8 | 17 | 1.5e12 / 1.4e12 | 4.8e7 / 1.6e8 | 6.0e5 / 8.7e4 | 1.5e9 / 4.9e4 ; 1.5e9 / 1.7e5 |
| segment 1BQF | 6970 / 5910 | 6 / 5 | 24 | 4.0e14 / 3.7e14 | **1.4e7 / 1.0e7** | 3.4e6 / 0 | 4.1e11 / 1.4e4 ; 3.8e11 / 1.0e4 |
| segment QSVT | 6970 / 5910 | 8 / 7 | 26 | 1.3e15 / 4.9e15 | 8.1e8 / 3.7e8 | 3.4e7 / 0 | 1.4e12 / 8.4e5 ; 5.0e12 / 3.8e5 |

- At T = 100, the component gates per event are:
  - one-bit hit: 2.1e6–5.3e6;
  - polynomial: 4.4e6–7.3e6;
  - 1BQF: 1.6e5–1.7e5;
  - QSVT: 5.3e6–1.2e7.

  These are against 3.5e9–1.7e10 (and 1.5e12–4.9e14 for the QSVT) for the whole register.
- At T = 400, the 1BQF components need 7.7e5–1.5e6 gates against 6.4e11–6.7e11 for the whole
  register.
- **Largest component, toy, all points:** ≤ 6 qubits (one-bit hit), ≤ 8 (polynomial), ≤ 7
  (1BQF) and ≤ 9 (QSVT), all with ancillas.
- **Largest component in items:** ≤ 10 hits and ≤ 16 segments.
- At t = 8, the toy needs 0–8 circuits per event.

### Classical cost of the same problem (measured)

Time per event to eigendecompose every component with ≥ 2 items in Python (`t_eigh`, median):

| event set | hits | segments |
|---|---|---|
| Run 3 | 0.09–0.10 s | 0.15–0.22 s |
| toy, T = 100 | 0.03 s | 0.04–0.06 s |
| toy, T = 400 | 0.12–0.14 s | 0.6–0.8 s |
| toy, T = 1000 | 0.3–0.6 s | 38–68 s |

- The toy T = 1000 segment time is dominated by slicing each component out of the 3.9 M-row
  sparse matrix in a Python loop, not by the eigendecompositions. The components have ≤ 16
  segments.
- Every component here is at most 81 items (Run 3) and 16 items (toy). Its exact solution is a
  dense eigendecomposition of at most an 81 × 81 matrix, which is trivial on a laptop.
- **The decomposition that makes the circuits small also makes the problem classically
  trivial.** Any quantum advantage claim must be made against this classical baseline, and on
  these events there is none to make.

## Kill criterion (`outputs/kill_criterion.txt`)

| solver | Run 3 events with a component > 10 qubits (with ancillas) | gates per event: components vs whole | verdict |
|---|---|---|---|
| hit one-bit 8I − C | 0 % (largest 7) | 1.3e8 vs 1.4e11, lower in 100 % of events | pass |
| hit polynomial d = 6 | 0 % (largest 9) | 3.6e8 vs 5.1e11, lower in 100 % | pass |
| segment 1BQF | 0 % (largest 9) | 2.2e7 vs 3.4e11, lower in 99 % | pass |
| segment QSVT | 1.9 % (largest 11; system register ≤ 7) | 2.9e9 vs 8.0e14, lower in 97 % | pass |

**Verdict: C is not stopped.**

## Findings

1. **Splitting is exact and cuts the cost by 3–5 orders of magnitude.**
   - The cut is 1e3 for the hit register, 1e4 for the segment 1BQF and 3e5 for the QSVT (Run 3
     medians). On the toy at T = 1000 the cut for the 1BQF is 0.9e7–3.5e7.
   - The saving comes from three places:
     - no padding;
     - small registers, so cheap multi-controlled rotations (≤ 7 controls instead of 11–24);
     - fewer shots: Σ m ln m over components is below N ln N of the whole register, and no
       shots are spent where nothing can be found.
2. **With components, the segment 1BQF is the cheapest quantum route on Run 3.**
   - It needs 2.2e7 gates per event, against 1.3e8 for the one-bit hit register.
   - Per found track it needs 1.1e5, against 9.2e5.
   - It also finds more tracks: 0.86–0.87 against 0.64.
   - The hit register's advantage came from a smaller whole register. With components, segment
     chains are cheaper than hit cliques: they have fewer pairs per item and need 2 product steps
     instead of 4.
3. **Every circuit fits in ≤ 9 qubits on Run 3, except QSVT components at 11 qubits in 1.9 % of
   events.** The QSVT width comes from its 4 ancillas. The system register is ≤ 7 qubits.
   Circuits per event: 258–313, or 36–50 when components of ≤ 8 items are solved classically.
4. **Solving the components of ≤ 8 items classically leaves 36–50 circuits per Run 3 event and
   0–8 per toy event.** It cuts the gates by a further factor of 6–40 (Run 3: 26 / 39 / 8.6 / 5.9 for one-bit hit / polynomial / 1BQF / QSVT).
5. **The same split makes the problem classically trivial.** The total eigendecomposition time is
   0.1–0.2 s per Run 3 event.

## Caveats

- Gates are a cost model, and amplitude amplification is ideal.
- Empty components are identified from the exact output. A real run needs a stopping rule: P
  floor 1e-4, at most 79 rounds.
- Shots per component follow β m ln m, a fixed allocation. An adaptive one (stop a component once
  it is read) would cost less.
- The QSVT comb uses the global spectral domain on every component.
- Run 3 hits are not clusters.

## Figures (`figures/`, PDF and PNG, paper 1 style)

- `C01_qubits_vs_T`: largest-component qubits against T per vertex count, with the whole register
  (G).
- `C02_circuits_vs_T`: circuits per event, for every coupled component and for components > 8
  items.
- `C03_total_gates_vs_T`: gates per event with shots, for the whole register, components and the
  t = 8 hybrid.
- `C04_gates_per_track_vs_T`: gates per found track, whole register against components.
- `C05_run3_bars`: Run 3 width, gates per event and gates per found track.
- `C06_component_sizes`: components per event by size, toy T = 1000 and Run 3.
