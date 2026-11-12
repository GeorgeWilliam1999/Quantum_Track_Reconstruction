# Experiment G: readout shot budget (12 Nov 2026)

Status: **provisional**. Plan item G of the Notion research plan "cheap and effective VELO track
finding" (https://app.notion.com/p/4155d544b9d98186919cd5024e93e8cb); to-do
https://app.notion.com/p/4165d544b9d981db9a30ffcd22dc2550.

## Question

With the exact output, every hit filter that passes line 2 gives the same tracks (experiment B).
A quantum solver is read by measuring its register. This study asks three things for each solver:

- How many shots N does it need to reach its exact-output track efficiency?
- What is the total two-qubit gate cost per event, and per found track, once the shots are counted?
- Do the filters separate once the shot cost is counted?

## What is measured and what is a model

- **Measured:** efficiency and ghost rate against N.
  - Shots are drawn from the **exact** post-selected output distribution of each solver, by a
    multinomial draw.
  - Track finding and scoring are the real readout and the EventValidator.
  - No circuit is simulated, so no noise and no compilation effects are in these numbers.
- **Model:** gates per post-selected shot. This is the section 4.7 Givens pricing, as in
  experiments B and A, with ideal amplitude amplification (rounds = max(1, π/(4√P_padded))).
  Total gates = N90 × gates per shot.
- **Resolution:** N is on half-decade steps (10, 32, 100, … 1e8). N90 is therefore known to a
  factor of 3.16. Ratios of N90 below that factor are not resolved.

## Code

| what | where |
|---|---|
| sampler | package `LHCb_VeLo_Toy_Model`, branch `claude/cheap-velo-G` (claude/cheap-velo-B + claude/coupled-readout merged), `src/lhcb_velo_toy/solvers/quantum/shots.py`: `output_distribution`, `ShotSampler` (nested multinomial), `on_items`, `shot_grid`; tests `tests/test_shots.py` |
| shots per event | `scripts/01_shots.py`: hit readout line 160, hit p_min line 169, segment readout line 191, segment p_min line 205, segment gates per shot line 211, saturation stop line 120 |
| N90, gates, reproduction, figures | `scripts/02_summary.py`: N90 target line 86, total gates line 95, gates per track line 96 |
| reused | `Cheap_VELO/B_graded_clique_filter/scripts/b_lib.py` (hit operator, filters, B readout, gates); `Vertexing/Focusing_scan/scripts/07_full_sweep.segment_solutions` (segment solves); package `coupled_hit_groups` (experiment A readout) |

To rerun: `Q_env/bin/python scripts/01_shots.py real 12` (~12 min), `scripts/01_shots.py toy 10`
(~12 min), then `scripts/02_summary.py` (prints to `outputs/02_summary.log`).

## Method

- **Events:**
  - Toy: the B / A events (T = 100, 400, 1000 × 1, 5, 10, 20 vertices; seeds of 07_full_sweep;
    52 events).
  - Run 3: the 103 Gauss Sim10 events (segment ε = 5 mrad, as run_real.py).
- **Solvers:**
  - Hit register at the focusing-scan vertices, three filters:
    - the one-bit filter on 8I − C (the default; 2 calls);
    - the degree-6 step polynomial on the shifted domain (6 calls);
    - the ideal step on every line ≥ 2 (not a circuit, so it has no gate count).
  - Segment register: the one-bit filter (1BQF) and the QSVT comb (degree 40), at the truth-free
    reading I.
  - References without shots: the classical solver (reading I, coupled readout) and the
    no-solver control of experiment A (score "deg").
- **Shots:**
  - Probabilities are p_i = |ψ_i|² / Σ over the real register items. Padding is excluded after
    post-selection, as the brief asks.
  - For the polynomial the padded rows carry f(0)², recorded as `pad_leak`. It is up to 0.3 %
    of the post-selected probability on the toy and up to 12 % on Run 3 (largest event). Those
    shots would land on empty rows, so the polynomial's N90 on Run 3 is optimistic by up to
    about 12 %. That is well inside the grid resolution.
  - The draws are nested multinomial draws at N = 10 … 1e8, with 5 seeds.
- **When an item is on.** There are two rules, each with k = 1 and k = 2:
  - **count rule:** the item is seen at least k times (the rule of the brief);
  - **estimate rule:** the count rule and, in addition, c_i/N ≥ p_min.
  - p_min is the exact-output reading written as a probability:
    - hit register: weight ≥ τ ⇔ p_i ≥ τ²/(nP) (B's per-filter τ);
    - segments: amplitude above the isolated-segment value (reading I).
  - Only the estimate rule converges to the exact readings of A and B. The count rule switches
    on every item with p_i > 0.
  - **The main numbers use the estimate rule with k = 1.**
- **Readout:**
  - Hit register: the connected components of C over the on hits, with ≥ 3 hits.
  - Segments: the coupled readout with greedy hit uniqueness, where the slot winner is the
    largest count.
  - EventValidator: purity ≥ 0.7, reconstructible = ≥ 3 hits.
- **N90 and costs:**
  - N90 is the smallest N with the seed-mean efficiency ≥ the exact-output efficiency − 0.02.
  - Total gates = N90 × gates per shot.
  - Gates per found track = total / (efficiency at N90 × reconstructible tracks).
  - Medians are over the events of a point.
  - A curve stops after two consecutive N within 0.005 of its N → ∞ value, and keeps that value
    above.

## Reproduction check: passed (`outputs/reproduction.txt`)

At N → ∞, the estimate rule gives B's hit-register efficiencies exactly on every event (|diff| = 0):
- toy: one-bit 8I − C 0.9453, polynomial 0.9453, ideal step 0.9057;
- Run 3: 0.6406, 0.6438, 0.4241.

The exact reading I + coupled readout recomputed here gives A's efficiencies exactly (|diff| = 0):
- toy: classical 0.9881, 1BQF 0.9879, QSVT 0.9815;
- Run 3: 0.8619, 0.8646, 0.6890.

At N → ∞ the estimate rule equals the exact reading on every event.

## Results

### Toy (`outputs/summary_toy.csv`; estimate rule, k = 1; medians over events)

N90 (1 / 5 / 10 / 20 vertices):

| T | hit one-bit 8I − C | hit polynomial d = 6 | hit ideal step | segment 1BQF | segment QSVT |
|---|---|---|---|---|---|
| 100 | 3.2e3 / 1e4 / 1e4 / 1e4 | 3.2e3 (all) | 3.2e3 / 3.2e3 / 1e3 / 1e3 | 3.2e3 (all) | 1e4 / 3.2e6 / 3.2e4 / 3.2e6 |
| 400 | 1e4 / 1e4 / 1e4 / **1e5** | 1e4 (all) | 1e4 (all) | 1e4 (all) | 3.2e7 / 1e5 / 1e5 / 1e5 (reached in 1 / 0.8 / 0.6 / 0.6 of events) |
| 1000 | 3.2e4 / 3.2e4 / 3.2e4 / **3.2e5** | 3.2e4 (all) | 1e4 / 3.2e4 / 3.2e4 / 3.2e4 | 1e6 / 3.2e5 / 1e6 / 1e6 | 1e5 / 2e5 / 3.2e5 / 3.2e5 (0.67 at 5 vertices) |

Total two-qubit gates per event, shots included:

| T | hit one-bit 8I − C | hit polynomial d = 6 | segment 1BQF | segment QSVT |
|---|---|---|---|---|
| 100 | 3.5e9 / 1.0e10 / 9.5e9 / 9.3e9 | 9.1e9 / 8.6e9 / 8.2e9 / 5.6e9 | 1.5e10 – 1.7e10 | 1.5e12 / 4.9e14 / 5.0e12 / 4.9e14 |
| 400 | 5.8e10 / 5.6e10 / 5.6e10 / 5.3e11 | 1.5e11 / 1.5e11 / 1.4e11 / 1.4e11 | 6.4e11 – 6.7e11 | 3.6e16 / 1.9e14 / 2.1e14 / 2.0e14 |
| 1000 | 7.1e11 / 7.0e11 / 7.0e11 / 6.6e12 | 1.5e12 / 1.5e12 / 1.5e12 / 1.4e12 | 4.0e14 / 1.1e14 / 3.8e14 / 3.7e14 | 1.3e15 / 2.9e15 / 4.2e15 / 4.9e15 |

Gates per found track at T = 1000:
- one-bit 8I − C: 7.0e8–7.2e8 (7.0e9 at 20 vertices);
- polynomial: 1.5e9;
- 1BQF: 1.2e11–4.1e11;
- QSVT: 1.4e12–5.0e12.

Exact-output efficiencies at T = 1000:
- hit filters: 0.97 / 0.99 / 0.99 / 0.94;
- 1BQF: 0.98–0.99;
- QSVT: 0.97–0.98;
- classical: 0.99;
- no-solver control: 0.95–0.97.

### Run 3, 103 events (`outputs/summary_run3.csv`; estimate rule, k = 1)

| solver | exact eff. | N90 (median) | eff. at N90 | ghost at N90 | qubits | gates per shot | gates per event | gates per found track |
|---|---|---|---|---|---|---|---|---|
| hit, one-bit 8I − C | 0.641 | 3.2e4 | 0.631 | 0.054 | 13 | 6.8e6 | 1.4e11 | 1.1e9 |
| hit, polynomial d = 6 shifted | 0.644 | 3.2e4 | 0.632 | 0.055 | 15 | 1.6e7 | 5.1e11 | 3.5e9 |
| hit, ideal step | 0.424 | 3.2e3 | 0.417 | 0.041 | — | — | — | — |
| segment 1BQF + coupled readout | 0.865 | 3.2e4 | 0.855 | 0.050 | 19 | 1.6e7 | 3.4e11 | 1.7e9 |
| segment QSVT + coupled readout | 0.689 | 1e6 (reached in 0.97 of events) | 0.671 | 0.056 | 21 | 7.1e8 | 8.0e14 | 4.9e12 |
| classical + coupled readout | 0.862 | no shots | | 0.041 | | | | |
| no-solver control | 0.858 | no shots | | 0.052 | | | | |

With the count rule and k = 1, the hit filters need N90 = 1e4: 6.8e10 gates per event for the
one-bit filter (4.8e8 per found track) and 1.6e11 for the polynomial. On the toy, the count rule
reaches a slightly higher limit than the exact reading (0.958 against 0.945, mean over toy
events), because it also keeps weak hits of broken cliques.

### N90 against the number of kept items (`outputs/n90_scaling.csv`)

The fit is N90 ∝ m^a, where m = the items the exact reading keeps. The ratio is N90 / (m ln m),
median.

| solver | a (all events) | N90 / (m ln m), toy / Run 3 |
|---|---|---|
| hit one-bit 8I − C | 0.82 | 1.0 / 3.5 |
| hit polynomial d = 6 | 0.98 | 0.8 / 3.3 |
| hit ideal step | 1.01 | 0.8 / 0.7 |
| segment 1BQF | 1.24 | 1.2 / 1.9 |
| segment QSVT | 1.04 | 9.8 / 151 |

- Against the register size n, the segment exponents are lower (0.58 for 1BQF). Most segments
  carry almost no probability, so the shots are set by the kept items, not by the register.

## Findings

1. **The shot count is coupon collection over the kept items.** N90 ≈ (0.7–3.5) × m ln m for
   the hit filters and the segment 1BQF.
   - The exception is the segment QSVT: 10× (toy) to 150× (Run 3) above m ln m. The comb gives
     a small amplitude to some segments the readout needs. It passes the 5-hit chain lines and
     suppresses the shorter chains (section 8 / limit T7). Not checked track by track here.
2. **The filters separate only a little.**
   - Up to 10 vertices, and on Run 3, the one-bit filter on 8I − C and the degree-6 polynomial
     need the same N90 within one grid step (a factor of 3.16).
   - Per event the one-bit filter is then cheaper at T = 400 and 1000 (2.0–2.6×) and on Run 3
     (3.6×), because each shot costs fewer gates.
   - At T = 100 with 5–20 vertices the two cost about the same: the one-bit filter needs one
     grid step more shots (1e4 against 3.2e3).
   - At 20 vertices (T = 400 and 1000), the one-bit filter needs 10× more shots (3.2e5 against
     3.2e4). The polynomial is then 5× cheaper per event: 1.4e12 against 6.6e12 at T = 1000.
   - **Inference, not tested:** the one-bit response puts 4× less probability on a 3-hit
     clique (f(2)² = 0.15) than on a 5-hit clique (f(4)² = 0.5). The polynomial is nearly flat
     above line 2, so it spreads the shots more evenly.
   - The ideal step needs the fewest shots (it has no pair or broken-clique leakage), but it is
     not a circuit.
3. **On the toy, counting shots makes the hit-register advantage larger.** At T = 1000, total
   gates per event are 7e11 (one-bit hit) against 1.1e14–4.0e14 (segment 1BQF): 160–570×. With
   one shot it was 24× (experiment B). The segment register needs 10–30× more shots on the toy
   at T = 1000 (N90 3e5–1e6 against 3e4).
4. **On Run 3, counting shots removes most of the hit-register advantage.**
   - Per event the one-bit hit filter still costs 2.4× less: 1.4e11 against 3.4e11 for the
     segment 1BQF.
   - Per found track it costs only 1.5× less: 1.1e9 against 1.7e9.
   - The segment 1BQF with the coupled readout finds more tracks: 0.865 against 0.641.
   - The segment QSVT is the most expensive solver everywhere: 8e14 gates per event on Run 3.
5. **The no-solver control (0.858) and the classical solver (0.862) match the best quantum
   efficiency on Run 3 with no shots and no gates.** This repeats experiment A's finding: at the
   truth-free reading the solve adds little.

## Caveats

- The shots are from the exact distribution. There is no noise, no compilation, and no
  imperfect amplitude amplification. Amplitude amplification is assumed ideal, with rounds
  = max(1, π/(4√P)).
- The gates are a cost model.
- N90 has half-decade resolution.
- The QSVT N90 is a median over the events that reach it (0.6–1 of the events on the toy, 0.97
  on Run 3).
- Run 3 hits are not clusters.

## Figures (`figures/`, PDF and PNG, paper 1 style)

- `G01_efficiency_vs_shots`: efficiency against N per solver (toy T = 100 / 400 / 1000 pooled
  over vertices, and Run 3). The classical solver and the no-solver control are drawn as lines.
- `G02_N90_vs_T`: N90 against T per vertex count.
- `G03_total_gates_vs_T`: total gates per event against T, shots included. `G03b`: the same per
  found track.
- `G04_qubits_vs_T`: logical qubits against T.
- `G05_run3_shots_and_gates`: Run 3 N90, gates per event and gates per found track.
- `G06_N90_vs_register_size`: N90 against the kept items m, with the m ln m line.
