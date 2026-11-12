# Experiment B: graded clique filter for the hit register (12 Nov 2026)

Status: **provisional**. Plan: Notion "Research plan: cheap and effective VELO track finding",
section B (https://app.notion.com/p/4155d544b9d98186919cd5024e93e8cb).

## Question

The hit register passes only the clique lines {3, 4} of the per-vertex hit operator C (tracks with
4 and 5 hits). So it cannot find 3-hit tracks, and on Run 3 its success probability P is 0.18.
This study asks what happens when the filter passes every line ≥ 2 (all cliques of m ≥ 3 hits).
It measures P, gates, track efficiency and ghost rate.

## What is measured and what is a model

- **Measured (exact):** every filter is applied exactly, by eigendecomposition per connected
  component with the uniform probe. This gives P, padded P, the per-hit output and the tracks.
  No circuit is simulated.
- **Model:** the gate counts. They use the Givens price per product step (05_resources.py) with
  the section 4.7 corrections:
  - 4 product steps per call on the hit operator (2 on segments);
  - padded P;
  - calls = degree d for a polynomial and 2 for a one-bit filter;
  - rounds = max(1, π/(4√P_padded)).
- **Readout:** this is the exact per-hit output, not sampled shots, so the shot cost is not in any
  number (plan item G).

## Code

| what | where |
|---|---|
| filters, exact application, readout | package `LHCb_VeLo_Toy_Model`, branch `claude/cheap-velo-B`, `src/lhcb_velo_toy/solvers/quantum/clique_filters.py`; tests `tests/test_clique_filters.py` (18 tests; full package suite 110 passed) |
| per-event evaluation, pricing, readout threshold | `scripts/b_lib.py` (`evaluate`, line 116; Givens price per step line 128; α line 132, λ_min line 133; gates line 151; threshold τ line 152) |
| toy sweep | `scripts/01_toy_sweep.py` → `outputs/toy_events.csv`, `outputs/toy_tracks.csv`, `outputs/toy_reproduction.txt` |
| Run 3 | `scripts/02_run3.py` → `outputs/run3_events.csv`, `outputs/run3_tracks.csv`, `outputs/run3_reproduction.txt` |
| tables, kill criteria, figures | `scripts/03_summary.py` → `outputs/summary_{toy,run3,hits,run3_categories}.csv`, `outputs/kill_criteria.txt`, `figures/B0*.{pdf,png}`; log `outputs/03_summary.log` |

The Vertexing scripts (`Vertexing/Focusing_scan/scripts/*`, untracked in the main checkout, and
`Vertexing/Real_events/scripts/*`) are imported read-only from
`/data/bfys/gscriven/Quantum_Track_Reconstruction`. To rerun: `Q_env/bin/python scripts/01_toy_sweep.py 14`
(~25 s), `scripts/02_run3.py 14` (~100 s), then `scripts/03_summary.py`.

## Set-up

- **Operator:** C is the per-vertex hit operator at 4w (`09_vertex_offset.operator_matrix`, which is
  the `07_full_sweep.per_vertex_operator` rule). It is built at the focusing-scan vertices ("scan")
  and at the truth vertices ("truth").
- **Filters**, as functions f(c) of an eigenvalue c of C (a clique of m hits has its top line at
  c = m − 1):

  | key | filter | calls | qubits |
  |---|---|---|---|
  | ideal34 | ideal projector on lines {3, 4} (the published one) | — | — |
  | step2 | ideal projector on every integer line ≥ 2 | — | — |
  | band | every eigenvalue > 1.5, integer or not | — | — |
  | poly6/10/16 | bounded Chebyshev step at 1.5 on [−α, α], α = ρ(C) of the event | d | ⌈log2 n⌉ + 4 |
  | poly6s/10s/16s | the same on the shifted domain [λ_min, α] (needs an identity term in the block encoding, not priced) | d | ⌈log2 n⌉ + 4 |
  | onebit | one-bit filter on 4I − C: f = cos(π(4 − c)/8) | 2 | ⌈log2 n⌉ + 2 |
  | onebit8 | one-bit filter on 8I − C: second zero moved from c = 8 to c = 16 | 2 | ⌈log2 n⌉ + 2 |

  - The polynomials fit an erf step of width (domain length)/(2d). They are scaled so that
    |f| ≤ 1 on the domain (`clique_filters.py:149`).
  - The ideal filters are not circuits, so they have no gate count.
- **Readout:** a hit survives when √n·|f(C)u|_i ≥ τ. On a perfect m-hit clique this weight equals
  |f(m − 1)|.
  - τ = 0.5 for the ideal filters.
  - τ = (|f(1)| + |f(2)|)/2 for the circuits. This is a per-filter threshold, placed between the
    pair line and the 3-hit line.
  - Surviving hits are grouped by the connected components of C. A group of ≥ 3 hits is a
    candidate.
  - Candidates are scored with the package EventValidator (purity ≥ 0.7; reconstructible = ≥ 3
    hits).
- **Baselines:** classical, one-bit filter and QSVT on the segment Hamiltonian, and the
  focused-cell hit register. They come from the per-event and per-track CSVs of section 8
  (`full_sweep_*.csv`) and of the Run 3 study (`real_*.csv`) on the same events. They are not
  rerun.
  - Segment gates use the same pricing: 2 product steps, padded P, and calls = 2 or 40 for the
    degree-40 comb (`03_summary.py:63-71`).
  - Paper 1 counted 160 calls for its phase-sequence route. With that count the QSVT gates
    would be 4× higher.

## Reproduction checks: both passed

- **Toy** (`outputs/toy_reproduction.txt`, check at `01_toy_sweep.py:83`):
  - On all 52 events, three quantities equal the published values event by event: the {3, 4} P,
    the number of coupled pairs, and the Givens price per step.
  - The match is to |diff| ≤ 1e-15 against `full_sweep_events.csv` (hitV_P, hitV_k,
    hitV_cx_givens).
  - At T = 1000, P is **0.972 / 0.989 / 0.955 / 0.770** for 1 / 5 / 10 / 20 vertices (published
    0.97 / 0.99 / 0.96 / 0.77).
- **Run 3** (`outputs/run3_reproduction.txt`, checks at `02_run3.py:77-78`):
  - On all 103 events, the {3, 4} P matches hitV_P to 8e-16, median **0.181** (published 0.18).
  - The ideal step on lines ≥ 2 matches hitV_P_all to 3e-15, median **0.361** (published 0.36).
- **Baseline categories:** the focused-cell baseline reproduces the Run 3 README. Toy-like
  efficiency is 0.897 (README 0.90) and overall efficiency is 0.585 (README 0.585).

## Results

### Toy (scan vertices; mean over events; `outputs/summary_toy.csv`)

The table gives padded P, then efficiency and ghost rate, for 1 / 5 / 10 / 20 vertices.

| T | filter | padded P | efficiency | ghost rate | gates (1 vertex) |
|---|---|---|---|---|---|
| 100 | ideal {3, 4} | 0.93 / 0.92 / 0.77 / 0.57 | 0.97 / 0.97 / 0.81 / 0.60 | 0 | — |
| 100 | ideal step ≥ 2 | 0.95 / 0.94 / 0.79 / 0.58 | 1.00 / 0.99 / 0.84 / 0.63 | 0 | — |
| 100 | soft band | 0.95 / 0.94 / 0.85 / 0.62 | 1.00 / 1.00 / 0.92 / 0.68 | ≤ 0.002 | — |
| 100 | poly d = 6 / 6 shifted | 0.74 / 0.76 / 0.68 / 0.52 ; 0.93 / 0.87 / 0.81 / 0.56 | 1.00 / 1.00 / 0.92 / 0.68 | ≤ 0.002 | 2.8e6 |
| 100 | poly d = 16 | 0.68 / 0.71 / 0.63 / 0.49 | same | same | 7.4e6 |
| 100 | one-bit 4I − C | 0.93 / 0.92 / 0.82 / 0.60 | same | same | 9.3e5 |
| 100 | segment one-bit ≤ 1 % / QSVT | 1.9e-3 / 9.1e-4 | 0.94 / 0.94 / 0.96 / 0.94 | ≤ 0.004 | 5.3e6 / 1.5e8 |
| 1000 | ideal {3, 4} | 0.58 / 0.59 / 0.57 / 0.46 | 0.97 / 0.99 / 0.96 / 0.78 | ≤ 0.0004 | — |
| 1000 | ideal step ≥ 2 | 0.59 / 0.59 / 0.58 / 0.48 | 0.97 / 0.99 / 0.97 / 0.84 | ≤ 0.006 | — |
| 1000 | soft band | 0.60 / 0.60 / 0.59 / 0.54 | 0.97 / 0.99 / 0.99 / 0.94 | ≤ 0.013 | — |
| 1000 | poly d = 6 shifted | 0.59 / 0.59 / 0.57 / 0.47 | 0.97 / 0.99 / 0.99 / 0.94 | ≤ 0.013 | 4.7e7 |
| 1000 | poly d = 16 | 0.50 / 0.48 / 0.49 / 0.46 | same | same | 1.4e8 |
| 1000 | one-bit 4I − C | 0.58 / 0.59 / 0.58 / 0.51 | 0.97 / 0.99 / 0.99 / 0.94 | ≤ 0.005 | 1.6e7 |
| 1000 | one-bit 8I − C | 0.30 / 0.29 / 0.28 / 0.24 | same as poly | ≤ 0.013 | 2.2e7 |
| 1000 | segment one-bit ≤ 1 % | 7e-4 | 0.60 / 0.90 / 0.90 / 0.89 | ≤ 0.006 | 3.8e8 |
| 1000 | QSVT comb ≤ 1 % | 2e-4 | 0.85 / 0.86 / 0.86 / 0.85 | ≤ 0.016 | 1.4e10 |
| 1000 | focused cells | — | 0.99 / 1.00 / 0.97 / 0.81 | ≤ 0.002 | — |

T = 400 is in the CSV and in the figures. Qubits at T = 100 / 400 / 1000:
- polynomial: 13 / 15 / 17;
- one-bit filter on hits: 11 / 13 / 15;
- segment one-bit filter: 18 / 22 / 24;
- segment QSVT: 20 / 24 / 26.

Efficiency by hits per track, scan vertices (`outputs/summary_hits.csv`), for 3 / 4 / 5 hits:

| | T = 100 | T = 400 | T = 1000 |
|---|---|---|---|
| ideal {3, 4} | 0 / 0.91 / 0.85 | 0 / 0.93 / 0.91 | 0 / 0.94 / 0.93 |
| any filter passing lines ≥ 2 (band, polynomials, one-bit) | 0.90 / 0.93 / 0.89 | 0.96 / 0.97 / 0.98 | 0.97 / 0.96 / 0.98 |
| segment classical / one-bit / QSVT at ≤ 1 % | 0 / 0.76 / 0.99 | 0 / 0.52–0.66 / 0.97 | 0 / 0.003–0.007 / 0.82–0.94 |
| focused cells | 0 / 0.90 / 0.86 | 0 / 0.92 / 0.93 | 0 / 0.94 / 0.95 |

### Run 3, 103 events (`outputs/summary_run3.csv`, `outputs/summary_run3_categories.csv`)

P, gates and qubits are medians over events. Efficiency is pooled over truth tracks. Ghost rate
is the mean over events.

| filter (scan vertices) | P | padded P | qubits | gates | efficiency | ghost rate |
|---|---|---|---|---|---|---|
| ideal {3, 4} | 0.181 | 0.125 | — | — | 0.281 | 0.027 |
| ideal step ≥ 2 | 0.361 | 0.253 | — | — | 0.428 | 0.039 |
| soft band > 1.5 | 0.664 | 0.460 | — | — | 0.647 | 0.055 |
| poly d = 6 / 10 / 16 | 0.42 / 0.49 / 0.57 | 0.309 / 0.348 / 0.397 | 15 | 1.9e7 / 2.9e7 / 4.4e7 | 0.640 | 0.055 |
| poly d = 6 / 10 / 16, shifted | 0.55 / 0.57 / 0.59 | 0.388 / 0.394 / 0.411 | 15 | 1.6e7 / 2.7e7 / 4.3e7 | 0.640–0.642 | 0.055 |
| one-bit 4I − C | 0.451 | 0.311 | 13 | 6.1e6 | 0.570 | 0.052 |
| one-bit 8I − C | 0.393 | 0.268 | 13 | 6.8e6 | 0.637 | 0.055 |
| segment one-bit, ≤ 1 % false | — | 6.6e-3 | 19 | 1.6e7 | 0.029 | 0.011 |
| segment QSVT, truth-free | — | 8.3e-4 | 21 | 7.1e8 | 0.575 | 0.059 |
| segment classical, truth-free | — | — | — | — | 0.327 | 0.110 |
| focused cells | — | — | — | — | 0.585 | 0.026 |

At the truth vertices every hit-register filter gains 0.01–0.04 in efficiency (band 0.687,
polynomials 0.677–0.680) and P changes by ≤ 0.02.

Run 3 efficiency by category, scan vertices (fraction of tracks in brackets):

| | 3 hits (16 %) | \|t\| > 0.25 (21 %) | p < 1 GeV (22 %) | secondaries (22 %) | electrons (13 %) | toy-like (50 %) |
|---|---|---|---|---|---|---|
| ideal {3, 4} | 0.023 | 0.033 | 0.077 | 0.128 | 0.158 | 0.429 |
| polynomial d = 16 shifted | 0.116 | 0.169 | 0.204 | 0.244 | 0.291 | 0.929 |
| one-bit 4I − C / 8I − C | 0.116 / 0.116 | 0.165 | 0.198 / 0.201 | 0.233 / 0.242 | 0.281 / 0.289 | 0.806 / 0.922 |
| focused cells | 0.029 | 0.013 | 0.133 | 0.278 | 0.364 | 0.897 |
| QSVT truth-free | 0.027 | 0.301 | 0.290 | 0.309 | 0.186 | 0.765 |
| classical truth-free | 0.346 | 0.437 | 0.300 | 0.190 | 0.106 | 0.314 |

By hits per track (3 / 4 / 5 / > 5): ideal {3, 4} 0.02 / 0.36 / 0.55 / 0.25; polynomials
0.12 / 0.47 / 0.70 / 0.85; one-bit 4I − C 0.12 / 0.47 / 0.70 / **0.71**.

## Kill criteria (`outputs/kill_criteria.txt`)

| criterion | value | verdict |
|---|---|---|
| K1: best polynomial of degree ≤ 16, padded P on Run 3 < 0.30 | 0.411 (d = 16, shifted; scan vertices, median); 0.397 on the symmetric domain; 0.427 at the truth vertices | **pass** |
| K2: toy-like Run 3 efficiency < 0.85 | polynomials 0.929 (scan) / 0.956 (truth); one-bit 8I − C 0.922; one-bit 4I − C 0.806 | **pass** for the polynomials; the one-bit filter on 4I − C alone would fail it |
| K3: toy 3-hit efficiency < 0.5 with ghost rate > 0.02 | 3-hit efficiency 0.90 / 0.96 / 0.97 at T = 100 / 400 / 1000; worst ghost rate 0.013 | **pass** |

None fires, so B continues.

## Findings

1. **Passing every line ≥ 2 recovers 3-hit tracks on the toy.** Their efficiency goes from 0 to
   0.90–0.97, at ghost rate ≤ 0.013. Overall toy efficiency at 20 vertices, T = 1000, rises from
   0.78 to 0.94. That is above the focused cells (0.81) and the segment methods (0.85–0.89).
2. **On Run 3 the graded filter is the best method tested** (0.64–0.65 against 0.58 for the
   focused cells and the truth-free QSVT, and 0.33 for the truth-free classical solver).
   - The gain is in tracks with more than 5 hits (0.25 → 0.85) and in 4- and 5-hit tracks.
   - **3-hit tracks stay at 0.12 on Run 3**, against 0.35 for the truth-free classical solver.
     Not yet tested: the likely cause is that most Run 3 3-hit tracks are not cliques of the
     vertex-conditioned operator (large slope, low momentum, secondaries).
3. **P doubles or more on Run 3.**
   - The soft band gives 0.46 padded (0.66 unpadded), against 0.125 padded for {3, 4}. It passes
     broken cliques that the exact-line step rejects, so it beats the ideal step (0.25).
   - The best polynomial of degree ≤ 16 reaches 0.41.
4. **Once a filter passes line 2 and rejects line 1, the tracks no longer depend on the filter.**
   - Every polynomial, the band and the one-bit filters give the same tracks to ±0.005. With the
     exact per-hit output, the readout reduces to the components of C with ≥ 3 surviving hits.
   - So the filter sets P and cost, not efficiency. The exact-output readout hides this; the
     shot budget (plan G) is needed to separate the filters on track quality.
5. **The cheapest effective filter is a one-bit filter with a larger shift.**
   - The one-bit filter on 4I − C has a second zero at c = 8. It therefore loses Run 3 tracks
     with ≥ 8 hits (> 5-hit efficiency 0.71; toy-like 0.81, which fails K2).
   - Moving the shift to 8I − C keeps 2 calls and recovers the polynomials' efficiency (0.637,
     toy-like 0.92), at P 0.27 padded. It costs 6.8e6 gates per solved event, 2.4× fewer than the
     cheapest polynomial (d = 6 shifted, 1.6e7).
   - On the toy, the one-bit filter on 4I − C is the cheapest choice. Toy cliques have ≤ 5 hits,
     so c ≤ 4.
6. **Polynomial degree buys little.** P grows from d = 6 to d = 16 only on Run 3, where α ≈ 15
   (0.31 → 0.40 symmetric, 0.39 → 0.41 shifted). Gates grow linearly with d.
   - On the toy (α ≈ 4–7), d = 6 shifted is as good as any higher degree.
   - The shifted domain [λ_min, α] is worth +0.08 in P at d = 6 on Run 3, because the spectrum is
     nearly one-sided (λ_min ≈ −3.5, α ≈ 15).
7. **Gates against the segment register (model).**
   - Toy, T = 1000: the hit one-bit filter needs 1.6e7 gates and d = 6 shifted needs 4.7e7,
     against 3.8e8 for the segment one-bit filter and 1.4e10 for the segment comb. That is
     24× / 8× / 300× fewer.
   - Run 3: one-bit 8I − C needs 6.8e6, against 1.6e7 for the segment one-bit filter at ≤ 1 %
     (whose efficiency is 0.03) and 7.1e8 for the segment comb.
   - Padding multiplies P by n/2^n = 0.60 at T = 1000 (mean over events). It is the largest
     single P loss on the toy.

## Caveats

- Run 3 hits are Geant4 points smeared by 12 µm, not clusters (Run 3 README, item 1).
- The readout uses the exact output, with no shots.
- α = ρ(C) is computed per event from the exact spectrum. A circuit needs a bound known in
  advance; a looser α lowers P for the symmetric polynomials.
- The identity term of the shifted domain is not priced.
- Ghost rates on Run 3 (0.055) are about twice those of the focused cells (0.026).

## Figures (`figures/`, PDF and PNG; paper 1 style from `Focusing_scan/scripts/plot_style.py`)

- `B01_P_vs_T`: padded P against T per vertex count, every filter and the segment one-bit filter
  and comb.
- `B02_gates_vs_T`: two-qubit gates per solved event against T.
- `B03_qubits_vs_T`: logical qubits against T.
- `B04_efficiency_vs_T` (scan vertices) and `B04b_efficiency_vs_T_truth_vertices`: efficiency
  against T, all methods.
- `B05_ghost_rate_vs_T`: ghost rate against T.
- `B06_efficiency_by_hits`: efficiency by hits per track, toy T = 100 / 400 / 1000 and Run 3.
- `B07_run3_P_and_categories`: Run 3 padded P per filter, and efficiency per category.
- `B08_responses`: the filter responses at a toy α and at the Run 3 median α.
