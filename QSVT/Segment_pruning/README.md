# Segment pruning: dropping the segments that cannot be on a track

Started 2026-10-13 on a suggestion a colleague made to George: the Hamiltonian is
built from segments, and every coupling is an acceptance-angle test between two
segments that share a hit. A segment with no compatible neighbour on either side
can never be part of a reconstructed track, so it need not be in the matrix at
all. This folder tests what that buys: how the matrix size and the register
width scale with track count once those segments are gone, and whether the
one-bit filter and the QSVT behave any differently.

## The idea in one paragraph

`A = sI - C` is block-diagonal over the connected components of the coupling
graph (segments are vertices, compatible head-to-tail pairs are edges). A segment
with no edge is a 1 x 1 block `[s]`. Every solver we use is a matrix function of
`A` applied to the uniform vector `b`: the classical inverse, the one-bit
filter's `cos(At/2)` and `sin(At)`, and the QSVT polynomial `p(A)`. A matrix
function of a block-diagonal matrix acts block by block, so deleting whole blocks
leaves the solution on the remaining segments **unchanged**. What changes is the
dimension `n` (so the system register `ceil(log2 n)`), the normalisation of the
quantum state (the post-selection success probability rises by the fraction of
the register that was decoupled), and the number of controls on every
multi-controlled gate of the one-bit circuit. The decoupled segments are exactly
the "isolated" segments of the error atlas: the classical solver puts them at
the attractor `delta/s = 1/4` and the one-bit filter sends them to zero, so no
solver ever read them as track segments anyway. The same argument removes any
union of whole components, so dropping lone compatible pairs (`min_component_size
= 3`) is also exact; whether that is wanted is a separate choice.

The one honest cost: a *true* segment is decoupled when its track lost a hit on
both sides. Pruning removes it, and the study counts those separately. No
segment-pair solver can recover such a segment either (the one-bit filter already
zeroes it), so this is a floor shared by every method, not a loss caused by the
prune.

## What was added to the package

`LHCb_VeLo_Toy_Model` (branch `centralise-2.1`, uncommitted at the time of
writing):

* `solvers/hamiltonians/pruning.py`: `decoupled_mask`, `prune_system`,
  `expand_solution`, `segment_degree`, `coupling_pattern`, `qubits_for`.
* `Hamiltonian.prune(min_degree=1, min_component_size=1)` and
  `Hamiltonian.expand(vec, fill=0)` on the base class, so every builder has
  them; `kept_index`, `n_segments_full`, `prune_info`, and `<array>_full` copies
  of the per-segment arrays.
* `construct_hamiltonian(..., prune=True | dict)` on `SimpleHamiltonianFast`,
  `SimpleHamiltonian` and `ExtendedHamiltonian`. The extended builder prunes the
  final operator after its penalty terms (the occupancy term couples every
  co-hit pair, so there the prune removes nothing, as it must).
* `segment_truth_mask(ham, full_space=True)` for the truth over the original
  index space of a pruned Hamiltonian.
* `tests/test_pruning.py` (11 tests): kept set against a brute-force recount;
  classical, one-bit (package circuit, matrix-free) and QSVT solutions against
  the full solve restricted to the kept segments; the one-bit success
  probability scaling exactly by the register ratio; component prune against
  `connected_components`; the reference builder; the extended builders; expand.

## Files

| file | what it does |
|---|---|
| `common.py` | configurations, event loading from the qtrk store, the four solver lanes, full-space metrics |
| `01_run_study.py` | per event: matrix census (full / decoupled dropped / components < 3 dropped), register widths, true segments decoupled; then every solver on the three variants with exactness check, success probability, time and memory |
| `02_figures.py` | figures and `outputs/summary.md` |
| `03_circuit_cost.py` | small events: the package one-bit circuit transpiled, full against pruned (qubits, two-qubit gates, depth); QSVT resource model |
| `04_activation_spectrum.py` | paper corpus, T = 400 and 1000: amplitude histograms of true and false segments per solver with what each prune removes, and the eigenvalue spectrum of A split by true/false weight and by component type, with the three responses drawn over it |
| `05_pruned_response.py` | re-tunes the filters for the pruned spectrum: one-bit evolution-time scan and comb degree scan (current comb against a design weighted on the surviving false lines), exact, on both pruned variants |
| `06_recompute_metrics.py` | recomputes every metric column from the stored amplitudes with each variant's working point set from the segments that variant keeps |
| `outputs/<config>_scaling.csv`, `<config>_solves.csv`, `circuit_cost.csv`, `summary.md` | results |
| `figures/` | `fig1_matrix_size`, `fig2_qubits`, `fig3_success`, `fig4_metrics`, `fig5_cost`, `fig6_circuit_cost`, `fig7_activations_T{400,1000}`, `fig8_eigenvalues_T{400,1000}`, `fig9_onebit_tscan`, `fig10_comb_degree_scan` |

Configurations (phi_max = 0.2, five planes, gamma = 3, delta = 1, formula
acceptance with kappa = 3): `paper` = 5 µm, 0.15 mrad, 1 % hit loss (the paper-1
corpus, 10 repetitions); `no_noise` = no smearing, 0.1 mrad, no hit loss (3
repetitions); `wide` = 15 µm, 0.45 mrad, 1 % hit loss (3 repetitions). T = 10 to
1000. Events are the cached qtrk-store events, so every number here is on the
same events as the paper.

Solvers: classical inverse; the one-bit filter as the package circuit
(`solve_one_bit_matrixfree`, George's "1BQF"); the exact one-bit readout of the
paper (`p_i = [cos^2(At/2)u]_i^2 + [sin(At)u/2]_i^2`); the QSVT (degree 40 line
comb on the base Hamiltonian, exact statevector, the same polynomial for the full
and pruned systems since the spectral bounds are taken from the full matrix).
Every solver is read at the efficiency-first working point in the full index
space; pruned segments get amplitude zero.

Run: `conda activate Q_env; python 01_run_study.py --config paper --reps 10;
python 01_run_study.py --config no_noise --reps 3; python 01_run_study.py
--config wide --reps 3; python 03_circuit_cost.py; python 02_figures.py`.

## Results

All numbers below are from `outputs/summary.md` (regenerated 2026-10-13 after
the full runs: paper corpus 80 events, the two bracket configurations 24 events
each). "Worst repetition" means the largest register over the repetitions at
that T, which is what a hardware budget has to cover.

### 1. The matrix and the register

| configuration | T | segments (all) | with a compatible neighbour | in a component of 3 or more | qubits all / neighbour / component |
|---|---|---|---|---|---|
| paper corpus | 400 | 632 424 | 2 471 | 1 588 | 20 / 12 / 11 |
| paper corpus | 1000 | 3 925 351 | 17 065 | 4 459 | 22 / 15 / 13 |
| no noise | 1000 | 4 000 000 | 4 909 | 4 010 | 22 / 13 / 12 |
| wide (3x paper) | 1000 | 3 926 345 | 112 429 | 33 983 | 22 / 17 / 16 |

Fitted power laws over T >= 100 (medians): all segments go as T^2.00 (the
4T^2 of the candidate construction); the coupled segments go as T^1.08 with no
detector effects (essentially the 4T true segments), T^1.48 on the paper corpus
and T^2.34 on the wide configuration, where the compatible false grass is
already the bulk of what is kept at T = 1000. The components-of-three-or-more
count stays within a factor 1.1 of 4T on the paper corpus all the way to T =
1000 (T^1.03). So the register saving is 7 to 9 qubits at T = 1000 on the
paper corpus and 9 qubits without noise, but it shrinks to 5 qubits once the
acceptance window is three times wider: the prune buys most exactly where the
window is tight. Figures: `fig1_matrix_size`, `fig2_qubits`.

### 2. The solutions do not change

For every event, solver and prune variant, the pruned solve agrees with the
full solve restricted to the kept segments to 4e-7 (classical, the MINRES
tolerance on two different systems) and to float32 round-off for the one-bit
filter and 1e-16 for the QSVT. Efficiency at false rate <= 1 %, the false rate
at 99 % efficiency and the AUC are identical between the full matrix and the
decoupled-segments-dropped matrix at every T for all three solvers
(`fig4_metrics`, solid against dashed). The true segments that the prune
removes are those whose track lost a hit on both sides: 154 of 39 240 true
segments over the ten T = 1000 paper-corpus events (0.4 %), none without hit
loss. The one-bit filter already sends them to zero, so no segment-pair solver
loses anything it had.

### 3. What the quantum lanes gain

* **Post-selection probability.** The one-bit circuit's success probability
  rises by exactly 2^(qubits saved): 7.5e-4 to 0.096 at T = 1000 on the paper
  corpus (x128), 5.5e-4 to 0.14 at T = 400 (x256). The QSVT's rises by
  n_full / n_kept minus the comb's small leak at the decoupled line: 4.3e-4 to
  0.097 at T = 1000 (x224). Amplification cost 1/P drops by the same factor
  (`fig3_success`).
* **Statevector cost.** The matrix-free one-bit emulation at T = 1000 goes from
  a 2^24-amplitude state (268 MB, 15-30 s) to 2^17 (2 MB, well under a second)
  (`fig5_cost`).
* **Gate cost.** On transpiled one-bit circuits for T = 2 to 15 the number of
  pair gates is unchanged (it is the number of couplings) but each pair gate
  has fewer controls: two-qubit gates drop 2.0-2.6x and depth 3.3-4.8x at T =
  10-15 (T = 15: 870 to 58 segments, 12 to 8 qubits, 12 301 to 5 051 two-qubit
  gates, depth 28 063 to 7 630). QSVT register: 17 to 13 qubits (LCU) or 14 to
  10 (phase sequence) at T = 15 (`fig6_circuit_cost`, `outputs/circuit_cost.csv`).

### 4. An unplanned finding: the 99 % working point is set by two-segment pieces

From T = 400 on the paper corpus the false rate at 99 % efficiency is about
0.34 (T = 400) to 0.77 (T = 1000) for every solver, while the efficiency at
false rate <= 1 % is still 0.97 (classical, T = 400) and 0.92 (QSVT, T = 1000).
The reason is visible once the component census is in hand: about 1 % of the
true segments sit in two-segment pieces (a track that lost its second or
fourth hit), and those have the amplitude of a false compatible pair in every
solver. The 99 % rule then has to place the threshold at the pair level and
admits every false pair, of which there are thousands. Dropping the components
of fewer than three segments (`min_component_size = 3`, equally exact) removes
the true pairs from the threshold set and the false pairs from the matrix:
false rate at 99 % efficiency 0.34 to 0.008-0.015 at T = 400 (QSVT 0.008,
classical 0.012, one-bit 0.015) and 0.77 to 0.06-0.13 at T = 1000 (classical
0.064, QSVT 0.126, one-bit 0.129), with the 99 % threshold now set from the
true segments that variant keeps and efficiency over all true segments
0.990-0.996 (the segments in pairs are the price). The efficiency at false
rate <= 1 % does not move. (Numbers corrected 13 Oct evening by
`06_recompute_metrics.py`; the first pass set this variant's threshold from
the wrong segment set.)
This is the same structural-miss floor as the decoupled true segments, one
component size up, and it is a working-point statement, not a solver
statement: George should decide whether the papers read the solvers on
components that can hold a track (three or more segments) or on everything.

### 4b. Activations and the eigenvalues behind them (`fig7`, `fig8`)

`fig7_activations_T{T}`: per solver, the amplitude histograms of true and
false segments of the full solve (amplitudes relative to the median true
amplitude, so the three solvers share an axis), with the false segments that
survive each prune drawn as outlines. The decoupled prune removes the one
giant bar: the classical attractor at 0.25 (0.69 of the true median), the
exact zero of the one-bit filter and the comb's small leak at the decoupled
line. What is left of the false population after it is the compatible pairs,
which sit at 0.6-0.7 of the true median for the one-bit filter (inside the
true band's lower edge, which is why its efficiency at false rate under 1 %
is stuck at 0.5 from T = 100) and at about 0.5 for the comb (below the true
band). The components-under-3 prune removes that population too.

`fig8_eigenvalues_T{T}`: the spectrum of A over the ten events, every
eigenpair weighted by its overlap with true segments against false segments
(left) and stacked by component type (right), with the three responses
normalised at the first track line drawn above. The isolated line at s = 4
carries 6 x 10^6 false eigenvalues at T = 400 and nothing true; the four
track lines carry 1.5 x 10^4; the false pairs are the two bars at s -/+ 1
(8 x 10^3), the chains the bars at s -/+ sqrt 2 and s. After the decoupled
prune the bar at s is gone entirely (the only eigenvalues left there are the
middle modes of three-segment chains, a few hundred), and after the
components-under-3 prune the pair bars are gone too. Class counts are in
`outputs/eigen_classes_T{T}.csv`.

What that means for the responses (George's question of 13 Oct): the
one-bit filter's notch at s, its only design freedom, now points at an empty
line, and the comb no longer has to hold its deepest null against a line that
outnumbers the tracks 400 to 1. `05_pruned_response.py` re-tunes both on the
pruned systems; see section 4c.

### 4c. Re-tuning the filters for the pruned spectrum (`fig9`, `fig10`)

Both scans are exact (per-component eigendecomposition), paper corpus, ten
events at T = 400 and ten at T = 1000, on the full matrix (decoupled
segments added analytically at their 1 x 1 value), the decoupled-dropped
matrix and the components-under-3 matrix. Medians over events;
`outputs/pruned_onebit_tscan.csv`, `outputs/pruned_comb_scan.csv`.

**One-bit filter, evolution time t (`fig9_onebit_tscan`).** The notch at s
(t = pi/s = 0.785) is the worst place to be once the decoupled segments are
gone. On the decoupled-dropped matrix at T = 400 the efficiency at false rate
<= 1 % is 0.49 at t = pi/s and 0.97 for t between 1.0 and 1.15, which is the
notch moved to the false-pair line s - 1 = 3 (t = pi/3 = 1.05). That is the
classical solver's and the comb's level (0.97), reached by a degree-one
filter. On the full matrix the same t is a disaster (efficiency 0: the
isolated line sits at |cos(2 pi/3)| = 0.5 and six million segments light up),
which is why the notch had to be at s before. The false rate at 99 %
efficiency stays at 0.34 on the decoupled-dropped matrix for every t (the
two-segment pieces again) and is 0.008-0.015 on the components-under-3
matrix, where any t outside 0.6-1.0 gives efficiency 0.98. At T = 1000 no
evolution time takes the one-bit filter above 0.49 on any matrix; its false
rate at 99 % on the decoupled-dropped matrix improves from 0.77 to 0.45 at
t = 1.0-1.15.

**Comb, degree (`fig10_comb_degree_scan`).** The efficiency at false rate
<= 1 % depends on the degree and not on the prune: the current line comb
reaches its plateau at degree 12 (T = 400, 0.97) and degree 20 (T = 1000,
0.94), on all three matrices alike. What the prune changes is the 99 %
reading: on the full matrix it is degree-sensitive, because the comb's leak
at s times six million decoupled segments is above the 99 % threshold at
degrees 24 and 30 (false rate 0.998) and below it at 40 (0.34), so degree 40
worked by the size of its leak, not by design; after the decoupled prune the
99 % reading is the same at every degree from 8 to 40. The null at s was the
fragile part of the design and it is no longer needed. The alternative design
fitted with its weight on the surviving false lines (pairs at s -/+ 1, chain
modes at s -/+ sqrt 2 and s) brings nothing: equal at T = 400 and one step
later to the plateau at T = 1000 (degree 24 against 20). So the
recommendation is to keep the line comb and halve the degree: 20 walk calls
instead of 40 at T = 1000, 12 at T = 400, with the register already 7-9
qubits smaller.

**In one line.** After the prune the one-bit filter can be re-tuned to the
false-pair line and then matches the comb at T = 400 (efficiency 0.97 at
false rate <= 1 %), and the comb needs half its degree; at T = 1000 only the
comb (degree >= 20) clears 0.9, as before.


### 5. What does not change, and caveats

* The extended Hamiltonian with the occupancy term is untouched by the exact
  prune (every co-hit pair is coupled), so the occupancy-refit results keep
  their full register unless one accepts a different operator (suggestion 7).
* The acceptance window drives everything: the prune's value is set by the
  number of compatible false pairs, which is what the wide configuration shows
  growing faster than T^2.
* The prune is a classical pre-processing step with the same cost as the
  Hamiltonian build (the segment degree is a by-product of building A), plus an
  O(n) relabelling onto a compact register; it must be charged as such in any
  resource table.
* The package feature is uncommitted on `centralise-2.1`; the study CSVs,
  figures and stored amplitudes are in this folder and not in the qtrk store.

**Recommendation.** Keep it. It is exact, it costs nothing beyond the build,
it takes 7-9 qubits off the register at T = 1000 on the paper corpus, raises
the post-selection probability by two orders of magnitude, and makes the
quantum lanes' register scale with the coupled segments (close to 4T when the
window is tight) instead of the 4T^2 candidates. The decision that needs
George is the second one: whether the stronger, still exact, component-size
prune becomes the convention for reading the 99 % working point.


## Suggested further analysis and tests

Ordered by how much they would change the decision.

1. **Resource accounting for the papers.** Redo the qubit and call counts of paper
   1 (and the per-call price of the hit-priced block encoding, paper 4) on the
   pruned index set. The prune is free on top of the Hamiltonian build (the
   degree of every segment is a by-product of building `A`), but mapping the
   kept segments onto a compact register is a classical O(n) relabelling that
   must be charged in the same column as the build. The honest statement is
   "the quantum register holds the coupled segments, not the candidate
   segments", and that changes the `ceil(log2 4T^2)` line in every table.
2. **Re-tune the filters for the pruned spectrum.** Done 13 Oct (section
   4c): one-bit notch to the false-pair line (t = pi/3) recovers 0.97 at T =
   400; comb degree 40 to 20. Open: the re-tuned one-bit filter on the package
   circuit (the scan used the exact readout) and the degree-20 comb's phase
   factors and resource row for the paper.
3. **Stronger prunes.** `min_component_size = 3` (lone pairs out) and `4` (only
   components that can hold a complete track) are exact too; measure the true
   segments each one drops on the hit-loss corpus and at Run-3 hit
   inefficiencies, and the extra qubits saved. A chain-length criterion (the
   longest head-to-tail path in the component) is the natural next refinement.
4. **Acceptance dependence.** The pruned size is the number of segments with a
   compatible neighbour, which grows with the acceptance window; repeat the
   scaling at the formula acceptance with kappa = 2, 3, 4, 5. The prune buys
   most where the window is tight, which is also where the classical solver is
   best, so this pins down where the quantum lanes still have a case.
5. **Real events.** Run the prune on the Run-3 VELO events
   (`Toy_Characterisation/Run3_Verification`): real hit loss makes decoupled
   true segments common, and the prune then doubles as the structural-miss
   count that any segment-pair solver must report.
6. **Padding waste.** `ceil(log2 n_kept)` wastes up to half a register when
   `n_kept` sits just above a power of two; dropping the smallest components
   first (or the least likely by amplitude) until the next power of two would
   be an exact way to recover that qubit on most events.
7. **Occupancy term.** The exact prune does nothing with the occupancy penalty
   (every co-hit pair is coupled). The alternative, prune on the base operator
   and then add the penalty to the pruned system, is a different operator;
   test whether the refit result (false rate 0.039 at efficiency 0.97, the
   fragment floor broken) survives it, since that is the operator the paper 3
   plan needs.
8. **Gate cost model.** The pair-gate count is unchanged by the prune; what
   shrinks is the number of controls per gate. Fit the transpiled two-qubit
   count against (pairs x register width^2) from `circuit_cost.csv` to give a
   closed form for the saving, and extend to a noisy simulation (IBM noise
   model, small T) full against pruned to see whether the shallower circuit
   buys fidelity.
9. **Beyond T = 1000.** With the quantum lanes now sized by the coupled
   segments, T = 2000 and 5000 are affordable for the one-bit filter and the
   QSVT; the limit becomes the Hamiltonian build and the classical solve.
10. **Hit-uniqueness post-selection on top.** The prune and the post-selection
    (`solvers/postselect.py`) remove different false types (decoupled segments
    against contested hits); run both to see whether the heavy-corpus false
    rate drops further than either alone.

