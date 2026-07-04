# Occupancy — the Denby per-hit occupancy (hit-uniqueness) term

Subproject for the second Denby–Peterson penalty (split out from
`../Bifurification/` per George, 2026-07-04). **Step kernel only** for now.

The per-hit occupancy constraint E_occ = α Σ_h (o_h − 1)², o_h = Σ_{s∋h} x_s,
expands to the sparse, 1BQF-compatible system
`A_occ = A0 + 4α·I + 2α·B_all`, `b += 4α` (B_all = ALL co-hit segment pairs,
any angle). The isolated-segment attractor shifts to (δ+4α)/(γ+δ+4α) — and is
badly mis-centred in practice (B_all coupling is never negligible), so working
points use the τ-sweep, not the attractor. NOTE: Denby's GLOBAL count form
α(Σx − N)² is rank-one dense → classical-only; the per-hit form here is the
quantum-compatible reading (awaiting George's confirmation).

## Shared machinery
Code lives in `../Bifurification/` (one implementation, two subprojects):
`dp_terms.dp_system(ham, alpha=...)` builds the system; `bif.fork_graph` is
B_all. Data: `../Bifurification/results/dp_{pilot,classical_spectrum,costs,
working_points}.csv` cover the α axis; this folder holds the occupancy-focused
analysis + figures.

## The four questions (George, 2026-07-04) → `occupancy_analysis.ipynb`
1. Does occupancy improve segment efficiency / false rate vs the original —
   classical AND 1BQF?
2. Extra cost in qubits and 2-qubit gates for the 1BQF?
3. How is the sparsity of the matrix affected? (B_all is ~16× the base
   off-diagonal at T=10 — the dominant sparsity cost of the two DP terms.)
4. Does the activation spectrum change — fewer distinct sets of peaks?

Key results so far (pilot + classical-first spectral study, 2026-07-04):
attractor-τ zeroes efficiency at all α while AUC≈1; Youden τ≈0.05–0.08 is
classically perfect at T=50; spectrum stretches to λ_max≈24 (α=0.1) / 64
(α=0.3) leaving ~1% of solution weight in the qsvt comb window — quantum
solvers need a retargeted comb/notch.
