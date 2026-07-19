# Block_Encoding — benchmarking literature block encodings on the 1BQF / QSVT systems

**Question (George, 2026-07-19):** can any of

1. **S-FABLE / LS-FABLE** — Kuklinski & Rempfer, arXiv:2401.04234
2. **Almost-optimal basis-state transpositions** — Herbert, Sorci & Tang (Quantinuum), arXiv:2309.12820
3. **Dictionary-based block encoding** — Yang et al., arXiv:2405.18007 (Quantum 9, 1805)
4. **Explicit circuits for structured sparse matrices** — Camps, Lin, Van Beeumen & Yang, arXiv:2203.10236

improve the block encoding used by our 1BQF / QSVT segment solvers?  If not, understand
*why not* and design a better route.

## What we encode

The segment Hamiltonian from the qtrk store (regenerated on demand, never stored):
`A = (γ+δ) I − C` at the production point γ=3, δ=1, so **A = 4I − C**, `b = 1`.
- **step kernel**: C binary symmetric (all couplings exactly 1), nnz ≈ O(n_seg),
  max degree Δ set by hit-sharing occupancy.
- **erf kernel**: C entries continuous in (0,1] — the value-diversity axis.
- The current production encodings this must beat:
  - **QSVT** (`lhcb_velo_toy.solvers.quantum.QSVT`): *exact dense dilation*
    U = [[X,√(1−X²)],[√(1−X²),−X]], X = affine(A) — subnormalization-optimal
    (spectrum fills [−1,1] exactly) but the SELECT gates are dense `UnitaryGate`s:
    O(4^{n_s}) matrix elements, simulation-only. This is the D5 gap
    (`QSVT/Paper_planning/RESOURCE_REDUCTION.md`).
  - **1BQF** (`OneBQF`): Hadamard test on e^{-iAt} from `interaction_pairs`
    (per-pair 2-level rotations), cost ∝ nnz per call.

## Benchmark axes (per method × matrix)

- subnormalization α (with every rescale factor folded in), **verified** encoding error
  ‖A/α − ⟨0|U|0⟩‖₂ by explicit block extraction (small n),
- qubits (system + ancilla), gate counts after transpile to {1q, cx}, depth,
- classical preprocessing cost/time,
- **system-level**: QSVT comb degree multiplier ≈ α / α_dil (α_dil = span(A)/2 — the
  dilation's effective subnormalization), success probability, total 2q gates per
  comb solve and per 1BQF-notch (degree-1) solve.

## Matrices

- Real store events (deterministic keys): exact-verification tier T ∈ {2,3,5};
  structure/scaling tier T ∈ {10,…,400}; step & erf kernels; clean & noisy.
- Canonical clusters (P4 chain, K(1,m) hub) for interpretability.

## Pre-registered expectations from the papers themselves

- FABLE α = 2^n·max|a| — degree disaster independent of gate counts.
- S-FABLE/LS-FABLE §5.3: breakdown on *binary / nonnegative / structured* matrices —
  exactly our C. Measure the rotation-count-vs-error curve on real C.
- Camps: α = s_pad (padded sparsity) but needs an *arithmetic* index oracle; our
  pattern is event-random ⇒ generic oracle pays Ω(nnz·n) (transposition-compiled).
- Dictionary: α = Σ_l|A_l| ≈ χ′(C) ≈ Δ for the *binary* step kernel (near-optimal),
  degenerates to Σ|a_ij| for erf; depth O(log ns) costs O(n²s) ancillas.
- Camps §5 Szegedy-form: α = 1 but encodes the *degree-normalized* D^{-1/2}C D^{-1/2}
  — spectral-lines experiment: does the comb survive normalization?

## Files

- `be_lib.py` — verified implementations + analytic cost models
- `01_verify_and_measure.py` — small-n exact benchmark (block extraction) → CSV
- `02_structure_scaling.py` — real-event structure metrics + resource scaling → CSV+figs
- `03_endtoend_filters.py` — comb-degree inflation, success prob, normalized-walk physics
- `outputs/` — figures + CSVs (provenance: this repo, commit hash in file headers)
