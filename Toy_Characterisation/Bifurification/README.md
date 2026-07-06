# Bifurification — the Denby–Peterson fork (bifurcation) term

One coherent folder for the fork-penalty subproject (merged 2026-07-04 from
`Bifurification` v2 + `Bifurification_initial` v1, per George).

The fork term penalises co-hit segment pairs within a mutual-angle window ε_B:
`A_fork = A0 + β·B_fork` (off-diagonal mode, attractor unchanged). Only the
ε-windowed form is viable — the dense all-pair fork uniformly rescales and
breaks the 1BQF (v1 finding). **Step kernel only** for now (not ERF).

## Layout
- `bif.py` — v1 machinery, still live: `fork_graph` (all co-hit pairs, used by
  the occupancy term), `fork_graph_eps` (ε-windowed fork), `auc`.
- `dp_terms.py` — the shared DP system builder `dp_system(ham, beta, eps_B,
  alpha, ...)` (fork + occupancy compose; the `Occupancy/` subproject imports
  from here).
- `dp_pilot.py` → `results/dp_pilot.csv` — fork/occupancy pilot (T∈{50,100,200}
  ×3 rep, classical + 1BQF).
- `dp_spectrum_classical.py` → `results/dp_classical_spectrum.csv` — the
  classical-first spectral study (which eigenvalues to comb).
- `dp_costs.py` → `results/dp_costs.csv` — 1BQF qubit / 2-qubit-gate cost and
  sparsity of the modified systems (questions 2+3).
- `dp_working_points.py` → `results/dp_working_points.csv` — matched-event
  τ-sweep working points, classical + 1BQF (question 1) + pooled spectra /
  peak structure (question 4).
- `dp_matrix_characterisation.py` → `results/dp_matrix_{census,spectrum,classes}.csv`
  + `figures/dp_matrix_*.png` — the MATRIX study (George, 2026-07-06): sparsity
  census across T × noise (fork-window dial, co-hit degrees, C ⊥ B supports),
  closed-form γ*(β,α) map from measured extremes (γ shifts the band rigidly,
  never shrinks it), and per-false-class separability for the classical solve
  AND the exact 1BQF filter (x_Q = |(e^{iAt}u + u)/2| via expm_multiply —
  circuit-free, validated cos=1.000000000000 vs dense eigh).
- `dp_gamma_validation.py` → `results/dp_gamma_validation.csv` — empirical test
  that γ = γ* does NOT restore 1BQF separation under the DP terms
  (distance-to-notch is γ-invariant; a penalty revives notched segments by
  coupling them — the fork/occ admission fractions are frozen from γ=3 to 8/236).
- `dp_occupancy_proof.py` → `figures/dp_occupancy_proof.png` +
  `results/dp_occupancy_proof.csv` — the four-panel PROOF that occupancy
  cannot help a notch-equipped solver: (a) per-segment revival scatter,
  (b) full τ sweep (no threshold escape; classical contrast), (c) γ scan
  (no γ escape), (d) post-selected mass diverted to false segments.
- `dp_analysis.ipynb` — joint pilot exploration; `fork_analysis.ipynb` — the
  fork-specific answers to questions 1–4.
- `initial_v1/` — the v1 study (notebooks 01–04, `bifurcation_hamiltonian.md`,
  dense-fork characterisation + ε-windowed fix). Reference only; not edited.

## The four questions (George, 2026-07-04)
1. Do the added terms improve segment efficiency / false rate vs the original —
   classical AND 1BQF?
2. Extra cost in qubits and 2-qubit gates for the 1BQF?
3. How is the sparsity of the matrix affected?
4. Does the activation spectrum change — fewer distinct sets of peaks?

Sister subproject: [`../Occupancy/`](../Occupancy/) (the per-hit occupancy
term). Notion write-up: "Denby–Peterson penalties — pilot & classical-first
spectral characterisation".
