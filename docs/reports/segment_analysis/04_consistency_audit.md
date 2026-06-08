# Consistency audit

<!-- STATUS: final -->
<!-- SOURCES: segment_level_analysis.ipynb parameter cells; results/run_summary.json -->

## Cross-section parameter table

| Section | particle | PV σ (mm) | module half (mm) | σ_s (mrad) | σ_r (µm) | ε model | angle method |
|---|---|---|---|---|---|---|---|
| §3 baseline | pion | (0,0,1) | 40 | 0.1 | 0 | computed | triplet |
| §4 | pion | (0,0,1) | 40 | swept | 0 | computed | triplet |
| §5 | pion | (0,0,1) | 40 | swept | 0 | computed | triplet |
| §6 | pion | (0,0,1) | 40 | 0.1 | swept | computed | triplet |
| §7 | pion | (0,0,1) | 40 | 0.1 | 0 | computed | triplet |
| §10 | MIP | (1,1,1) | 80 | 0.1 | 5 | fixed 2 mrad | triplet |
| §10b | MIP | (1,1,1) | 80 | 0.1 | 5 | fixed 2 mrad | pairwise |
| §11 | pion | (0,0,1) | 40 | 1e-8 | 0 | computed (zero floor) | triplet |
| §12 | pion | (0,0,1) | 40 | varies | varies | computed | triplet |
| §13d | MIP | (1,1,1) | 80 | 0.1 | 5 | fixed 2 mrad | pairwise |
| §14–§17 | pion | (0,0,1) | 40 | 0.1 | 5 | fixed 2 mrad | triplet |

**Finding**: all differences are deliberate and documented in the notebook markdown.
- §10 vs §10b: only the angle-pair definition differs; geometry & generator are aligned.
- §12e vs §10/§10b: documented as NOT directly comparable (different particle, PV σ, module size) — flagged in-notebook.
- §14–§17: back to the "default" pion generator so knee analysis is consistent with §7/§12 density scans.

No hidden disagreements were found.

## Condor parameter grid
[source: `results/run_summary.json`]
112 jobs total (56 verify_scatt + 56 verify_res). Grid: angle_settings [0.02, 0.2], track_counts [10, 20, 50, 100], n_events 20, with σ_s ∈ {5e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3} for verify_scatt and σ_r ∈ {0, 5, 10, 15, 20, 30, 50} µm for verify_res. σ_r_fixed = 0, σ_s_fixed = 1e-4 mrad. These cover §5 and §6 of the notebook and are consistent with the pkl-cached arrays loaded at the top of §5 and §6.

## Cache & reproducibility
All large sweeps (§4 histograms, §5, §6, §7, §11, §12, §14, §15) read their results from `outputs/segment_analysis/cache/*.pkl` if present, otherwise recompute and persist. Re-running the notebook end-to-end should reproduce all figures bit-identical from cache.

## Sanity checks that passed inside the notebook
- §2 pure-Python vs §2b Numba triplet kernel: identical counts [source: §2b "Numba JIT compiled & validated"].
- §10b pure-Python vs §13d Numba pairwise kernel: identical counts [source: §13d validation stdout].
- §14 custom `_build_ham_vectorized` vs library `SimpleHamiltonianFast.construct_hamiltonian`: "A matrices equal: True, b vectors equal: True" [source: §14 cell #VSC-e18e095c].
- §15 smoke test with `n=5` tracks matches the knee-analysis sweep [source: §15a cell].

No sanity check failed.
