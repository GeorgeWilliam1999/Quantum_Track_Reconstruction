# Inventory — Quantum segment-level analysis

<!-- STATUS: final -->
<!-- SOURCES: see paths below -->

## Headline notebook
`Toy_Characterisation/Verify_new_results/Quantum_segment_level_analysis.ipynb`
(25 cells; backups `.bak_pre{exec,refactor,style,14e,7d,7e}`).

Sections after the §7e τ-sweep insertion:
- §1 Setup (constants, paths, style)
- §2 Sweep configuration → Condor CSVs
- §3 HTCondor submit (markdown)
- §4 Job status
- §5 Aggregator (per-event metric extraction)
- §6 Load both readouts (E1 statevector, E2 sampling) and pre-aggregate
- §7 Figure 1 — segment metrics
- §7b Activation distributions vs T
- §7c Relative-threshold (per-event normalised) fix
- §7d / §8 / §9 / §10 fidelity, tracking, timing figures
- §11 Output artefacts (after τ-sweep §7e cells 21–23)

## Worker
`Toy_Characterisation/Verify_new_results/condor_obqf/run_event.py` — single Condor job, builds `SimpleHamiltonianFast`, solves classically and via `OneBitHHL`, computes segment + track metrics, writes one pickle [source: lines 1–280].

Submission file: `condor_obqf/submit_gpu.sub` — 1 GPU/job, memory tiers 16/32/64/128 GB by T.

## Underlying library (HEAD `13ef495`, `lhcb_velo_toy v2.0.0`)
- Hamiltonian: `LHCb_VeLo_Toy_Model/src/lhcb_velo_toy/solvers/hamiltonians/fast.py` → `SimpleHamiltonianFast`.
- Quantum solver: `LHCb_VeLo_Toy_Model/src/lhcb_velo_toy/solvers/quantum/OneBQF.py` (re-exported as `OneBitHHL`).
- Track finders: `LHCb_VeLo_Toy_Model/src/lhcb_velo_toy/solvers/reconstruction/track_finder.py` → `get_tracks` (connected components, threshold 0.0), `get_tracks_layered` (greedy, threshold 0.45 default, here passed 0.35 + `min_hits=3`).
- Validator: `lhcb_velo_toy.analysis.validation.validator.EventValidator.match_tracks` — purity ≥ 0.70, `hit_efficiency_min = 0.0`, `min_rec_hits = 3`.

## Aggregate CSV outputs
- `outputs/quantum_segment_analysis/seg14e_T1000_statevector/aggregate.csv` (E1, 9 rows × 68 cols, `n_trk ∈ {2,5,10,20,50,100,200,500,1000}`)
- `outputs/quantum_segment_analysis/seg14e_T1000_sampling/aggregate.csv` (E2, identical grid)
- `outputs/quantum_segment_analysis/gamma_sweep_ntrk2.csv` (γ-scan at T=2, used by OneBQF_parameter_audit)
- `outputs/quantum_segment_analysis/sweep_statevector_gamma1.csv` (T ∈ [2,10] at γ=1 fix)

## Existing write-ups (reference, not duplicated)
- `Verify_new_results/OneBQF_parameter_audit.tex` (`.pdf` compiled): code-parity audit + γ-scan + γ=1 ablation at T ∈ [2,10]. Conclusions verbatim:
  - Implementation is byte-identical to upstream `Xenofon-Chiotopoulos/OneBQF`.
  - Padding to next $2^n$ is exact at machine precision.
  - Shot noise is **not** the cause of the 75% efficiency floor — reproduced under shot-free statevector.
  - 75% floor at γ=3 = 8 of 16 eigenvalues sit at the QPE rejection notch φ=0.5.
  - γ=1, δ=1, ε=2 mrad restores perfect efficiency for T ∈ [2,10].
- `docs/reports/segment_analysis/` 4-file scratch suite + `segment_level_report.tex` — the classical (`SimpleHamiltonianFast`-only) reference doc already covers the same Hamiltonian / event-generation / track-finder details. This new report set focuses on the **quantum** side and the **direct comparison**.

## Repetition counts (per (T, rep) combination)
[source: `_add_seg14e.py` and §2 of notebook]
| T | reps | shots (E2) |
|---:|---:|---:|
| 2 | 30 | 8 192 |
| 5 | 30 | 8 192 |
| 10 | 30 | 10 000 |
| 20 | 30 | 40 000 |
| 50 | 20 | 250 000 |
| 100 | 20 | 1 000 000 |
| 200 | 10 | 4 000 000 |
| 500 | 5 | 25 000 000 |
| 1000 | 3 | 100 000 000 |

E1 (statevector) uses `shots=1` for the single coherent run; the readout is exact.
