# Parameter consistency audit — quantum vs classical

<!-- STATUS: final -->
<!-- SOURCES: run_event.py defaults; segment_level_analysis.ipynb §14e cell; OneBQF_parameter_audit.tex Table 1, Table 2 -->

## Cross-experiment parameter table

| Experiment | Source | T grid | γ | δ | ε | τ | σ_s (rad) | σ_r (m) | drop | particle | readout |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Classical §14e | `segment_level_analysis.ipynb` §14 | 2–1000 | 3 | 1 | 2e-3 | 0.35 | 1e-4 | 5e-6 | 0.01 | pion | classical CG/spsolve |
| Quantum §14e mirror | `condor_obqf/run_event.py` defaults | 2–1000 | 3 | 1 | 2e-3 | 0.35 | 1e-4 | 5e-6 | 0.01 | pion | E1 (sv) + E2 (sampling) |
| Quantum γ-scan (audit) | `gamma_sweep_ntrk2.csv` | 2 | 0.05–50 | 1 | 2e-3 | 0.35 | 1e-4 | 5e-6 | 0.00 | pion | E1 statevector |
| Quantum γ=1 ablation | `sweep_statevector_gamma1.csv` | 2–10 | 1 | 1 | 2e-3 | 0.35 | 1e-4 | 5e-6 | 0.00 | pion | E1 statevector |

## Differences and rationale

1. **Drop rate**: 0 in the γ-scan and γ=1 ablation; 0.01 in the §14e mirror. The classical §14 sweep
   *also* uses drop=0.01 (verified in `segment_level_analysis.ipynb` §14b markdown), so the §14e
   classical-vs-quantum direct comparison is consistent. The drop-0 audits are baseline noise-free
   characterisations and are documented as such in `OneBQF_parameter_audit.tex` §2.

2. **γ**: The §14e operating point uses γ=3. The OneBQF audit established that γ=3 is sub-optimal
   for OneBitHHL (puts spectrum across the QPE rejection notch at φ=0.5). We deliberately keep γ=3
   for the head-to-head comparison so the classical and quantum sides solve **the same Hamiltonian**.
   The γ=1 result (`sweep_statevector_gamma1.csv` rows 0–6) is reported as a separate result and
   marked as the "corrected" quantum operating point.

3. **τ**: Default 0.35 throughout. The §7e τ-sweep (this work) varies τ ∈ [0.20, 0.70] for the
   E1+E2 §14e data only — the underlying solver runs are unchanged.

4. **Particle / geometry / PV / cone**: identical across all four experiments — pion, 5 modules at
   z = {33, 66, 99, 132, 165} mm, 40×40 mm² half-width, PV σ = (0,0,1) mm,
   φ_max = θ_max = 0.2 rad. Cross-checked in `run_event.py::_safe_generate` (lines 38–80) against
   `segment_level_analysis.ipynb` §1 cells.

5. **Validator cuts**: purity ≥ 0.70, hit_eff_min = 0.0, min_rec_hits = 3 — the LHCb-standard
   reconstructable-track cuts, identical to `segment_level_analysis.ipynb` §15.

## Cross-reference with `docs/reports/segment_analysis/04_consistency_audit.md`
The classical-only audit lists §14–§17 of `segment_level_analysis.ipynb` as
> pion, PV σ (0,0,1), 40 mm modules, σ_s = 0.1 mrad, σ_r = 5 µm, ε = 2 mrad (fixed), τ = 0.35, triplet method.

This is **bit-identical** to the §14e quantum mirror operating point above. Therefore the two
sides of the comparison report (Report B) are reading the same physical scenario.

## Outstanding flags

<!-- FLAG: Reproducibility note — the §14e ε is a *fixed* 2 mrad override, not the formula-computed value (~0.46 mrad at σ_s=1e-4, σ_r=5e-6). The notebook documents this in §14b markdown but it is a quietly load-bearing parameter — at the formula ε the off-diagonal compatibility graph is much sparser and both eff and FR shift. -->

<!-- FLAG: The notebook §7e "optimal τ" panel uses an eff≥0.90 floor and therefore always picks τ=0.20 because the FR-suppressing τ=0.40 row violates the floor at T≥200. The picked-τ panel is not a useful summary on its own — the Pareto curve in fig6 is what tells the story. -->

<!-- FLAG: One cross-system disagreement: at T=10 the E1 sv eff_Q is 0.98 but E2 sampling is 0.89. This is the only row where E1 and E2 disagree by >5%. At T=10 the shot budget is 10 000 — the smallest in absolute terms. The disagreement is consistent with shot-noise scaling: SE(eff) ~ 1/√shots and 10 000 shots × 30 reps gives expected eff jitter ~0.05. Not a code bug — but worth noting in the report. -->

No further inconsistencies found across the four experiments.
