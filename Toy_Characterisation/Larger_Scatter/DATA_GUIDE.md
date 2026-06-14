# Larger_Scatter — data guide

> Read `../DATA_GENERATION_GUIDE.md` first. **Source of truth:** Notion *Data Coverage*
> (`37d5d544-b9d9-8183-8c50-e76544e53093`); *Data & Metrics* (`3795d544-b9d9-81b5-adda-d1ec5e474d7e`).

## What it is
Robustness of the segment reconstruction to **large multiple-scattering** (σ_scatt up to 1e-3)
and **hit inefficiency** (random dropout) — the two axes that fragment true tracks and create the
coupled-false load.

## Envelope (authoritative: `standard_specs()` → `larger_scatter`)
- `kernel="step"`, `γ=3`, `δ=1`, `σ_res=0`, `φ_max=0.2`.
- `σ_scatt ∈ {1, 3, 5, 7, 10}×1e-4` (full range, scattering-dominated).
- `hit_ineff ∈ {0, .01, .02, .05, .10}` (5 dropout levels).
- ε = **formula** per cell (5 effective ε values, provenance `formula`).
- T grid `{10…1000}`; classical 20 reps, 1BQF statevector 3→1.
- Store coverage (Notion): classical ~3520 · 1BQF ~440 (the largest study).

## Generate / solve / refresh
```bash
$PY .../condor/build_submission.py --lanes events,classical,quantum_cpu --only-missing --submit
$PY .../build_metrics.py
```
Store-backed analysis: `store_analysis.py` → `results/ls_store_summary.csv` + `figures/ls_*.png`.

## Quirks / gotchas
- **hit_ineff fragments tracks** into P3/P2/P1 sub-chains whose eigenvalues fall **off** the P4
  comb lines — the efficiency cost is the fragment population, not noise per se (see the QSVT
  hit-drop study and `feedback-solver-thresholds`). Report efficiency-first working points.
- **Shared clean cell** (σ_scatt=1e-4, drop=0) is shared with Epsilon_study_2 / Verify — attribute
  by the `studies` membership column.
- σ_scatt and hit_ineff are an **independent cross product** here (unlike ERF's paired noise).

## Blocked work — now unblocked
8 1BQF solves at high T. **Matrix-free engine** clears them; resubmit `quantum_cpu --only-missing`.
