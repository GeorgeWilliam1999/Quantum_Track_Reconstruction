# High-T 1BQF refresh — running log (2026-06-14)

Matrix-free 1BQF engine (default) + cKDTree A-build (committed `LHCb_VeLo_Toy_Model@17db26f`)
unblocked the high-T 1BQF solves that the legacy Aer path OOM'd on. This log tracks the
data regenerated, figures refreshed, and write-ups corrected.

## TASK 1 — data (DONE, verified)

- **62 previously-missing 1BQF solves** generated locally matrix-free
  (`run_shard.py` on `shards/rescue_missing_1bqf.csv`): 16×T400(CPU), 16×T700(CPU),
  30×T1000(GPU-keyed). `done=62 skipped=0 failed=0` (3672 s incl. wide-ERF O(T³) A-builds).
  Studies: Epsilon_study_2, ERF, Larger_Scatter, Larger_Scatter_Density.
- **Verify_new_results 1BQF rep top-up at T=700/1000** (`Verify_new_results/topup_1bqf_highT.py`):
  +32 rows mirroring every qsvt T700/1000 rep≥1 on the SAME (event,ham) → 1BQF now matches
  qsvt rep counts (γ=3: 5 reps; γ=1,2: 3 reps; both hit_ineff). Appended to solutions.csv (.bak).
- **metrics.csv refreshed** (`build_metrics.py`, recomputed view, absolute γ-aware τ).
- **Verification:** solutions.csv 13848 rows, **0 missing**; metrics 13848 rows; solver counts
  classical 11643 / quantum 1526 (+32) / **qsvt 679 (UNCHANGED — manifest not rebuilt)**.

### Re-derived numbers (from refreshed metrics.csv)

**Verify 3-solver benchmark, γ=3 clean, fixed ε=2 mrad (eff% / far%):**

| T | classical | 1BQF (NEW, reps) | qsvt comb |
|---|---|---|---|
| 100 | 100 / 0.22 | 74.95 / 0.07 (5) | 99.6 / 0.00 |
| 400 | 100 / 2.25 | 74.71 / 1.73 (3) | 97.23 / 0.02 |
| 700 | 100 / 9.64 | **74.45 / 8.05 (5)** | 94.29 / 0.17 |
| 1000 | 100 / 20.48 | **74.25 / 18.79 (5)** | 92.14 / 0.93 |

Old QSVT II table had 1BQF 74.1/7.49 (T700) and 74.0/19.8 (T1000) from single-rep anchors →
update to the 5-rep means above (now with error bars).

**Per-study high-T 1BQF (rescue cells), eff / far / purity / cos_QC:**

| study | T700 | T1000 |
|---|---|---|
| Epsilon_study_2 | 0.733 / 0.425 / 0.575 / 0.959 | 0.772 / 0.595 / 0.405 / 0.839 |
| ERF (wide/heavy) | 0.962 / 0.770 / 0.230 / 0.439 | 0.970 / 0.826 / 0.174 / 0.537 |
| Larger_Scatter | 0.698 / 0.192 / 0.808 / 0.967 | 0.698 / 0.310 / 0.690 / 0.965 |
| Larger_Scatter_Density | 0.788 / 0.437 / 0.564 / 0.816 | 0.790 / 0.507 / 0.493 / 0.810 |

## Condor note
56 stale **held** jobs (clusters 4826704 quantum_cpu T400/700, 4826705 quantum_gpu T1000)
are the old Aer-OOM shards for exactly these 62 solves — now superseded. `condor_rm` was
blocked by the permission classifier (jobs not created this session); **left for the user to
remove** (`condor_rm 4826704 4826705`). They are held/idle and harmless.

## TASK 2 — analyses/figures regenerated (DONE)

Re-run with the refreshed metrics view (Q_env, matrix-free engine, **never** MPLBACKEND=Agg
for notebooks):
- **Epsilon_study_2** `gen_metrics_vs_T.py` → `eff_fr_vs_T_sscatt_family.png`,
  `eff_fr_vs_T_sres_family.png` — 1BQF now plotted to **T = 1000** in every noise row.
- **ERF** `store_landscape.py`, **Larger_Scatter** `store_analysis.py`,
  **Larger_Scatter_Density** `store_analysis.py` — CSVs refreshed (quantum rows now to T=1000);
  ERF/LSD store figures are classical (byte-unchanged). LS `fig_quantum` title fixed
  ("T<=200" → "to T=1000") and `ls_quantum.png` regenerated (1BQF cos≈0.966 to T=1000).
- **QSVT nb01** `01_qsvt_segment_efficiency` — stale "1BQF stops at T=400" text fixed (cells
  intro/§1/plot-comment); 5 figs regenerated; headline table 1BQF now 74.45/8.05 (700),
  74.24/18.79 (1000), quantum metric rows 220→252. **nb02** re-run (T=400 only, unchanged).
- **Verify** `segment_level_store`, `Quantum_segment_level_store` (1BQF metrics now paired to
  T=1000, 372 points), `Quantum_segment_level_analysis_new_data` — all executed clean,
  figures + inline plots regenerated (image/png preserved).

## TASK 3 — write-ups corrected (DONE, all re-derived from refreshed metrics.csv)

- **QSVT II** (`37c5d544…baca1c`): 2×2 table 1BQF T700 74.1/7.49→**74.45/8.05**,
  T1000 74.0/19.8→**74.24/18.79**; "single-rep anchors at T≥400" prose replaced; Note updated.
- **QSVT hub** (`37b5d544…ed22`): WP2 callout "where the 1BQF statevector never reached" fixed;
  qsvt far 0.04→0.17 % (700), 1.0→0.93 % (1000), eff 94.5→94.3 %.
- **Paper draft** (`37c5d544…4d56f0`): fixed-τ Table 1 1BQF cells → 74.45/8.05, 74.24/18.79.
- **Application compendium** (`37b5d544…14`): §10 scale table — **dashed 1BQF cells filled**
  ("— (statevector did not scale)" → 74.45/8.05; "—" → 74.24/18.79); qsvt 0.04→0.17, 1.0→0.93;
  prose updated.
- **Epsilon_study_2 §7** (`3795d544…4725`): "1BQF … to T=400 (clean row only)" → "to T=1000 in
  σ_res ≤ 0.02 rows"; false-rate-tracks-classical clause extended to T=700/1000 (0.33/0.56 vs
  0.34/0.59 at σ_res=0.01); "Extend to T=400–1000 (jobs in progress)" bullet marked done.
- **Larger_Scatter (T3)** (`3795d544…c35e`): "quantum for T≤200" → "to T=1000"; high-T 1BQF
  numbers added (σ_scatt=1e-3 far 22→61→78 % at T=400/700/1000; cos≈0.96–0.97).
- **Verify write-up** (`3795d544…c35e`): data-source callout — row count 2676→2708, 1BQF
  T700/1000 top-up note.
- ERF / Larger_Scatter_Density write-ups: classical-focused, no stale 1BQF-high-T claim → no edit.

**Figure cache-busters:** the regenerated figures keep their raw-GitHub URLs; the `?v=<commit>`
tags must be bumped to the commit that lands these figures once the user pushes, so Notion
re-fetches (else the cached old image persists).

## Concurrency note
A parallel session committed the matrix-free infra (`86493be0`) + SYNC commits and advanced
`main`/this branch to `f0a1c862` (origin/main 1 ahead). Its uncommitted work — `PROJECT_STATUS.md`,
`Epsilon_study_2/theory.md`, the `dp*` detector-physics figures + `gen_detector_physics.py` — was
left untouched; only this refresh's files were committed.
