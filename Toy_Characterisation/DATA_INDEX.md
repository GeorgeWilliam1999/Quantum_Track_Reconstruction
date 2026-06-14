# Toy_Characterisation — Data Index

Which directories use (or will want to use) the **NEW** decoupled pipeline data.

- **NEW data** = the content-addressed store at `/data/bfys/gscriven/qtrk_store`
  (env `QTRK_STORE`), produced by `_shared/qtrk_pipeline`. Layers: `events/*.json.gz`,
  `solutions/*.npz`, and the recomputed views `manifest/solutions.csv`,
  `manifest/events.csv`, `manifest/metrics.csv`. Read it via `qtrk_pipeline` (`qp`):
  `qp.load_metrics(...)`, `qp.load_event(...)`, `qp.load_solution(sol_key)`,
  `qp.rescale_to_signal(...)`. Metrics are always **recomputed**, never trusted from disk.
- The store's `manifest` currently spans **5 studies** (solve-row counts):
  `Larger_Scatter` 3960 · `ERF` 3240 · `Epsilon_study_2` 2154 · `Verify_new_results` 1800
  · `Larger_Scatter_Density` 1440.
- "Requires old data" below = reads its own local `results/`/pickles from the pre-pipeline
  era; **not** wired to the store. Listed only so we know to leave it alone.

_Last refreshed: 2026-06-09 (after the signal-support metrics rebuild)._

> **Operational guide (start here):** `DATA_GENERATION_GUIDE.md` — schematics, where metrics are
> defined, the store/keys, solvers, and how to run Condor well. Per-project guides:
> `<study>/DATA_GUIDE.md` for Epsilon_study_2 / ERF / Larger_Scatter / Larger_Scatter_Density /
> Verify_new_results. **Notion remains the single source of truth** (Data Coverage
> `37d5d544-…`, Data & Metrics `3795d544-…`); the guides are the operational companion.
>
> **Scaling / performance reference:** `SCALING_DEEP_DIVE.md` (2026-06-14) — profiled bottleneck
> map. Key facts (now ACTIONED): `A` stays sparse but the **build was O(T³)** → exact cKDTree fix
> (~25×@T1000 vs the §A.1-measured ~100 s original O(T³) build, `max|ΔA|=0`); the **1BQF host OOM was the transpiled circuit fed to Aer**
> (~7 KB/gate × millions), *not* the statevector → **bit-identical matrix-free engine** is now the
> default (statevector-only memory, seconds/solve). The 62 high-T 1BQF solves that were blocked are
> now trivial (clean T=700: 6 s/0.9 GB; noisy T=400: 3 s/0.9 GB).

---

## 1. The pipeline itself

| Directory | Role | Data |
|---|---|---|
| `_shared/qtrk_pipeline` | **Defines & produces** the store | Generates events, builds sparse `A`, solves classical/quantum, writes `manifest/*.csv`. `build_metrics.py` → `metrics.csv`; `condor/` shards the campaign. This is the source of all NEW data. |

## 2. Produce NEW data (params feed the manifest) — analysis **ported (2026-06-14)**

These directories' `gen_params*.py` define a manifest study. As of 2026-06-14 each has a
**store-backed analysis script** that reads `qp.load_metrics(study=...)` directly (no local
`results/` pkls, no re-solve). The original `analysis.ipynb` notebooks are retained but are
no longer the canonical source. **Shared-event gotcha:** clean cells (e.g. p_drop=0,
σ_res=0) are shared across studies, so per-study selection must use the comma-separated
`studies` membership column, **not** the primary `study` column (else low-σ_scatt clean
cells are attributed to Epsilon_study_2 and look missing).

| Directory | Manifest study | Store-backed analysis (NEW canonical) | Legacy notebooks (superseded) |
|---|---|---|---|
| `Epsilon_study_2` | `Epsilon_study_2` (2125) | headline re-derived from store, unchanged (store-verified) | `analysis.ipynb`, `deep_analysis.ipynb`, `segment_metrics_calc_epsilon.ipynb` |
| `ERF` | `ERF` (3190, **paired noise**: clean/moderate/heavy) | `store_landscape.py` → `results/erf_store_landscape.csv` + `figures/erf_landscape_*.png` | `analysis.ipynb`, `hamiltonian_comparison.ipynb` |
| `Larger_Scatter` | `Larger_Scatter` (3920) | `store_analysis.py` → `results/ls_store_summary.csv` + `figures/ls_*.png` | `analysis.ipynb` |
| `Larger_Scatter_Density` | `Larger_Scatter_Density` (1418) | `store_analysis.py` → `results/lsd_store_summary.csv` + `figures/lsd_*.png` | `analysis.ipynb` |

## 3. Consume NEW data (already store-backed)

| Directory | How it reads the store | Notes |
|---|---|---|
| `Verify_new_results` | `qtrk_view.py` adapter → `qp.load_metrics()`, `qp.load_solution`, `paired/aggregate` | **Also produces** manifest study `Verify_new_results` (1800). Store-backed notebooks: `segment_level_store.ipynb`, `Quantum_segment_level_store.ipynb`, `Quantum_segment_level_analysis_new_data.ipynb`. Remaining `segment_level_analysis*.ipynb`, `reconstruction_metrics_verification.ipynb`, `Untitled-*` still read old `results/`. |
| `Segment_level_studies` | `seg_store.py` → `qp.load_metrics`, `solutions.csv`, `qp.load_solution` | Pure **consumer** (no params of its own; uses fixed-ε store rows). Notebooks `01`–`06` import `seg_store`; `07_segment_amplitude_atlas` derives from their outputs. |
| `FR_SE_tradeoff` | `clean_vs_noisy_400.ipynb` → `solutions.csv`, `qp.load_event`, `qp.load_solution`, `qp.rescale_to_signal` | Consumer (clean vs σ_res=0.05 at T=400). |
| `Presentation_mini_q_workshop_UM` | `scripts/_common.py` → `metrics.csv`; `fig_aggregate/quantum/per_event/concept/anim.py` | Figure/animation generation off the store. |
| `Bifurification` | `bif.py` → reuses `qtrk_store` **events**, recomputes solutions with an added bifurcation term | Consumer of events; writes its own derived solutions (not into the manifest). |

## 4. Require OLD data only (leave as-is)

`condor_pipeline`, `EpsilonStudies`, `Initial`, `Quantum_Toy_Study`,
`Recovery_Seperation_analysis`, `Segment_Grass`, `hit_competition_outputs` — all
**require old data**; none reference the store.

---

### TODO implied by this index
- ✅ Section-2 analyses (`Epsilon_study_2`, `ERF`, `Larger_Scatter`, `Larger_Scatter_Density`)
  ported to `qp.load_metrics(study=...)` via store-backed scripts (2026-06-14).
- Finish migrating the remaining old `Verify_new_results` notebooks; retire
  `Quantum_segment_level_analysis.ipynb` (reads local `results/`).
- ERF: Youden-J / EER threshold optimisation still needs pooled per-segment scores
  (`qp.load_solution` + truth) — existing store data, not yet done.

_Section-2 update: 2026-06-14._
