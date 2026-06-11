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

---

## 1. The pipeline itself

| Directory | Role | Data |
|---|---|---|
| `_shared/qtrk_pipeline` | **Defines & produces** the store | Generates events, builds sparse `A`, solves classical/quantum, writes `manifest/*.csv`. `build_metrics.py` → `metrics.csv`; `condor/` shards the campaign. This is the source of all NEW data. |

## 2. Produce NEW data (params feed the manifest) — analysis **not yet ported**

These directories' `gen_params*.py` define a manifest study, so the store holds their
high-stat data. Their **analysis notebooks still read local `results/` (old)** and need
porting to `qp.load_metrics(study=...)`.

| Directory | Manifest study | Produces | Analysis notebooks (still old → want NEW) |
|---|---|---|---|
| `Epsilon_study_2` | `Epsilon_study_2` (2154) | `gen_params.py`, `gen_params_seg14e.py`, `gen_params_topup.py` | `analysis.ipynb`, `deep_analysis.ipynb`, `segment_metrics_calc_epsilon.ipynb` |
| `ERF` | `ERF` (3240) | param sweep (paired noise) | `analysis.ipynb`, `hamiltonian_comparison.ipynb` |
| `Larger_Scatter` | `Larger_Scatter` (3960) | `gen_params.py` | `analysis.ipynb` |
| `Larger_Scatter_Density` | `Larger_Scatter_Density` (1440) | `gen_params.py` | `analysis.ipynb` |

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
Port the section-2 analysis notebooks (`Epsilon_study_2`, `ERF`, `Larger_Scatter`,
`Larger_Scatter_Density`) from local `results/` to `qp.load_metrics(study=...)`, and
finish migrating the remaining old `Verify_new_results` notebooks.
