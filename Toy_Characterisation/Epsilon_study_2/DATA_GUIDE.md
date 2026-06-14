# Epsilon_study_2 — data guide

> Read `../DATA_GENERATION_GUIDE.md` first (shared geometry, metrics, store, Condor).
> **Source of truth:** Notion *Data Coverage* (`37d5d544-b9d9-8183-8c50-e76544e53093`) for
> coverage; the §7 ε-sensitivity write-up page `3795d544-b9d9-8141-a647-d1ec5e474d7e`;
> *Data & Metrics — Source of Truth* (`3795d544-b9d9-81b5-adda-f20676fc541d`) for definitions.

## What it is
Noise-sensitivity characterisation of the **1BQF** segment reconstruction: how detector noise
(σ_scatt, σ_res) moves segment efficiency / false-rate / purity, with the **formula ε** replacing
the old hand-tuned acceptance.

## Envelope (authoritative: `qtrk_pipeline/manifest.py::standard_specs()` → `eps2`)
- `kernel="step"`, `γ=3`, `δ=1`, `φ_max=0.2`, `hit_ineff=0`.
- `σ_scatt ∈ {1, 3, 5}×1e-4`  ×  `σ_res ∈ {0, .01, .02, .05}` → **12 noise cells** (full cross product).
- **ε = formula** per cell (`compute_epsilon`), provenance `formula`.
- T grid `{10,20,50,100,200,400,700,1000}`; classical 20 reps, statevector 1BQF 3 reps (taper 1 at high T).
- Store coverage (Notion): classical 1920 · 1BQF ~234 rows.

## Generate / solve / refresh
```bash
# manifest is built from standard_specs(); submit via build_submission (see master guide §5)
$PY .../condor/build_submission.py --lanes events,classical,quantum_cpu --only-missing --submit
$PY .../build_metrics.py            # refresh the metrics VIEW after solves land
```
Analysis is **store-backed**: headline re-derived from `qp.load_metrics(study="Epsilon_study_2")`.
ε-sensitivity scripts: `gen_epsilon_sensitivity_scan.py`, `gen_metrics_vs_T.py`,
`gen_T400_eval.py`, `gen_sparsity_vs_epsilon.py` → `figures/epsilon_sensitivity/*.png` +
`outputs/*.json`. Legacy `analysis.ipynb`/`deep_analysis.ipynb` read old `results/` (superseded).

## Quirks / gotchas
- **Shared-event attribution:** clean low-σ cells are shared across studies — select rows by the
  comma-separated `studies` membership column, **not** the primary `study` column.
- **σ_res=0.05 at high T is hub-indefinite** (true amplitudes go negative; no τ reaches 99 % eff at
  T=400) — a solver-level breakdown, not a threshold artefact. Expected, documented in §7.
- The `[0.34…0.906]` "eps" sweep in the quantum notebook is a **solution-vector τ sweep** (analysis
  only, no jobs), *not* a Hamiltonian ε.
- High-T pickles in the old `results/` carried O(T³) `false_angles` (251 GB bloat) — that data is
  superseded by the store; do not revive it.

## Blocked work — now unblocked
12 1BQF solves at T=400/700/1000 (σ_res cells) were OOM-blocked. **The matrix-free engine solves
them in seconds at <1.5 GB** (clean default now). Resubmit `quantum_cpu --only-missing`.
