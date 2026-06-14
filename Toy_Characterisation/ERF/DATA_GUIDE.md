# ERF — data guide

> Read `../DATA_GENERATION_GUIDE.md` first. **Source of truth:** Notion *Data Coverage*
> (`37d5d544-b9d9-8183-8c50-e76544e53093`) and *Data & Metrics* (`3795d544-b9d9-81b5-adda-d1ec5e474d7e`
> for ε/τ; row `3795d544-…` Source-of-Truth for A sign conventions).

## What it is
Soft-threshold (ERF) Hamiltonian study: replaces the step coupling with
`val = 1 + erf((ε − angle)/(√2·θ_d))` to test a smooth acceptance vs the hard step, across a
**θ_d (erf_sigma)** sweep and **paired** noise points.

## Envelope (authoritative: `standard_specs()` → `erf`)
- `kernel="erf"`, `γ=3`, `δ=1`.
- **Paired noise** (NOT a cross product): `noise_pairs = [(σ_scatt,σ_res)] = [(1e-4,0), (3e-4,.01), (5e-4,.02)]`
  = clean / moderate / heavy.
- `erf_sigma (θ_d) ∈ {1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3}` (6 widths).
- T grid `{10…1000}`; classical 20 reps, 1BQF statevector 3→1.
- Store coverage (Notion): classical ~2880 · 1BQF ~360.

## Generate / solve / refresh
```bash
$PY .../condor/build_submission.py --lanes events,classical,quantum_cpu --only-missing --submit
$PY .../build_metrics.py
```
Store-backed analysis: `store_landscape.py` → `results/erf_store_landscape.csv` +
`figures/erf_landscape_*.png`. Legacy `analysis.ipynb`/`hamiltonian_comparison.ipynb` (old data).

## Quirks / gotchas — the ERF densification trap (critical)
- The ERF branch writes a value at every surviving pair, so it **must prune** `val > 1e-9`
  (`fast.py`, already in place). Without it the COO stored **O(T³) explicit near-zero entries** →
  150 GB at T=1000 → the historic 64 GB OOM. The pipeline guard *warns* (does not crash) past
  `nnz/n_seg > 250` because wide θ_d at high T is legitimately denser — **confirm feasibility
  before mass-submitting wide-θ_d high-T corners.**
- The **cKDTree A-build speedup (2026-06-14) is step-only** — the ERF branch is unchanged and
  still uses the dense block (validated `max|ΔA|=0`). ERF builds are heavier than step by design.
- Youden-J / EER threshold optimisation needs pooled per-segment scores
  (`qp.load_solution` + truth) — existing store data, analysis not yet done (DATA_INDEX TODO).

## Blocked work — now unblocked
27 1BQF solves (the largest blocked share) at T=400/700/1000. **Matrix-free engine** clears them;
resubmit `quantum_cpu --only-missing`. (ERF A-build is the slower part — still seconds–minutes.)
