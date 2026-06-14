# Verify_new_results — data guide

> Read `../DATA_GENERATION_GUIDE.md` first. **Source of truth:** Notion *Data Coverage*
> (`37d5d544-b9d9-8183-8c50-e76544e53093`) and *Data & Metrics — Source of Truth*
> (`3795d544-b9d9-81b5-adda-d1ec5e474d7e`). This is the **canonical 3-solver benchmark**.

## What it is
The shared cross-solver point: **classical vs 1BQF vs QSVT** on the *same* events, across the
full T range and a γ sweep. It is the reference all other studies are compared against, and the
home of the 1BQF→T=1000 and QSVT→T=1000 coverage.

## Envelope (authoritative: `standard_specs()` → `verify` + `verify_fixed`)
- `kernel="step"`, `δ=1`, `σ_scatt=1e-4`, `σ_res=0` (clean), `φ_max=0.2`.
- `γ ∈ {1, 2, 3}` (the γ=1 case is **indefinite** → MINRES classical, QSVT γ-aware band floor).
- `hit_ineff ∈ {0, 0.01}`.
- **Two ham_keys, same events:**
  - `verify` — ε = **formula** (provenance `formula`).
  - `verify_fixed` — ε = **0.002 (2 mrad) fixed** (provenance `set`, `eps_values=[0.002]`): the
    actual acceptance used by the analysis notebooks (`FIXED_EPSILON`/`EPS_LOC`).
- T grid `{10…1000}` (13 points incl. the densify fill {30,75,150,300,550}); classical 10–20 reps,
  QSVT 10 (≤400) / 5 (550–1000), 1BQF 10 (≤75) / 5 (100–200) / 3 (400) / 1 (700/1000).
- Store coverage (Notion): classical 1870 · 1BQF 300 · QSVT 506.

## Read / generate / refresh
Store-backed via the **`qtrk_view.py`** adapter → `qp.load_metrics()`, `qp.load_solution`,
paired/aggregate helpers. Store-backed notebooks: `segment_level_store.ipynb`,
`Quantum_segment_level_store.ipynb`, `Quantum_segment_level_analysis_new_data.ipynb`. QSVT rows are
registered by the **QSVT campaign scripts**, not `build_manifest` — see the ⚠️ below.
```bash
$PY .../condor/build_submission.py --lanes events,classical,quantum_cpu --only-missing --submit
$PY .../build_metrics.py
```

## Quirks / gotchas
- ⚠️ **QSVT rows are not in the specs.** A `build_manifest` rebuild **drops qsvt rows** from
  `solutions.csv` (the `.npz` survive on disk); re-run the QSVT campaign script
  (`QSVT/qsvt_store_campaign.py`) to re-register. Never rebuild the manifest without re-registering.
- The `[0.34…0.906]` "eps" sweep in `Quantum_segment_level_analysis.ipynb` is a **solution-vector τ
  sweep** (analysis-only), *not* a Hamiltonian ε.
- γ=1 is the indefinite operating point — classical uses MINRES (never CG); filtered solvers use the
  γ-aware design. Don't judge the filtered solvers at the classical τ=0.35.
- Legacy notebooks `segment_level_analysis*.ipynb`, `reconstruction_metrics_verification.ipynb`,
  `Untitled-*` still read old `results/` — being retired (DATA_INDEX TODO).
- A=γ+δ on the diagonal, off-diagonal −1 attractive (Source-of-Truth §2.4); a globally-flipped
  sign solves identically but use the canonical form.

## Blocked work — now unblocked
1BQF "didn't scale past T=700" on the statevector — that was the Aer host-RAM OOM, **not** the
statevector. **The matrix-free engine makes T=1000+ 1BQF routine** (12 s / 1.5 GB), closing the
1BQF column that QSVT had to cover alone. Resubmit the missing 1BQF high-T reps via
`quantum_cpu --only-missing`.
