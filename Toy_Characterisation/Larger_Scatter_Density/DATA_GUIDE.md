# Larger_Scatter_Density — data guide

> Read `../DATA_GENERATION_GUIDE.md` first. **Source of truth:** Notion *Data Coverage*
> (`37d5d544-b9d9-8183-8c50-e76544e53093`); *Data & Metrics* (`3795d544-b9d9-81b5-adda-d1ec5e474d7e`).

## What it is
Track-**density** axis: narrowing the production cone `φ_max` packs tracks closer together at
fixed T, raising the local occupancy and the coupled-false load — isolating *geometric density*
from track multiplicity.

## Envelope (authoritative: `standard_specs()` → `lsd`)
- `kernel="step"`, `γ=3`, `δ=1`, `σ_res=0`, `hit_ineff=0`.
- `φ_max ∈ {0.2, 0.1, 0.05, 0.02, 0.01}` (5 cone widths; smaller = denser).
- `σ_scatt ∈ {1, 3}×1e-4`.
- ε = **formula** (2 values per the Notion row: {4.2e-4, 1.3e-3}).
- T grid `{10…1000}`; classical 20 reps, 1BQF statevector 3→1.
- Store coverage (Notion): classical ~1280 · 1BQF ~160.

## Generate / solve / refresh
```bash
$PY .../condor/build_submission.py --lanes events,classical,quantum_cpu --only-missing --submit
$PY .../build_metrics.py
```
Store-backed analysis: `store_analysis.py` → `results/lsd_store_summary.csv` + `figures/lsd_*.png`.

## Quirks / gotchas
- **`φ_max` changes the EVENT** (it is an `event_key` axis), so each cone is a distinct event set —
  not reusable across the φ_max sweep.
- Smaller cones raise density → watch the sparse-A guard at high T with the larger ε (1.3e-3);
  keep ε at the formula value (a hand-wide ε at high density goes dense — see master guide §7).
- Clean σ_scatt=1e-4 cells overlap other studies — attribute by `studies`.

## Blocked work — now unblocked
15 1BQF solves at high T. **Matrix-free engine** clears them; resubmit `quantum_cpu --only-missing`.
