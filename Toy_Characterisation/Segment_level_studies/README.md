# Segment-Level Studies (fixed ε = 2 mrad)

New-data remake of the solver segment-level analysis, built on the decoupled
`qtrk_store` pipeline. These notebooks **only consume the store** — the Condor
solves are read back; nothing is re-solved here. Segment metrics are the
recomputed VIEW (never trusted from disk) at the γ-aware absolute threshold
τ = δ/(δ+γ) + 0.10 (= 0.35 at γ=3). Definitions are the single source of truth
on the Notion page; this project reads them, it does not redefine them.

## Data

- **Store:** `/data/bfys/gscriven/qtrk_store` (events regenerated on demand,
  solution vectors cached, metrics recomputed).
- **Campaign:** fixed Hamiltonian acceptance ε = 2 mrad (`eps_provenance='set'`,
  study `Verify_new_results`). Selected by `seg_store` via
  `eps_provenance=='set' & epsilon≈0.002`.
- **Parameter schema:** σ_scatt = 1e-4, σ_res = 0, φ_max = 0.2, step kernel,
  δ = 1; swept over **γ ∈ {1, 2, 3}**, **hit_ineff ∈ {0, 0.01}**,
  **n_trk ∈ {10, 20, 50, 100, 200, 400, 700, 1000}**.
  Classical: ≈20 reps/point. Quantum (1BQF, statevector): 3 reps low-n,
  1 rep at 700/1000.

## Data layer — `seg_store.py`

- `fixed_eps_metrics()` — fixed-ε rows of the recomputed metrics view.
- `agg_by_ntrk(M, solver, gamma, hit_ineff)` — per-n_trk mean/sem for the 2×2.
- `solves_index(solver, gamma, hit_ineff)` — solve inventory (skips solves whose
  `.npz` is not yet on disk).
- `load_vectors(row, classical_partner=...)` — stored solution vector (quantum
  rescaled to the matching classical L2 norm) + truth mask + per-segment angle.
- `build_A(row)` — regenerate the matrix A for one solve (spectral diagnostics).

## Notebooks

| Notebook | Produces | Source |
|---|---|---|
| `01_solver_segment_efficiency.ipynb` | **The 2×2 figure** (i efficiency, ii false rate, iii pair counts, iv active pairs) vs n_trk — γ=3 headline (classical + quantum, ±1% drop) and a γ∈{1,2,3} companion. | metrics view (no re-solve) |
| `02_solution_amplitudes_and_angles.ipynb` | Solution-amplitude histograms (true vs false, classical & quantum) and per-segment polar-angle histograms (true vs false, counts vs θ). | stored vectors + regenerated geometry |
| `03_hamiltonian_spectrum.ipynb` | λ_min/λ_max/κ/Gershgorin/sparsity vs multiplicity + the true-vs-false separation diagnostic vs n. | regenerated A + stored vectors |

Figures land under `outputs/<notebook>/` as PDF (600 dpi) + PNG (300 dpi).

## Key findings (γ = 3, fixed ε = 2 mrad)

- **Segment efficiency** stays high across multiplicity: classical ≈ 98 %,
  quantum ≈ 99.5 % (1 % drop). **False rate** grows with n_trk — classical
  0 → 21 %, quantum 0 → 44 % (to n=400) — i.e. 1BQF promotes more false
  segments at high density.
- **min(true) = 0.364 at every n** (the outer-segment geometric fixed point),
  always above τ = 0.35 → 100 % true-segment acceptance; the false attractor
  sits at 0.25 with a heavy tail reaching ≈ 1.5 at high n.
- **Spectrum is benign:** κ ≈ 2.4 → ≈ 9 (mild, not explosive); λ_min stays
  positive (~0.8–2.4) even after the Gershgorin lower bound collapses to 0
  (n ≳ 400) — sparsity-driven, not guaranteed. So the solver/conditioning is
  not the failure mode; the false-positive growth is the segment-pairing effect.

## Running

```bash
PY=/data/bfys/gscriven/conda/envs/Q_env/bin/python
# (do NOT set MPLBACKEND=Agg - it suppresses the inline figures in the executed notebooks)
$PY -m jupyter nbconvert --to notebook --execute --inplace 01_solver_segment_efficiency.ipynb
# nb1 reads the metrics view (fast); nb2 reads vectors; nb3 runs ARPACK (slow at n=1000, cached).
```

`seg_store.py` sets `QTRK_STORE` and the `_shared` / toy-model paths itself, so
the notebooks need no extra environment beyond the `Q_env` kernel.
