# Bifurification — adding a Denby–Peterson fork term to the segment Hamiltonian

A study of what happens to the segment-level metrics (and the classical-vs-quantum
picture) when a **bifurcation / fork penalty** is added to the base segment
Hamiltonian. Builds on `../Segment_level_studies/` (the base model, the
eigenvalue/notch picture, the 1BQF).

## Contents
- **`bifurcation_hamiltonian.md`** — the mathematics. Derives the two forms from
  the Denby–Peterson occupancy penalty:
  - off-diag: `A' = (γ+δ)I − C + βB`, `b = δ1`
  - full: `A'' = (γ+δ+2β)I − C + βB`, `b = (δ+β)1`
  with `C` = continuation adjacency, `B` = fork adjacency (segments sharing a
  start- or end-hit). Both keep a **constant diagonal** (every segment touches
  exactly 2 hits) → both are 1BQF-compatible. Includes the analytic worked
  examples, the spectrum, the expected metric effect, and the classical-vs-quantum
  analysis.
- **`bif.py`** — helper: `event`, `base_hamiltonian`, `fork_graph`, `bif_system`
  (build A,b,τ for either form), `solve_classical`, `solve_quantum` (1BQF),
  `metrics`, `auc`. Events are **reused** from the qtrk_store generator; metrics
  are **recomputed** here because A has changed.
- **`01_construction_and_spectrum.ipynb`** — the fork graph (dense, O(T³)), the
  analytic small clusters, the spectrum (most of the notch survives), and the
  uniform down-scaling of the solution.
- **`02_metrics_vs_beta.ipynb`** — segment efficiency / purity / false-rate / AUC
  vs β for the **dense** fork: classical to **T = 100** (both forms, fixed and
  β-aware τ), and the **classical-vs-quantum** comparison at small T.
- **`03_epsilon_windowed_bifurcation.ipynb`** — the **ε-windowed** (sparse) fork
  `B_ε` (`bif.fork_graph_eps`): restores sparsity, **targeted** false-positive
  suppression with no collateral, and a **preserved 1BQF**. The production version.
- **`04_failure_types_and_phase.ipynb`** — deep dive on the **T = 200** false
  positives with the **classical** solver: per-FP failure type (hub 77 % / bridge
  23 %), classical amplitude, and the **dominant eigenvalue/QPE-phase** that carries
  it. Tests whether the **1BQF phase** can be tuned (a second notch) to erase the
  coupled false the way the single notch at λ=s erases the isolated false. Answer:
  **no** — the coupled false are a *spread that straddles the true band* (hubs at
  λ=(γ+δ)−√m just below 2.382, bridges just above), so no notch separates them from
  true tracks; the base single-notch point is Pareto-best. Confirms the fix must be
  Hamiltonian-level (the ε-fork term, nb03, for bridges) + track-level (for hubs),
  not a phase update. Figs: `fp_atlas_T200`, `phase_filter_map`, `two_notch_scan`.

## Findings (γ = 3, δ = 1, ε = 2 mrad, clean events)
1. True and false segments carry **equal fork degree**, so β acts as a near-uniform
   **down-scaling** of the whole solution. At a *fixed* τ this looks like segments
   switching off (efficiency and false-rate collapse together); at a **β-aware τ**
   or via **AUC** the separation is essentially unchanged (AUC ≈ 1). The fork
   penalty is **not** a targeted false-positive fix — cross-track bridges are
   coupled clusters that look like real tracks to it.
2. **Classical vs quantum is dramatic (dense fork).** Classically benign (AUC stays
   1). Quantumly **harmful**: even a tiny β collapses the 1BQF discrimination
   (AUC 1.0 → ≈ 0.5, random) and the dense fork graph breaks the sparse-A invariant
   (~20× slower solve).
3. **The ε-windowed fork fixes all of it (nb03, the production version).**
   Restricting the penalty to the acceptance window — co-hit pairs within ε of each
   other (the genuinely competing near-collinear continuations) — makes `B_ε`
   **sparse** ($O(n_{\rm seg})$), **targeted** (false-rate → 0 with ~2% efficiency
   cost and **no** down-scaling), and **1BQF-safe** (quantum AUC ≈ 0.96–0.99, solve
   ~1 s vs the dense fork's 0.55 / 86 s). This is the version to carry forward.

## Running
```bash
PY=/data/bfys/gscriven/conda/envs/Q_env/bin/python
# (do NOT set MPLBACKEND=Agg - it suppresses the inline figures in the executed notebooks)
$PY -m jupyter nbconvert --to notebook --execute --inplace 01_construction_and_spectrum.ipynb
$PY -m jupyter nbconvert --to notebook --execute --inplace 02_metrics_vs_beta.ipynb   # quantum ~5–8 min (cached)
```
`bif.py` sets `QTRK_STORE` and the shared/lib paths itself; only the `Q_env` kernel
is needed.
