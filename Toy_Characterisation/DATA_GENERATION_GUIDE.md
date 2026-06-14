# Data-generation & management guide — segment-level Toy (qtrk_store)

**Status:** operational companion to the Notion source-of-truth. Last updated 2026-06-14
(matrix-free 1BQF engine + cKDTree A-build landed; see `SCALING_DEEP_DIVE.md`).

> ## ⛳ Single source of truth (do not duplicate — link)
> The **Notion pages are authoritative** for *what data must exist, at what coverage, and what
> every metric means*. This guide says *how* to produce/manage it and never overrides them.
> - **Data Coverage — Unified Store (qtrk_store)** — `37d5d544-b9d9-8183-8c50-e76544e53093`
>   → the canonical per-study parameter envelope, solver row counts, completion %.
> - **Data & Metrics — The Source of Truth** — `3795d544-b9d9-81b5-adda-f20676fc541d`
>   → the canonical definitions of A, b, ε, τ, efficiency/false-rate/purity.
> - **Project hub "Quantum LHCb Toy"** — `3265d544b9d980b0befcef00eb67ab9c`.
> In-repo companions: `DATA_INDEX.md` (which dir uses which data) · `SCALING_DEEP_DIVE.md`
> (performance/scaling) · this guide · the three skills in `.claude/skills/`.
> **Rule:** if a number here disagrees with Notion, **Notion wins** — fix this guide.

---

## 0. The 60-second model

Events are generated once and shared; the matrix `A = sI − C` (s = γ+δ) is **regenerated on
demand, never stored**; each solver writes a `.npz` solution under a **content hash**; metrics
are a **recomputed view**, never trusted from disk. Three independently-keyed layers
(`event_key → ham_key → sol_key`) mean nothing is ever recomputed unnecessarily and any point
is reproducible from its parameters.

```
params ──key──> events/<event_key>.json.gz ──(regen A)──> solver ──> solutions/<sol_key>.npz
                                                                          │
                                          build_metrics.py (recompute) ──> manifest/metrics.csv  (the VIEW)
```

Store root: `$QTRK_STORE` (default `/data/bfys/gscriven/qtrk_store`). Library:
`Toy_Characterisation/_shared/qtrk_pipeline` (`import qtrk_pipeline as qp`). Conda env: `Q_env`.

---

## 1. Schematics — where the geometry & operator live

| Thing | Where it is defined | Notes |
|---|---|---|
| **Detector geometry** | `_shared/helpers.py::make_geometry()` | **5 planes**, Δz = 33 mm, z = 33,66,…,165; half-size 40 mm; PV σ = (0,0,1). *Single source* — every study imports it, so geometry stays in lockstep. |
| **Event generation** | `_shared/helpers.py::safe_generate()` → `lhcb_velo_toy.generation.StateEventGenerator` | retries until every track ≥ 3 hits; `measurement_error`=σ_res, `collision_noise`=σ_scatt; φ_max/θ_max cone. |
| **Hit inefficiency** | `_shared/helpers.py::apply_hit_inefficiency()` | random p_drop dropout, prunes track & module hit-id lists. |
| **Segments + matrix A** | `LHCb_VeLo_Toy_Model/.../hamiltonians/fast.py::SimpleHamiltonianFast` | `construct_segments(materialize_segments=False)` then `construct_hamiltonian(convolution=False|True)`. **A = sI − C**, symmetric, constant diagonal s = γ+δ, off-diagonal −1 (step) where two segments share a middle hit and the kink angle < ε. `n_seg = 4T²`. |
| **Pipeline A wrapper + guard** | `qtrk_pipeline/hamiltonian.py::build_hamiltonian()` | string `kernel` API + the **sparse invariant guard** (`nnz < 5·n_seg` for step). |
| **1BQF circuit schematic** | `LHCb_VeLo_Toy_Model/.../quantum/OneBQF.py` | 1-bit HHL: H|b⟩ → 1-bit QPE (controlled-U = Givens product) → ancilla notch → uncompute. `n_qubits = ⌈log₂4T²⌉ + 2`. |
| **QSVT circuit schematic** | `LHCb_VeLo_Toy_Model/.../quantum/QSVT.py` | LCU-of-Chebyshev over the qubitization walk; the 1BQF cosine is its degree-1 member. |

**Reference A figures / spectra** (geometry-pinned eigenvalues, P4 band, hubs, bridges) live in
`QSVT/Initial/outputs/` and `Epsilon_study_2/figures/epsilon_sensitivity/`, and are written up
in the Notion QSVT pages and the Epsilon_study_2 §7 page (`3795d544-b9d9-8141-a647-d1ec5e474d7e`).

---

## 2. Metrics — where they are defined (do not re-implement)

All metrics come from **`LHCb_VeLo_Toy_Model/src/lhcb_velo_toy/analysis/segment_metrics.py`**,
re-exported through `_shared/helpers.py`. Canonical definitions live on the **Data & Metrics —
Source of Truth** Notion page; this is the operational pointer.

- `compute_epsilon(sigma_res, sigma_scatt, dz=33, scale=3, theta_min=1.5e-5)` — the closed-form
  acceptance ε = √(2(scale·σ_scatt)² + 12 arctan²(scale·σ_res/Δz) + 2 θ_min²). **This is the
  Hamiltonian ε** (it changes A). Clean reference ≈ 4.25e-4 rad.
- `segment_truth_mask(ham)` — per-segment truth labels (a segment is true iff both endpoints
  share a non-negative track id). Truth is **recomputed from the event**, never stored.
- `solver_segment_metrics(truth, scores, threshold)` — eff / false-rate / purity after a cut.
- **Threshold convention (critical):** the store **metrics view is recomputed at ABSOLUTE
  τ = 0.35** (γ-aware; `δ/(δ+γ)+0.10` at γ=3) — never a relative `τ·max`. **Reporting** uses
  **per-solver, efficiency-first working points** (place τ to hold ≥99 % eff, quote the false rate
  paid); the classical τ=0.35 is *not* the right cut for the filtered (1BQF/QSVT) solvers. Keep
  both views. See the `feedback-solver-thresholds` memory and the QSVT working-points figures.
- The quantum/QSVT solution vector is **rescaled to the classical signal support**
  (`qp.rescale_to_signal` / `rescale_quantum`) before thresholding.

**Golden rule:** metrics are a VIEW. Always recompute (`qp.metrics_at` / `build_metrics.py`);
never trust a metric baked into an old pickle (that bug class is closed structurally).

---

## 3. The store — layout, keys, idempotency

```
$QTRK_STORE/
  events/<event_key>.json.gz      float64-exact Event.to_dict (regen A is bit-identical)
  solutions/<sol_key>.npz         {'sol': float32 vector, 'meta': {...}}  + truth NOT stored
  manifest/events.csv             one row per event (the plan)
  manifest/solutions.csv          one row per solve (the plan)
  manifest/metrics.csv            the RECOMPUTED view (build_metrics.py)
  manifest/shards/<lane>/*.csv    Condor job shards (regenerated by build_submission)
  logs/                           Condor .out/.err/.log
```

Keys (`qtrk_pipeline/keys.py`, pure functions, blake2b of a canonical string, floats to 12 sig-fig):
- `event_key = (T, rep, σ_scatt, σ_res, φ_max, hit_ineff, ghost_rate)`
- `ham_key   = (ε + provenance['formula'|'set'], kernel['step'|'erf'], erf_sigma, γ, δ, fork_β, fork_ε)`
- `sol_key   = event_key ⊕ ham_key ⊕ (solver, device, readout)`

**Idempotency:** workers skip any `event_path`/`sol_key` already on disk → re-running is always
safe. **Key-safety rule:** never change what goes INTO a key for data that already exists, or
done solves look undone and recompute. (The matrix-free engine is *key-safe*: `device` stays in
the key, `engine` does not — a matrix-free CPU solve has the same `sol_key` as the old Aer one.)

---

## 4. Solvers (what runs, and the 2026-06-14 changes)

| solver | code | engine / cost | notes |
|---|---|---|---|
| **classical** | `qtrk_pipeline/solve.py::solve_classical` | `spsolve` (n_seg<5000) else **MINRES** | A is symmetric & (γ=1) **indefinite** → MINRES, never CG. 4–10 iters, seconds to T=1500. Supplies the signal-support reference. |
| **1BQF** (quantum) | `solve.py::solve_quantum` → `helpers.solve_quantum_statevector` | **DEFAULT = matrix-free** (NEW) | statevector-only memory (≤256 MB @T1000), seconds/solve, **bit-identical** to the old Aer path (validated cos=1.0, ΔP~1e-14). Removes the host-RAM OOM. `engine='aer'` (env `QTRK_OBQF_ENGINE=aer`) keeps the legacy AerSimulator path for cross-checks / sampling. |
| **qsvt** | `solve.py::solve_qsvt` | matrix-free Chebyshev recursion | γ-aware **line-comb** inverse, degree 40, default. `minimax`/`band` variants. ~1.5 s @T1000. |

**A-build (NEW):** `fast.py::construct_hamiltonian` step path now uses a **cKDTree fixed-radius
query** instead of the O(T³) full block dot — **bit-identical A** (validated `max|ΔA|=0` at
T=50–1000, step *and* ERF) and ~25× faster at T=1000 (vs the §A.1-measured ~100 s original O(T³) build; baseline-dependent, 8.5–25×). The ERF/`convolution=True` path is
unchanged. The sparse invariant (`nnz ≈ n_seg`) is unchanged and still guarded.

> These two changes mean **the 62 high-T 1BQF solves that were blocked by OOM are now trivial**
> (clean T=1000: 12 s / 1.5 GB; noisy σ_res=0.05 T=400: 2.85 s / 1.5 GB — both used to OOM at
> 95–119 GB). The Condor **GPU lane is no longer needed** for 1BQF statevector.

---

## 5. Running Condor well

Driver `qtrk_pipeline/condor/build_submission.py` shards `manifest/solutions.csv` into per-lane
jobs; worker `condor/worker.sh → run_shard.py` is idempotent (skips done work). Full operational
detail (with the corrected OOM model) is in the **`qtrk-condor` skill**.

```bash
export PYTHONPATH="…/Toy_Characterisation/_shared:…/LHCb_VeLo_Toy_Model/src:$PYTHONPATH"
export QTRK_STORE=/data/bfys/gscriven/qtrk_store
PY=/data/bfys/gscriven/conda/envs/Q_env/bin/python
# build the manifest from the specs (only if specs changed):
$PY -c "import qtrk_pipeline as qp; qp.build_manifest(qp.standard_specs())"
# submit (idempotent; --only-missing near completion; --suffix to not clobber live shards):
$PY …/condor/build_submission.py --lanes events,classical,quantum_cpu,quantum_gpu --submit
$PY …/condor/build_submission.py --lanes quantum_cpu --only-missing --suffix _rescue --submit
```

**Lanes & resources (post-matrix-free):**

| lane | device/cpus | memory | chunk |
|---|---|---|---|
| events | CPU / 2 | 16 GB | 120–500 by T |
| classical | CPU / 2 | 16 GB (≤3.3 GB used @T1500) | 15–300 by T |
| quantum_cpu | CPU / 8 | **flat 16 GB** (matrix-free is statevector-only) | 50–200 by T |
| quantum_gpu | (GPU req. now redundant) | **flat 16 GB** | 50–200 by T |

- **Memory:** with the matrix-free engine the old ε-aware 24/48/96 GB tiers are gone — `_mem_gb`
  returns a flat 16 GB (≈10× headroom). Big requests schedule slowly; small ones schedule fast.
- **GPU:** matrix-free runs on CPU and **beats Aer-GPU** (T=400: 2.6 s vs 365 s), so the
  `quantum_gpu` lane's `request_gpus=1` is now wasted. **Recommended follow-up** (not yet applied,
  to keep this change minimal): in `write_submit`, drop the GPU request for `quantum_gpu` (set it
  to CPU/8) or merge it into `quantum_cpu` — this retires the GPU-scarcity bottleneck entirely.
  Until then high-T quantum still routes to the GPU lane but only uses its 1 CPU.
- **Held (OOM) recovery:** should be a non-issue now. If you forced `engine=aer` and OOM at high
  T, the fix is **not** a bigger `request_memory` (cost ∝ gates) — switch back to the matrix-free
  engine. See the `qtrk-condor` skill "THE bigger one" section.
- **Never** re-shard a lane another (running) cluster is using — use `--suffix`.
- **Refresh the metrics view** after solves land: `$PY …/build_metrics.py`.

---

## 6. End-to-end, by hand (the qtrk-data-pipeline skill expands this)

```python
import qtrk_pipeline as qp
from helpers import segment_truth_mask, solver_segment_metrics, rescale_to_signal
ev, ekey = qp.ensure_event(n_trk=400, rep=0, sigma_scatt=1e-4, sigma_res=0.0,
                           phi_max=0.2, hit_ineff=0.0)
ham = qp.build_hamiltonian(ev, epsilon=qp.compute_epsilon(0.0, 1e-4), kernel="step",
                           gamma=3.0, delta=1.0)          # regenerates A (bit-identical)
truth = segment_truth_mask(ham)
solc, t = qp.solve_classical(ham)                          # classical reference
qd = qp.solve_quantum(ham, device="CPU")                   # 1BQF, matrix-free by default
# metrics are a VIEW — recompute at tau=0.35 (or your per-solver working point):
m = solver_segment_metrics(truth, solc, 0.35)
```

Loading existing data: `qp.load_event`, `qp.load_solution(sol_key)`, `qp.load_metrics(study=…)`.

---

## 7. Invariants & the "flawless" pre-flight checklist

Before any mass submission (use the `params-preflight` agent + these):

1. **Sparse A:** step `nnz/n_seg < 5` (guarded in `build_hamiltonian`). A wide ε at high T goes
   dense (`ε_dense ≈ θ₀√(8/3T)`) — keep ε at/near the formula value.
2. **A is regenerable bit-identically:** the selftest checks `max|ΔA| = 0`; never persist A.
3. **Metrics recomputed**, never read from disk; threshold = absolute 0.35 for the store view.
4. **Key-safety:** don't change key inputs for existing data; `engine` is not in the key.
5. **Quantum engine:** default matrix-free (statevector-exact). Use `engine=aer` only for
   sampling-readout studies or as a cross-check.
6. **Reps policy:** classical 20; statevector quantum 3 (exact) tapering to 1 at high T; sampling
   opt-in 20 reps. (Encoded in `StudySpec`.)
7. **Run the selftest** after any library/pipeline edit:
   `python qtrk_pipeline/_selftest.py` (must print ALL CHECKS PASSED, incl. `max|dA|=0`).
8. **Validate library edits** against an independent reference (the `_validate_*.py` scripts) and
   run `test_pipeline.py` + the `library-regression` checks before releasing to workers.

---

## 8. Per-project guides

Each manifest study has a short, project-specific guide (envelope, commands, quirks, blocked
work) that links back here and to its Notion coverage row:

- `Epsilon_study_2/DATA_GUIDE.md`
- `ERF/DATA_GUIDE.md`
- `Larger_Scatter/DATA_GUIDE.md`
- `Larger_Scatter_Density/DATA_GUIDE.md`
- `Verify_new_results/DATA_GUIDE.md`

Consumers (no params of their own) read the store via adapters: `Segment_level_studies`
(`seg_store.py`), `FR_SE_tradeoff`, `Presentation_mini_q_workshop_UM`, `Bifurification`,
and the `QSVT/` campaigns. See `DATA_INDEX.md` §3.
