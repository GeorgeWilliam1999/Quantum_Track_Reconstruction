# Scaling deep-dive — LHCb VeLo segment toy & solvers

**Goal:** a profiled, evidence-backed optimisation plan for running *larger* experiments —
higher track multiplicity `T` (past the T=1000 wall), deeper QSVT polynomials, and noisier
(large-ε) cells — across three layers: (A) the sparse matrix `A`, (B) the quantum circuit &
simulation, (C) HPC/Condor execution.

Measured on `stbc-i2.nikhef.nl` (28 cores, 250 GB RAM) with `Q_env`
(qiskit 2.4.1 / qiskit-aer, numpy/scipy). Every number tagged **[M]** is measured here or
read from the live store/queue; **[H]** is a hypothesis still to test. Profilers committed at
`Toy_Characterisation/_shared/_profile_layerA.py`, `_profile_layerB.py`, `_probe_aer_mem.py`,
`_probe_notranspile.py`.

Operating point unless noted: γ=3, δ=1 (s=γ+δ=4), clean noise (σ_res=0, σ_scatt=1e-4),
formula acceptance ε = 4.25e-4 rad, step kernel. `n_seg = 4T²`.

---

## 0. TL;DR — the three walls and what each really is

| Layer | What people *assume* the wall is | What it **measured** as | Real scaling law |
|---|---|---|---|
| **A. matrix A** | memory of A at high T | **build (candidate scan) time** — `A` itself is trivially small | `t_build ≈ O(T³)`; A storage & solve are cheap |
| **B. quantum** | the statevector (2ⁿ) | **host-side circuit gate count fed to Aer** (transpile output + Aer ingestion) | gates `≈ O(3T · g(n_sys))`, RAM tracks gates, *not* 2ⁿ |
| **C. Condor** | needs bigger memory tiers | **GPU scarcity + per-process multi-solve accumulation + CPU sim runtime** | CPU sim `t ≈ O(A_nnz·2^{n_sys})`; OOM ∝ gates |

The statevector is **never** the wall: 67 MB at T=400, 256 MB at T=1000, ~1 GB at T=1500 **[M]**.
The held jobs OOM at **24–119 GB** of *host* RAM while their statevector is < 0.3 GB.

---

## A. Sparse matrix `A = sI − C` (build → store → solve)

### A.1 Measured scaling (clean, γ=3)

`SimpleHamiltonianFast.construct_segments(materialize_segments=False)` +
`construct_hamiltonian(convolution=False)`, then the pipeline classical solve
(`spsolve` for n_seg<5000 else `minres`):

| T | n_seg | t_seg (s) | **t_Abuild (s)** | t_solve (s) | nnz | nnz/n_seg | A (MB) | peak RSS (MB) | MINRES it | κ=λ_hi/λ_lo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 160 000 | 0.034 | **0.44** | 0.43 | 161 202 | 1.0075 | 2.6 | 179 | 4 | 2.36 |
| 400 | 640 000 | 0.22 | **2.96** | 0.45 | 642 430 | 1.0038 | 10.3 | 368 | 4 | 2.36 |
| 700 | 1 960 000 | 0.56 | **17.3** | 1.49 | 1 964 546 | 1.0023 | 31.4 | 929 | 6 | 2.72 |
| 1000 | 4 000 000 | 1.45 | **99.5** | 1.89 | 4 006 934 | 1.0017 | 64.1 | 1 064 | 8 | 9.47 |
| 1500 | 9 000 000 | 2.59 | **231.6** | 11.4 | 9 012 024 | 1.0013 | 144.1 | 3 308 | 10 | 8.57 |

**Findings (all [M]):**

1. **`A` stays sparse end-to-end.** `nnz/n_seg` ∈ [1.0013, 1.0075] across all T at the formula
   ε — i.e. `A` is essentially the diagonal plus `2·(3T)` true couplings. The sparse invariant
   holds; nothing densifies. `A` is tiny: **64 MB at T=1000, 144 MB at T=1500.** The 16 GB
   classical Condor tier is over-provisioned by ~50×; peak process RSS is **≤ 3.3 GB even at
   T=1500.** Storage / regeneration is a non-issue (events are stored, A regenerated; A is
   never the memory problem).

2. **The build is the bottleneck and it is super-quadratic, ≈ O(T³).**
   `t_Abuild` grows 0.44 → 231.6 s over T 200→1500 (7.5× in T, **526× in time → exponent ≈ 3.1**).
   At T=1000 the build (99.5 s) is **53× the solve** (1.9 s); at T=1500, 232 s vs 11 s.
   Root cause: `construct_hamiltonian` scans candidate segment pairs per middle hit via a dense
   block `cos_mat = vᵢ @ vⱼ.T` of shape `(in_deg × out_deg)` *before* the ε threshold prunes it.
   With ~T hits/module and 3 middle modules, the candidate flops are `Σ_h in_deg·out_deg ≈ 3·T·T² = 3T³`.
   The *output* is O(T²) sparse, but the *work* is O(T³). (The empirical exponent steepens above
   T≈700 from cache/temporary-allocation pressure on the (T×T) intermediates.)

3. **The classical solve is cheap and robust.** MINRES converges in **4–10 iterations** to
   rel-resid ≤ 1e-6 at every T (residual 2e-16 → 3.6e-6). This is the correct method: `A` is
   symmetric and (at γ=1) **indefinite**, so CG is wrong (it stalls — see `solve.py` docstring).
   No preconditioner is needed at these iteration counts.

4. **Conditioning is benign at γ=3 but degrades with density.** κ = λ_hi/λ_lo rises 2.36 → 9.47
   (λ_lo falls from 2.382 → 0.76) between T=200 and T=1000 as star/hub clusters K(1,m) with
   √m → s=4 push the smallest eigenvalue toward 0. Still well within MINRES's comfort zone. At
   γ=1 (s=2) the operator is genuinely indefinite (λ_lo < 0); MINRES handles it, CG does not.

### A.2 Can A be partitioned to grow sub-quadratically? (the "D3 constant-width" idea)

`C` is block-diagonal over the connected components of the coupling graph, and the memory
([[project-qsvt-filter]] D3) records the **max component saturates** (≤ 18 segments) while the
component *count* is O(T). So a per-component solve is exact (machine precision) and embarrassingly
parallel, but the **total** classical work is still Σ(component costs) = O(#components) = O(T) for
the solve — the solve was never the problem. **It does not help the O(T³) build**, which is the
all-pairs angular scan, not the linear algebra. The build is what must be attacked (optimisation **P3**).

---

## B. Quantum circuit & simulation

### B.1 The statevector is not the wall — gate count is

1BQF qubit budget `n_qubits = ⌈log₂4T²⌉ + 2` (system + 1 time + 1 ancilla). Statevector =
`2^{n_qubits}·16 B`:

| T | n_sys | n_qubits | **statevector** | interaction_pairs (=3T) | logical gates | transpiled gates (opt0 / opt1) |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 14 | 16 | 1.0 MB | 150 | 7 727 | 162 827 / 130 205 |
| 100 | 16 | 18 | 4.2 MB | 301 | 16 747 | 376 141 / 297 875 |
| 200 | 18 | 20 | 16.8 MB | 604 | 36 159 | 853 975 / — |
| 400 | 20 | 22 | 67 MB | 1 226 | 78 501 | 1 934 665 / 1 531 311 |
| 700 | 21 | 23 | 134 MB | ~2 100 | ~150 k | ~3.5 M (est) |
| 1000 | 22 | 24 | 256 MB | ~3 000 | ~210 k | ~5 M (est) |

**[M]** Each interaction pair becomes one multi-controlled-RX with `n_sys` controls
(`OneBQF._apply_direct_controlled_u`, Givens path). Transpiling that MCRX to basis gates
explodes it by **~540 gates/pair at T=50 (n_sys=14) up to ~1 580 gates/pair at T=400 (n_sys=20)** —
the per-MCRX cost grows with the control count. The transpiled circuit is **1.9 M gates at clean
T=400** and ~5 M at clean T=1000. *Noisy* cells multiply `interaction_pairs` by the false-coupling
load `κ·ε²` (≈10–20× at σ_res=0.05), so the transpiled circuit reaches tens of millions of gates.

### B.2 Where the host RAM actually goes (the OOM)

Staged peak RSS, full 1BQF statevector path (`helpers.solve_quantum_statevector`, opt1):

| stage | T=100 | T=400 |
|---|---:|---:|
| after A-build | 343 MB | 513 MB |
| after `build_circuit` (logical) | 343 MB | 513 MB |
| after `transpile()` (retained) | 415 MB | 971 MB |
| **after `sim.run()` (PEAK)** | **917 MB** | (held: ~13 GB/solve) |

**[M]** The `transpile()` call itself is modest — **~1 GB peak at clean T=400**, and opt1 is only
~460 MB more than the logical circuit. The blow-up is **`AerSimulator.run()` ingesting the
multi-million-gate transpiled circuit**: at T=100 it roughly doubles RSS (418 → 917 MB), and at
high T it dominates. `AER_MAX_MEM_MB` caps Aer's *statevector* buffers only — it does **not** bound
the circuit ingestion, which is why the held jobs OOM far above their statevector size.

> **Refinement of the standing lead.** The OOM is host-side *circuit*, not the statevector — that
> part of the lead is confirmed **[M]**. But the precise locus is **Aer's `assemble_circuit`
> ingesting the transpiled gate list**, not the `transpile()` pass (which is ~1 GB at T=400). Aer
> builds one C++/Python op per transpiled gate (~7 KB each): 1.9 M gates × 7 KB ≈ 13 GB — exactly the
> measured clean-T=400 single-solve cost. Scaled up by the higher per-MCRX gate count at larger
> `n_sys`, this reproduces the held 71 GB (T=700) and 95–119 GB (T=1000). Consequence:
> **`optimization_level=0` does NOT fix the OOM** — opt0 produces *more* gates (1.93 M vs 1.53 M at
> T=400; **[M]** T=100 opt0 peak 1031 MB vs opt1 911 MB). opt0's only win is transpile *time*
> (T=400: 2.2 s vs 34 s, **16×**).

### B.3 What is and isn't behaviour-preserving (measured)

- **[M] opt0 ≡ opt1, bit-identical.** Success probability 0.0035 = 0.0035 (T=50); 0.001749 = 0.001749
  (T=100). opt0→opt1 is safe; it buys transpile *time*, not memory.
- **[M] You cannot skip the host transpile.** Feeding the un-transpiled circuit straight to
  `sim.run()` **fails**: `AerError: 'unknown instruction: c16rx'` — Aer's statevector backend has no
  native 16-control RX, it *requires* the host-side MCRX→basis decomposition that causes the blow-up.
  So "skip transpile" is a dead end; the gate count must be cut another way.
- **[M] The idealised exact filter is close but not identical.** The clean 1BQF post-selected state is
  *intended* to be `(I + e^{−iAt})/2·|b⟩` (t=π/s); computed matrix-free (`expm_multiply`, **26 ms** at
  T=50) it agrees with the Aer circuit only to **cos = 0.9836, max|Δ| = 1.7e-2** — because the
  OneBQF circuit applies a **sequential product of 2-level Givens rotations** (Trotter-structured,
  non-commuting within a cluster), not the exact exponential. So a matrix-free 1BQF that must
  *reproduce stored vectors* has to replay the actual gate sequence, not the exact filter (see P2).

### B.4 The simulation-time wall (measured from the live store)

Real campaign solve times (`t_solve` harvested from `solutions/*.npz` meta, medians over reps/cells):

| T | n_qubits | **CPU t_med** | CPU t_max | **GPU t_med** | GPU t_max |
|---:|---:|---:|---:|---:|---:|
| 50 | 16 | 200 s | 1 074 s | — | — |
| 100 | 18 | **2 282 s** | 14 981 s | — | — |
| 200 | 20 | — | — | 85 s | 1 075 s |
| 400 | 22 | **14 208 s (4.0 h)** | 151 795 s (42 h) | 365 s | 3 821 s |
| 700 | 23 | **41 976 s (11.7 h)** | 140 325 s (39 h) | 526 s | — |
| 1000 | 24 | — | — | **6 282 s (1.7 h)** | 27 142 s (7.5 h) |

**[M]** Sim time scales `≈ O(A_nnz · 2^{n_sys})` (both ∝ T²) ⇒ ≈ T⁴ on CPU, softened on GPU.
**GPU is ~39× faster than CPU at T=400** (365 s vs 14 208 s). The 10× CPU spread within a T is the
ε/noise axis (more gates). The CPU lane is where the held T=400/700 tail rots: median 4–12 h, max
39–42 h, brushing the 3-day `MaxRuntime`.

### B.5 Levers to cut the host memory — what survives the measurements

Three candidate fixes were tested against the OOM mechanism (Aer assembling ~millions of gate ops):

| lever | bit-identical? | memory effect | verdict |
|---|---|---|---|
| `optimization_level=0` | **yes [M]** | *worse* (more gates) | only cuts transpile time, not the OOM |
| skip transpile, raw circuit → Aer | n/a | — | **fails [M]** (`unknown instruction c16rx`) |
| ancilla v-chain MCX (fewer gates) | yes (structural) | **catastrophic** | each ancilla doubles the 2ⁿ statevector → rejected |
| **matrix-free gate-replay (no qiskit/Aer)** | yes (replays same ops) | **statevector-only** (≤256 MB @ T=1000) | **the fix** — P2 |

The only memory the 1BQF *fundamentally* needs is the 2ⁿ statevector (256 MB at T=1000). Aer's gate
explosion is an implementation artefact of routing through a transpiled qiskit circuit. Applying the
OneBQF's 2-level Givens rotations directly to a numpy/cupy statevector array (index-arithmetic, O(1)
extra memory per op) keeps memory at the statevector and is bit-identical by construction — see P2.

### B.6 QSVT depth / qubits (matrix-free today, circuit later)

**[M]** Production = degree-40 γ-aware line-comb, solved **matrix-free** (`QSVT.solve_statevector`,
d sparse mat-vecs). Resources:

| T | n_sys | LCU q | total q | ‖c‖₁ | P_success | matrix-free t_solve |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 18 | 6 | 25 | 0.51 | 2.2e-3 | 0.22 s |
| 400 | 20 | 6 | 27 | 0.51 | 1.1e-3 | 0.50 s |
| 700 | 21 | 6 | 28 | 0.51 | 6.4e-4 | 1.48 s |
| 1000 | 22 | 6 | 29 | 0.51 | 4.5e-4 | 1.49 s |

The comb adds only **+7 qubits** (LCU 6 + BE 1) on top of `n_sys`, flat in T; `P_success ∝ 1/T`
(2.2e-3 → 4.5e-4 over 200→1000 = 4.9× for 5× T). **Matrix-free QSVT is essentially free** (≤ 1.5 s at
T=1000) — it is not a bottleneck, which is why the store extends QSVT past the 1BQF wall to T=1000.
Resource reduction: **minimax degree-10** gives LCU 4 (total −2 qubits), ~4× fewer walk calls, and
**P_success 0.137** (≈300× higher → far fewer amplitude-amplification rounds) at a small metric cost.

---

## C. HPC / Condor

### C.1 The held high-T tail — exact census

**[M, live queue]** 56 held jobs (cluster 4826704) + **62 undone solves total**, all 1BQF:

| undone | T | device | studies blocked |
|---:|---:|---|---|
| 16 | 400 | CPU | ERF, Epsilon_study_2, Larger_Scatter, Larger_Scatter_Density |
| 16 | 700 | CPU | (same) |
| 30 | 1000 | GPU | (same) |

All held with `over cgroup memory limit`, across **every** tier:

| shard | tier req. | measured RSS at hold |
|---|---:|---:|
| quantum_cpu T=400 clean (m016, 2 solves/job) | 24 GB | **26 GB** |
| quantum_cpu T=400 σ_res=0.05 (m096) | 96 GB | **95–119 GB** |
| quantum_cpu T=700 clean (m048, 1 solve/job) | 48 GB | **71.5 GB** |
| quantum_gpu T=1000 clean (m064, 1 solve/job) | 64 GB | **71.5–119 GB** |

The statevector for these is 67–256 MB. The 26–119 GB is **entirely host-side circuit** (B.2).
Two compounding effects:
- **Per-process multi-solve accumulation.** `run_shard.process_solves` runs all rows of a shard in
  **one** process. Clean T=400 chunks 2 solves/job → the second solve's Aer ingestion stacks on
  un-reclaimed memory from the first → **26 GB for 2× a ~13 GB solve.** Python/Aer don't return the
  freed gate-list memory to the OS (fragmentation).
- **Tier inflation doesn't help.** Bumping m016→m096 (someone qedit'd these up) still OOMs because
  the cost grows with gates, not with the slot — a 96 GB slot used 119 GB.

### C.2 GPU scarcity

**[M]** GPU-capable nodes matching `GPUs_Capability ≥ 7.0` that this campaign actually lands on are
`wn-lot-008/009` (2 GPUs each) — almost always `Claimed`. The high-T quantum_gpu lane (1 GPU/job)
therefore starves, while GPU is exactly where the runtime problem is solved (B.4). CPU slots are
~25–50× more plentiful but 39× slower *and* hit the same host-RAM OOM.

### C.3 Tiering logic (today)

`build_submission._mem_gb(lane,T,ε)`: ε-aware 16/24/48/96 GB, +`max(48)` at T≥700, +`max(64)` at
T≥1000. `_chunk`: quantum_cpu 2 solves/job at T=400, 1 at T=700; quantum_gpu 5 at T≤400, 1 above.
quantum_cpu = 8 CPU, quantum_gpu = 1 CPU + 1 GPU. The ε-awareness is correct in spirit (noisy = more
gates = more RAM) but the tiers are calibrated to an under-estimate of the circuit cost (the skill
text says "transpiled circuit ~1.3 GB at T=400"; the measured single-solve host cost is ~13 GB).

---

## 1. Prioritised optimisations

Ranked by (unblock value × safety / effort). Each: file·function·change, measured/estimated win, risk.

### P0 — Per-solve subprocess isolation in the quantum worker  *(the quickest safe win)*
- **Where:** `_shared/qtrk_pipeline/condor/run_shard.py` · `process_solves`.
- **Change:** run each *quantum* row's `_solve_one(r)` in a fresh `multiprocessing`/`subprocess`
  child (fork, run, exit), instead of looping all rows in one long-lived process. Classical/event
  lanes unchanged.
- **Why it works [M]:** the held clean-T=400 shard used **26 GB for 2 solves** because Aer's
  ingested gate-list memory from solve #1 is not returned to the OS before solve #2. One solve/process
  caps the high-water mark at a single solve (~13 GB at clean T=400) and reclaims it fully on exit; an
  OOM/segfault then kills one solve, not the shard (idempotent workers just re-pick it).
- **Win:** clean T=400 quantum_cpu drops from 26 GB (2 solves) to ~13 GB + overhead per process
  (a 24 GB tier holds it with margin; 16 GB is borderline), and eliminates the *accumulation* class of
  OOM entirely so the tier finally bounds the cost. **Risk: very low** — behaviour-preserving (same
  `_solve_one`, same store writes), only the process boundary changes. ~15 lines.

### P1 — Route the held tail to GPU and bound its host RAM
- **Where:** Condor ops (no code): release the 62 undone 1BQF solves onto the GPU lane.
- **Why [M]:** GPU is **39× faster** (T=400: 365 s vs 14 208 s) and the held CPU T=400/700 tail is
  both slow *and* OOM-prone. GPU host RAM still pays the circuit-ingestion cost (T=1000 GPU held at
  95–119 GB), so combine with P0 (1 solve/proc) and a realistic tier (C.3 → use the *measured*
  single-solve host cost, ~13/40/70 GB for clean T=400/700/1000, ×2 for noisy).
- **Win:** turns 4–12 h CPU solves into 6–9 min GPU solves and stops the runtime-tail starvation.
  **Risk: low**, but GPU slots are scarce (C.2) — throughput-limited, not a code change. Note the
  `sol_key` device gotcha: flipping a *done* CPU solve to GPU changes its key (statevector is
  device-independent) → copy the npz rather than recompute.

### P2 — Matrix-free 1BQF statevector readout (replace the qiskit-circuit + Aer path)
- **Where:** `_shared/helpers.py` · `solve_quantum_statevector` — add a `method='matrixfree'` path
  that emulates the OneBQF circuit by applying its 2-level Givens rotations directly to a numpy
  (CPU) / cupy (GPU) statevector array, *without* building a qiskit circuit, transpiling, or invoking
  Aer. The op list is exactly `OneBQF.phase_estimation` / `uncompute_phase_estimation` (the same CX
  ladders + controlled-RX), applied as in-place index-arithmetic updates on the 2ⁿ vector.
- **Why [M, mechanism proven; two dead-ends ruled out]:** the OOM is Aer's `assemble_circuit` holding
  ~7 KB × (millions of transpiled gates). Two cheaper-looking fixes were *measured to fail*: opt0 makes
  it worse, and feeding the raw circuit to Aer errors (`unknown instruction c16rx`). A gate-replay
  needs only the 2ⁿ statevector (**≤ 256 MB at T=1000, ~1 GB at T=1500 [M]**) plus O(1) per op — a
  **50–500× host-RAM cut**. It is bit-identical *by construction* (same operations, same order), which
  matters because the exact filter `(I+e^{−iAt})/2` is **only 98.4 %-aligned [M]** to the circuit
  (the circuit is a Trotter product of non-commuting Givens rotations) — so you must replay, not
  short-cut. QSVT already ships exactly this pattern (`QSVT.run_circuit(streaming=True)`), so the
  approach is proven in-repo.
- **Win:** removes the OOM at the source for *all* T and ε (noisy T=1000 fits a 16 GB slot); removes
  transpile time entirely; runtime stays `O(pairs·2ⁿ)` (same compute as Aer) but on cupy matches the
  current GPU speed *without* the host OOM. **Risk: medium** — new code; must prove bit-identity to a
  sample of stored Aer vectors (cos ≥ 1−1e-9) at T=10/50/100 before adoption. ~80 lines, self-contained.

### P3 — Kill the O(T³) candidate scan in the A-build
- **Where:** `LHCb_VeLo_Toy_Model/.../hamiltonians/fast.py` · `construct_hamiltonian` inner loop
  (the per-middle-hit `cos_mat = vᵢ @ vⱼ.T` dense block).
- **Change:** only near-collinear (prev→mid)/(mid→next) pairs survive the ε cut, yet every pair is
  dotted. Replace the full `(in_deg × out_deg)` block with an **angular pre-filter**: bin outgoing
  segment directions (per middle hit) on a coarse (θ,φ) grid of cell ~ε and, for each incoming
  segment, test only the ~O(1) bins within ε. Candidate work drops from `3T³` to `≈ 3T² · (cone
  occupancy)`, i.e. near-O(n_seg).
- **Win:** at clean ε the surviving pairs are only 3T, so the achievable build is **O(T²) not O(T³)**
  — projected T=1000 build ~3–5 s (vs 99.5 s **[M]**), T=1500 ~10 s (vs 232 s). **Risk: medium** —
  library change touching the matrix; must prove `max|ΔA| = 0` vs the current build (use the
  pipeline's `assert_sparse` guard + an explicit equality test on stored events) before adoption.
  Library change → blast-radius check across both repos (api-impact).

### P4 — QSVT resource reduction: minimax degree + per-cluster width
- **Where:** `_shared/qtrk_pipeline/solve.py` · `solve_qsvt(filter_design="minimax")` (already wired);
  D3 per-component solving for the circuit path.
- **Why [M]:** minimax d=10 → LCU 4 (vs 6), ~4× fewer walk calls, **P_success 0.137 vs 4.5e-4**
  (matrix-free metrics within the documented small cost). Per-cluster (D3) gives register width
  `⌈log₂(max cluster≈18)⌉+1 ≈ 5–6 qubits` *constant in T* and P=O(1) → kills the √T amplification.
- **Win:** the deep-/wide-circuit envelope (degree-40 line-comb, 29 qubits at T=1000) shrinks to
  ~23–27 qubits and ~10 walk calls — keeps the *circuit* path (when built) feasible at high T.
  **Risk: low** for the matrix-free store (metrics already characterised); medium for the real
  circuit (parity/angles still unbuilt).

### P5 — Right-size the classical tier; raise classical T ceiling for free
- **Where:** `build_submission._mem_gb` (classical lane) and `_chunk`.
- **Why [M]:** classical peak RSS is **≤ 3.3 GB at T=1500**; the 16 GB tier wastes ~5× and the
  per-shard chunking (`25/15` at T≥700) is set for an over-estimate. With P3 (fast build), classical
  T=1500–2000 is minutes/solve in < 8 GB.
- **Win:** more classical throughput per slot; classical reference extends past T=1000 cheaply.
  **Risk: trivial.**

---

## 2. Scale-up roadmap (feasible envelope after each step)

| step | classical T | 1BQF (statevector) feasible T | noisy (σ_res=0.05) | QSVT (matrix-free) | unblocks |
|---|---|---|---|---|---|
| **today** | 1000 (build 100 s) | ≤700 CPU / 1000 GPU, but OOM-flaky | ≤400 (96 GB tier) | 1000 | — |
| **+P0** (subproc isolation) | — | clean ≤700 CPU stable in tier; T=1000 GPU stable | ≤400 stable | — | the 62-solve held tail (clean cells) |
| **+P1** (GPU route) | — | T=400/700 in 6–9 min not 4–12 h | T=400 noisy in minutes | — | runtime tail; ERF/Eps2/LS high-T 1BQF |
| **+P2** (matrix-free 1BQF, *after bit-identity check*) | — | **T=1000+ 1BQF in a 16 GB slot, any ε** | **≤1000+** | — | every blocked 1BQF cell incl. σ_res=0.05 |
| **+P3** (O(T²) build) | **1500–2000 in seconds** | (build no longer gates the solve) | — | — | larger-T classical & QSVT studies |
| **+P4** (minimax/D3) | — | — | — | deeper combs / per-cluster, const width | QSVT degree studies at high T |

**Mapped to Notion data indices (blocked-by-scale):**
- *Epsilon_study_2 / ERF / Larger_Scatter(_Density)* 1BQF at T=400/700/1000 → blocked purely by the
  1BQF host-RAM OOM + CPU runtime. **P0+P1 unblock the clean cells immediately; P2 unblocks the
  noisy (large-ε) cells**, which are the ones the ε-aware tiers can't size for.
- *Verify_new_results 1BQF beyond T=700* — the statevector "didn't scale" per the store; it's not the
  statevector (256 MB) — **P0+P2 make T=1000+ 1BQF routine**, closing the 1BQF column that QSVT had to
  cover alone.
- *QSVT degree / deep-polynomial* studies — matrix-free already reaches T=1000 cheaply (B.6); **P4**
  is what keeps the *circuit* realisation tractable when it's built.

---

## 3. The quickest safe win (to unblock the current held tail, today)

**P0 + a correct re-tier, applied as Condor ops on the 62 undone 1BQF solves — no library edit.**

1. **Stop tier-chasing.** The held jobs were bumped 16→96 GB and still OOM because cost ∝ gates, not
   slot. Instead, **isolate per solve** (P0) so clean T=400 fits 16 GB and clean T=700 fits ~40 GB.
2. **Re-route the CPU T=400/700 tail to GPU** (P1) at one solve/job (already the chunk at T=700;
   set T=400 chunk→1 for the resubmit) — 6–9 min/solve instead of 4–12 h, dodging the runtime tail.
3. **Resubmit only the undone** with `build_submission.py --only-missing --suffix _rescue` (never
   re-shard a live lane). Use the **measured** host costs for the tier (~16/48/80 GB clean T=400/700/1000),
   not the under-estimate.

This is safe (idempotent workers, no schema/key change, no live-lane re-shard), needs no code change
to land the clean cells, and clears the 62-solve backlog. The *noisy* σ_res=0.05 cells still need
P2 (matrix-free 1BQF) to be sized sanely — that's the one code change worth validating next.

**Caveat (measured):** P0 alone caps a *clean* T=400 solve at ~13 GB — that fits a 16 GB slot, but a
single noisy σ_res=0.05 T=400 solve was ~95–119 GB even isolated, because its gate count is ~15×
higher. So P0+P1 fully unblock the **clean** ε cells; the **noisy** cells are only truly fixed by P2.

---

## 4. Open tests (verified-vs-hypothesis ledger)

- **[M] confirmed:** A stays sparse (nnz≈n_seg) to T=1500; build ≈O(T³); MINRES 4–10 it; statevector
  ≤256 MB at T=1000; opt0≡opt1 bit-identical (success 0.0035=0.0035); opt0 uses *more* RAM than opt1;
  transpile ~1 GB at T=400 (not the OOM); the OOM is Aer `assemble_circuit` (~7 KB/gate × millions);
  raw circuit → Aer **fails** (`unknown instruction c16rx`); exact filter `(I+e^{−iAt})/2` only 98.4 %
  aligned (Trotter); GPU 39× CPU; held tail = 62 1BQF solves OOM at 24–119 GB host RAM.
- **[H] to test before adopting:**
  - **P1:** GPU host RAM still pays the Aer assemble cost (T=1000 GPU held at 95–119 GB) — confirm P0
    (1 solve/proc) + the measured tier holds a single GPU solve; P2 makes this moot.

---

## 5. Validation results (the [H] items, now measured)

P2 and P3 were prototyped and run against the live code paths (scripts:
`_shared/_validate_matrixfree.py`, `_validate_fastbuild.py`). Both reproduce the existing
results exactly.

### 5.1 P2 — matrix-free 1BQF is bit-identical to the Aer path [M]

The emulator applies the **same** `OneBitHHL.build_circuit()` gate list to a numpy statevector
(multi-controlled-RX by control-mask index arithmetic — never a 2ᵏ matrix), then the **same**
`get_solution_from_statevector`. Compared to the live `helpers.solve_quantum_statevector(...,'CPU')`:

| T | n_qubits | Δsuccess | cos(sol_mf, sol_aer) | max\|Δsol\| | t_matrixfree | t_aer | peak RSS (mf) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 11 | 1.1e-14 | 1.0000000000 | 5.2e-15 | 0.1 s | 0.9 s | 294 MB |
| 20 | 13 | 1.3e-14 | 1.0000000000 | 5.9e-15 | 0.2 s | 2.8 s | 296 MB |
| 50 | 16 | 1.3e-14 | 1.0000000000 | 8.3e-15 | **10 s** | **153 s** | **312 MB** |

**Bit-identical to machine precision** (cos = 1.0000000000, max\|Δ\| ~1e-14 — pure float
reassociation), and **already 15× faster at T=50** (it skips transpile + Aer `assemble`). It is a
memory *and* speed win at small/medium T; at large T it stays statevector-only memory (≤256 MB at
T=1000) and is the only thing that fits noisy T=1000 in a normal slot. (Pure-numpy is single-threaded,
so the speed crossover vs Aer's threaded C++ is at larger T — a cupy backend keeps GPU speed without
the host OOM.) **The exact-filter short-cut `(I+e^{−iAt})/2` is correctly rejected: it is only
98.4 %-aligned (Trotter), so the implementation must replay the gate ops — which it does.**

### 5.2 P3 — accelerated A-build is exact [M]

A cKDTree fixed-radius neighbour query replaces the per-middle-hit full `(in_deg×out_deg)` dot, then
applies the **identical** `cos > cos(ε)` test. Vs `SimpleHamiltonianFast.construct_hamiltonian`:

| T | nnz match | **max\|ΔA\|** | t_current | t_fast | speedup |
|---:|:---:|---:|---:|---:|---:|
| 200 | ✓ exact | **0.0** | 0.19 s | 0.15 s | 1.2× |
| 400 | ✓ exact | **0.0** | 1.46 s | 0.48 s | 3.0× |
| 700 | ✓ exact | **0.0** | 6.85 s | 1.43 s | 4.8× |
| 1000 | ✓ exact | **0.0** | 35.5 s | 4.17 s | **8.5×** |

**`max|ΔA| = 0` and nnz identical at every T** — same matrix, bit-for-bit. Speedup grows with T
(O(T³)→O(T²log T)) and is **baseline-dependent**: this micro-benchmark's reference build
(`_validate_fastbuild.py`, `t_current` = 35.5 s @T1000, a warm/already-grouped reference) gives
**8.5×**, while the **§A.1 full-pipeline original build (99.5 s @T1000)** gives **~25×** vs the
same ~4 s fast build. The **headline/Notion figure is ~25×** (vs the full O(T³) build); 8.5× is the
conservative micro-benchmark floor. Behaviour-preserving for **all** solvers (they all build the same A).

### 5.3 P0 / opt0 — behaviour-preserving by construction

- **P0 (per-solve subprocess isolation):** runs the identical `_solve_one(r)`; the store is
  content-addressed by a deterministic `sol_key` (event regenerated from seed → A → solve, all
  deterministic) and the worker is idempotent. Same `sol_key` ⇒ same `.npz`. Only the process
  boundary changes — no result can differ. (Safety/throughput change, not a numerical one.)
- **opt0:** already bit-identical to opt1 [M] (success 0.0035 = 0.0035). Safe, saves transpile time
  only — *not* a memory fix (it is superseded by P2).

---

## 6. Exact change list — which files, which functions, what changes

Grouped by *blocking* (unblocks the held tail / prevents silent failure) vs *speed-only*.
"Key-safe" = does **not** change any `sol_key`/`ham_key`/store schema, so existing solutions stay
valid and nothing recomputes.

### 6a. LHCb_VeLo_Toy_Model package

| file · function | change | status | risk |
|---|---|---|---|
| `src/lhcb_velo_toy/solvers/hamiltonians/fast.py` · `construct_hamiltonian` (step branch, the per-middle-hit `cos_mat = vᵢ@vⱼ.T` loop) | replace the full `(in_deg×out_deg)` dot with a **cKDTree fixed-radius query** on the segment direction vectors, then the *same* `cos > cos(ε)` test (prototype: `_shared/_validate_fastbuild.py::fast_build`). No signature change; output identical. | **validated exact** `max\|ΔA\|=0`, 8.5× @T=1000 [M] | medium — library matrix code; run `api-impact` + the equality test on stored events first. ERF/`convolution=True` branch can get the same query but keep its `1+erf` value test (separate, optional). |
| `src/lhcb_velo_toy/solvers/quantum/OneBQF.py` | **leave unchanged** (it is the byte-identical verified reference). The matrix-free engine reuses its `build_circuit()` op list from the *pipeline* layer (6b), so the library stays frozen. | — | none |

### 6b. Condor data pipeline (`Toy_Characterisation/_shared`)

| file · function | change | status | risk |
|---|---|---|---|
| `helpers.py` · `solve_quantum_statevector` | add a **matrix-free engine**: build `OneBitHHL(...).build_circuit()` then apply its gates to a numpy (CPU) / cupy (GPU) statevector via control-mask index arithmetic — **no `transpile`, no Aer** (prototype: `_shared/_validate_matrixfree.py::emulate`). Make it the engine behind the *existing* `device='CPU'/'GPU'` (statevector readout) so the result and `sol_key` are unchanged. **This removes the OOM.** | **validated bit-identical** (cos=1.0, max\|Δ\|~1e-14; 312 MB @T=50) [M] | medium — new code; keep the Aer path available for sampling/noise studies; production needs a vectorised/cupy inner loop for high-T *speed* (the simple prototype is fast to T≈50, then slows — memory is the proven win). |
| `qtrk_pipeline/condor/run_shard.py` · `process_solves` | run each **quantum** `_solve_one(r)` in a fresh `multiprocessing` child (fork→solve→exit); classical/events unchanged. Bounds the per-process high-water mark to one solve and reclaims it; an OOM kills one solve, not the shard. | behaviour-preserving by construction [M] | very low — same `_solve_one`, same store write, ~15 lines. **Key-safe.** |
| `qtrk_pipeline/solve.py` · `solve_quantum` | pass the chosen engine through (one kwarg). | trivial | very low. Key-safe. |
| `qtrk_pipeline/condor/build_submission.py` · `_mem_gb`, `_chunk` | **after P2 lands:** quantum memory is statevector-only, so collapse the ε-aware 16/24/48/96 tiers to a flat ~16 GB at all T/ε, and raise `_chunk` (more solves/job) since memory no longer accumulates. **Before P2:** keep ε tiers but set them from the *measured* single-solve host cost (~13/40/70 GB clean T=400/700/1000), not the under-estimate. | planning [M-informed] | low — submission only, no result change. |
| `helpers.py` · `solve_quantum_statevector` (Aer path) `transpile(..., optimization_level=1)` → `0` | only if you keep the Aer path: saves transpile time (34→2 s @T400), bit-identical. **Superseded by the matrix-free engine** — do this only as an interim. | bit-identical [M] | very low. Key-safe. |

### 6c. Not changing (and why)
- **Store schema / keys / `manifest`** — untouched; every change above is key-safe so the 13.7k done
  solves stay valid.
- **`solve_classical` / MINRES** — already correct and cheap (4–10 it, indefinite-safe). No change.
- **QSVT solver** — matrix-free already; only P4 (minimax/per-cluster) if you later build the real
  circuit. No change needed for current studies.

### 6d. Recommended order
1. **`run_shard.py` subprocess isolation** (P0) + re-tier from measured costs — *Condor-ops only, unblocks the clean held tail today.*
2. **`helpers.py` matrix-free engine** (P2) — *the real fix; unblocks noisy cells & T>1000; validate bit-identity on a sample of stored vectors, then flip the engine.*
3. **`fast.py` cKDTree build** (P3) — *exact, 8.5×@T=1000; helps every solver's build; do the `api-impact` pass first.*
4. **`build_submission.py` tier/chunk simplification** (after P2) and drop the interim opt0.

---

## 7. IMPLEMENTATION STATUS — actioned 2026-06-14 (committed in the 2026-06-14 sync)

All validated changes are now **committed**. The **P3 cKDTree A-build landed in
`LHCb_VeLo_Toy_Model` at the canonical `17db26f`** (the "consolidate quantum solvers +
segment_metrics; exact O(T²) cKDTree A-build" commit, pushed) — the earlier note that it was
uncommitted and riding on a parallel session is now resolved. The QTR pipeline changes
(matrix-free engine, flat tiers) are committed in the sync branch (see `SYNC_LOG.md`).

| change | file(s) | status |
|---|---|---|
| **P3 cKDTree A-build** (exact, step path) | `LHCb_VeLo_Toy_Model/.../hamiltonians/fast.py::construct_hamiltonian` @ `17db26f` | **DONE & committed** — `max\|ΔA\|=0` at T=50–1000 (step+ERF, noisy; re-verified 2026-06-14), **~25× @T1000** vs the §A.1-measured ~100 s original O(T³) build (the §5.2 micro-benchmark's 8.5× used an already-grouped 35 s reference; multiplier is baseline-dependent, 8.5–25×); `test_pipeline.py` + selftest pass |
| **P2 matrix-free 1BQF engine** | `_shared/obqf_matrixfree.py` (new) + `_shared/helpers.py::solve_quantum_statevector` (engine dispatch, default `matrixfree`, env `QTRK_OBQF_ENGINE`) | **DONE** — bit-identical to Aer (cos=1.0, ΔP~1e-14, clean+noisy); selftest metrics identical via both engines |
| **Condor tiers/chunks** | `_shared/qtrk_pipeline/condor/build_submission.py::_mem_gb,_chunk` | **DONE** — flat 16 GB (was ε-aware 24/48/96), chunks 50–200 |
| **P0 subprocess isolation** | `run_shard.py` | **SUPERSEDED** — matrix-free removes the OOM it worked around; not implemented (avoids fork overhead) |
| GPU-lane retirement | `build_submission.py::write_submit` | **documented, not applied** — matrix-free beats Aer-GPU on CPU; dropping `request_gpus` retires the GPU bottleneck (left for a focused follow-up) |

**End-to-end proof (real `qp` API, throwaway store):** the two cases that used to OOM now solve in
seconds at <1 GB —

| case (held cost) | build / classical / 1BQF | peak RSS |
|---|---|---|
| clean T=700 (71.5 GB) | 2.0 / 0.4 / 6.1 s | 919 MB |
| noisy σ_res=0.05 T=400 (95–119 GB) | 1.2 / 2.0 / 2.9 s | 919 MB |

**To clear the 62 blocked solves:** `build_submission.py --lanes quantum_cpu --only-missing
--suffix _rescue --submit` (matrix-free is now the default engine). Validation scripts kept in
`_shared/`: `_validate_p3_inplace.py` (A bit-identity), `_validate_matrixfree.py` (engine
bit-identity), `_profile_*`/`_probe_*` (the profiling evidence).

**Not committed.** Recommend: review, then commit the pipeline repo changes; the
`LHCb_VeLo_Toy_Model` `fast.py` edit rides on top of the parallel session's uncommitted changes —
coordinate before committing that repo.
