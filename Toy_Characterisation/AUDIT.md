# Toy_Characterisation — Task Audit & Tracker

_Last audited: 2026-06-06_

This file tracks every study/experiment currently living under
`Toy_Characterisation/`, its purpose, its run status, and the next action
required. It is a living document — update the status table when jobs are
submitted, complete, or analysed.

All Python uses `/data/bfys/gscriven/conda/envs/Q_env/bin/python`.
Batch jobs run on the Nikhef HTCondor pool (`taai-007.nikhef.nl`).

---

## 0. Executive status

| Task | Study dir | Stage | Status |
|------|-----------|-------|--------|
| T1 | `Initial/` | Baseline + hit competition characterisation | ✅ Done (reports written) |
| T2 | `Epsilon_study_2/` | Calculated ε + sensitivity grid | ✅ 1102 pkls + **report.pdf (22 pp, +stability §, 6/06)**; held jobs cleared |
| T3 | `Larger_Scatter/` | Larger scattering + hit inefficiency | 🟢 **Submitted 6/06** (1225 jobs, 4799949–52) |
| T4 | `Larger_Scatter_Density/` | Tighter cone × larger scattering | 🟢 **Submitted 6/06** (490 jobs, 4799953–56) |
| T5 | `ERF/` | Smooth ERF angular cost, histogram thresholds | 🟢 **Submitted 6/06** (882 jobs, 4799957–60) + 162 local pkls |
| — | `Verify_new_results/` | Quantum (OneBQF) segment-level verification (seg14e) | 🟡 outputs under `outputs/`; notebook to finish |
| — | `Quantum_Toy_Study/` | Earlier quantum/classical OneBQF benchmarking | ✅ Runs complete, analysis ongoing |
| — | `Segment_Grass/` | "Grass" sub-peak / geometric clipping study | ✅ Done (report) |
| — | `Recovery_Seperation_analysis/` | Near-coincident track degeneracy | ✅ Done (report PDF) |
| — | `EpsilonStudies/` | Legacy ε / param-opt sweep (older codebase) | ⚪ Superseded by Epsilon_study_2 |

**Next action (2026-06-06):** the long pole is **T3 + T4 — submit to Condor now**
(both still 0 pkls after >1 week). Then submit T5 ERF (882 rows). Write the T2 report
in parallel (data complete). Triage the 33 held T=1000 jobs (§6b).

**Worker fix (2026-06-06):** `_shared/run_worker.py` default-metric threshold changed
from relative `tau_default*max(sol)` to **absolute `tau_default` (0.35)** on both `sol_C`
and the rescaled `sol_Q`, matching `segment_level_analysis.ipynb` `SOLVER_THRESHOLD`.
The relative rule collapsed below the 0.25 attractor / 0.375 plateau and gave a spurious
FAR≈0.98. Shared by all four studies → recompute metrics from stored vectors for any
pkl written before this fix; do not trust `*_metrics_default`. (File is untracked in git.)

**Blocker — RESOLVED 2026-05-30 (history):** 286 Condor jobs were HELD (166 walltime,
120 memory). Fixed in-place via `condor_qedit` + `condor_release` (OOM→128/192 GB, all→
*long* + 3-day). See §6. **A new wave of 33 holds appeared 2026-06-06 — see §6b.**

---

## 1. Supervisor task series (segment-level algorithm)

The core programme is a 5-task series studying the segment-level
reconstruction discriminator under increasingly realistic conditions. All
share the `lhcb_velo_toy` v2.0.0 package, the standard
`gen_params.py → submit.sh → _shared/run_worker.py → aggregate.sh → analysis.ipynb`
pipeline, and the calculated ε formula:

$$\varepsilon(\sigma_{\rm res},\sigma_{\rm scatt}) = \sqrt{2(s\,\sigma_{\rm scatt})^2 + 12\,\arctan^2\!\big(\tfrac{s\,\sigma_{\rm res}}{\Delta z}\big) + 2\,\theta_{\min}^2}$$

with $\Delta z = 33$ mm, $s = 3$, $\theta_{\min} = 1.5\times10^{-5}$ rad.

### T1 — `Initial/` (baseline & hit competition) — ✅ DONE
- `baseline.ipynb`, `characterisation.ipynb`, `hit_competition_study.ipynb`.
- Reports compiled: `baseline_report.pdf`, `characterisation_report.pdf`,
  `hit_competition_study_report.pdf`.
- Establishes the default config and the high-density degradation baseline.

### T2 — `Epsilon_study_2/` (calculated ε + sensitivity) — 🟡 MOSTLY DONE
Replace the hand-tuned ε with `compute_epsilon` and probe sensitivity to
$(\sigma_{\rm res}, \sigma_{\rm scatt})$.
- **Validation gate**: rerun the canonical Verify point with calculated ε.
  Results: `results/validation/` — **40 pkls present**.
- **Sensitivity grid**: $\sigma_{\rm res}\in\{0,0.01,0.02,0.05\}$,
  $\sigma_{\rm scatt}\in\{1,3,5\}\times10^{-4}$, $T\in\{10..1000\}$.
  Results: `results/sensitivity/` — **468 pkls present**.
- **seg14e calc-eps** (classical+quantum companion): `results/seg14e_calc_eps/`
  — **541 pkls present**; aggregated CSVs written.
- Aggregated CSVs exist: `eps2_summary.csv`, `eps2_events.csv`, `eps2_angles.csv`.
- `analysis.ipynb` + `segment_metrics_calc_epsilon.ipynb` (re-touched 5/30).
- ⚠️ Outstanding: clusters **4696265, 4696266, 4719401, 4719402, 4720866**
  are HELD (OOM at high T). Top-up params already generated
  (`gen_params_topup.py`, `eps2_topup_mem32/64.csv`).
- **Next action:** resubmit held high-T rows at higher `request_memory`
  (or cap T), then re-aggregate and refresh `analysis.ipynb`.

### T3 — `Larger_Scatter/` (scattering × hit inefficiency) — 🔴 NOT SUBMITTED
- Cartesian grid $5\times5$: $\sigma_{\rm scatt}\in\{1,3,5,7,10\}\times10^{-4}$,
  $p_{\rm drop}\in\{0,0.01,0.02,0.05,0.10\}$, $T\in\{10..1000\}$.
- Params generated (`params/ls_mem16/32/64.csv`, 1224 rows) and empty
  result subdirs created on 5/27, but **no logs and no result pkls** →
  never submitted.
- `analysis.ipynb` re-touched 5/30 (scaffolding only — no data yet).
- **Next action:** run `submit.sh`. ⚠️ heed the T2 OOM lesson — the high-T
  rows will need the 64 GB+ memory tier.

### T4 — `Larger_Scatter_Density/` (tight cone × scattering) — 🔴 NOT SUBMITTED
- Grid: $\phi_{\max}\in\{0.2,0.1,0.05,0.02,0.01\}$,
  $\sigma_{\rm scatt}\in\{1,3\}\times10^{-4}$, $T\in\{10..1000\}$.
- Params generated (`params/lsd_mem*.csv`, 489 rows), empty result subdirs,
  **no logs/results** → never submitted.
- **Next action:** run `submit.sh` (same memory caveat).

### T5 — `ERF/` (smooth angular cost) — 🟡 LOCAL RUNS DONE, CONDOR NOT SUBMITTED
- Only experiment with `convolution=1`. Sweep soft-threshold width
  $\theta_d\in\{10^{-6}..10^{-3}\}$ × three $(\sigma_{\rm scatt},\sigma_{\rm res})$
  rows × $T$. $\theta_d=10^{-6}$ is the step-function regression check.
- Params generated (`params/erf_mem16/32/64/128.csv`, 882 rows incl. the 128 GB tier
  for T=1000 that the old `gen_params.py` mis-tiered). Condor: **no logs/results yet**.
- **Local progress (2026-06-06):** `results/` holds **162 CPU pkls** at T∈{10,20,50}
  (all 18 θ_d × noise cells × 3 reps, status=ok), generated inline from `analysis.ipynb`.
- **New `hamiltonian_comparison.ipynb`** (local, standalone): fixed event (T=50, 5 layers)
  comparing ERF vs step Hamiltonian — eigenspectra, classical+quantum solution vectors,
  segment metrics vs the Verify_new_results baseline, across clean / 1%-noise / σ-scan.
  Corrected finding: with the **absolute 0.35 threshold**, the step function is *perfect*
  at low noise (eff=1.0, FAR=0.0); ERF only wins under heavy resolution smearing
  (σ_res=0.02: +0.15–0.30 abs efficiency, θ_d=1e-3 best) and roughly doubles quantum
  cos_QC (0.84 vs 0.47). See repo memory `project_segment_threshold.md`.
- Analysis plan (Condor): pool true/false segment scores, pick threshold by Youden's
  J and EER, compare to default $\tau=0.35$.
- **Next action:** run `submit.sh` (882 rows; 720 beyond the local T≤50 head start).

---

## 2. Quantum verification — `Verify_new_results/` — 🟡 ACTIVE

Quantum (OneBQF / OneBitHHL) segment-level analysis verifying the paper's
segment-level results, focused on the **seg14e** configuration. Heavy GPU
Condor usage.
- Main notebooks: `Quantum_segment_level_analysis.ipynb` (touched 5/30),
  `segment_level_analysis.ipynb`, `reconstruction_metrics_verification.ipynb`.
- Outputs under `outputs/quantum_segment_analysis/`: seg14e at T=100, T=500,
  T=1000 (statevector + sampling + hires), T=2000 sampling, shots appendix.
- `accelerated_hamiltonian.py` — local accelerated builder
  (see repo memory `accelerated_hamiltonian_notes.md`).
- Compiled report: `OneBQF_parameter_audit.pdf`.
- Generated CSVs (sec7j–sec7o) for false-segment leakage / matched-threshold.
- ⚠️ Clusters **4671050 (T2000), 4716246 (T1000 hires)** HELD (OOM).
- **Next action:** resubmit held seg14e high-T sampling jobs with more memory;
  finish `Quantum_segment_level_analysis.ipynb`.

---

## 3. Quantum benchmarking — `Quantum_Toy_Study/` — ✅ runs done

Earlier, broader OneBQF classical-vs-quantum benchmarking. All Condor runs
appear complete (large populated `logs/` dirs, no held jobs traced here).
- `outputs/`: `seg14_OneBQF` (drop0/drop1), shot-count variants
  (s8192 / s81920), `seg18d_T500`, `seg18e_T500_gamma2` (incl. s1000000).
- Notebooks: `quantum_classical_comparison.ipynb`,
  `segment_level_OneBQF.ipynb`, `quantum_bench_init.ipynb`.
- Note: `Untitled-2.ipynb` is scratch — consider cleaning up.
- **Next action:** consolidate findings into the segment report; likely
  superseded by `Verify_new_results/` for the final numbers.

---

## 4. Supporting / completed studies

### `Segment_Grass/` — ✅ DONE
Root-cause analysis of occupancy "grass" sub-peaks (geometric clipping at the
cone edge). Report `segment_grass_report.pdf` compiled; results + figures
present. `segment_grass_experiment.ipynb` re-touched 5/30 (likely figure
polish).

### `Recovery_Seperation_analysis/` — ✅ DONE
Near-coincident track degeneracy / recovery dips / condition-number study.
Report `recovery_separation_analysis.pdf` compiled, ~30 figures.
Scratch notebooks `Untitled-2/3/5.ipynb` present — candidates for cleanup.

### `EpsilonStudies/` — ⚪ SUPERSEDED
Legacy ε / Hamiltonian param-opt sweep (`experiment_plan.md`, batched
`job_xxxxx/` dirs, `aggregated/`, 90 plot files) on an older codebase.
Superseded by `Epsilon_study_2/`. Keep for provenance; do not extend.

---

## 5. Shared infrastructure — `_shared/` and `condor_pipeline/`

- `_shared/`: `run_worker.py` (worker entry), `helpers.py` (event gen +
  physics), `aggregate.py`, `submit_base.sub` (memory-tier wrapper).
  Every T2–T5 study's `submit.sh` is a thin wrapper over `submit_base.sub`.
- `condor_pipeline/`: older parameterised pipeline + `results_opt/`
  (param-opt outputs). Predecessor of the `_shared` pipeline.
- Memory tiers: each study emits `*_mem16/32/64.csv` splitting rows by
  expected RAM. The OOM holds show the **64 GB tier is still insufficient
  for the very largest T** — see §6.

---

## 6. Condor holds — RESOLVED 2026-05-30

286 jobs were HELD. Two distinct causes (per-proc `HoldReasonCode` +
`MemoryUsage`):
- **166 walltime** — hit `+MaxRuntime=86400` (24 h) under the *medium*
  category; memory was fine.
- **120 memory** — used ~71.5 GB at a 64 GB request (code 34); cluster
  4671050 used 143 GB at a 128 GB request.

| Cluster | Study | Cause | Fix applied |
|---------|-------|-------|-------------|
| 4696265 (51) | Epsilon_study_2 sensitivity | walltime (32 GB ok) | long + 3-day |
| 4719401 (12) | Epsilon_study_2 sensitivity | walltime (32 GB ok) | long + 3-day |
| 4696266 (78) | Epsilon_study_2 validation | 40 OOM / 38 walltime | OOM→128 GB, long + 3-day |
| 4719402 (130) | Epsilon_study_2 validation | 65 OOM / 65 walltime | OOM→128 GB, long + 3-day |
| 4720866 (3) | Epsilon_study_2 seg14e calc-eps | OOM | →128 GB, long + 3-day |
| 4716246 (10) | Verify_new_results seg14e T1000 hires | OOM | →128 GB, long + 3-day |
| 4671050 (2) | Verify_new_results seg14e T2000 | OOM at 128 GB | →192 GB, long + 3-day |

**Fix mechanism (in-place, preserves exact parameter rows):**
```bash
# OOM 64→128 GB, 128→192 GB
condor_qedit -constraint 'JobStatus==5 && RequestMemory==65536 && MemoryUsage>58000' RequestMemory 131072
condor_qedit -constraint 'JobStatus==5 && RequestMemory==131072 && MemoryUsage>120000' RequestMemory 196608
# longer walltime for everything (medium 24h -> long 3 days)
condor_qedit -constraint 'JobStatus==5' JobCategory '"long"'
condor_qedit -constraint 'JobStatus==5' MaxRuntime 259200
condor_release -constraint 'JobStatus==5'
```
Result: **0 held**, all 286 running/idle.

**Permanent fix for T3–T5 (Larger_Scatter, Larger_Scatter_Density, ERF):**
✅ **applied** — `_shared/submit_base.sub` now has `+JobCategory = "long"` and
`+MaxRuntime = 259200`, and each study's `gen_params.py` emits 16/32/64/128 GB tiers
(ERF includes the `erf_mem128.csv` T=1000 tier the old logic missed). The remaining
gap is the **24 h GPU walltime site cap** — see §6b.

---

## 6b. Condor holds — 2026-06-06 (RESOLVED)

**Resolution:** the 33 held jobs were cleared. The 31 eps2 jobs (T=200 walltime,
T=700 "GPU unused") were re-run **classical-only** (`run_quantum=0`, clusters
4799974/75) after the decision that T≥500 quantum is intractable (T=200≈6 h,
sampling no cheaper). The 2 dead Verify T2000 jobs (incl. the 256 GB OOM) were
removed. Per-T quantum cutoff (`run_quantum=1` for T≤200 only) was baked into all
T3/T4/T5 `gen_params.py` so the new submissions cannot reproduce these holds. Queue:
**0 held.** Root-cause notes retained below.

**Second wave (T3 T=1000 OOM, RESOLVED):** after submitting T3/T4/T5, the T=1000
classical jobs (128 GB tier, densest `he=0`/`he=0.01` cells) OOM'd at 130.8 GB. Root
cause was **`collect_segment_pair_angles` (O(T³))** in `run_worker.py` — its raw
per-triplet angle arrays exceed 128 GB at T=1000 (~3e9 triplets). The worker's existing
guard only skipped it for `shots>0`, not for high-T classical jobs. Fixed: skip the raw
arrays for `n_trk > 300` (the cheap `fast_segment_metrics` angle-efficiency summary is
kept). T=400 classical now completes in 3.4 s (was OOM). Held T3 jobs released; the
not-yet-run T4/T5 T=1000 jobs are protected by the same guard.

Queue snapshot 2026-06-06 12:47 (pre-fix): **90 jobs — 51 idle, 6 running, 33 HELD**,
Epsilon_study_2 T=200/T=700 quantum. Three distinct causes — **did not blind-release**:

| Cluster(s) | Count | Hold reason | Action needed |
|------------|-------|-------------|---------------|
| 4789478, 4790812 | 24 | `job exceed MaxWallTime of 86400` (24 h) | The `MaxRuntime=259200` override does **not** clear this — it is a **site cap on the GPU category**. Either split T=1000 into shorter chunks, run fewer reps, or accept CPU statevector. |
| 4790865, 4671050 | 8 | `GPU claimed but no GPU usage` | GPU statevector jobs not exercising the GPU (fell back to CPU or no cuStateVec). Verify `device=GPU` path in `helpers.solve_quantum_statevector` actually loads the GPU backend before re-running. |
| 4776162 | 1 | OOM at **262144 MB (256 GB)** — used 261624 MB | A single T-large job overflowing even the 192/256 GB tier. Cap T or skip this row. |

**Decision required:** is T=1000 *quantum* (GPU) essential for the T2 report, or can the
report rest on the classical T=1000 + quantum up to T≤500? T2's headline (validation gate +
sensitivity heatmaps) does not strictly need T=1000 quantum.

---

## 7. Housekeeping notes
- Scratch notebooks to triage/remove: `Quantum_Toy_Study/Untitled-2.ipynb`,
  `Verify_new_results/Untitled-4.ipynb`,
  `Recovery_Seperation_analysis/Untitled-2/3/5.ipynb`.
- Backups present (`*_BACKUP.ipynb`, `results_v1_backup/`,
  `quantum_bench_init_backup.ipynb`) — keep until reports are final.
