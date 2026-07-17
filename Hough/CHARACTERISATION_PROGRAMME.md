# Hough characterisation programme — quantum + classical, compared with 1BQF & QSVT

**Goal (George, 2026-07-03):** a functional, well-characterised **quantum and classical Hough
implementation**, compared like-for-like with the existing 1-bit quantum filter (1BQF) and QSVT work.

This file is the **canonical local mirror** of the Notion programme to-do — work from here; Notion is
the system of record but this file carries everything needed to implement without the MCP server.
Keep the two in sync: when a phase completes, tick it here AND in the Notion parent's Progress list,
and append a dated worklog entry to the matching Notion to-do body.

## Notion to-do registry

| Phase | To-do | Notion page ID |
|---|---|---|
| parent | Hough — programme: build & characterise quantum + classical Hough vs 1BQF/QSVT | `3925d544-b9d9-81fe-8052-f86401dd769a` |
| 0 (Now, **In Progress**) | Make Hough and the segment solvers comparable (segment↔track bridge) | `3925d544-b9d9-81be-80de-cb8b949bd270` |
| 1 (Next) | Classical Hough under realistic noise on shared store events | `3925d544-b9d9-81dc-b129-fcbaf7223f3e` |
| 2a (Next) | Quantum vote-state simulator (P1 amplitude voting) | `3925d544-b9d9-81bd-8950-e86c7a0e636d` |
| 2b (Later) | Dürr–Høyer demonstrator (~12 q, gate count vs 1BQF) | `37c5d544-b9d9-81b7-b3da-c8054fa39ada` |
| 2c (Later) | Resource model + scoreboard vs 1BQF/QSVT/D3 | `37c5d544-b9d9-8161-8c35-cd17e50bb19b` |
| 3a (Later) | 4-D displaced-vertex stress (the quantum-win motivation) | `37c5d544-b9d9-81bc-8a5c-e53127b5d363` |
| 3b (Later) | Run-3 real-geometry test (honest negative acceptable) | `37c5d544-b9d9-817b-8d55-e4cd0dc9c5d3` |
| (legacy) | Resolution & smear law — N_bins axis DONE (deep dive); σ-sweep subsumed by Phase 1; George to tick/narrow | `37c5d544-b9d9-813a-be2c-f8d24d995f51` |

Write-up: **Hough I** `37c5d544-b9d9-81aa-b0ef-d505797f7b91` (§1–10; needs Trust=Provisional +
Provenance stamp; grows a four-solver section from Phase 0).
Project pages: Hough Transform `37c5d544-b9d9-80f4-94a9-ef0a24f48d02`, Quantum LHCb Toy
`3265d544-b9d9-80b0-befc-ef00eb67ab9c`. To-do DB data source `e535955f-0756-4222-bed8-47f25e2b020f`.

## The structural claims being tested

1. **No coupling floor.** The segment solvers' dominant failure (cross-track P4-isomorphic bridges;
   the floor theorem `f(PAP^T)b = P f(A)b`) is absent by construction — a Hough ghost needs ≥3
   accidentally collinear hits through the origin (measured ~0 %). vs classical far →20 %, 1BQF
   wp99 far →83 % at T=1000.
2. **Register constant in T.** `⌈log2 N_bins⌉` qubits (16 @ 256²) set by angular resolution, not
   occupancy — vs `⌈log2 4T²⌉` (16→24) for 1BQF, +7 for QSVT (29 q @ T=1000). Same win as QSVT-D3
   cluster decomposition (5–6 q const), but native.
3. **Honest costs (do not oversell):** (a) the resolution wall — bin ≳ smear; locus voting removes
   the *smear* part (deep dive: locus-2048 eff 0.92 @T=1000 vs point-256 0.79), the density/merge
   part remains; (b) displaced vertices break the 2-D reduction → 4-param Hough (the analogue of the
   QSVT comb's real-geometry breakdown — test on Run-3); (c) classical 2-D peak-find is already
   cheap (~150 ms @T=1000) — **the quantum claim lives in high dimension** (4-D vertex / 5-D
   curvature, N_bins 1e9+): register still constant in T, O(√N_bins) readout where the classical
   accumulator cannot be materialised, while segment solvers pay 4T² width AND the floor.

## The characterisation standard (what "well-characterised" means here)

1. **Shared events** via qtrk_store (`event_key`/`ham_key`); Hough runs as a store *consumer*
   (Model B) on bit-identical events.
2. **Metric trio everywhere:** efficiency + false/ghost rate + **activation spectrum**
   (Hough analogue = accumulator peak-height distribution, matched vs ghost peaks; quantum
   analogue = `acc²` sampling spectrum). ROC/AUC/Youden-J/EER = diagnostics only.
3. **Per-solver, efficiency-first thresholds.** classical τ = `δ/(δ+γ)+0.10` (=0.35 @ γ=3);
   1BQF/QSVT = wp99 (`working_point_threshold`, τ just below the 1 % true-amplitude quantile);
   Hough = efficiency-first peak-acceptance threshold. Never judge one solver at another's cut
   (the "1BQF is 75 % efficient" artefact). Pooled τ* rule is unstable under resampling — wp99 is
   the stable headline.
4. **Model breakdown ≠ solver breakdown.** Acceptance wall / fragment degeneracy / real-data m≤2
   floor hit all solvers identically → report the honest target ("non-fragment trues", "m≥3").
5. **Resource accounting vs the 1BQF anchor:** width formula + measured curve, oracle calls incl.
   amplification, success probability vs T, presented as a scoreboard row (vs 1BQF / QSVT / D3).
6. **Dequantisation honesty (QSVT-WP6 rule):** state what is classically simulable; confine the
   quantum claim to the resource profile.
7. **Low-but-nonzero noise** for size/density studies; **explicit parameter tables** in every
   notebook/figure/write-up; heavy visualisation; every study carries the negative result that
   makes it credible.
8. **Statistics:** classical 20 reps · exact quantum 3–5 · sampling 20 reps × 1e5 shots;
   T grid {10,20,50,100,200,400,700,1000} (+ densify {30,75,150,300,550} where present).

## Phase 0 — the comparison bridge (IN PROGRESS)

Hough is **track-level**; the store metric machinery (`build_metrics.py` → `solver_segment_metrics`)
assumes a **segment activation vector**. Build BOTH projections in `Hough/bridge_lib.py`:

**(a) Segment-projection of Hough → segment metrics.** A found Hough track = claimed hits (≤1/plane,
≥3 planes). For each consecutive claimed plane pair, the joining segment is active → binary
activation vector over the ham's 4T² segments → `solver_segment_metrics(truth, activation, 0.5)`
with `truth = segment_truth_mask(ham)`. Map `(hit_i, hit_j) → segment index` from the ham's segment
endpoint arrays (`materialize_segments=False`). Hough's activation spectrum analogue = peak-height
distributions.

**(b) Track-projection of segment solvers → track metrics.** Per stored solution: threshold at the
per-solver τ (classical 0.35 γ-aware; 1BQF/QSVT wp99 after `rescale_to_signal` with the classical
partner) → active segments → **union-find over shared hits** → connected components = track
candidates → the Hough evaluator (majority ≥70 % hit match, ≥3 hits) → track eff/ghost/clone.
Equivalent to `lhcb_velo_toy/solvers/reconstruction/track_finder.py::get_tracks` but WITHOUT
materialising O(4T²) Segment objects; validate union-find against `get_tracks` on a T≤50 event.
Tangles (multi-track components at high T): first pass no splitting; **measure and report the
tangle rate** — itself a finding (segment solvers need a track-builder step Hough doesn't).

**(c) Four-solver benchmark.** The 160 shared clean Verify events (σ_scatt=1e-4, σ_res=0, drop=0,
ghost=0, φ_max=0.2; T {10..1000} × 20 reps; solver coverage differs — classical ~20 reps,
1BQF/QSVT 3–10 — report coverage explicitly). Outputs → `Hough/outputs/four_solver/`:
segment 2×2, track eff/ghost/clone vs T, coverage.csv, summary.csv.
Palette: classical green / 1BQF red / QSVT purple / **Hough orange**.

## Phase 1 — classical Hough under realistic noise

Axes (mirror the segment studies): σ_scatt {1,3,5,7,10}e-4 · σ_res {0,0.01,0.02,0.05} mm ·
hit_ineff {0,0.01,0.02,0.05,0.10} · φ_max {0.2,0.1,0.05,0.02,0.01} (event_key axis) · the
Noisy_Realistic point (1e-4, 0.01 mm, 1 %) · T grid, 20 reps (Hough ~150 ms @T=1000).
Hough parameter scan (the (d,hw) analogue): N_bins {64²..2048²}, smoothing σ, **point vs locus
voting**, min-planes, peak threshold (fixed default AND efficiency-first wp point).
Known from the deep dive (don't redo): universal merge sigmoid (a=0.632, r0=2.49 bins), r0=2.5w at
every grid, locus voting kills vertex smear, fragmentation ~ event property (corr eff↔|z_pv| −0.8).
ghost_rate injection NOT implemented in `events.py` — leave 0.
Physics story: Hough failure modes (merge/resolution) are DISJOINT from segment-solver failure
modes (coupling floor / fragments) — map each axis onto the corresponding segment study
(drop ↔ Larger_Scatter; φ_max ↔ Larger_Scatter_Density; σ_res ↔ Epsilon_study_2).

## Phase 2 — quantum implementation

**2a — P1 vote-state simulator.** `|ψ⟩ = Σ_bin (acc[bin]/Z)|bin⟩`, q=⌈log2 N_bins⌉ (16 @ 256²).
Matrix-free simulator = normalised accumulator (exact; the analogue of the 1BQF matrix-free
statevector path). Readout: sampling P ∝ acc² (quadratic peak contrast vs classical acc). Measure
P(top-T peaks | n_shots), eff/ghost/clone vs shots (20 reps × 1e5 shots convention). Deflation:
sample → claim hits → re-vote. Apply the Phase-0 projections to shot-recovered tracks.
**2b — Dürr–Høyer demonstrator** (~12 q qiskit, coarse accumulator, one event; vote oracle from hit
adjacency; fullest bin in O(√N_bins); gate count vs 1BQF). The hardware candidate.
**2c — Resource model.** Qubits & oracle calls vs (T, resolution, dimension); scoreboard row vs
1BQF (16–24 q, 5–30 calls), QSVT (20–29 q, ~50–1500 walk calls, P∝1/T), QSVT-D3 (5–6 q const).

## Phase 3 — the quantum-win regime

**3a — 4-D displaced-vertex stress.** Release the PV pin (PV_SIGMA x,y > 0) → full (tx,ty,x0,y0)
Hough; measure classical accumulator blow-up + where the 2-D reduction fails. This measurement
MOTIVATES the quantum claim. **3b — Run-3 real geometry** (WP5 events, variable track length +
real vertices; honest negative acceptable).

---

## Key repo facts (from the 2026-07-03 surveys; enough to implement without re-survey)

### Store & pipeline
- Store `/data/bfys/gscriven/qtrk_store` (`$QTRK_STORE`): ~7.8k events, ~17.5k solution npz,
  `manifest/{events,solutions,metrics}.csv`. Pipeline `Toy_Characterisation/_shared/qtrk_pipeline`
  (`import qtrk_pipeline as qp`; `_shared` must be on sys.path). Conda env **Q_env**.
- Keys (blake2b): `event_key = f(T, rep, σ_scatt, σ_res, φ_max, hit_ineff, ghost_rate)`;
  `ham_key = f(ε+provenance, kernel, erf_sigma, γ, δ, fork_β, fork_ε)`;
  `sol_key = event ⊕ ham ⊕ (solver, device, readout)` — solver **hard-validated** to
  {classical, quantum, qsvt} → Hough stays a store consumer (Model B), no sol_key.
- `qp.ensure_event` load-or-generate (deterministic `seed_for(event_key)`; events stored
  float64-exact gzip JSON → regenerated A is bit-identical). `qp.load_event(qp.store.event_path(k))`.
- `qp.build_hamiltonian(event, ham_params)` regenerates A on demand (`SimpleHamiltonianFast`,
  `materialize_segments=False`, step kernel → `convolution=False`); A = sI−C, s=γ+δ, n_seg=4T²;
  sparse guard nnz < 5·n_seg (step).
- Solvers: `solve_classical` (spsolve/MINRES — MINRES because γ=1 is indefinite),
  `solve_quantum` (1BQF matrix-free statevector), `solve_qsvt` (γ-aware line-comb inverse d=40).
- `qp.load_solution(sol_key)` → `{'sol', **meta}`. `build_metrics.build(STUDY)` = the metrics VIEW.
- **Gotchas:** `build_manifest` is additive (never overwrite solutions.csv; campaign rows must be
  re-registered after a spec rebuild); metrics are a VIEW (always recompute; never relative τ·max);
  positions float64 (float32 corrupts ~4e-4 rad classification); solutions.csv writes not
  concurrent-safe (serialize campaigns); `studies` membership column (comma-sep) selects shared
  events, not the primary `study` column.

### Metric definitions (do not re-implement)
- `lhcb_velo_toy/analysis/segment_metrics.py`: `compute_epsilon(σ_res, σ_scatt, dz=33, scale=3)`;
  `segment_truth_mask(ham)` (both endpoints share a non-negative track_id; uses cached
  `ham._segment_track_ids`); `solver_segment_metrics(truth, scores, threshold)` →
  eff = n_true_active/n_true_all · purity = n_true_active/n_active ·
  **false_rate = n_false_active/n_active (=1−purity)** + counts.
- `qtrk_pipeline/metrics.py`: `ABS_THRESHOLD=0.35`; `threshold_for(γ,δ,margin=0.10)`;
  `WP_TARGET_EFF=0.99`; `metrics_at`; `rescale_to_signal(sol_Q_raw, sol_C, τ)` (**signal-support
  rescale** — classical actives only, NOT full norm); `quantum_metrics` (+`cos_QC` on signal
  support); `working_point_threshold(sol, truth, 0.99)` → wp99 (scale-invariant eff/far);
  `metrics_at_wp`, `quantum_metrics_wp`.
- Validity gate: `max|x| ≤ 50` (λ_min→0 explosions); exclude before pooling spectra.
- Track-level: `lhcb_velo_toy/solvers/reconstruction/track_finder.py::get_tracks(ham, solution,
  event, threshold)` exists but needs materialized segments — use the Phase-0 union-find instead.

### Hough code & results (all commits pushed to origin/main)
- `Hough/hough_prototype.py`: `hit_directions` → `hough_accumulate` (256² over [−0.25,0.25]²,
  Gaussian smooth σ=1 bin) → `extract_tracks` (maximum_filter local maxima, nearest-peak hit
  assignment, ≥3 planes) → `evaluate` (LHCb majority ≥70 %, MIN_HITS≥3 → eff/ghost/clone).
- `Hough/hough_store_run.py`: Model-B store consumer — selected the 160 clean Verify events from
  events.csv, per-event CSV + summary → `outputs/hough_store_{per_event,summary}.csv`.
- `Hough/hough_study_lib.py` + deep-dive scripts 01–05 (16 figs `outputs/deep_dive/`): vote model
  d_k = t(1−z_pv/z_k)+s_k/z_k; radial vertex smear rms=|t||z_pv|·8.79e-3/mm; merge sigmoid;
  parameter-free eff(T) ≤0.96 % error; complete-linkage fragmentation criterion; LOCUS voting
  (d(ζ)=(x,y)/(z−ζ), dedup + height-priority claims).
- Baseline numbers (clean store events): track eff 99 %@T10 → 95 %@200 → 90 %@400 → 77 %@1000;
  ghost <4 %; 0 clones; 150 ms@T1000. Locus-2048: 0.92 @T=1000.

### Comparison anchors (clean, Verify_new_results)
- classical: eff 100 %, far 0.2 %→20 % (T10→1000), 20 reps.
- 1BQF: wp99 ~100 % eff, far 5 %→83 %; fixed-τ 74.5 % eff is the threshold artefact. Coverage:
  10 reps T≤75, 5 @100–200, 3 @400, 1–5 @700/1000.
- QSVT comb (d40 hw0.18): eff 92–100 %, far ~0→1.0 % (T=1000); wp99 far 0/0.31/1.44 @{100,400,1000}.
- QSVT resources: 29 q @T=1000; d≈16 → ~50 flat walk calls to T=700; P_anc ≈ 0.44/T.
- 1BQF programme context (George 2026-07-03): the 1BQF zeros UNCONNECTED false segments; the study
  suite probes that capacity; expected failure = connected falses in large/dense events; activation
  spectrum required for every study (QSVT-VII-style).

### Survey provenance
Distilled from the two 2026-07-03 sub-agent surveys of `QSVT/` (README, AUDIT_2026-06-12,
PAPER_PLAN, RESOURCE_REDUCTION, 20 notebooks, 5 campaigns) and `Toy_Characterisation/`
(DATA_INDEX, DATA_GENERATION_GUIDE, qtrk_pipeline source, study READMEs), cross-checked against
memory `project_qsvt_filter.md`, `project_1bqf_programme.md`, `project_hough_tracking.md` and the
Notion write-up inventory (48 write-ups across Quantum LHCb Toy + QSVT).
