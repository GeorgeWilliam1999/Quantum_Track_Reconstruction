# Efficiency_Frontier — recovering the QSVT segment-efficiency drop at near-zero false rate

**Question (George, 2026-08-24):** the production comb retains a *relative drop in segment
efficiency* against the classical solver (and against unity). Can we recover that efficiency
while keeping the comb's tiny false rate — by moving along the (efficiency, false-rate)
frontier over **polynomial degree** and **Hamiltonian set-out** — and if not, can we *prove
where the residual deficit lives*?

One experiment ≈ this folder ≈ one Notion write-up (at results time). To-do tracks progress.

> **Scope ruling (George, 2026-08-25): this is quantum-computing work.** Classical
> post-processing — specifically the hit-uniqueness post-selection gate, which exists as its
> own prior study (`qtrk_pipeline.postselect`, write-up 2026-07-18) — is **out of scope**
> here. No composed points, no gate arms, no slot-contest loss channel. Every number in this
> experiment is a property of the quantum filter p(A) alone. (Reaffirms the 2026-07-18
> mission-creep retraction.)

## 1. The deficit, as currently measured (baselines to beat)

| Regime | Best known QSVT point | Classical reference | Deficit |
|---|---|---|---|
| clean/moderate T≤400 | comb d=40: eff ~0.97–1.00 @ far ≈ 0 | eff 1.00 @ far 0.023 (T=400) | up to ~3% eff |
| heavy T=200 (quantum-only) | XII fitted response: far 0.061–0.096 at matched eff 0.97–0.98 | far 0.124 at both | ~2–3% eff below target |
| normalized walk (BE study 03) | ±½ comb d=20: **eff 1.000 @ far 0.000** (step-clean T=100/200) | — | none (clean) |

The last row is the standing hint that **full recovery is possible on clean events** — the
normalized-discriminant spectrum separates better than raw A. Its behaviour under noise,
erf, and composition is exactly what this experiment must map.

## 2. Known loss channels (pre-registered decomposition)

Every missed true segment at an operating point must be attributed to one of:

- **L1 — twin/fragment floor** (theorem-bound): hit-drop fragments and P4-bridge twins are
  graph-isomorphic with uniform b ⇒ *no* spectral filter separates them. Irreducible for
  any p(A) on the same A; only a *changed Hamiltonian* (occupancy term, ε shrink) or
  classical-side info can break it. Count directly (Codesign/05_direct_twin_count.py
  machinery).
- **L2 — response ripple**: eigenvalue lands in a comb null / ripple trough. Degree curves
  are **non-monotone** (BE study 05: d=24 spikes to 0.45 while 20/28 are clean — ripple ×
  tangle) ⇒ every operating degree needs a d & d+4 stability check. Fixable by
  degree/response choice.
- **L3 — threshold overlap**: filtered amplitude of a true segment sits below τ at the far
  target because a *false* population overlaps from above. Fixable only by a response that
  widens the amplitude gap.

**The central deliverable is the stacked L1–L3 budget per configuration** — it converts
"the comb loses 2%" into "1.2% is twins, 0.5% is ripple, 0.3% is overlap", and says
which knob (if any) can pay each part back.

## 3. Axes

### A. Response family × degree (the "polynomial degrees" axis)
| Family | Source | Degrees |
|---|---|---|
| production line-comb inverse (hw 0.18) | `design_line_comb_inverse` | d ∈ {8,12,16,20,24,28,32,40,48,64} |
| sharp high-purity comb | QSVT VIII (d=120 variant) | d ∈ {80, 120} |
| band-limited inverse | `design_band_limited_inverse` | same grid |
| **fitted response** (refit per set-out) | Codesign/04_fit_comb_to_measured_spectrum.py | d ∈ {20,28,40} |
| **normalized ±½ comb** (discriminant walk) | Block_Encoding/03 (α=1 route) | d ∈ {12,20,28,32,40} |

Every frontier point carries its d and d+4 twin as a stability margin (report both; a
config whose d/d+4 pair disagrees by >1% eff is flagged unstable, per the BE-05 ripple
finding). Minimax/Remez combs stay OUT of scope (needs the networked machine — separate
deferred to-do).

### B. Hamiltonian set-out (the "different Hamiltonians" axis)
Baseline: `A = (γ+δ)I − C`, γ=3, δ=1, step kernel, formula ε, no fork, no occupancy.
Varied one knob at a time from baseline, then the best combination:

- **γ ∈ {1, 2, 3}** (gap scaling; seg_store carries γ per row already),
- **kernel**: step vs erf (erf = the doubled-coupling step at θ_d→0; ±½ comb is the only
  known erf filter at wp99 — re-test here on the frontier),
- **ε scale**: s ∈ {1.5, 2.12, 3.0} × RMS-kink formula (the ε-driven competing-candidate
  trade-off is the suspected L3 driver — links the open ε to-do 37a5d544…),
- **occupancy in-matrix**: A_occ α ∈ {0, 0.05, 0.10} **with the filter refit at each α**
  (the XIII reversal: unfitted = artifact no-go; fitted α=0.05 broke the fragment floor
  rep1 0.654→0.062) — the only knob that attacks L1 directly,
- **normalization**: raw A vs degree-normalized discriminant D^{-1/2}CD^{-1/2}
  (spectrum {±1, ±½}; hubs/P3/isolated rejected exactly by the ±½ comb),
- **fork (bifurcation) β ∈ {0, 0.5, 1.0}** at clean AND moderate AND heavy, **always with the
  response refit** (George, 2026-08-26). *Scope correction:* the earlier "clean only, XI says
  NO-GO" guardrail was stale — XI measured the fork under the **unfitted** production comb,
  and XII then *redeemed* the fork by refitting (b0.5@γ\* fitted 0.085/0.046, beating
  fitted-β0). Refitting per set-out is exactly this experiment's method, so the fork must be
  judged under it. Also run the **occupancy × fork combination** (the "best combo" of §3B).

### C. Events / noise / size
- Store-backed events, deterministic keys (`qtrk_pipeline.ensure_event`) — the standard
  three regimes per `Bifurification/dp_matrix_characterisation.NOISES` (σ_scatt 1e-4
  everywhere): clean (res 0, drop 0), moderate (σ_res 10 µm, **drop 1%**), heavy
  (σ_res 20 µm, drop 1%), formula ε, γ per config. [Drop=1% confirmed by George
  2026-08-24; this plan's earlier "0.5%" was a transcription slip vs the study code.]
- T = 200 (workhorse grid), T = 400 (winners only), ≥3 reps per cell for spread.

### D. Frontier extraction + composition (the working-point discipline)
- Per config: **full ROC** via `quantum_metrics_wp` (τ sweep on stored amplitudes) — the
  frontier is the object, not a point. Never judge at the classical τ=0.35
  (per-solver-thresholds rule). Report: the (eff, far) curve, wp99/wp995/wp999 cuts,
  far@eff∈{0.99, 0.995, 0.999}, and eff@far∈{0.001, 0.01}.
- Pareto set across all configs per regime; the heavy quantum-only reference to dominate
  is the XII fitted response (far 0.061–0.096 at matched eff 0.97–0.98, T=200).

## 4. Stages

**Stage 0 — inventory + baseline frontier (no compute).** Harvest every existing cell:
QSVT II store campaign (506 solves), Segment_level_studies degree_sweep + activation
spectra, Codesign 04/06 (fitted comb, occupancy refit), Block_Encoding 03 CSV (normalized
comb). Assemble the *existing* (eff, far, d,
set-out) frontier first; freeze the config matrix to only the missing cells. Output:
`outputs/00_inventory.csv` + the gap list.

**Stage 1 — degree × response frontier at baseline set-out** (clean + moderate, T=200).
The cheap core: statevector, matrix-free. Answers "is the deficit ripple (L2/L3) at all?"
and finds the 2–3 response families worth carrying.

**Stage 2 — Hamiltonian set-out axis** on the carried responses (one knob at a time, then
best-combo). Fitted response refit *per set-out* (H3 below). Clean + moderate first;
includes the erf column.

**Stage 3 — heavy-noise frontier + loss decomposition.** The full L1–L3 budget per
config (twin count, ripple test = filtered amplitude vs comb-line placement,
threshold-overlap census). The decisive stage.

**Stage 4 — confirmation + write-up.** T=400 on the Pareto winners, d/d+4 stability
stamp, per-rep spread; figures; Notion write-up (Intro · Aims · Method · Results ·
Conclusion) with provenance; PROJECT_STATUS + paper-feed note (candidate §9 upgrade).

## 5. Pre-registered hypotheses

- **H1**: on clean/moderate the deficit is dominated by L2/L3 (ripple + threshold overlap),
  not the floor ⇒ recoverable to eff ≥ 0.995 at far ≤ 0.01 by degree/response choice alone.
  (The BE-03 normalized row is the existence proof on clean.)
- **H2**: on heavy the deficit splits into a *measurable* L1 irreducible part and an
  L2/L3 recoverable part; the budget quantifies the ceiling of any spectral-only fix.
- **H3**: the fitted response *per Hamiltonian set-out* beats any fixed comb across
  set-outs (activations move ⇒ polynomial refits — the XII mechanism, now used as a knob).
- **H4**: occupancy α=0.05 (fitted) + normalized ±½ comb is the best single combination at
  heavy — it attacks L1 (occupancy) and L2/L3 (cleaner normalized spectrum) at once.

## 6. Success / failure criteria

- **Success (recovery)**: a config with eff ≥ 0.995 @ far ≤ 0.01 on moderate T=200 *and*
  a point strictly Pareto-better than the XII fitted response (far 0.061–0.096 at eff
  0.97–0.98) on heavy T=200, stable at d & d+4, confirmed at T=400.
- **Success (honest negative)**: the L1–L3 budget shows the residual deficit is ≥90%
  L1 ⇒ the drop is *provably* not recoverable by any spectral filter on the same A —
  closing the question with the floor theorem rather than a scan, and pointing the
  recovery at Hamiltonian design (occupancy/ε) with quantified headroom.
- Either outcome feeds the paper (§9 fitted response / §10 limits).

## 7. Guardrails (house rules that bite here)

- Per-solver working points; ROC always; absolute-τ comparisons only as a labelled
  extra column. wp99-degenerate corners use the frontier-best τ (matched-eff convention).
- Degree stability margin d & d+4 on every quoted operating degree.
- Store discipline: events by deterministic key, A regenerated never stored, erf pruning
  (val > 1e-9) so A stays sparse; metrics as a view.
- Matrix-free statevector solves; Condor sweeps via params CSV + params-preflight; heavy
  1BQF-style OOM traps don't apply (no Aer assembly in the QSVT emulation path) but
  per-solve subprocess isolation stays the pattern for big grids.
- **Block encodings are a COSTING column, not a simulation path** (decided 2026-08-24):
  the hit-oracle/Szegedy encodings add +49–69 ancilla qubits, so circuit-level statevector
  simulation is out by factors of 2^49+; the matrix-free Chebyshev recursion computes the
  *identical* amplitudes in seconds (the response physics is encoding-independent). The
  oracle enters the experiment as (i) the α subnormalization → effective-degree bookkeeping
  (deg ∝ α, BE-III degree law) and (ii) the measured CX/call prices (BE-II) attached to
  every frontier point at write-up time — zero extra compute.
- Figures: house palette, every panel labelled with its config; numbers in CSVs under
  `outputs/`, figures under `figures/`.

## 8. Files (planned)

- `PLAN.md` — this document
- `00_inventory.py` — Stage 0 harvest → `outputs/00_inventory.csv` + gap list
- `01_degree_response_frontier.py` — Stage 1 grid → `outputs/01_frontier_clean_moderate.csv`
- `02_hamiltonian_setouts.py` — Stage 2 knobs (+ refit hook into Codesign/04) → `outputs/02_setout_frontier.csv`
- `03_highT_frontier.py` — Stage 1+2 extension to T=1000 (+ stored-1BQF reference) → `outputs/03_highT_frontier.csv`, `outputs/03_1bqf_reference.csv` ✅
- `04_heavy_frontier.py` — Stage 3 heavy quantum-only frontier → `outputs/04_heavy_frontier.csv`
- `05_loss_decomposition.py` — L1–L3 budget per config → `outputs/05_loss_budget.csv`
- `06_confirm_T400.py` — Stage 4 winners → `outputs/06_confirmation.csv`
- `07_writeup_figs.py` — frontier plots, stacked loss budgets, Pareto overlays
