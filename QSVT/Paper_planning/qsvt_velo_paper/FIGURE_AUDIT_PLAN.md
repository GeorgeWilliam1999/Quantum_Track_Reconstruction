# Figure audit plan — are the paper's figures sensible choices? (2026-09-02, for George)

Purpose: decide, figure by figure, whether the DATA behind each plot should change —
configuration, threshold convention, T grid, repetitions, solver set — as opposed to the
caption/label sweep already recorded in `QSVT/Efficiency_Frontier/FIGURE_DATA_SOURCES.md`
(the provenance ledger, 24 entries, 2026-09-02). The ledger answers "where does this
figure come from"; this audit answers "is that the right thing to have plotted".

## 1. The rubric — seven questions per figure

| # | Criterion | Fails when… |
|---|---|---|
| C1 | **Configuration fidelity** | the figure's acceptance / noise / T / γ is not the one the surrounding text argues about (e.g. a 5 mrad figure in a 2 mrad argument; ERF "moderate" ≠ frontier "moderate") |
| C2 | **Convention consistency** | curves in one figure are compared across threshold conventions without saying so, or a solver-vs-solver claim is read at unmatched working points |
| C3 | **Statistical adequacy** | the claim is comparative ("A beats B", "crossover at T") but rests on 1–3 reps, or single-rep points are drawn like multi-rep ones, or sem is absent/zero by construction |
| C4 | **Claim support** | the plotted quantity does not actually measure the sentence it is cited for (e.g. "measured depth" that is a formula of measured P) |
| C5 | **Provenance** | untracked data, hand-copied figure outside the factory, in-figure recomputation that can drift from the CSV |
| C6 | **Representativeness** | a single event (rep 0) or one operating point (one α, one T) stands in for a population without a check that the others look the same |
| C7 | **Economy** | two figures carry the same data/claim, or a figure adds nothing a neighbour does not |

Decision per figure — one of: **KEEP** · **RELABEL** (caption / in-figure text only) ·
**RE-AGGREGATE** (same data, different aggregation, more reps from the store) ·
**REGENERATE** (different data: configuration, convention, grid, or new solves) ·
**MERGE** · **DROP**.

Cost tiers for a data change: **T0** caption only · **T1** re-plot from existing
data (minutes) · **T2** re-run a driver on stored events (hours, local) · **T3** new
Condor solves (days).

## 2. Master table — every figure in `main.tex` (26 graphics, 24 figure environments)

Priority: **H** = a data-choice question that can change a claim; **M** = a defensible
choice that needs a stated reason or a cheap consistency check; **L** = keep, note only.

| Fig | Label · section | Data / configuration (from the ledger) | Suspected issue (criterion) | Candidate data change · tier | Pri |
|---|---|---|---|---|---|
| 1 | `fig:atlas` §2.3 | analytic canonical clusters, fixed τ = 0.35; hand-copied from notebook 07 | C5 (outside factory) | none; record md5 / move into factory · T0 | L |
| 2 | `fig:motifs` §2.5 | analytic motif spectra | — | none | L |
| 3 | `fig:responses` §3.1 | analytic; comb refit on domain (0, 8) and renormalised to 1, not the production (0.2, 7.8) / 0.95 cap | C5 (not the literal production polynomial) | plot the production polynomial · T1 | M |
| 4 | `fig:threshold` §3.2 | stored T = 400 rep 0 clean, fixed 2 mrad; τ sweep with fixed-τ and wp99 markers; under-constrained manifest filter | C6 (single event), C5 (filter hazard) | check reps 1–2 give the same curves; tighten filter · T1 | M |
| 5 | `fig:ladder` §3.2 | analytic chain ladder | C5 (hand-copied) | none · T0 | L |
| 6 | `fig:clean2x2` §5.2 | store campaign, fixed 2 mrad clean; classical + comb at fixed τ, 1BQF at wp99 (+ faded fixed τ); reps 3–20 | C2 (mixed conventions by design), C3 (comb 5 reps at T ≥ 550, 1BQF 3 reps at T = 300–550 vs classical 20) | (a) keep mixed now that the companions exist, or make it one convention; (b) top up comb/1BQF reps to 10 at T ≥ 300 · T3; (c) no hit-drop figure although §5.3 quotes drop numbers · T1 from store | **H** |
| 6A/6B | `fig:fig6fixed`, `fig:fig6matched` (new) | as fig 6; fitted series reps 0–2 at T ∈ {10, 50, 100, 200, 400, 700, 1000} | C3 (3 reps for the fitted series; no T = 300/550) | extend fitted series to 5 reps and add T = 300, 550 · T2 | M |
| 7 | `fig:resources` §5.4 | `qsvt_width_and_P.pdf`, `qsvt_total_depth.pdf` — **not in the ledger**; hand-copied 2026-07-18 from notebook 05 (`degree_sweep.csv`); claims: 21→29 qubits, P ≈ 0.44/T, d = 16 comb ≈ 50 calls flat to T = 700 | C5 (untraced), C7 (same CSV as fig 23), C3 (T = 700/1000 single rep) | trace; MERGE with fig 23 or make fig 7 the d = 16 story and fig 23 the d = 40 readout · T1 | **H** |
| 8 | `fig:twins` §5.5 | analytic | — | none | L |
| 9 | `fig:tangles` §6 | heavy formula ε, T = 200 rep 0, component structure | C6 | component-size histogram over reps 0–2 · T1 | L |
| 10 | `fig:pileup` §6 | heavy T = 200 rep 0, base A (β = 0), mode weights; isolated segments carry no weight in the histogram | C4 (empty bin at λ = s misreads), C6 | add the isolated-segment weight; state β = 0 · T1 | M |
| 11 | `fig:occproof` §7.2 | heavy formula ε, T = 200 rep 0, **α = 0.3**, τ sweep + wp99 dots; crop strips the provenance footer | C1 (α = 0.3 here, α = 0.05 in §9.4/§10), C6, and consistency with the corrected §7.1 mean-field reading (uniform rescale on events) | regenerate at α = 0.05 (or two-α panel) on reps 0–2; keep footer · T2 | **H** |
| 12 | `fig:erf` §8.3 | formula ε at σ_scatt 3e-4/5e-4 ("Larger_Scatter" pairs: 3.40/6.65 mrad) — **not** the frontier's moderate/heavy (1e-4: 3.18/6.31); T = 200 rep 0; CSV untracked | C1 (naming collision), C6, C5 | regenerate at the frontier noise pairs, reps 0–2, commit the CSV · T2 | **H** |
| 13 | `fig:fit` §9.3 | heavy formula ε T = 200; (a, b) rep 0 in-process refit, (c, d) reps 0–2 CSV; matched eff **0.985** | C2 (0.985 here, 0.99 everywhere in §10), C5 (fit loop duplicated in the figure) | re-read at 0.99 or state the 0.985 reason on the figure; read (a, b) from the CSV · T1 | M |
| 14 | `fig:occfit` §9.4 | heavy T = 200 reps 0–2, α ∈ {0.05, 0.3}, matched 0.985, d = 40…960 | C2 (0.985 vs 0.99) | as fig 13 · T1 | M |
| 15 | `fig:degmod`, `fig:degheavy` §10.1 | degree scan, T = 200, 10 reps, formula ε, eff@far ≤ 1 % and far@eff 0.99 | — (clean omitted at the median; stated) | none | L |
| 16 | `fig:degmarg` §10.1 | difference of medians on a non-uniform degree grid, no uncertainty | C4 (not per-degree), C3 (no band), C7 (the sigmoid figures already show it) | bootstrap band per step or DROP · T1 | M |
| 17 | `fig:linesmod` §10.2 | analytic | — | none | L |
| 18 | `fig:respmod` §10.2 | moderate T = 200 rep 0; d = 40 base/fork, d = 80 occupancy/both (labelled) | C6 | none; note the degree choice per panel in the caption · T0 | L |
| 19 | `fig:budget` §10.3 | **not in the ledger as its own entry**: `outputs/05_loss_budget.csv` (writer `05_loss_budget.py`), fitted family at its best degree per operator, median over reps | C5 (untraced), C4 ("best degree" = max over degrees, then decompose — check it is not a selection effect), C3 (reps?) | trace; record reps and the L1–L3 definitions; consider fixed d = 40 alongside best-degree · T1 | M |
| 20 | `fig:density5` §10.4 | formula ε, 3 regimes, T = 200 (10 reps) … 1000 (3); **classical at fixed τ, every other series matched 99 %**; pooled medians across CSVs with mixed rep counts | C2 (mixed within the figure), C3 (rep provenance per point unrecorded) | re-read the classical curve at matched 99 % (`classical_invA` far_e990 exists in every driver CSV — no new solves); record reps per point · T1 | **H** |
| 21 | `fig:rescmp` §11.5 | width analytic; gates on clean fixed 2 mrad rep 0, T ≤ 400; per-cluster comb at d = 16 vs global d = 40; HHL band estimate | C1 (config unlabelled), C4 (d = 16 vs d = 40 in one panel), C3 (rep 0) | label; same degree for both comb points or say why; reps 0–2 · T1–T2 | M |
| 22 | `fig:readout` §11.7 | `degree_sweep.csv`, fixed 2 mrad clean, d = 40; T = 50–400 3 reps, T = 700/1000 1 rep; 0.44/T hardcoded; "measured walk calls" is a formula of measured P | C3, C4, C7 (overlaps fig 7) | MERGE with fig 7; add reps at T = 700/1000 · T2 | **H** (with fig 7) |
| 23 | `fig:run3` §11.9 | real Run-3 events, **ε = 5 mrad** (a third acceptance), fixed τ = 0.35, 8 events, no in-image configuration | C1 (why 5 mrad?), C2 (fixed τ only where the paper's headline is matched-efficiency) | regenerate at the benchmark 2 mrad and/or a formula ε for real hits, and add a matched-working-point read-out · T2 | **H** |

Housekeeping (not figures in the paper): `figures/` still holds unused files
(`classical_2x2_g3_base.pdf`, `qsvt_2x2_gamma3_drop0.pdf`, `quantum_2x2_g3_base.pdf`,
`fit_comb.png`, `fork_quantum_noisy.png`, `erf_exact_evolution_check.png`) — confirm unused
by `grep includegraphics`, then delete.

## 3. The high-priority cards — what each audit step actually checks

**A-6 Figure 6 (`fig:clean2x2`).** Claim it supports: the §5.2 bullets (all store-exact)
and the abstract's 20.5 % → 0.9 %. Questions: (i) now that Figures A/B show each
convention alone, is a mixed-convention headline still the right first picture, or
should figure 6 become one of the two (drop the faded 1BQF)? (ii) rep imbalance: at
T ≥ 550 the comb has 5 reps and the 1BQF 3; a comparative claim at T = 1000 (0.9 vs
20.5 %) has sem ≈ 0.1 pp on the comb, so adequate — but a top-up to 10 reps at T ∈
{300, 550, 700, 1000} for comb/1BQF would cost ≈ 20 Condor solves. (iii) §5.3 quotes
hit-drop numbers with no figure; the store has the `hit_ineff = 0.01` slice (3–10 reps) —
a one-line factory addition. Check: `seg_store.agg_by_ntrk` rep counts per T per solver;
list what a 10-rep top-up needs.

**A-7/22 Figures 7 and 22 (`fig:resources`, `fig:readout`).** Both draw
`QSVT/Segment_level_studies/outputs/depth_and_qubits/degree_sweep.csv`. Figure 7 is not
in the ledger (hand-copied). Steps: trace the notebook cell that wrote the two PDFs;
confirm the numbers in §5.4 (21→29 qubits, 0.44/T, d = 16 ≈ 6 % / ≈ 50 calls to T = 700)
against the CSV; list which columns each figure uses; decide whether one figure can carry
both (§5.4 the scaling, §11.7 the readout consequence) — or keep two but from the
factory. Single-rep T = 700/1000 points: either add reps 1–2 (T2, statevector at d = 40)
or mark them.

**A-11 Figure 11 (`fig:occproof`).** The paper's occupancy operating point is α = 0.05
(§9.4 fitted, §10 everything); this figure is α = 0.3 on one heavy T = 200 event. With
the §7.1 correction (on events the term is a uniform rescale × the per-hit
non-uniformities), the figure's panel (a) "base vs occupancy amplitudes" should be
re-read: does it show the collapse (all amplitudes scaled by one factor) with the co-hit
bulk separated only in the ranking? Steps: regenerate `dp_occupancy_proof.py` at α = 0.05
and 0.3 on reps 0–2; check the §7.2 sentence "efficiency-first false rate falls from 0.650
to 0.210" holds at α = 0.05 (the §9.4 numbers say the matched-0.985 far at α = 0.05 is
what the paper actually uses later). Keep the provenance footer in the crop.

**A-12 Figure 12 (`fig:erf`).** "moderate/heavy" in this figure are σ_scatt = 3e-4/5e-4
(3.40/6.65 mrad); everywhere else in the paper they are σ_scatt = 1e-4 (3.18/6.31 mrad).
Steps: rerun `erf_exact_evolution_check.py` at the frontier pairs (needs the stored 1BQF
erf-kernel circuit solutions at those pairs — check the store has them; if not this is
T3), reps 0–2; commit the CSV; if the rerun is not possible, rename the x-ticks to the σ
values and say so in the caption.

**A-20 Figure 20 (`fig:density5`).** The one paper figure that still mixes conventions
between curves. The fix needs no new solves: every driver CSV carries
`classical_invA` rows with `far_e990`/`eff_e990`. Steps: re-plot with the classical
series at matched 99 %; keep a faint fixed-τ classical if the "production baseline"
reading is wanted; record reps per (regime, T) on the figure (10/5/3/3 — and which CSV
each point pools). Check the §10.4 numbers (0.10/0.68 1BQF; 0.9905/0.952 moderate) are
unchanged (they are matched-point numbers already).

**A-23 Figure 23 (`fig:run3`).** ε = 5 mrad is a third acceptance convention with no
stated reason, read at fixed τ only. Steps: find why 5 mrad (notebook history); compute
the formula ε for the Run-3 geometry if σ estimates exist, else use the benchmark 2 mrad;
add the matched-99 % read-out per solver (the notebook's section 3 already computes
working points); state "real data, ε, γ, τ" in the caption whatever the outcome.

## 4. The medium items in one line each

- **fig 3**: draw the literal production comb (domain (0.2, 7.8), spectral bounds (0.5,
  7.5), 0.95 cap) — T1, removes a caption clause.
- **fig 4**: overlay reps 1–2 as faint curves (T1); constrain the manifest filter by
  study and noise (code hygiene).
- **fig 10**: put the isolated-segment weight into the histogram or label the empty bin;
  say β = 0.
- **figs 13/14**: one matched target for the paper — either re-read §9 at 0.99 (T1, the
  CSVs carry `_e990` columns? check) or print "0.985: the mixed-motif floor exceeds the
  1 % budget" on the figures.
- **fig 16**: bootstrap band over reps per degree step, normalise per degree, or drop.
- **fig 19**: ledger entry; reps; "best degree" check.
- **fig 21**: label the configuration; d = 16 vs d = 40; reps 0–2.
- **figs 6A/6B**: fitted series to 5 reps and T = 300/550 (T2; T = 1000 fits are the
  cost).

## 5. Process

1. **Complete the ledger** (figs 7, 19, 6A/6B) — same card format as the 24 entries.
2. **Fill one audit card per figure** (the table above is the seed): the sentence(s) in
   `main.tex` the figure is cited for (line numbers), configuration, per-curve convention,
   reps, C1–C7 verdicts, candidate data change, tier, recommendation.
3. **George decides per figure** (KEEP / RELABEL / RE-AGGREGATE / REGENERATE / MERGE /
   DROP), starting from the six H items.
4. **Execute in one batch**: T1 items through `make_paper_figures.py` and the
   Efficiency-Frontier factory scripts; T2 items as local driver runs; T3 items as one
   Condor submission (params preflight, then `condor-monitor`).
5. **Re-verify numbers**: after regeneration, the number-specification sweep (Notion
   to-do `3ce5d544-b9d9-8130-8adf-f1478f127c63`) re-checks every quoted number against
   the new CSVs; the ledger is updated; provenance stamped on the Notion write-ups.

The card-filling step (2) parallelises cleanly (one agent per figure, read-only, same
schema as the 2026-09-01 `figure-data-audit` workflow) — it is not launched unless George
opts in; done serially it is roughly a day.

## 6. What the audit does NOT do

It does not change any figure by itself, does not touch the physics conclusions, and
does not set Trust levels. Every regenerated figure comes back as Provisional with a new
provenance stamp for George to verify.
