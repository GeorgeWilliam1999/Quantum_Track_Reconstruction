# Integration plan — the modified-Hamiltonian results into the paper
**2026-09-01.** Source study: `QSVT/Efficiency_Frontier/` (commits `3a858c09`..`ab95dca8`).
Contradiction audit: `Efficiency_Frontier/outputs/paper_contradiction_map.json`
(29-agent sweep + adversarial verify, adjudicated below). George's directives:
hybrid integration (theory beside each term's definition, results as one new
section), current structure preserved, every figure carries its parameters.

## 0. The master adjudication: two different "clean" configurations
Five audit findings (abstract L68, §5 L868/L871/L877, conclusion L1996) reduce
to one fact the paper must now state loudly: **the paper's clean campaign runs
at fixed ε = 2 mrad; the Efficiency Frontier study's clean regime uses the
formula ε ≈ 0.42 mrad.** At 2 mrad the acceptance admits ~5× more false
candidates and the composition wall is real at T=1000 (comb wp99 far ≈ 0.20 —
the paper's own store data, correct as printed). At the formula acceptance the
same multiplicity is solved outright (eff 0.9998 @ far ≤ 0.1 %, d=40).
**The audit's "replace the clean numbers" verdicts are REJECTED; the
disambiguation edits are ADOPTED.** This is not a retreat — it measures the
paper's own thesis that ε is the most consequential knob: *the acceptance, not
the multiplicity alone, sets the composition wall.* Every touched sentence
gains its ε qualifier; §5's crossover paragraph gains the bridge sentence.

## 1. Genuine must-fix edits (adjudicated UPHELD)
1. **Abstract L81 + intro echo L160:** "roughly halving the fitted false rate
   at every matched efficiency" — false in both directions (moderate eff 0.99:
   ×30 improvement 0.314→0.010; heavy matched-eff 0.97/0.98: occupancy 0.087
   is *worse* than base 0.077). Replace with the split-by-axis phrasing:
   the polynomial buys efficiency at a fixed false-rate budget; the operator
   buys purity at the matched high-efficiency point (×30 moderate, ×~2 heavy,
   T=200).
2. **Abstract L68, §5 L866–882, conclusion L1996:** add the ε = 2 mrad
   qualifier; in §5 add the formula-ε bridge sentence (adjudication §0).
   Keep the historical numbers — they are correct at their acceptance.
3. **eq:occ prose (§7.1):** the energy is written role-blind
   (E = αΣ_h(o_h−1)²) but the matrix and the verbal description are
   per-(hit, role) — write E = αΣ_h Σ_{role∈{in,out}} (o_h^role − 1)², matching
   `dp_terms`/`bif.fork_graph` (verified: start-hit and end-hit groups built
   separately; consecutive chain segments never coupled).

## 2. New theory, woven in place (hybrid)
* **§2 (after eq. 7):** one forward-reference sentence: the ladder's (s, δ)
  dependence is kept explicit because §7/§8 re-use it verbatim under the
  modified operators.
* **§7.1 occupancy** gains the closed forms (from `09_modified_atlas.py`,
  micro-verified to 1e-12): occupancy is a pure (s, δ) → (s+4α, δ+4α) shift on
  every pure chain (no same-role pair inside a chain), so eq. 7 generalises
  verbatim (r'+1/r' = s+4α); attractor → (δ+4α)/(s+4α) = 0.2857 at α=0.05;
  chain lines shift rigidly by +4α; a same-role pair splits to s+2α / s+6α and
  its level drops to (δ+4α)/(s+6α) = 0.2791 — **below** the shifted attractor
  by ≈ 2αδ/s². The floor-theorem invariant grows from the C-graph to the
  (C, B_all) pair: twins must now also match same-role hit degrees — the
  analytic seed of the measured twin collapse (2.7 % → 0.76 % moderate,
  87 % → 1.3 % heavy).
* **§8.1 fork** gains: exact β-invariance of every pure-true chain (b
  unchanged, no fork edge joins two true segments — the atlas AND ladder are
  untouched, stronger than the existing spectral-lines remark); the competing
  pair drops to δ/(s+β) = 0.222 (β=0.5), lines s±β; the chain+prong motif
  solved exactly (both ends move — state it); B_fork ⊂ B_all so under both
  terms a forked pair carries 2α+β (level (δ+4α)/(s+6α+β) = 0.250 at the
  production knobs — numerically AT the base attractor, worth a remark).
* **Figures:** modified motif gallery (fig-2 analogue) and modified response
  functions (fig-3 analogue) — one panel per operator, base ghosted.
  [BUILD PENDING: gallery + response figure code in make_paper_figures.py]

## 3. The new results section
Insert **after §9 (sec:fit), before Limitations**: "The operator axis at
scale: degree, density, and the loss budget" (~2.5 pp):
1. The degree scan (xiv_degree_scan_{moderate,heavy}) — the sigmoid, its
   operator-dependent onset, the flat moderate base curve (the L1 proof made
   visible), the unfitted-comb counterpart (xiv_degree_comb_*), the marginal
   step-gain figure (xiv_degree_marginal).
2. The L1–L3 loss budget + twin fractions (xiv_loss_budget) — on base A the
   deficit is 100 % twins; the operator sets the twin population, the
   response cannot touch it.
3. Metrics vs density for all five series (xiv_metrics_vs_T_*) — matched
   1BQF cosine everywhere; occupancy's span wall (λ_max ≈ 4α·max hit degree:
   143 @ T=700, 202 @ T=1000); the formula-ε composition wall closing the
   moderate frontier between T=700 and T=1000.
4. The corrected two-axis summary table (moderate T=200): classical/1BQF/best
   fitted all pay far 0.3137 at eff 0.99 on base A; occupancy+refit pays
   0.0102; occ+fork best at heavy (far@eff0.99 0.225 vs base 0.633).
* §7's "destroys the notch" verdict gets its sharpened data point: the matched
  1BQF reads exactly 0.000 under every occupancy operator at every regime,
  including clean, where QSVT at d=4 keeps 1.000.
* Limitations §10: occupancy = low-T technique (span law) + the erf L3
  headroom (0.25 % twin floor, 1.27 % threshold overlap) as open response
  engineering.

## 4. Wording-risk sweep
The 50 tension/wording-risk items in `paper_contradiction_map.json` are
folded during the section edits; none blocks. Recurring themes: (a) "fork =
regression" phrasings predating the refit redemption — soften toward §8's
existing "operator and response are one design" doctrine; (b) unscoped
superlatives that the T-axis breaks — add "at T=200" scopes; (c) numbers
quoted from pre-2026-08-26 pooled baselines — re-derive from
`outputs/{06_degree_scan,06b_highT_onebqf,10_degree_numbers}.csv`.

## 5. Provenance for every new number
repo `Quantum_Track_Reconstruction` @ `ab95dca8` ·
`QSVT/Efficiency_Frontier/{06_degree_scan,06b_highT_onebqf,09_modified_atlas,10_degree_figs,08_writeup_figs}.py` ·
store events via `qtrk_pipeline.ensure_event` (deterministic keys), T=200
reps 0–9, T=400 reps 0–4, T=700/1000 reps 0–2 · CSVs in
`Efficiency_Frontier/outputs/`.
