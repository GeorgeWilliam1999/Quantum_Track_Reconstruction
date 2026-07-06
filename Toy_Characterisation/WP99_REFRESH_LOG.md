# wp99 working-point refresh — running log (2026-06-14)

**Mandate (user):** every 1BQF segment result must be reported at the
**high-efficiency working point**, NOT the low-false-rate fixed τ=0.35 absolute cut.
The image we want: *we recover all the true segments* (killing false segments is how
LHCb works). The fixed τ=0.35 cut chops the 1BQF's outer true band (~0.18–0.28) and
pins efficiency at the artificial ~74–75 % plateau — "the cut, not lost physics."

## Working point definition (wp99)

Per solver, per config (per T / noise cell): place τ just **below the 1 % quantile of
the TRUE segment amplitudes** → ~99 % segment efficiency; quote the false rate paid.
Same convention the QSVT notebooks already use. Classical naturally sits at ~100 %
eff already; the change is for the **1BQF (quantum)** rows.

- Absolute cut being replaced: `_shared/qtrk_pipeline/metrics.py` `ABS_THRESHOLD=0.35`.
- Metrics are a recomputed VIEW, so re-thresholding is cheap (no re-solve needed).

## Decisions (2026-06-14)

1. Working point = **wp99** (1 % true-amplitude quantile). Not literal min-true 100 %,
   not Youden/EER (Youden stays a diagnostic only).
2. Scope = **canonical studies + their Notion write-ups**. Legacy scratch left alone
   (EpsilonStudies, Initial, Quantum_Toy_Study, Recovery_Seperation_analysis; all
   `Untitled-*` / `*_BACKUP` notebooks).
3. Reports: **replace fixed-τ=0.35 as the headline** with wp99, BUT preserve any
   deliberate two-threshold comparison — it was the discovery mechanism for the 1BQF.

## Studies in scope

Epsilon_study_2 · ERF · Larger_Scatter · Larger_Scatter_Density · Verify_new_results ·
Segment_level_studies · FR_SE_tradeoff · Run3_Verification · Bifurification ·
Presentation_mini_q_workshop_UM. QSVT pages already lead with wp99 (secondary
fixed-τ cells only need a relabel).

## TASK 0 — shared working-point machinery (DONE)

- [x] `_shared/qtrk_pipeline/metrics.py`: `WP_TARGET_EFF=0.99`,
  `working_point_threshold(sol,truth,target_eff)`, `metrics_at_wp`, `quantum_metrics_wp`
  (exported via `qtrk_pipeline.__init__`). wp99 eff/far are scale-invariant (τ is a
  quantile of the solution's own true amplitudes), so they do NOT depend on the quantum
  rescale convention — only `cos_QC` does.
- [x] `_shared/qtrk_pipeline/build_metrics.py`: writes `segment_{efficiency,purity,
  false_rate}_wp`, `tau_wp`, `n_{active,true_active,false_active}_wp` for EVERY solve
  (classical + quantum + qsvt), so all store-backed studies inherit wp99 from one path.
- [x] `Verify_new_results/qtrk_view.py`: `paired`/`aggregate` surface `eff_*_wp`,
  `far_*_wp`, `pur_*_wp`, `tau_wp_*` (graceful fallback when columns absent).
- [x] `Segment_level_studies/seg_store.py`: `fixed_eps_metrics`/`agg_by_ntrk` expose
  `se_wp_*`, `fr_wp_*`, `tam_wp`, `fam_wp`.

### Validation (single clean γ=3 T=400 1BQF solve, Verify store)
sol_C false median 0.25 (Hopfield attractor), true median 0.41; sol_Q false ≈ 0 (1BQF
kills false), true ≈ 0.40, outer band dips to ~0.18.
- fixed τ=0.35 → **eff 75.0 % / far 0.0 %**  ← the artefact
- wp99 (τ≈0.18) → **eff 100.0 % / far 2.68 %**  ← recovers all true segments

### Verify benchmark, 1BQF γ=3 clean (fixed τ=0.35 → wp99)
| T | fixed eff/far | wp99 eff/far (τ≈0.18) |
|---|---|---|
| 100 | 74.95 / 0.07 | 99.9 / 5.1 |
| 400 | 74.71 / 1.73 | 99.5 / 43.9 |
| 1000 | 74.25 / 18.79 | 100.0 / 82.7 |

classical wp99 (γ=3 clean): 100 % eff, far 0 → 10 % (T=10→1000) — far the cheaper
solver at equal efficiency. The 1BQF is a ~100 %-efficiency algorithm whose false-rate
cost grows with multiplicity, NOT a "75 %-efficiency" algorithm.

## TASK 1 — store-wide metrics.csv rebuild

- metrics.csv backed up → `metrics.csv.pre_wp99.bak` (13848 rows).
- Full rebuild with wp columns: IN PROGRESS (loads all high-T vectors; ~15–20 min).

## TASK 1b — metrics.csv rebuilt (DONE)
Full store rebuild: 13848 rows (matches backup exactly), wp columns present, 0 nulls.
`quantum_metrics_wp` later given a `signal_threshold` arg + build passes the γ-aware τ
so `tau_wp`/`cos_QC` are correct for γ≠3 (eff/far were always correct — scale-invariant).
The CURRENT metrics.csv `tau_wp` for γ≠3 quantum rows is on the 0.35-rescale (cosmetic
only; a final rebuild will refresh it). eff/far/purity_wp are correct everywhere.

## TASK 2 — figure generators converted + RE-RUN at wp99 (Q_env; never MPLBACKEND=Agg)

DONE (executed, figures regenerated, verified):
- [x] `Epsilon_study_2/gen_metrics_vs_T.py` — eff/far-vs-T family figs (1BQF clean
  0.75→1.00 eff); console prints both fixed-τ and wp99.
- [x] `Segment_level_studies/01_solver_segment_efficiency.ipynb` — 5 figs (2×2 ×4 +
  γ-sweep); wp99 headline + faded fixed-τ; summary CSV gains eff_wp/far_wp.
- [x] `Verify_new_results/Quantum_segment_level_store.ipynb` — 6 figs; Fig1/Fig4 wp99
  headline + faded fixed-τ; Figs 5/6 (abs-vs-rel, AUC) intentionally unchanged.
- [x] `ERF/store_landscape.py`, `Larger_Scatter/store_analysis.py`,
  `Larger_Scatter_Density/store_analysis.py` — eff/far now wp99 (eff_fix/far_fix kept).
  Classical headlines ~unchanged (well-separated); LS `fig_quantum` 1BQF now wp99.

SUPERSEDED (legacy, left alone per scope): `ERF/analysis.ipynb` (pkl-based),
`Larger_Scatter/analysis.ipynb` + `Larger_Scatter_Density/analysis.ipynb` (local
`*_events.csv`) — replaced by the store_*.py scripts above.

PRESERVE (two-threshold discovery notebooks — the mechanism that motivated the 1BQF):
`FR_SE_tradeoff/clean_vs_noisy_400.ipynb` (already sweeps τ_Q),
`Bifurification/02_metrics_vs_beta.ipynb` (already labels "fixed τ=0.35 … collapse
together" as the artefact), `Segment_level_studies/04_quantum_false_rate_investigation`.

NO THRESHOLD HEADLINE (amplitude/angle/spectral — skip): `Segment_level_studies/02`,
`07_segment_amplitude_atlas`.

STILL TO DO (report 1BQF eff/far, store-backed): Verify `segment_level_store.ipynb`
(classical-focused), `Quantum_segment_level_analysis_new_data.ipynb`; Run3_Verification
(live solve); Epsilon analysis/deep_analysis (confirm canonical vs local); UM-talk scripts.

## TASK 3 — write-ups corrected

DONE (verified rendered):
- [x] **📐 Source of Truth** (`3795d544…c541d`): added §6.5 "HEADLINE operating point — wp99"
  (definition + fixed-vs-wp99 table + scale-invariance); reframed §7 "~75 %" sentence; updated
  §11 bottom line. This page WINS → wp99 is project-canonical.
- [x] **Epsilon_study_2 1BQF report** (`3795d544…4d7e`): §2.6 callout reframed (75 % = fixed-τ
  cut, wp99 recovers ~100 %); Abstract finding 2 attributes ~100 % to wp99. §7.7/§7.10
  (efficiency-first τ) preserved as the discovery analysis.
- [x] **1BQF eigenfilter false-rate** (`37a5d544…0778`): Bottom line now leads with wp99
  (~100 % eff); plateau-halving mechanism + τ=0.35-vs-τ_Q comparison preserved (discovery).
- [x] **Quantum-vs-classical comparison** (`3795d544…c40f`, legacy 27-May): caveat extended
  → wp99 canonical; τ=0.35 table marked historical.
- [x] **OneBQF / quantum-algorithm** (`3795d544…98afc`, legacy 27-May): caveat extended → wp99;
  "75 % floor" + relative-τ fix marked as fixed-τ diagnostic.
- [x] **Amplitude atlas** (`37a5d544…435e`, discovery): bottom line — F1 endpoint-halving now
  shown recovered by wp99 (lever 1 realised).

PENDING (lower priority / figure-gated): Reviewer-response (`3795d544…708f`, historical doc —
likely leave as record), Larger_Scatter T3 page, UM-talk page (both figure-heavy, need the
push first). QSVT pages already lead with wp99 — only secondary fixed-τ cells would need a relabel.

⚠️ **User git push required** before Notion raw-URL figure embeds refresh: regenerated
figures under Epsilon_study_2/figures/epsilon_sensitivity/, Segment_level_studies/outputs/
solver_segment_efficiency/, Verify_new_results/outputs/quantum_segment_analysis/store_backed/,
ERF/figures/, Larger_Scatter/figures/, Larger_Scatter_Density/figures/. After push, bump the
`?v=<commit>` cache-busters on the embeds.

## Optional cleanup — DONE
- Final `build_metrics.py` rebuild (γ-aware `signal_threshold`) complete: 13848 rows, 0 nulls,
  eff/far unchanged. `tau_wp` now sane for all healthy solves (median 0.18; γ=3 clean = 0.181).
  A few genuinely DEGENERATE solves (1BQF ≈ orthogonal to the classical signal support — failed
  high-noise / qsvt cells) still report a large `tau_wp`; harmless (eff/far are scale-invariant).
- metrics.csv backups: `.pre_wp99.bak` (original) and `.wp99_v1.bak` (0.35-rescale tau_wp).

## SCOPE CORRECTION (2026-06-14, user) — wp99 is the 1BQF HEADLINE ONLY

The classical solver is NOT switched to wp99: it keeps its fixed γ-aware cut
(τ=0.35 at γ=3), its established operating point (classical never had the 75 %
dilemma — at 0.35 it already gets ~all true segments with low false). Applying
wp99 to classical inverted the classical-focused narratives (ERF resolution
recovery, LS drop-costs-efficiency, density) and contradicted the Epsilon abstract
— so it was reverted. Final convention: **classical = fixed τ=0.35; 1BQF = wp99.**

Files fixed (classical reverted, 1BQF kept wp99) + re-run:
- `ERF/store_landscape.py`, `Larger_Scatter_Density/store_analysis.py` → classical fixed
  (both classical-only studies).
- `Larger_Scatter/store_analysis.py` → classical figs fixed; `fig_quantum` 1BQF uses
  new `eff_wp` column (wp99).
- `Epsilon_study_2/gen_metrics_vs_T.py` → per-solver: classical `segment_efficiency`,
  1BQF `segment_efficiency_wp`.
- `Segment_level_studies/01` → `plot_2x2(wp_headline=)`: classical fixed, 1BQF wp99+faded;
  γ-sweep classical fixed / 1BQF wp99.
- `Verify_new_results/Quantum_segment_level_store.ipynb` → classical `eff_C` (fixed),
  1BQF `eff_Q_wp` headline + faded `eff_Q`.
Classical headline numbers now match the existing page text again (LS clean 100/0.1,
max-drop 84.6, etc.). Notion: LS-T3 1BQF text reframed to wp99.

## Commit / push — DONE
- `508fd455` on `highT-1bqf-refresh-2026-06-14` (pushed to origin). Code + 2 notebooks + all
  regenerated figures/CSVs + this log. Parallel-session files (PROJECT_STATUS.md, theory.md,
  dp*/detector_physics.*) deliberately excluded. **Notion `?v=` cache-busters should be bumped
  to `508fd455`** so the corrected pages re-fetch the new figures.

## SESSION 2 (2026-06-15) — Epsilon §7 sensitivity-scan + §7.13 DP4 figures → wp99

The §7 ε/σ sensitivity-scan figures and the §7.13 DP4 soft-track figure were the
remaining Epsilon_study_2 plots still baking the fixed τ=0.35 1BQF (the artefactual
~75 % plateau) — `gen_metrics_vs_T.py` had been converted in session 1 but these
had not. User decision: **wp99 ONLY for the 1BQF (drop the fixed-τ 1BQF curve);
classical stays fixed τ=0.35.** Discovery plots preserved.

Files edited + re-run (Q_env):
- `gen_epsilon_sensitivity_scan.py`: `solve_cell` now also computes
  `quantum_metrics_wp` → `eff_Q_wp`/`far_Q_wp`/`tau_Q_wp` in the JSON. Figs 1+2
  (`eps_scan_fixed_sres/sscatt`), Fig 4 (`sigma_scan_formula_eps`) plot 1BQF at
  `*_Q_wp`; Fig 6 (`store_grid_highT`) reads `segment_efficiency_wp`/
  `segment_false_rate_wp` for the quantum rows. Titles → "classical τ=0.35, 1BQF wp99".
  eff ylim → (0.9, 1.005). The fixed-τ `eff_Q`/`far_Q` kept in JSON for the record.
- `replot_sigma_scan_fig.py`: mirrors Fig 4 (reads `*_Q_wp` from the JSON).
- `gen_detector_physics.py`: DP4 `solve_quantum` → `quantum_metrics_wp`; panel (a)
  annotated "classical τ=0.35, 1BQF wp99".

PRESERVED (discovery, unchanged in intent, regenerated with fresh stats):
- `eff_universal_collapse.png` (motif ladder vs τ=0.35 + classical-only collapse).
- `roc_tau_sweep.png` (swept-τ working-point ROC, markers at τ=0.35).

Validation (formula-ε scans, T=30): 1BQF fixed-τ effQ ≈ 0.74–0.75 → wp99 effQ
0.997–1.000; wp99 far 0 → 0.18 as σ_res 0→0.05 (LHCb-style trade). DP4 wp99 effQ
≈ 1.0 across the soft-track sweep (pT 11→0.36 GeV). All 7 figures regenerated,
non-zero, visually confirmed.

TODO (handoff): **user must git push** the regenerated Epsilon figures, then bump
the Notion `?v=` cache-busters on the §7 / §7.13 embeds so the write-up re-fetches
the wp99 plots (the page text was already reframed to wp99 in session 1).

## SESSION 3 (2026-07-03) — Notion cache-buster closure. DONE → todo closed.

Verified sessions 1+2 are merged to origin/main (`508fd455`, `981ad948`, `55bbf923`
via merges `69a167dd`, `04ce295f`); Eps2 figures identical branch↔main. Swept every
figure embed on the six study write-ups (git last-change vs `?v=` buster commit date):

- **Eps2 detector-noise page:** 7 embeds stale (the §7 scan set at `?v=f9163421`
  06-13, dp4 at `?v=5fa4f020` 06-14 — both predating the 06-15 `55bbf923` regen)
  → bumped all 7 to `?v=55bbf923` via surgical in-URL `update_content` (safety-tested
  on a scratch page first: image blocks survive in-URL edits, no literalisation).
  Verified post-edit: 7×`v=55bbf923`, 0 old busters, 0 escaped `\![`.
- **LS / LSD / Segment_level / Verify / ERF pages:** all embeds already current
  (session-1 bump to `69a167dd` 23:51 postdates the last figure change `981ad948`
  23:40; no-buster embeds unchanged since before each page build). No action.

wp99 refresh COMPLETE end-to-end: code + figures (main), page text, cache-busters.

## SESSION 4 (2026-07-06) — the QSVT side (user re-raise: "plots showing ~70 % are not correct")

Sessions 1–3 covered Toy_Characterisation; the standing note "QSVT pages already lead
with wp99" was wrong for the HEADLINE surfaces: QSVT nb01's 2×2 plotted ALL solvers at
fixed τ, and the paper draft §7.2 led with the fixed-τ Table 1 (1BQF 74–75 %).

Fixed (convention: classical & comb at fixed τ=0.35; **1BQF headline = wp99**, fixed-τ
kept faded as the labelled cut artefact):
- `QSVT/Segment_level_studies/01_qsvt_segment_efficiency.ipynb` — 2×2 (clean + 1 % drop)
  and γ companion re-plotted (1BQF wp99 solid + faded fixed-τ); headline cell prints both
  conventions and saves `outputs/segment_efficiency/headline_both_conventions.csv`.
  Re-executed in Q_env (no errors, figures verified visually). Same filenames.
- **Paper draft** (Notion `37c5d544…d56f0`): §7.2 retitled "The headline comparison —
  solver operating points" + rewritten (1BQF wp99 ladder 5.1→43.9→69.5→82.6 % over
  T=100/400/700/1000); Table 1 1BQF column → wp99; Table 2 τ* rows → stable wp99 view
  rows (clean: 100/1.5 · 99.5/43.9 · 99.1/1.4; drop: 99.4/41.7 · 99.5/44.3 · 99.4/44.0 —
  fragment degeneracy now visibly solver-independent); Fig 6 swapped
  headline_at_tau_star.png → segment_efficiency/qsvt_working_points.png; τ* demoted to an
  explicit instability caveat; abstract "cheapest at every multiplicity / up to 44 %" →
  "cheapest through T≈400, 1.4 % vs 43.9 % (82.6 % at T=1000)"; §7.3 crossover honesty
  (comb behind classical at matched 99 % beyond T≈400: 9.1 vs 4.9 @700, 20.7 vs 10.0 @1000).
- **QSVT II** (`37c5d544…aca1c`): headline table 1BQF column → wp99 + convention
  paragraph; alt text + bottom-line callout rewritten (fixed-τ 74.x numbers preserved in
  prose as the artefact record). **QSVT project hub**: status callout "vs 1BQF 75 %/1.7 %"
  → wp99 framing; status-table T-label fix (0.9 % vs 20.5 % is T=1000).
- Numbers source: store view (seg_store.fixed_eps_metrics, wp cols; validity gate drops
  144/3189 rows). Table 2's fixed-τ discovery row deliberately kept.

HANDOFF: **user must git push** nb01 + regenerated figures (qsvt_2x2_gamma3_drop0/1pct,
qsvt_gamma_companion, qsvt_working_points, headline_both_conventions.csv), then bump
`?v=` busters: QSVT II 2×2 embed (`?v=7ddc068a` → new commit) and ADD busters to the
paper-draft Fig 5/5d embeds (currently unbusted, content changed).
NOT swept (deliberate/discovery, unchanged): nb04 per-solver-threshold study, nb06 §5
conventions comparison, nb07 comb-design fixed-τ analysis; compendium §7 (threshold
mechanism/staircase) and §17 (carry the audit's τ* caveats).

SESSION 4b (same day, user asked "are the write-ups correct?") — verification sweep:
- QSVT III: checked, already clean (working-point based, no 1BQF headline).
- **Application compendium** (`37b5d544…c7414`): WAS still stale → fixed + verified:
  2×2/drop/γ-companion alt-texts + captions rewritten to the new figure content (1BQF
  wp99 + faded artefact curve); §9 mid-programme scoreboard flagged "historical record,
  kept verbatim" (fixed-τ + τ* rows predate the convention); §10 prose corrected (the
  1BQF DOES now have T=700/1000 points — matrix-free engine; "no data points there"
  was stale) and its scale-table 1BQF column → wp99 (99.5/43.9 · 99.5/69.5 · 100/82.6)
  with the fixed-cut values kept as a one-line artefact note.
- §18 verdict table: checked, τ-clean (no 1BQF fixed-τ claims).

## SESSION 5 (2026-07-06, second session in parallel) — full-repo sweep of the remainder

Two exhaustive scans (every .py + every .ipynb under Toy_Characterisation/, QSVT/,
Hough/) against the convention; SESSION 4 (above) covered the QSVT surfaces, this
session everything else. Convention throughout: classical fixed γ-aware τ; **1BQF
headline = wp99**; fixed-τ 1BQF kept only faded/labelled.

**Library fix — `metrics.py` `working_point_threshold` degenerate corner (Run-3-found):**
when >1 % of TRUE segments have ~zero amplitude (the 1BQF structurally zeroes ISOLATED
clusters — λ = s exactly, notch → ~1e-17 — and real events have isolated-true from
missing hits) the 1 %-quantile τ went NEGATIVE (st[k]−1e-9), re-admitting every
structurally-killed segment (Run-3: ~1e6 isolated-false → far 0.92, purity 0.08 —
meaningless). Now floored at `WP_TAU_FLOOR = 1e-6`: the working point reads "admit
every segment the solver can PHYSICALLY recover"; the achieved efficiency honestly
reports the structural miss. Healthy solves (quantile ~0.18) unaffected.

**Fixed + re-executed/regenerated (1BQF → wp99 headline, fixed-τ kept as record):**
- `Run3_Verification/Run3_segment_characterisation.ipynb` — quantum cell now computes
  `quantum_metrics_wp`; per-class figure cut at the 1BQF's own wp99 τ. **HEADLINE
  CHANGE: 1BQF wp99 eff 0.93 / pur 0.97 / far 0.030 — BEATS classical (0.86/0.98/0.023)
  on Run-3.** The ~7 % residual inefficiency is STRUCTURAL (isolated-true zeroed by the
  same notch that kills the ~1e6 isolated-false). Old "eff ≈ 0.7" = the cut artefact.
  `quantum_vs_classical.csv` gains effQ_wp/purQ_wp/farQ_wp/tauQ_wp.
- `Run3_Verification/make_error_rates_vs_size.py` — 1BQF overlay reads the wp columns.
- `Noisy_Realistic/noisy_analysis.ipynb` — ladder figure: quantum+qsvt panels → wp99
  headline + faded fixed-τ (wp99 1BQF eff 0.997–1.000 vs the 0.70–0.73 artefact); also
  exact-token membership + validity gate ported in (matches noisy_first_look.py).
- `Epsilon_study_2/build_fp_census.py` → census gains `act_Q_wp`/`tau_Q_wp` (scale-
  invariant, valid on the legacy-rescale pickles); `make_fp_notebook.py` +
  `false_positive_types.ipynb` quantum panels → wp99, fixed-τ kept in the summary CSV.
- `Epsilon_study_2/gen_epsilon_pmiss_scan.py` — Part B + Fig 9 1BQF → wp99.
- `Epsilon_study_2/gen_T400_eval.py` — store-1BQF overlay star → wp columns (captions
  relabelled "store 1BQF (wp99)").

**Bannered LEGACY (not re-executed; superseded, kept as record):** Eps2 `analysis.ipynb`
+ `segment_metrics_calc_epsilon.ipynb`, ERF `analysis.ipynb` + `hamiltonian_comparison
.ipynb`, LS/LSD `analysis.ipynb`, Verify `Quantum_segment_level_analysis_new_data.ipynb`
+ `segment_level_analysis{,_new_data}.ipynb`; one-line LEGACY headers on the Verify
`_restyle/_add_seg14e/_add_tau_sweep/_collect_and_plot.py` generators,
`condor_obqf/run_segment_event.py`, `_shared/run_worker.py`.

**Flagged, deliberately NOT touched:** UM-talk scripts (`Presentation_mini_q_workshop_UM/
scripts/gen_quantum.py, gen_feedback.py, fig_quantum.py` — delivered-talk assets, 11 Jun;
convert only if the deck is reused), `QSVT/Initial/qsvt_helpers.py` (fixed-τ-only helper
library for the superseded Initial band-design notebooks), `QSVT/Initial/02` (self-
caveated sandbox), `build_metrics.py` main() console summary (diagnostic, no figure).
PRESERVE list honoured (FR_SE_tradeoff, Bifurification 02_metrics_vs_beta,
Segment_level_studies/04, ERF youden diagnostics).

## SESSION 4 CLOSURE (2026-07-06, evening) — pushed + busters bumped. DONE.

Commit `0576268d` fast-forwarded to origin/main (`git push origin HEAD:main` from the
branch — dirty-tree checkout was the blocker on the first attempt). Verified: raw
GitHub content-length matches local bytes for all three changed figures. Cache-busters
applied + fetch-verified: QSVT II 2×2 `?v=7ddc068a`→`?v=0576268d`; paper draft Fig 5 +
5d and Application-compendium 2×2 clean/drop + γ-companion got `?v=0576268d` ADDED
(were unbusted). `qsvt_working_points.png` embeds left unbusted (content unchanged).
QSVT II Provenance property re-stamped to `0576268d`. wp99 todo `37f5d544…01dc`
closed (Done). The QSVT lane of the wp99 refresh is complete end-to-end:
figures → prose → tables → embeds → provenance.
