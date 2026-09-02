# Paper additions plan — the two figure-6 companions (for George's check, 2026-09-02)

Nothing in `main.tex` has been changed yet. This file is the complete plan; each item is
either an unconditional edit (E) or a decision George has to make (D). After the OK the
edits are applied in one pass, the paper recompiled, and the ledger updated.

## 0. What the two figures are

Both come from `QSVT/Efficiency_Frontier/13_fig6_overlay_fig.py`, numbers in
`outputs/13_fig6_companion_numbers.csv`. One configuration throughout, identical to
figure 6's: fixed ε = 2 mrad, σ_scatt = 1e-4, σ_res = 0, no hit drop, φ_max = 0.2, γ = 3,
δ = 1, step kernel, T = 10…1000, the stored `Verify_new_results` events. Efficiency axis
zoomed to 90–100 %, false-rate axis log with a measured 0 % drawn at the 10⁻³ floor.
Every curve on a figure is read at ONE threshold convention, named in the title and footer.

**Figure A — `xiv_fig6_fixed_tau.pdf`** (label `fig:fig6fixed`). Figure 6's three solvers,
all at the fixed absolute threshold τ = 0.35 (1BQF and comb amplitudes rescaled to the
classical signal-support norm first, as in the store). Store mean ± sem over reps
(classical 5–20, comb 5–10, 1BQF 3–10 per T). The 1BQF sits at ≈75 % — below the zoom —
and is annotated with an arrow ("1BQF ≈ 75 % at every T: the endpoint cut, off this
scale"). No fitted-response series: the fitted polynomial has no absolute activation
scale, so it cannot be read at a fixed τ (stated in the footer).

| T | classical eff / far | comb eff / far | 1BQF eff / far |
|---|---|---|---|
| 200 | 100 / 0.76 % | 98.9 / 0.00 % | 75.0 / 0.33 % |
| 400 | 100 / 2.25 % | 97.2 / 0.02 % | 74.7 / 1.73 % |
| 700 | 100 / 9.64 % | 94.3 / 0.17 % | 74.5 / 8.05 % |
| 1000 | 100 / 20.5 % | 92.1 / 0.93 % | 74.3 / 18.8 % |

**Figure B — `xiv_fig6_matched.pdf`** (label `fig:fig6matched`). Every solver at the
matched 99 %-efficiency working point: the store's three (classical, production comb,
1BQF; threshold just below the 1 % quantile of true amplitudes, so efficiency reads
≥ 99 %) plus the four refitted d = 40 responses (base A, +occupancy α = 0.05,
+bifurcation β = 0.5, both; threshold at exactly the 99 % rank; same stored events,
reps 0–2; driver `12_fig6_overlay.py`). The driver's classical solve reproduces the store
rep-for-rep at both conventions (worst difference 0.0000 pp), so the two families are on
the same footing.

| T | classical | comb | 1BQF | fitted base | +occ | +fork | both |
|---|---|---|---|---|---|---|---|
| 200 | 0.65 % | 0.31 % | 15.7 % | 0.00 | 0.00 | 0.00 | 0.00 |
| 400 | 1.49 % | 1.44 % | 43.9 % | 0.27 | 0.83 | 0.06 | 0.50 |
| 700 | 4.92 % | 9.06 % | 69.5 % | 1.04 | 3.84 | 1.13 | 1.60 |
| 1000 | 10.0 % | 20.7 % | 82.6 % | 2.80 | 5.72 | 2.55 | 3.54 |

(false rate at the matched point; the store curves at T = 300/550 have no fitted
counterpart because the driver ran T = 10, 50, 100, 200, 400, 700, 1000.)

## 1. Placement (D1 — George decides; recommendation given)

- **Figure A → §5.2 "Clean-event results"**, directly after `fig:clean2x2` (after line 868,
  before the crossover paragraph). It is pure §5 content: the same three solvers, one
  convention, and it makes the fixed-τ bullet numbers (lines 832–853) visible on a scale
  where the comb's 100 → 92.1 % sag and the classical 0 → 20.5 % false-rate rise are both
  legible (figure 6 squeezes them into a 0–100 axis).
- **Figure B → §10.4 "Density: the five solvers to T = 1000"**, directly after
  `fig:density5` (after line 1904). *Recommended over §5.2* because (i) the four refitted
  operators are only defined in §7, §8 and §10, so in §5 they would be unexplained
  curves; (ii) beside `fig:density5` (formula ε, same five series, same T range) it is
  the direct acceptance comparison the paper keeps promising in words ("at the formula
  acceptance the same T = 1000 multiplicity is solved outright"); (iii) it carries the
  §10 finding that at 2 mrad none of the operator terms beats base A (occupancy is the
  worst of the four at T ≥ 400), which belongs with §10.4's "does not scale" argument.
  §5.2 gets a one-line forward reference.
- Alternative: both in §5.2 as a pair. Reads well as "figure 6 split by convention" but
  forward-references three sections from a §5 figure. Not recommended.

## 2. Captions (E — drafted; wording open to edits)

**Figure A**
> The figure~\ref{fig:clean2x2} campaign read at one threshold convention: every solver at
> the fixed absolute cut $\tau=0.35$ (1BQF and comb amplitudes rescaled to the classical
> signal-support norm before the cut). (a) Segment efficiency on a 90–100\% axis, (b)
> segment false rate (log; a measured 0\% is drawn at $10^{-3}$). Same configuration as
> figure~\ref{fig:clean2x2}: fixed $\varepsilon=2$\,mrad, $\gamma=3$, no hit drop, mean $\pm$
> standard error over stored event repetitions (5–20 classical, 5–10 comb, 3–10 1BQF per
> $T$). The 1BQF's $\approx75\%$ plateau is the endpoint cut of section~\ref{sec:onebit}
> and lies below this axis (arrow). No fitted response appears: a fitted polynomial has no
> absolute activation scale and is read only at a matched working point
> (figure~\ref{fig:fig6matched}).

**Figure B**
> The same campaign at the matched $99\%$-efficiency working point, every series alike:
> the three stored solvers of figure~\ref{fig:clean2x2} (threshold just below the 1\%
> quantile of each solve's true amplitudes, so efficiency reads $\ge99\%$) and the four
> refitted $d{=}40$ responses of section~\ref{sec:atscale} on base $A$, $+$occupancy
> ($\alpha=0.05$), $+$bifurcation ($\beta=0.5$) and both (threshold at exactly the 99\%
> rank; the same stored events, repetitions 0–2, refit per operator). Fixed
> $\varepsilon=2$\,mrad throughout — compare figure~\ref{fig:density5}, the identical five
> series at the formula acceptance. At this acceptance the production comb ties the
> classical solver at $T=400$ ($1.44\%$ vs $1.49\%$) and is behind it from $T=550$
> ($20.7\%$ vs $10.0\%$ at $T=1000$); the refitted base-$A$ response stays at $2.8\%$, and
> no operator term improves on it (occupancy is the worst of the four beyond $T=400$).

## 3. Text edits (E unless marked D)

**T1 — §5.2 opening (lines 825–830).** After "…the 1BQF at its efficiency-first point),
the picture is:" insert the clause "— figures~\ref{fig:fig6fixed} and~\ref{fig:fig6matched}
repeat the campaign one convention at a time —". (Pure cross-reference.)

**T2 — §5.2 crossover paragraph (lines 870–880) — D2.** The current numbers ("$1.3\%$
vs classical $3.1\%$ on matched fresh solves" at $T=400$; "$\approx21\%$ comb vs $17.5\%$
classical, crossover near $T=700$"; footnote "as low as $10\%$") have no locatable
source: not the store (20-rep or reps 0–2), not the new driver, no file in the repo. The
store (the same aggregation as figure 6 and every bullet above it) reads: comb ahead
through $T=300$ (0.31 vs 0.65 at 200, 0.76 vs 1.02 at 300), tie at $T=400$ (1.44 vs
1.49), behind from $T=550$ (5.2 vs 2.9; 9.1 vs 4.9 at 700; 20.7 vs 10.0 at 1000). The
2026-06-14 working-point refresh (`Toy_Characterisation/WP99_REFRESH_LOG.md`, "§7.3
crossover honesty") already adjudicated exactly these store numbers for the Notion draft;
`main.tex` inherited the older ones from `main_v1_backup.tex`. **Recommendation: replace
with the store numbers and delete the footnote.** Draft:

> At the matched $99\%$-efficiency point (figure~\ref{fig:fig6matched}) the comb is ahead
> of the classical solver through $T=300$ ($0.31\%$ vs $0.65\%$ at $T=200$), ties it at
> $T=400$ ($1.44\%$ vs $1.49\%$) and is behind it from $T=550$ on ($9.1\%$ vs $4.9\%$ at
> $T=700$, $20.7\%$ vs $10.0\%$ at $T=1000$): at extreme density the composition effect
> dominates everyone — the false-to-true ratio $n_F/n_T=T-1$ converts even excellent
> per-segment separation into a growing impurity, and the comb's four narrow passes,
> tuned to the isolated-chain lines, admit the tangle eigenvalues that appear at this
> acceptance from $T\approx400$ on (section~\ref{sec:comb:limits}). One qualifier matters enough
> to state here rather than in a caption: the wall is set by the \emph{acceptance}, …

(the rest of the paragraph, from "the wall is set by the acceptance", unchanged; the
sentence "the two regimes … bracket the operational truth" stays as is — at fixed τ the
comb is still 22× below classical at $T=1000$.) If George prefers to keep the old
numbers, they need a source and the footnote must say which aggregation each is.

**T3 — §7.1 "The term" (lines 1140–1141 and 1160–1162) — D4.** Two sentences are wrong on
events and must go: "and the threshold tracks it" and "The whole true-track atlas moves as
one block: nothing about the term touches the structures we want to keep". The closed
form (eq.~\ref{eq:laddocc}) is exact for an isolated chain; on an event every hit carries
$T-1$ false candidates per role, so the same-role sums $o_h^{\,r}$ are hit-independent
to $\approx0.2\%$ and the term acts as a uniform rescale of the whole solution. Measured
on the stored $T=200$ events at $\alpha=0.05$: $x_{\rm occ}=0.1087\,x_{\rm base}$ for every
segment (p5–p95 0.1087–0.1088 clean; 0.1093–0.1099 moderate; 0.1082–0.1098 heavy); the
isolated-false level is $(\delta+4\alpha)/(s+4\alpha+4\alpha(T-1))=0.0273$ (measured
0.0272), not the chain-only $0.2857$; the fixed threshold $\tau=0.386$ activates nothing
at any $T$. The ranking is preserved exactly at clean (false rate at $99\%$ efficiency
0.0029 both operators) and improved at moderate (0.345 vs 0.444) and heavy (0.299 vs
0.648) through the per-hit non-uniformities — which is why every occupancy result in
§9.4 and §10 is read at a matched working point and none of them changes. Draft
replacement for lines 1140–1141: "The isolated attractor of a lone chain moves to
$(\delta+4\alpha)/(\gamma+\delta+4\alpha)$; on an event it does not — see below." Draft
new paragraph after line 1162 (replacing its last sentence):

> On an isolated chain, that is; an event is not a collection of isolated chains. Every
> hit carries $T-1$ false candidates in each role, so the same-role sums $o_h^{\,r}$
> that the term penalises are, to within $\approx0.2\%$, the same number at every hit,
> and the term acts on the whole solution as one uniform rescale: on the stored $T=200$
> events at $\alpha=0.05$, $x_{\rm occ}=0.109\,x_{\rm base}$ segment by segment, the
> isolated-false level is $(\delta+4\alpha)/(s+4\alpha+4\alpha(T-1))=0.027$ rather than
> the chain-only $0.286$, and the fixed threshold activates nothing. The atlas does not
> move as a block — it shrinks by a $T$-dependent factor — so an occupancy operator is
> read only at a matched working point. What survives the rescale is the ranking, and
> that is where the term earns its keep: at clean noise the false rate at $99\%$
> efficiency is unchanged (0.0029 on both operators); at moderate and heavy noise it
> falls from 0.444 to 0.345 and from 0.648 to 0.299, through the per-hit departures from
> the mean field, which are exactly the co-hit structures the base Hamiltonian cannot
> see.

Numbers: `QSVT/Efficiency_Frontier/outputs/cache/06_coef_{regime}_*_rep{r}.npz` (sol_C,
truth) at formula ε, T = 200, reps 0–2, computed 2026-09-02 (worklog in the Notion
to-do). The §7.2 heavy T = 200 α = 0.3 number ($0.650\to0.210$) is a matched-point number
and stands.

**T4 — §10.4 new paragraph after `fig:density5` and Figure B (after line 1929).**

> Figure~\ref{fig:fig6matched} repeats the five series on the benchmark acceptance,
> fixed $\varepsilon=2$\,mrad, where figure~\ref{fig:density5}'s clean row is no longer
> solved outright: the refitted base-$A$ response reads $2.8\%$ false at the matched
> $99\%$ point at $T=1000$ (against $0\%$ at the formula acceptance on the same
> multiplicity), the classical solver $10.0\%$, the production comb $20.7\%$ and the
> 1BQF $82.6\%$. The operator terms do not help here: the bifurcation term is neutral
> ($2.55\%$), both terms together cost ($3.54\%$), and occupancy is the worst of the
> four from $T=400$ on ($5.7\%$ at $T=1000$) — the same span-growth mechanism as under
> noise, now driven by the $4T^2$ false candidates the wide window admits rather than by
> fragments. The two figures together are the measurement behind the two-axis rule
> below: the acceptance sets the false population the response has to separate, and no
> response or operator recovers what a five-times-too-wide window lets in.

**T5 — `fig:density5` caption (lines 1893–1902).** Make the mixed convention explicit:
replace "classical (fixed $\tau$), the matched 1BQF at its $99\%$-efficiency working
point, and the fitted $d{=}40$ response" with "the classical solver at the fixed cut
$\tau=0.35$ (the production baseline; its matched-$99\%$ curve is in
figure~\ref{fig:fig6matched}), and — all at the matched $99\%$-efficiency working point —
the 1BQF and the fitted $d{=}40$ response". (Audit question A-15 below asks whether the
classical curve in this figure should instead be re-read at the matched point so the
whole figure is one convention; that is a data change and is for the audit, not for
this pass.)

**T6 — §11.1 Limitations (after line 1994 "…entire co-hit cliques.").** Add:

> The mean-field collapse of section~\ref{sec:occupancy} sharpens the first of these:
> at fixed $\alpha$ the same-role sums grow as $(T-1)\alpha$, so the whole solution
> scale shrinks $\propto1/T$ along with the span; $\alpha\propto1/T$ would pin both.

**T7 — §5.2 1BQF bullet (line 843–847)** already says the fixed-τ plateau is an artefact;
add "(figure~\ref{fig:fig6fixed})" after "faded curve in figure~\ref{fig:clean2x2}".

**T8 — §2.4 metrics (lines 414–452).** No change needed; the three conventions are
defined there. Check only that "matched-efficiency" and "efficiency-first (99 %)" are
both named, since Figure B's footer uses "matched 99 %" for the store's efficiency-first
point and the driver's exact-rank point.

**T9 — abstract / conclusion.** No echo of the crossover numbers found (the abstract quotes
the fixed-τ 20.5 % → 0.9 % pair, which is store-exact). No edit.

## 4. Mechanics (E, after the OK)

1. Copy `QSVT/Efficiency_Frontier/figures/xiv_fig6_fixed_tau.pdf` and `xiv_fig6_matched.pdf`
   into `QSVT/Paper_planning/qsvt_velo_paper/figures/`.
2. Apply T1–T7; add the two `figure` environments (`[t]`, `width=0.95\textwidth`,
   `captionsetup{width=0.92\textwidth}`) at the placements of §1.
3. Recompile (`pdflatex` ×2 + bibtex if needed); confirm the page count and that no
   `??` references remain (`grep -c '??' main.log`).
4. Update `QSVT/Efficiency_Frontier/FIGURE_DATA_SOURCES.md`: the `xiv_fig6_overlay` entry
   is already replaced by the two companion entries; add the paper labels.
5. Commit; append the worklog to the Notion to-do; patch the two Notion write-ups'
   occupancy sentences (theory addendum `3ce5d544-b9d9-8105…`, XIV `3ce5d544-b9d9-8164…`)
   with the T3 paragraph.

## 5. Decisions for George

| # | Question | Recommendation |
|---|---|---|
| D1 | Where does Figure B go — §10.4 beside `fig:density5`, or §5.2 beside Figure A? | §10.4 |
| D2 | Replace the untraceable crossover numbers (1.3 vs 3.1; 21 vs 17.5, crossover ≈700) with the store's (tie at 400; behind from 550; 20.7 vs 10.0 at 1000) and drop the footnote? | Yes |
| D3 | Add the classical +bifurcation series at fixed τ to Figure A (98.3 % / 13.5 % at T = 1000 vs base 100 % / 20.5 %)? It is a real fixed-τ reading (the fork term keeps the activation scale) but it is not one of figure 6's solvers. | No — keep Figure A as figure 6's three solvers |
| D4 | §7.1: replace the two "moves as one block / threshold tracks it" sentences and add the mean-field paragraph (T3) in this pass, or leave §7 for the audit? | Do it now — the sentences are wrong as written |
| D5 | Figure 6 itself: unchanged (as "additional figures" implies), or drop its faded 1BQF fixed-τ curve now that Figure A shows it? | Unchanged |
