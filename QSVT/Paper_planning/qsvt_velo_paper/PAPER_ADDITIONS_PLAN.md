# Paper additions and the working-point correction — plan (2026-09-02, revised)

Repo: Quantum_Track_Reconstruction. Paper: `QSVT/Paper_planning/qsvt_velo_paper/main.tex`
(39 pp at commit 04b731a4). Nothing in `main.tex` has been edited; this file lists what
should change, why, and the exact numbers, for George to approve section by section.

## 1. The three rules this revision follows

1. **One working point.** Every solver, the classical inversion included, is read at its
   efficiency-first working point: the lowest threshold that admits the true solutions,
   implemented as the threshold just below the 1 % quantile of the solver's own true-segment
   amplitudes (99 % of true segments kept). The fixed absolute threshold
   τ = δ/(γ+δ) + 0.10 = 0.35 was a bug: it sits inside the one-bit filter's reshaped true band
   (chain endpoints at 0.258), so it lowered the efficiency and flattered the false rate.
   No figure, table, or number is read at it any more.
2. **One solver name.** The comb is the QSVT. Its results are labelled by polynomial degree
   and by the Hamiltonian the problem is set up on (base A; A + occupancy term; A +
   bifurcation term; both). "Fitted response", "production comb", "line comb" are not solver
   names.
3. **Plain technical English** in every plan, label, to-do and message.

## 2. What was regenerated today (figures and data — done)

| Figure | File | What changed |
|---|---|---|
| Paper fig. 6, the clean-benchmark headline | `qsvt_velo_paper/figures/clean_headline.pdf` | All three solvers now at the efficiency-first point; the faded fixed-threshold 1BQF curve removed; efficiency axis 90–100 %; labels "classical", "QSVT (d = 40, base A)", "1BQF" |
| Paper fig. 4, the threshold sweep | `figures/threshold_tradeoff.pdf` | The grey "fixed τ = 0.35" line removed; each solver's efficiency-first point stays; "line comb" → "QSVT (d = 40)" |
| Paper figs. 3, 7, 13, 14, 22 | `response_functions`, `resource_comparison`, `fit_comb_paper`, `occupancy_fitted`, `readout_probability` | Labels only: "comb" / "fitted response" → "QSVT, degree d" |
| Paper fig. 20, the formula-acceptance composite | `Efficiency_Frontier/figures/xiv_metrics_composite.pdf` | The classical curve is now read at the efficiency-first point (it was the only fixed-threshold curve on the figure); QSVT labels carry degree and Hamiltonian |
| Efficiency-Frontier figures 15–19 | `xiv_degree_scan_*`, `xiv_degree_marginal`, `xiv_degree_comb_*`, `xiv_responses_modified`, `xiv_loss_budget` | Labels only |
| The new companion figure | `Efficiency_Frontier/figures/xiv_fig6_efficiency_first.pdf` | Replaces the two convention-split companions: every solver at the efficiency-first point, benchmark campaign plus the four degree-40 refits on the same events, 90–100 % efficiency axis |

Removed: the fixed-threshold companion figure and its numbers file, and the fixed-threshold
classical data file `06c_classical_fixed_tau.csv` (it served only the bug).

Consistency anchor: the new classical solves reproduce the benchmark campaign rep-for-rep at
the efficiency-first point (worst difference 0.0000 percentage points, reps 0–2, every T).

## 3. The numbers the paper should now carry (clean benchmark, fixed ε = 2 mrad, γ = 3)

Segment false rate in % at the efficiency-first working point; every series keeps ≥ 99 % of
true segments (classical 100 %, benchmark QSVT 99.1–100 %, 1BQF 99.5–100 %, refits 99.0–100 %).

| T | classical | QSVT d = 40, base A, benchmark line-comb coefficients | 1BQF | QSVT d = 40, base A, coefficients refit | + occupancy α = 0.05 | + bifurcation β = 0.5 | + both |
|---|---|---|---|---|---|---|---|
| 200 | 0.65 | 0.31 | 15.7 | 0.00 | 0.00 | 0.00 | 0.00 |
| 400 | 1.49 | 1.44 | 43.9 | 0.27 | 0.83 | 0.06 | 0.50 |
| 700 | 4.92 | 9.06 | 69.5 | 1.04 | 3.84 | 1.13 | 1.60 |
| 1000 | 10.02 | 20.67 | 82.6 | 2.80 | 5.72 | 2.55 | 3.54 |

Benchmark series: mean over all valid campaign repetitions (classical 5–20, QSVT 5–10, 1BQF
3–10 per T). Refit series: repetitions 0–2 on the same stored events. Source:
`Efficiency_Frontier/outputs/13_fig6_efficiency_first_numbers.csv`.

Reading: with the benchmark line-comb coefficients the QSVT is ahead of the classical solver
through T = 300, ties at T = 400, and is behind from T = 550 (the clean-event design was made
for a tangle-free spectrum; at T ≥ 400 the 2 mrad events develop tangles, section 5.4). With
the coefficients refit to the events the same degree-40 QSVT reads 2.8 % at T = 1000 against
the classical 10.0 %, on every Hamiltonian between 2.5 % and 5.7 %.

Formula acceptance (paper fig. 20), classical at the efficiency-first point, false rate in %:
moderate noise 31.4 (T = 200) → 65.8 → 84.6 → 92.0 (T = 1000); heavy noise 63.3 → 87.6 →
95.5 → 93.7. Clean 0.00 → 0.12 → 0.07 → 0.07. The QSVT refits on the modified Hamiltonians
were already read at this point; those curves are unchanged.

## 4. Passages in `main.tex` that read at the fixed threshold (to be rewritten)

Line numbers refer to commit 04b731a4. Each item gives the current wording and the proposed
replacement in plain terms; exact LaTeX is drafted once George approves the direction.

1. **Abstract, lines 67–71.** "at the wide fixed 2 mrad acceptance it drives the segment false
   rate from 20.5 % (classical, T = 1000) to 0.9 % at 92 % efficiency". Both numbers are
   fixed-threshold readings. Proposed: "at the wide fixed 2 mrad acceptance, keeping 99 % of
   true segments, a degree-40 polynomial refit to the events reads a 2.8 % segment false rate at
   T = 1000 tracks where the classical solver reads 10.0 %". The formula-acceptance clause
   ("solved outright") is unaffected.
2. **Introduction, lines 148–149.** "the classical solver's false rate has climbed to 20.5 %"
   → 10.0 %.
3. **Section 2.3, lines 392–412 (equation for τ).** Presents the fixed threshold as "the
   classical convention, kept for comparability" and efficiency-first as the convention for
   filtered solvers only. Proposed: one working point for every solver, defined once; the fixed
   cut mentioned only as an earlier convention that was found to sit inside the filtered true
   band and is not used. Whether the equation stays as history or goes: George's call.
4. **Section 2.4, lines 426–436 (metric definitions).** Remove "the fixed threshold of
   eq. (tau)" from the list of working points. One definition: the lowest threshold that keeps
   the wanted fraction of true segments — 99 % unless stated; sections 8–9 use the same rule at
   other fractions (the "matched-efficiency" wording can stay as the name for that variant).
5. **Fig. 1 caption, line 386** ("with the fixed threshold τ = 0.35 marked") and the figure
   file itself (a hand-copied notebook figure with the τ line drawn). Proposed: regenerate the
   atlas with the lowest true level marked instead (4/11 classical; 0.258 filtered) — that line
   is the efficiency-first threshold of the analytic picture. Same for fig. 5 (chain ladder,
   caption lines 637–642).
6. **Section 3.2, lines 583–609.** The "≈ 0.73–0.75 plateau" is the bug described as a
   property of the filter. Keep the endpoint-halving physics (real, closed form); rewrite the
   plateau as the reason the fixed cut was wrong, and state that at the efficiency-first point
   the 1BQF keeps 99.5–100 % at every T. Drop "the fixed τ = 0.35 of eq. (tau) is retained only
   where the classical baseline is quoted on its own established operating point".
7. **Fig. 4 caption, lines 620–630.** Drop "The grey vertical is the inherited fixed τ = 0.35"
   and "The fixed cut sits above the 1BQF's reshaped endpoint band, the origin of its 75 %
   plateau" (the line is no longer on the figure).
8. **Section 5.1, line 803.** "At fixed threshold its false rate reached tens of percent by
   T = 400" (the band design) → re-read at the efficiency-first point or drop the number.
9. **Section 5.2, lines 826–853 and the fig. 6 caption, lines 855–868.** Every number is a
   fixed-threshold reading (classical 2.25 % / 20.5 %; QSVT 0.02 % / 0.17 % / 0.9 % at 92.1 %;
   1BQF plateau). Replace with the table in section 3 above. The 1BQF bullet's parenthesis
   about the plateau goes.
10. **Section 5.2, lines 870–893 (crossover paragraph and footnote).** The quoted
    "1.3 % vs 3.1 %" and "≈ 21 % vs 17.5 %, crossover near T = 700" have no locatable source;
    the benchmark campaign reads: ahead through T = 300, tie at 400 (1.44 vs 1.49 %), behind
    from 550 (9.1 vs 4.9 at 700; 20.7 vs 10.0 at 1000). The sentence "The two regimes,
    fixed-τ where the comb wins by an order of magnitude ... bracket the operational truth"
    must go: there is no fixed-τ regime. The paragraph's real content becomes: the clean-event
    coefficients stop being right when tangles appear; refitting the same degree-40 polynomial
    to the events restores the win (2.8 % vs 10.0 % at T = 1000), which is the subject of
    section 9 and the companion figure.
11. **Fig. 20 caption, lines 1893–1903.** "classical (fixed τ)" → "classical" (now at the
    efficiency-first point); "the fitted d = 40 response on base A, +occupancy, +fork, and
    both" → "the degree-40 QSVT on base A, A + occupancy, A + bifurcation, and both";
    "all five series" → "the classical solver, the 1BQF, and the QSVT on four Hamiltonians".
12. **Fig. 23 caption, line 2360** ("dotted line is the fixed threshold"): the Run-3 figure is
    an imported PNG drawn at the fixed threshold only. Regenerate at the efficiency-first point
    (source script under `QSVT/Run3/`; cost to be checked in the figure audit).
13. **Conclusion, lines 2380–2389.** "from a 20.5 % false rate (the classical solver at its
    fixed threshold) to 0.9 % (the comb at the same threshold, at 92 % segment efficiency): a
    factor twenty fewer fakes" → the efficiency-first statement of item 1 (a factor 3.6 at
    T = 1000 with the refit; the benchmark coefficients alone do not win at that density).
14. **Section 7.1, lines 1140–1141 and 1160–1162 (unchanged from the earlier plan).** "the
    threshold tracks it" and "the whole true-track atlas moves as one block" are wrong on
    events: the occupancy term is a uniform rescale (x = 0.1087 · x_base at α = 0.05, T = 200,
    to 0.2 %), the ranking is what changes (false rate at 99 % efficiency: clean 0.29 % both;
    moderate 34.5 % vs 44.4 %; heavy 29.9 % vs 64.8 %). Proposed paragraph drafted in the
    previous version of this plan; every matched-point result in sections 9–10 stands.

## 5. How the paper describes the QSVT (naming check)

Counts in `main.tex`: "the comb" 31, "line comb" 8, "fitted response" 16, "refitted" 20,
"QSVT" 26. The structure is already right — section 5 is "Fitting the polynomial to the clean
events: the line comb", section 9 is "Fitting the polynomial to noisy events" — but the
results text treats "the comb" and "the fitted response" as different solvers (section 10.4
"the five solvers"; the table rows; the resource section title "HHL, the 1BQF, and the comb
side by side"). Proposed:

- One naming sentence at the start of section 5: the QSVT solver — the comb, so called because
  it combs the false tracks out of the spectrum — is one solver: a bounded polynomial p of
  degree d applied to the Hamiltonian A. Every QSVT result is labelled by d and by A. How the
  coefficients of p are chosen (the closed-form line design of section 5 for clean events, the
  least-squares fit of section 9 for measured events) is a method step, stated where it
  matters, not a solver name.
- Replace "the fitted response" / "the refitted response" in results text by "the QSVT, degree
  d, coefficients fit to the events" (or just "the QSVT" once the section has said so);
  "the comb" may stay as the solver's name.
- Section 10.4: "the five solvers" → "three solvers, the QSVT on four Hamiltonians".
- Section 12.5 title: "HHL, the 1BQF, and the QSVT side by side".

## 6. The new figure in the paper

`xiv_fig6_efficiency_first.pdf` (every solver at the efficiency-first point, benchmark campaign
plus the four refits). Recommended place: section 10.4, directly after fig. 20 — the four
Hamiltonians are only defined by then, and its point is the acceptance contrast with fig. 20
(same solvers, same working point, 2 mrad vs the formula acceptance). Fig. 6 stays in
section 5.2 with its three benchmark solvers, now at the same working point. Caption drafted:

> The clean benchmark of fig. 6 (fixed ε = 2 mrad, γ = 3, no hit drop) with every solver at
> its efficiency-first working point (99 % of true segments kept; panel (a) shows the
> convention holding): the classical solver, the 1BQF, and the degree-40 QSVT with the
> clean-event line-comb coefficients (benchmark campaign, mean ± s.e.m. over 3–20
> repetitions) together with the same degree-40 QSVT refit on base A, A + occupancy
> (α = 0.05), A + bifurcation (β = 0.5) and both (repetitions 0–2 of the same events). The
> line-comb coefficients stop winning once the 2 mrad events develop tangles (T ≥ 400); the
> refit restores the margin, 2.8 % against the classical 10.0 % at T = 1000, on every
> Hamiltonian. A measured 0 % false rate is drawn at the 10⁻³ floor.

Mechanics once approved: copy the PDF into the paper's figures directory; add the figure
environment after fig. 20 and a cross-reference sentence in section 10.4; apply the passage
edits of section 4; recompile (expect 39 pp, no unresolved references); update the figure
ledger; commit; append the worklog to the Notion to-do; patch the occupancy statements in the
two write-ups (theory addendum, QSVT XIV).

## 7. Questions for George (each changes what gets written)

1. **Where the new figure goes.** Section 10.4 beside fig. 20 (recommended), or section 5.2
   next to fig. 6, or in place of fig. 6.
2. **The headline claim.** Replace "20.5 % → 0.9 % at 92 % efficiency" in the abstract,
   introduction and conclusion with the efficiency-first statement (classical 10.0 %, QSVT
   refit 2.8 %, at T = 1000, 99 % of true segments kept)? Recommended yes — the old headline
   was a fixed-threshold artefact, and the benchmark coefficients alone read 20.7 % at that
   density.
3. **Which degree-40 base-A series is "the QSVT" in the paper.** Two series share degree and
   Hamiltonian and differ only in how the coefficients were chosen: the clean-event line-comb
   design (20.7 % at T = 1000) and the least-squares refit (2.8 %). Recommended: the refit is
   the QSVT everywhere results are compared; the line-comb design is shown in section 5 as the
   clean-event closed form with the statement that its coefficients must be refit once tangles
   appear. If George prefers, the label can carry a third attribute ("coefficients: line design
   / fit").
4. **The 1BQF's threshold.** Keep the same rule as every other solver (lowest threshold keeping
   99 % of true segments; in classical units ≈ 0.26 for the 1BQF), or read the 1BQF at a
   threshold just above zero (every non-zero segment counts, so all coupled false segments
   count)? The second raises the 1BQF false rates. Recommended: the same rule for every solver.
5. **99 % or 100 %.** "Admits all of the solutions" read literally is 100 % of true segments;
   under hit drop a true segment whose chain is broken can read exactly zero, so 100 % is not
   always attainable and the 99 % quantile has been the headline rule since 2026-06-14.
   Recommended: 99 % unless stated.
6. **Section 7.1 rewrite** (the mean-field correction): yes / later.
7. **The analytic figures 1 and 5**: regenerate with the lowest true level marked instead of
   the fixed cut: yes / drop the line entirely.
8. **The Run-3 figure 23**: regenerate at the efficiency-first point (cost to be checked), or
   drop it until then.
