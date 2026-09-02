# Review of section 1 (Introduction) — 2026-09-02

Scope: `main.tex` lines 101–208 at commit 419c3bdb, with the abstract, section 5.2, 9.4, 10.4,
11 and 12 read for cross-checks. Method: five independent reviewers (numbers re-derived from the
raw data; naming and working-point rules; claims against the body; citations via the
bibliography and alphaXiv; prose and logic), then an independent adversarial re-check of every
flagged item. 113 items examined, 53 flagged, 47 upheld after re-check, 6 refuted. Reviewers on
Opus for judgement work and Sonnet for mechanical checks. Every number below was re-derived from
the campaign store or the frontier CSVs at the efficiency-first working point (99 % of true
segments kept); the file and route are named at each item.

## A. What holds (checked, no change needed)

| Statement | Value re-derived | Source |
|---|---|---|
| Classical false rate at T = 1000, 2 mrad, 99 % efficiency: 10.0 % | 10.015 % (efficiency 100 %) | benchmark campaign, `seg_store` efficiency-first columns |
| QSVT degree 40, coefficients refit, T = 1000: 2.8 % | 2.803 % (mean of reps 0–2; efficiency 99.7 %) | `12_fig6_overlay.csv` |
| Ratio "factor 3.6" (conclusion) | 3.57 | the two above |
| "Four million segments" at T = 1000 | n_seg = 4T² = 4.00 M (3.89–4.00 M measured) | store rows |
| Five-hit chain, four eigenvalues in closed form | s − 2cos(kπ/5) = 2.382, 3.382, 4.618, 5.618 | section 2.5 |
| The comb below the classical solver "up to T ≈ 400" | win through T = 300 (0.76 vs 1.02 %), tie at 400 (1.44 vs 1.49 %), behind from 550 | campaign |
| 1BQF is the degree-one member (one zero) | consistent with the data encoding and section 9.5 | — |
| Working point as stated in the abstract | matches section 2.3/2.4 | — |
| Bibliography entries nicotra2023, hhl2009, trackhhl_epj, trackhhl, gslw2019, martyn2021 exist and match their papers | titles, authors, venues verified on alphaXiv | inline `thebibliography` |
| Nicotra et al. = linear system solved with HHL; trackhhl_epj = restricted (1-bit) phase estimation; trackhhl = single-ancilla cosine with its zero on the isolated-fake eigenvalue; QSVT cost linear in degree | verified against the papers' text | alphaXiv |

## B. Findings to act on

### B1. Statements that are wrong or violate a rule (high)

1. **"Both Denby–Peterson penalties destroy the one-bit filter" (line 165–166) is false for the fork.**
   Only the occupancy term destroys the 1BQF (false rate pinned at the 0.995 no-separation
   ceiling on every occupancy Hamiltonian at every noise level; AUC 0.996 → 0.634). The fork
   term *helps* it: 0.199 → 0.145 at matched 98 % efficiency (heavy, T = 200, `fit_comb.csv`,
   three-event means, = table `tab:fit`); on the clean benchmark at the 99 % point 0.021 → 0.014
   (T = 400), 0.085 → 0.051 (700), 0.212 → 0.121 (1000), lower in 9 of 9 repetitions
   (`12_fig6_overlay.csv`, family `onebqf_cos`). The abstract (line 85) already has this right;
   section 1 contradicts it. Verified by three routes.

2. **"Both become net wins once the polynomial is refitted" (line 167) is not true for the fork at the 99 % point.**
   At the efficiency-first working point, T = 200, medians of 10 repetitions: fork Hamiltonian
   with coefficients refit reads 0.321 at degree 40 and 0.307 at the best degree against base A
   0.314 (moderate) — a tie; and 0.720 (degree 40) / 0.653 (best) against 0.633 (heavy) — a
   regression. The fork wins only *below the fragment floor*: 0.061 → 0.046 at matched 97 %
   (heavy, T = 200). The paper's own table `tab:fit` row "either / 0.99" agrees. Same problem in
   the abstract, line 92–94 ("behaves the same way ... a net win").

3. **"Thirtyfold at moderate noise and roughly in half at heavy noise" (lines 170–172, and abstract 89–90) carries no polynomial degree, and the ratios depend on it.**
   Rule 2 says a QSVT result is labelled by degree and Hamiltonian. Occupancy α = 0.05,
   coefficients refit, T = 200, 99 % point, medians of 10 repetitions:
   - moderate: 0.314 → 0.010 at **degree 80** (30.8×); at degree 40 it is 0.314 → 0.018 (17×).
     Two files agree to six digits (`02_setout_frontier.csv`, `06_degree_scan.csv`).
   - heavy: 0.633 → 0.356 at **degree 160** (1.78×) per `04_heavy_frontier.csv`; at degree 40
     it is 0.504 (1.26×). **But** `06_degree_scan.csv` gives 0.484 (1.31×) for the same cell:
     the two drivers disagree on repetition 0 only (0.144 vs 0.497). Cause found: `eigsh` is
     called without a fixed start vector in both drivers, so ARPACK's random start perturbs the
     spectral bounds and the ridge-weight tie-break (μ = 0.139 vs 0.300 at identical efficiency
     at 1 % false rate). "Roughly in half" holds for one file and not the other. Needs
     adjudication before either number is quoted (see D1).
   Section 10.5 lines 2020–2022 quote the same 0.356.

4. **"It was demonstrated on trapped-ion and superconducting devices" (line 125) overstates the 1BQF paper.**
   arXiv:2601.07766 benchmarks against device-calibrated *noise models and emulators* (IBM
   Heron via Qiskit noise models; Quantinuum H2 via the noisy emulator) and names execution on
   physical devices as future work. Verified from the arXiv abstract and the paper text, plus
   the OneBQF code README. George is a co-author: please confirm (D3).

5. **Footnote at line 130 ("a trigonometric function realised at degree one in the Chebyshev basis of the walk operator") is mathematically wrong.**
   Degree one in that basis is T₁(X) = X, a straight line with its zero at λ = 0, not a cosine
   with its zero at λ = s. The paper itself realises the cosine as a degree-60 Chebyshev series
   (section 10, line 1726). "Degree one" in this paper counts spectral zeros.

6. **"99.98 % efficiency at a false rate below 0.1 %" (abstract line 76; section 5.2 line 900) is not an efficiency-first number.**
   It is the efficiency at a false-rate budget of 0.1 % (`eff_f001`, median 0.99975), i.e. a
   false-rate-first reading. At the 99 % rule the degree-40 refit on the formula acceptance at
   T = 1000 reads **99.85 % efficiency at zero measured false rate** (reps 99.93 / 99.85 /
   99.70, false rate 0.0000 on all three; `03_highT_frontier.csv`, `08_writeup_numbers.csv`).
   Sections 10.4 and 12 already carry 99.85 %.

7. **Roadmap: "Section 3 ... establishes the working-point conventions used throughout" (line 181–182) is stale.**
   Since the 2026-09-02 edit the conventions live in section 2.3/2.4; section 3 applies them.

### B2. Imprecise, unsupported, or incomplete (medium)

8. "Separates true from false essentially perfectly at low and moderate multiplicity" (line
   151) is unquantified. Campaign: exactly zero false actives at T = 10, 20, 30, 75, 100, one
   repetition in ten admits four at T = 50 (0.196 %), mean false rate under 0.2 % through
   T = 150, 0.31 % at T = 200 and 0.76 % at T = 300 against the classical 0.65 % and 1.02 %.
9. Item 1 (lines 152–156) says the coefficients "must be refit" at T = 1000 but never says what
   the un-refit comb reads there: 20.7 ± 0.6 % at 99.2 % efficiency, behind the classical
   solver from T = 550 (5.16 vs 2.92 %).
10. "The quantum false rate drops below the classical one" (line 161) is ambiguous between the
    two quantum solvers and false for the 1BQF on the base Hamiltonian (0.199 against the
    classical 0.124 at matched 98 %, heavy T = 200); only the comb with refit coefficients
    (0.096; 0.061 at 97 %) drops below. On the fork system the 1BQF does drop below (0.145 vs
    0.463), so the caveat is specific to base A.
11. Same sentence: "at every efficiency below a floor" is unscoped. Demonstrated at heavy noise,
    T = 200, on base A and the fork system; by T = 1000 under noise the 1 %-budget frontier has
    closed for every operator (section 10.4).
12. "The correct response cannot be written down a priori" (line 160) overstates: section 2.5
    writes the motif line positions in closed form before noise; what must be measured is which
    lines the pile-up populates and how heavily, and where the contaminated-true modes move.
13. "Inside any naive pass window" (line 159): the pair line sits 0.38 and the triple lines
    0.20 from the nearest true line. A band design passes them all; the degree-40 comb rejects
    the pair line (0.019 of its true-line response) but returns 0.27–0.54 on the triple lines,
    because 0.20 is at its resolution limit 2W/d.
14. The occupancy win is stated with only "(T = 200)" as scope; the body (sections 10.4 and
    11.1) shows the margin thins by T = 400 (1.3×) and is gone by T = 700.
15. "Denby–Peterson penalties" are used cold (line 165): no citation until line 251 and no
    gloss of what the two terms are; "fork" appears in the roadmap (line 191) undefined.
16. "The false rate" is never glossed before its first quantitative use (abstract line 70,
    intro line 154); with n_seg = 4T² against n_true = 4T an impurity and a false-positive rate
    differ by three orders of magnitude.
17. "A comb" (line 150) is not identified as the QSVT of the paper; the intro uses the solver
    name five times before section 5.1 fixes it.
18. The roadmap omits section 10 entirely (the section that houses both headline numbers the
    intro advertises), mis-describes section 6 ("introduces the noise model" — that is section
    2.1; section 6 sets the heavy operating point and diagnoses all three solvers), and lists
    six of section 11's nine subsections (missing: the density limit of the occupancy win,
    success probability and readout, classical simulability).
19. "The measured hardware-resource comparison of HHL, the 1BQF and the comb" (line 196): HHL
    is a structure-level estimate (2(2^n_c − 1) = 126–510 calls at n_c = 6–8 times the measured
    per-call cost), measured only for the 1BQF and the comb; the figure ledger flags the same
    overclaim in the figure's panel title.
20. The opening claim that the community has tested "GPUs, FPGAs, neural networks, and ...
    quantum computers" (line 109) has no citation. Three verified references: Aaij et al.,
    *Allen: a high-level trigger on GPUs for LHCb*, Comput. Softw. Big Sci. 4 (2020) 7,
    arXiv:1912.09161; Terzuoli et al., *A real-time demonstrator of track reconstruction with
    FPGAs at LHCb*, CTD 2023 proceedings, arXiv:2605.02753; Correia et al., *Graph neural
    network-based track finding in the LHCb vertex detector*, JINST 19 (2024) P12022,
    arXiv:2407.12119.

### B3. Style (low)

21. "In my opinion this last question carries most of the physics, and most of this work is
    spent answering it" (line 138): the hedge weakens the thesis sentence; by word count
    sections 6–10 are about half the paper, so "most" is defensible.
22. "We"/"I" mixing in the opening paragraph (line 105) — the rest of the paper uses "I" for
    authorial decisions and "we" for the shared narrative; fine if deliberate.
23. "The comb's coefficients must be refit to the events, and then read 2.8 %" (line 155):
    subject-verb sense slip ("coefficients ... read").
24. "Lifts the fragment floor" (line 172): the body says "breaks"/"dissolves"; "lifts" reads as
    "raises".
25. Item 3 largely duplicates the abstract; the roadmap paragraph is one unbroken 22-line
    block.
26. Bibliography line 2519: "M. H. Winands" should be "M. H. M. Winands" (the sibling entry has
    it right).

## C. Refuted after re-check (no action)

- "The occupancy density caveat is missing from the intro": the "(T = 200)" scope is present at
  line 172; the residual point is item 14 above.
- "The working point is not stated in the intro": the abstract states it; the intro's numbers
  say "at 99 % efficiency". A one-clause gloss (item 16) is enough.
- "The line comb is a third solver name" (line 186): it is a callback to vocabulary the abstract
  and item 1 already introduced; no rule violation.
- "HL-LHC platforms claim has no source" as originally filed: narrowed to item 20 (neural
  networks are well sourced later in the paper).
- "Deliberately verbose" remark: pure taste; the "we/I" argument against it does not hold.
- "Roughly in half at heavy noise should be re-stated at matched 0.98": the proposed text would
  have re-anchored the claim to a different working point; superseded by item 3 / D1.

## D. Decisions for George

- **D1. The heavy-noise occupancy number.** Two drivers disagree on one event because the
  eigenvalue solver starts from a random vector. Options: (a) fix the start vector, re-run the
  two drivers for that cell (minutes), and quote whichever median results; (b) quote the
  conservative "lowers it at heavy noise as well, from 0.633 to 0.36–0.48"; (c) drop the heavy
  ratio from the intro and abstract and keep only the moderate thirtyfold. Recommended: (a),
  and pin the start vector in every driver so this cannot recur (a to-do exists).
- **D2. State the degrees in the abstract and intro** ("degree 80" for the moderate thirtyfold,
  "degree 160" for the heavy factor), per rule 2, or move the numbers out of the intro and
  point to the figures (your stated preference for plots over point rates). Recommended: keep
  one sentence with the degrees stated; the figures carry the rest.
- **D3. The 1BQF hardware sentence** (item 4): confirm the paper's own description (noise
  models and emulators of IBM Heron and Quantinuum H2, hardware runs as future work) — you are
  a co-author and may know of a later run.
- **D4. Add the three platform citations** (item 20): yes / no.
- **D5. Roadmap rewrite** (items 7, 18, 19): the paste-ready paragraph in E below adds section
  10, fixes sections 3 and 6, and completes section 11's list.
- **D6. Section 1 style** (items 21–25): apply or leave.

## E. Paste-ready replacements (LaTeX), in order of appearance

**Abstract, line 76** (item 6): `outright ($99.98\%$ efficiency at a false rate below $0.1\%$)` →
`outright ($99.85\%$ efficiency at zero measured false rate)`. Same replacement at section 5.2
line 900, where the sentence should also say by whom: "solved outright by the same degree-40
polynomial with its coefficients refit ($99.85\%$ efficiency at zero measured false rate on
base $A$, section~\ref{sec:atscale})".

**Abstract, lines 88–91** (item 3; wording for option D1(a) once the heavy number is settled):
```
the strongest configuration in the study: at the matched $99\%$-efficiency
point ($T{=}200$) a polynomial with its coefficients refit to that spectrum
cuts the false rate thirtyfold at moderate noise ($0.31$ to $0.010$ at
degree $80$; seventeenfold at degree $40$) and roughly halves it at heavy
noise (degree $160$), and, at small coupling, it breaks
```

**Abstract, lines 92–94** (item 2):
```
The bifurcation (fork) term is likewise a regression under the unrefit
clean-event comb and likewise a net win once the response is refit to the
spectrum it creates --- though only below the fragment floor
($0.061\to0.046$ at matched efficiency $0.97$), since at the strict $99\%$
point both systems flood alike.
```

**Line 109** (item 20):
```
every alternative computing platform: GPUs~\cite{allen2020},
FPGAs~\cite{terzuoli2024}, neural networks~\cite{correia2024}, and, in
```
with three `\bibitem` entries (allen2020, terzuoli2024, correia2024) as given in item 20.

**Line 125** (item 4, pending D3):
```
hardware, and it was benchmarked against device-calibrated noise
models of superconducting and trapped-ion processors (IBM Heron and the
Quantinuum H2 emulator), with execution on the physical devices
themselves left to future work.
```

**Line 130 footnote** (item 5):
```
one\footnote{``Degree one'' counts spectral zeros, not Chebyshev order:
one application of the filter buys exactly one zero of the response
(section~\ref{sec:onebit}). The cosine is not a polynomial in $\lambda$
at all; expanded in the Chebyshev basis of the walk operator of
section~\ref{sec:qsvt} it is an infinite series whose truncation order
grows with the spectral span of $A$, and which this work realises at
degree $60$ (section~\ref{sec:atscale}). What carries over exactly is
the accounting: a degree-$d$ response owns $d$ zeros.}.
```

**Item 1 of the enumerate, lines 148–156** (items 8, 9, 17, 23):
```
\item On clean events the true-track spectrum is \emph{discrete}: detector
geometry pins every true track to the same five-hit chain, whose four
eigenvalues are known in closed form. A comb of narrow passes at those
four lines, and nothing else --- a degree-40 polynomial on the base
Hamiltonian $A$, fixed in section~\ref{sec:comb} and the one meant
wherever this work says ``the QSVT'' or ``the comb'' without
qualification --- separates true from false essentially perfectly up
to $T\approx300$ (a mean false rate under $0.2\%$ through $T=150$, and
$0.76\%$ at $T=300$ against the classical solver's $1.02\%$, every solver
read at the threshold that keeps $99\%$ of its true segments), and still
ties the classical solver at $T=400$. At the densities I can simulate
($T=1000$ tracks, four million segments) and the wide fixed $2$\,mrad
acceptance the picture reverses: the events grow tangles of false
segments whose eigenvalues fall inside those passes, and the comb reads a
false rate (the fraction of the selected segments that are fake) of
$20.7\%$ where the classical solver reads $10.0\%$. Refitting the
coefficients of the same degree-40 polynomial to the events recovers the
margin: it then reads $2.8\%$.
```

**Item 2, lines 157–163** (items 10–13):
```
\item On noisy events the false population moves: it concentrates on the
eigenvalue lines of specific small \emph{motifs} --- measurably the
hit-sharing pair line $\lambda=s-1$ and the triple lines $s\pm\sqrt2$.
These sit off the true lines, but only just: $0.38$ and $0.20$ away, the
latter no wider than the finest feature a degree-$40$ series can
resolve. A band design passes them wholesale; the comb suppresses the
pair line to a few percent of its true-line response but still returns a
quarter to a half of it on the triple lines, and noise unpins the lines
themselves. The line positions follow in closed form; which of them the
pile-up populates, and how heavily, does not, so the response must be
fitted to the measured pile-up. Once its coefficients are refit, the
comb's false rate drops below that of the classical solver on the same
operator at every efficiency below a floor that I can identify, count,
and explain segment by segment ($0.096$ against $0.124$ on base $A$ and
$0.085$ against $0.463$ on the fork system, heavy noise, $T{=}200$,
matched $98\%$ efficiency); the one-bit filter, whose single zero is
already spoken for, does not clear the classical solver on the base
Hamiltonian there ($0.199$). The margin is a low-multiplicity statement:
by $T{=}1000$ the $1\%$-budget frontier has closed for every operator
alike (section~\ref{sec:atscale:density}).
```

**Item 3, lines 164–175** (items 1, 2, 3, 14, 15, 24; heavy-noise clause pending D1):
```
\item Hamiltonian engineering and response engineering are coupled, and
neither knob can be evaluated without the other. Both Denby--Peterson
penalties~\cite{denby1988,peterson1989} --- an occupancy term, which
penalises a hit used by more than one selected segment in the same role,
and a bifurcation (fork) term, which penalises pairs of segments that
share a hit at small mutual angle --- read as regressions under the
clean-event comb applied without refitting its coefficients, and both
are rehabilitated once the coefficients are refit to the spectrum they
produce, though not equally: the occupancy term wins at the
efficiency-first $99\%$ point itself, the fork only below the fragment
floor ($0.046$ against $0.061$ at matched efficiency $0.97$). On the
one-bit filter the two terms act in opposite directions. The fork helps
it ($0.199\to0.145$ at matched $98\%$ efficiency, heavy noise,
$T{=}200$). The occupancy term is the sharpest case in the study: it
destroys the filter outright, its single zero having no way to follow
the co-hit coupling that lifts the isolated false segments off it, yet
with the polynomial's coefficients refit it cuts the false rate at the
matched $99\%$-efficiency point from $0.314$ to $0.010$ at moderate noise
(degree $80$) and from $0.633$ to $0.356$ at heavy noise (degree $160$),
both against base $A$ on the same events ($T{=}200$, $\alpha{=}0.05$),
and at small coupling it breaks the fragment floor by breaking the
spectral-twin degeneracy that causes it. Both are $T{=}200$ results and
both are, as measured, low-multiplicity: the term's spectral span grows
with the busiest hit's degree, so at fixed polynomial degree the margin
thins by $T{=}400$ and is gone by $T{=}700$
(sections~\ref{sec:atscale:density} and~\ref{sec:limits:occdensity}).
Changing the operator moves the activations, and moving the activations
changes the polynomial we have to fit.
```

**Roadmap, lines 178–199** (items 7, 18, 19):
```
This work is organised as follows. Section~\ref{sec:toy} builds the toy
model and the segment Hamiltonian from zero, including the exact
activation levels that make everything else quantitative and the
conventions used throughout: the efficiency-first working point, and the
segment efficiency, segment false rate and AUC definitions.
Section~\ref{sec:onebit} reviews the 1BQF as a spectral filter and applies
that working point to the true band its degree-one cosine reshapes, where
the earlier fixed cut fails. Section~\ref{sec:qsvt} develops the QSVT
machinery (block encoding, qubitization, and the
linear-combination-of-Chebyshev construction) at the level of detail
needed to count qubits and gates. Section~\ref{sec:comb} designs and
evaluates the comb on clean events, including its resource scaling and its
provable limits. Section~\ref{sec:noise} sets the heavy-noise operating
point and diagnoses what it does to each solver.
Section~\ref{sec:occupancy} dissects the occupancy term's collision with
the notch. Section~\ref{sec:fork} introduces the fork term and the design
rule that every operator change demands a response refit.
Section~\ref{sec:fit} fits the polynomial to the measured false pile-up
and presents the main comparison, including the refitted occupancy result.
Section~\ref{sec:atscale} then crosses the two axes systematically ---
every operator against every polynomial degree, with the coefficients
refit at each --- and reports the loss budget, the density scan to
$T=1000$ behind the clean-event numbers quoted above, and the two-axis
rule the study reduces to. Section~\ref{sec:limits} collects the
limitations: that the occupancy win does not survive density, the
realizability of a response whose coefficients have been refit, the
single-response question, Trotterization, the hardware-resource
comparison of HHL, the 1BQF and the comb --- measured for the 1BQF and the
comb, a structure-level estimate for HHL --- with the error-correction
requirements each implies, a hardware-native block encoding priced by
detector hits rather than matrix entries, the success probability and
readout cost, what the classical simulability of the response family does
and does not concede, and what a real detector geometry changes.
Section~\ref{sec:conclusion} concludes.
```

**Bibliography line 2519**: `M.~H.~Winands` → `M.~H.~M.~Winands`.

## F. Follow-ups outside section 1 found by this review

- Section 5.2 line 900 and the abstract carry the 99.98 % false-rate-first number (item 6).
- Section 10.5 lines 2020–2022 quote the disputed heavy 0.356 (D1).
- Drivers `04_heavy_frontier.py` and `06_degree_scan.py` (and any other caller of `eigsh`)
  should pass a fixed start vector so two runs of the same cell agree; then re-run the
  heavy, T = 200, occupancy, degree-160 cell in both and re-derive the median.
- The figure ledger already flags the resource figure's "measured" HHL panel title (item 19).
