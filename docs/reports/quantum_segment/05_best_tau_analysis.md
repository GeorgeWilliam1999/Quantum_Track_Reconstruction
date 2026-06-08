# Best-τ analysis — absolute vs relative threshold convention

<!-- STATUS: final -->
<!-- SOURCES: outputs/quantum_segment_analysis/seg14e_T1000_statevector/pickles/*.pkl,
              build_best_tau_figs.py, tau_sweep_{abs,rel}.csv, best_tau_rel_{f1,eff,dom}.csv -->

## Headline finding

The pre-compaction reports described the τ-sweep using the *absolute* threshold notation
(`sol_Q_scaled > τ`), but the numbers actually came from the notebook's §7e sweep, which
uses a *relative* threshold (`sol_Q_scaled > τ · qmax`).  These are NOT the same: the
quantum amplitudes scale with track multiplicity (`qmax` grows from 0.66 at T=2 to 10.99
at T=1000), so the operational default `sol_Q_scaled > 0.35` is at high T equivalent to
`> 0.35 / qmax` of the maximum, i.e. ≈3% of `qmax` — essentially "keep almost everything".

| T    | mean qmax | τ_abs = 0.35 ↔ τ_rel | absolute FR (%) | relative FR @ τ=0.40 (%) |
|------|-----------|----------------------|-----------------|--------------------------|
| 2    | 0.660     | 0.530                | 0.00            | 0.00                     |
| 10   | 1.171     | 0.299                | 0.00            | 0.00                     |
| 100  | 4.067     | 0.086                | 4.64            | 4.80                     |
| 500  | 9.000     | 0.039                | 54.53           | 1.29                     |
| 1000 | 10.985    | 0.032                | 82.57           | 0.00                     |

[source: tau_sweep_abs.csv, tau_sweep_rel.csv]

This re-frames the entire τ-sweep story:
* The "broken" operational default isn't the value `0.35` itself — it's the **choice of
  absolute scaling** on an amplitude vector whose dynamic range grows ∝T.
* The "fix" the pre-compaction report described (τ=0.40 → FR collapses at high T) is a
  *relative* threshold change, not an absolute one. The corresponding absolute thresholds
  at T=500 and T=1000 are τ_abs = 0.40 × 9.0 = 3.6 and 0.40 × 11.0 = 4.4 respectively.

## Optimal relative τ by criterion (E1 statevector)

Criterion (a) — max F₁ of segment efficiency and purity:

| T    | τ*  | eff (%) | FR (%)  | F₁    |
|------|-----|---------|---------|-------|
| 2    | 0.05| 97.5    | 0.00    | 0.987 |
| 10   | 0.05| 98.3    | 0.00    | 0.992 |
| 50   | 0.05| 98.2    | 1.55    | 0.983 |
| 100  | 0.05| 98.0    | 4.64    | 0.967 |
| 200  | 0.05| 97.9    | 15.58   | 0.907 |
| 500  | 0.40| 68.4    | 1.29    | 0.808 |
| 1000 | 0.30| 78.2    | 13.12   | 0.823 |

[source: best_tau_rel_f1.csv]

The F₁ optimum stays at the lowest threshold for T≤200 because purity is already high
there.  Above T=200 the false-positive flood becomes large enough that F₁ rewards
clipping the threshold — but the optimum is *not* a constant: it is τ*=0.40 at T=500 and
τ*=0.30 at T=1000.  The default τ_rel=0.35 sits awkwardly between the two regimes.

Criterion (b) — minimise FR subject to eff ≥ 0.90:

| T    | τ*  | eff (%) | FR (%) |
|------|-----|---------|--------|
| ≤200 | 0.05| ≥97.9   | ≤15.58 |
| 500  | 0.05| 98.0    | 54.5   |
| 1000 | 0.05| 97.7    | 82.6   |

[source: best_tau_rel_eff.csv]

For T≥500 no τ on the [0.05, 0.95] grid keeps eff ≥ 0.90 while reducing FR — the
constraint is infeasible because at T=500 even τ_rel = 0.05 only gives FR=54%, and any
larger τ trades eff for purity in steps of ~5%.

Criterion (c) — FR ≤ classical FR, then maximise eff:

| T    | τ*    | eff (%) | FR (%) | classical FR (%) |
|------|-------|---------|--------|------------------|
| ≤10  | 0.05  | ≥95.7   | 0.0    | 0.0              |
| 20   | 0.525 | 71.1    | 0.0    | 0.0              |
| 50   | 0.525 | 68.1    | 0.0    | 0.0              |
| 100  | 0.525 | 58.1    | 0.0    | 0.0              |
| 200  | 0.525 | 48.3    | 0.0    | 0.0              |
| 500  | 0.40  | 68.4    | 1.3    | 4.7              |
| 1000 | 0.30  | 78.2    | 13.1   | 23.1             |

[source: best_tau_rel_dom.csv]

At T=500 and T=1000 the quantum side actually dominates classical FR while keeping
respectable efficiency (68% and 78%).  At intermediate T (20–200) the classical FR is
exactly zero, so "dominating" forces τ high enough that quantum eff drops to ~50–70%.

## Reconciliation with previous reports

* The (T, τ) pivot tables in `03_results_seg14e_tau.md` are correct, but the column header
  was implicit about scale.  They are **relative-τ** numbers.
* The qualitative statement "increasing τ collapses the high-T FR" is correct only on the
  relative scale.  On the absolute scale you need to raise τ by an order of magnitude
  (1 → 3–4) to see any FR reduction at all.
* The §14e operating point τ=0.35 is *absolute* (set inside `run_event.py`).  In the
  notation of fig. 7 this is τ_rel ≈ 0.04 at T=1000, deep in the "keep everything" regime.

The operational implication is clear: replace the fixed absolute threshold with the
scale-aware rule `sol_Q_scaled > τ · max(sol_Q_scaled)` with τ ≈ 0.4 at T≈500 or
τ ≈ 0.3 at T≈1000.

<!-- TODO: independently verify by re-running run_event.py with the relative-threshold rule
     and checking the aggregate.csv matches the per-event sweep -->
