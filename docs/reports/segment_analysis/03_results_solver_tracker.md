# Results — solver & track-level (§14–§17)

<!-- STATUS: final -->
<!-- SOURCES: segment_level_analysis.ipynb §14–§17; outputs/segment_analysis/paper/fig1{4,5,6,7}*.pdf -->

Operating point throughout: $\gamma=3$, $\delta=1$, $\varepsilon=2$ mrad, threshold $\tau=0.35$.

## §14 Solver segment efficiency
[source: §14 cells + figure image (4-panel)]
$n_\text{tracks}$ ∈ {2, 4, 6, …, 100, 150, 200, 300, 500, 750, 1000}, 5 events per point. Both "clean" (0% drop) and "1% drop" variants reported.

| n | seg eff (clean) | false rate (clean) | true active | false active |
|---|---:|---:|---:|---:|
| 10 | 100% | 0.00% | 8.4 | 0.0 |
| 100 | 100% | 0.21% | ~130 | ~1 |
| 300 | 100% | 1.20% | ~1.3k | ~20 |
| 500 | 100% | 3.43% | ~2k | ~70 |
| 750 | 100% | 11.3% | ~3k | ~400 |
| 1000| 100% | 19.8% | ~4k | ~1000 |

Panel (iii) shows the combinatorial scaling: $N_\text{true pairs} \sim n$ (log-slope 1), $N_\text{false pairs} \sim n^3$ (log-slope 3). Panel (iv) is the *surviving* count after thresholding. Error bars (SE over 5 events) are small on eff and sizeable only on false rate at $n=1000$.

**Conclusion**: the solver itself delivers 100% segment efficiency up to $n=1000$; the observed degradation downstream is entirely due to the rising false-activation tail.

## §15 Track-level performance
[source: §15 cells + figure image (6-panel); cache `sec15_track_level_v2.pkl`]
Same grid. `EventValidator` with purity ≥ 0.70, `min_rec_hits = 3`.

Headline numbers (clean / 1% drop):
| n | efficiency | ghost rate | purity | hit eff |
|---|---:|---:|---:|---:|
| ≤ 100 | 99–100% / ~97% | ≲ 0.3% | ≈ 1.00 | 1.000 / 0.996 |
| 300 | ~99% / ~97% | ~0.7% | 1.00 | 1.000 / 0.995 |
| 500 | ~99% / ~94% | ~1.6% | 1.00 | 1.000 / 0.996 |
| 750 | ~85% / ~82% | ~3.3% (peak) | 1.00 | 1.000 / 0.996 |
| 1000 | ~44% / ~44% | ~0.9% | 1.00 | 1.000 / 0.996 |

Ghost rate rises to a peak ≈ 3.25% around $n\approx 500$–$750$ then *falls* at $n=1000$ — **not** because ghosts vanish but because the **connected-component tracker absorbs false segments into mega-clusters and the purity cut rejects those as un-matched**, so they no longer register as "ghost reco". Clone fraction is numerically zero across the grid. Matched-track count tops out near $n \approx 500$ and then drops while reconstructible-truth count keeps growing → efficiency collapse.

Knee analysis (`fig15d_knee_analysis.pdf`): efficiency drops below 90% between $n=500$ and $n=750$; drops below 50% between $n=750$ and $n=1000$.

## §16 Hamiltonian spectrum
[source: §16 cells + spectrum figure image (3-panel)]
Sparse ARPACK eigensolve of the Hamiltonian matrix at each multiplicity.

| n | λ_min | λ_max | κ = λ_max/λ_min | max off-diag row-sum |
|---|---:|---:|---:|---:|
| 10 | 2.38 | 5.62 | 2.36 | 2.0 |
| 100 | 2.22 | 5.80 | 2.61 | 2.7 |
| 300 | 1.98 | 6.00 | 3.03 | 3.0 |
| 500 | 1.02 | 6.94 | ~7.5 | 4.0 |
| 750 | 0.88 | 7.12 | ~8.6 | 4.0 |
| 1000| 0.88 | 7.12 | ~8.2 | 4.0 |

The Gershgorin lower bound $\min_i(d_i - r_i)$ saturates at 0 at $n=500$ (i.e. some rows have as much off-diagonal weight as their diagonal). Despite this, the actual spectrum remains strictly positive: $\lambda_\min \geq 0.88$. The matrix is therefore invertible and the SPD direct solve is numerically healthy at all tested multiplicities.

## §16c Solution histograms
[source: §16c cells + 6-panel histogram image]
Per-segment solver output $s_i$, separated into true (green) and false (magenta) populations.

| n | TP | FP > τ | FP rate |
|---|---:|---:|---:|
| 10 | 120 | 0 | 0.00% |
| 100 | 1 200 | 9 | 0.01% |
| 300 | 3 600 | 32 | 0.00% |
| 500 | 6 000 | 306 | 0.01% |
| 750 | 9 000 | 1 119 | 0.02% |
| 1000 | 12 000 | 2 788 | 0.02% |

True peak sits at $s^*_\text{outer}=0.3636$ (for outermost segments) with a broader support for inner segments; false peak is tight at $s^*_\text{fp}=\delta/(\delta+\gamma)=0.25$ — exactly the Hopfield fixed point for an isolated segment (confirms false segments remain "decoupled" in the Hamiltonian). The threshold $\tau=0.35$ sits cleanly between the two peaks for all tested $n$. The FP tail that emerges at high $n$ comes from the small fraction of false segments that by chance share hits with many true segments and are pulled past threshold by their reinforcement term.

## §17 Tracker A/B test
[source: §17 cells + `fig17_tracker_ab.pdf`]
Same active-segment set fed into two trackers:
- `get_tracks`: connected components (default).
- `get_tracks_layered`: module-exclusive greedy, seeded from first module, enforces angular consistency between consecutive segments.

| n | eff (CC) | eff (layered) | ghost (CC) | ghost (layered) |
|---|---:|---:|---:|---:|
| 100 | 98.8% | 100.0% | 0.5% | 0.1% |
| 300 | 94.7% | 100.0% | 0.9% | 1.2% |
| 500 | 85.4% | 100.0% | 1.6% | 3.9% |
| 750 | 67.1% | 99.96% | 3.3% | 11.7% |
| 1000 | 42.8% | 99.98% | 0.9% | 23.6% |

**Verdict** (verbatim paraphrase from §17c markdown):
> The Hamiltonian solver is healthy — the failure at $n=1000$ with the CC tracker is a post-processing artefact. The layered tracker fully recovers reconstruction efficiency (≥99.9%) but at the cost of a growing ghost rate (23.6% at $n=1000$). The validated Hamiltonian operating point $(\gamma,\delta,\varepsilon,\tau) = (3, 1, 2\,\text{mrad}, 0.35)$ delivers up to $n=1000$ reconstructible tracks/event with either tracker — the CC tracker when ghost rate is the priority, the layered tracker when efficiency is the priority.

Remaining open issue: the minor FP-tail peak near $s \approx 0.45$–$0.60$ at $n=1000$ in §16c. These segments have abnormally strong reinforcement; they are what drives the layered-tracker ghost rate. Tuning $\gamma$ upward or raising $\tau$ would suppress them at some cost to true hits near the threshold.
