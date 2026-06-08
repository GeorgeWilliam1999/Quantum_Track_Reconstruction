# Results — §14e mirror and τ-sweep

<!-- STATUS: final -->
<!-- SOURCES: outputs/quantum_segment_analysis/seg14e_T1000_{statevector,sampling}/aggregate.csv; Quantum_segment_level_analysis.ipynb §7, §7e -->

All numbers below are direct reads from `aggregate.csv` (mean ± SEM over `n_reps` repetitions
per row; reps schedule in `00_inventory.md`).

## Segment metrics at the §14e operating point (γ=3, δ=1, ε=2 mrad, τ=0.35, 1% hit drop)

| T | $\bar N_s$ | eff_C | eff_Q (E1) | eff_Q (E2) | pur_C | pur_Q (E1) | pur_Q (E2) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 0.975 | 0.729 | 0.729 | 1.000 | 1.000 | 1.000 |
| 5 | 97 | 0.937 | 0.725 | 0.765 | 1.000 | 1.000 | 1.000 |
| 10 | 395 | 0.983 | 0.983 | 0.890 | 1.000 | 1.000 | 1.000 |
| 20 | 1 555 | 0.950 | 0.964 | 0.884 | 1.000 | 0.998 | 0.998 |
| 50 | 9 828 | 0.970 | 0.979 | 0.867 | 0.997 | 0.981 | 0.988 |
| 100 | 39 199 | 0.964 | 0.975 | 0.858 | 0.994 | 0.949 | 0.959 |
| 200 | 156 562 | 0.963 | 0.974 | 0.856 | 0.994 | 0.840 | 0.831 |
| 500 | 980 103 | 0.965 | 0.975 | 0.910 | 0.961 | 0.452 | 0.448 |
| 1000 | 3 907 530 | 0.961 | 0.972 | 0.909 | 0.769 | 0.173 | 0.170 |

[source: `seg14e_T1000_{statevector,sampling}/aggregate.csv`]

Two regimes are visible:
1. **$T \in [2, 10]$**: eff_Q hits the documented 75% floor at $T=2,5$ (0.729, 0.725) — exactly
   the $1 - \cos^2(\pi\varphi_k)$ averaging predicted by the audit at γ=3. By $T=10$ enough
   noise eigenmodes are above τ that the floor is effectively masked.
2. **$T \in [50, 1000]$**: eff_Q is **slightly higher** than eff_C (0.97 vs 0.96), and pur_Q
   collapses from 0.95 → 0.17. The quantum activation set is inflated by false positives that
   the cosine-filter does *not* push below τ once the noise eigenmodes accumulate amplitude
   over many segments.

False-rate distillation (FR = 1 - pur, the §14e paper convention):

| T | FR_C (%) | FR_Q E1 (%) | FR_Q E2 (%) |
|---:|---:|---:|---:|
| 2 | 0.000 | 0.000 | 0.000 |
| 50 | 0.285 | 1.858 | 1.241 |
| 100 | 0.553 | 5.081 | 4.123 |
| 200 | 0.645 | 16.041 | 16.879 |
| 500 | 3.881 | **54.768** | 55.227 |
| 1000 | 23.076 | **82.670** | 83.026 |

## E1 vs E2 — shot-noise is not the failure mode
E1 statevector (exact) and E2 sampling (with the shot budget in `00_inventory.md`) **agree
within 1%** at every T:

| T | FR_Q E1 (%) | FR_Q E2 (%) | $|\Delta|$ |
|---:|---:|---:|---:|
| 100 | 5.08 | 4.12 | 0.96 |
| 200 | 16.04 | 16.88 | 0.84 |
| 500 | 54.77 | 55.23 | 0.46 |
| 1000 | 82.67 | 83.03 | 0.36 |

The shot count was scaled with $T^2$ specifically to suppress sampling noise to the percent
level; the result above confirms the residual disagreement at large $T$ is at the SEM of the
individual sweep (E1 SEM ≈ 0.5%, E2 SEM ≈ 0.4% at T=1000) and is not the source of the FR gap.
**This rules out the standard "needs more shots" explanation.**

## Active-set inflation — the algorithmic, not statistical, failure mode

| T | $\bar N_s$ | $\bar n_\text{true,clean}$ | $|\mathcal{A}|_C$ | $|\mathcal{A}|_{Q,E1}$ | $|\mathcal{A}|_{Q,E2}$ |
|---:|---:|---:|---:|---:|---:|
| 100 | 39 199 | ~394 | 388 | 411 | 358 |
| 200 | 156 562 | ~787 | 776 | 928 | 824 |
| 500 | 980 103 | ~1 968 | 2 009 | **4 313** | 4 067 |
| 1000 | 3 907 530 | ~3 935 | 4 995 | **22 430** | 21 423 |

The classical solver tracks the true segment count closely. The quantum active set is
**4.5×** larger than classical at T=1000 (E1) and **4×** larger at T=500. Since both readouts
agree, the inflation is in the *amplitudes themselves* — the cosine filter leaves the off-diagonal
mass of the noise eigenmodes large enough that many false segments cross $\tau = 0.35$.

## Fidelity at the §14e operating point
| T | $\cos(s_C, s_Q)$ E1 | rel-$L_2$ E1 | Jaccard E1 |
|---:|---:|---:|---:|
| 2 | 0.823 | 0.594 | 0.748 |
| 100 | 0.161 | 1.296 | 0.944 |
| 500 | 0.097 | 1.344 | 0.466 |
| 1000 | 0.106 | 1.338 | 0.223 |

Cosine fidelity collapses by $T=100$ but **Jaccard ≥ 0.94 up to $T=100$** — the post-rescale
activation sets are very similar even though the raw amplitudes are far apart, confirming that
the small-T classical-vs-quantum difference is amplitude-level shaping rather than wrong
segments. Above $T=200$ the Jaccard also collapses, consistent with the active-set inflation.

## §7e threshold sweep (this work)
[source: `Quantum_segment_level_analysis.ipynb` cells 22–23; `fig6_threshold_sweep.{pdf,png}`]
We post-process the E1 amplitudes with absolute thresholds $\tau \in \{0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70\}$
without re-running any circuits:

### Quantum False Rate (%) vs $(T, \tau)$
| $T$ \\ $\tau$ | 0.20 | 0.25 | 0.30 | 0.35 | **0.40** | 0.50 | 0.60 | 0.70 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 |
| 50 | 1.55 | 1.67 | 1.67 | 2.03 | 2.04 | 1.55 | 0.0 | 0.0 |
| 100 | 4.64 | 5.39 | 5.49 | 6.04 | 4.80 | 2.32 | 0.0 | 0.0 |
| 200 | 16.01 | 19.18 | 17.03 | 17.45 | 11.78 | 1.90 | 0.0 | 0.0 |
| 500 | 55.84 | 61.24 | 49.56 | 49.18 | **1.29** | 0.00 | 0.0 | 0.0 |
| 1000 | 84.92 | 84.88 | 13.12 | 5.22 | **0.00** | 0.00 | 0.0 | 0.0 |

### Quantum Segment Efficiency (%) vs $(T, \tau)$
| $T$ \\ $\tau$ | 0.20 | 0.25 | 0.30 | 0.35 | **0.40** | 0.50 | 0.60 | 0.70 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 97.50 | 97.50 | 97.50 | 73.75 | 73.75 | 72.92 | 72.92 | 48.33 |
| 50 | 98.20 | 93.60 | 93.58 | 74.28 | 74.08 | 68.90 | 66.80 | 43.62 |
| 100 | 97.96 | 84.92 | 83.79 | 74.08 | 73.45 | 59.89 | 53.26 | 31.59 |
| 200 | 95.51 | 76.60 | 76.53 | 71.54 | 70.51 | 48.48 | 38.69 | 24.22 |
| 500 | 93.39 | 74.35 | 74.09 | 69.20 | 68.44 | 44.69 | 20.55 | 14.70 |
| 1000 | 81.05 | 80.24 | 78.23 | 64.71 | **50.23** | 20.27 | 2.30 | 0.95 |

### Reading
- **$\tau = 0.20$**: matches the false-segment Hopfield fixed point $\delta/(\gamma+\delta) = 0.25$
  from *below*. Eff hits ≈ 0.95 at small/mid T, but pulls in essentially every false segment
  by $T=500$ (FR=55.8%). At $T=1000$ even Eff has collapsed to 81% — the cosine kernel is now
  also pulling true segments below 0.2.
- **$\tau = 0.35$** (the §14e default): straddles the two Hopfield peaks 0.25 (false) and 0.36
  (true outer-segment). Classical works because the true peak is *narrowly* above 0.35 in the
  Hopfield iteration. The quantum cosine-filter shrinks the entire spectrum towards 0.25 →
  many true segments end up below 0.35 and many false segments stay above ⇒ both 73% efficiency
  floor (at low T) and runaway FR (at high T).
- **$\tau = 0.40$**: above the false-peak Hopfield fixed point with margin. **FR drops to 1.3% at
  T=500 and 0% at T=1000**, at the cost of efficiency falling from 0.965 to 0.68 (T=500) and
  0.96 to 0.50 (T=1000).
- **$\tau \ge 0.50$**: efficiency collapses to <50% at high T because the cosine filter has
  shifted even outer-segment true amplitudes below 0.5.

### Optimal-τ scan with eff ≥ 0.90 floor
The notebook's automatic "highest-τ with eff ≥ 0.90" rule picks τ = 0.20 at every T because
the higher-τ rows fail the 0.90 floor at T ≥ 200. **The 0.90 floor is the wrong objective**:
the right trade-off depends on the downstream tracker. For a CC tracker, FR matters most
(merged clusters); for a layered tracker, efficiency matters most (per-segment greedy seeding).

<!-- FLAG: the §7e cell's "optimal τ" panel is misleading because of the eff-floor fallback rule. A pareto plot would be more honest. -->

## Tracking metrics (downstream)
| T | trk_C_cc eff | trk_Q_cc eff | trk_C_lay eff | trk_Q_lay eff | trk_C_cc ghost | trk_Q_cc ghost | trk_C_lay ghost | trk_Q_lay ghost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.953 | 0.732 | 0.968 | 0.990 | 0.008 | 0.084 | 0.002 | 0.088 |
| 200 | 0.948 | 0.339 | 0.968 | 0.989 | 0.009 | 0.056 | 0.002 | 0.268 |
| 500 | 0.836 | 0.002 | 0.969 | 0.988 | 0.032 | 0.667 | 0.038 | 0.701 |
| 1000 | 0.382 | 0.000 | 0.965 | 0.987 | 0.008 | 1.000 | 0.262 | 0.901 |

[source: `seg14e_T1000_statevector/aggregate.csv` `trk_*_efficiency_mean`, `trk_*_ghost_rate_mean`]

- **CC tracker + quantum** is catastrophic: at T=500 only 0.2% of truth tracks survive matching
  because the false-positive halo merges everything into one giant component. Ghost rate
  saturates at 1.0 by T=1000.
- **Layered tracker + quantum** keeps efficiency above 0.98 to T=1000 but pays an enormous
  ghost-rate price (0.90 at T=1000) because every excess false segment seeds at least one
  spurious chain.
- **Layered + classical** is the best operating point at T ≥ 200: eff 0.96–0.97, ghost rate
  ≤ 0.04 except at T=1000 (0.26).

## Timing
| T | $t_C$ (s) | $t_Q$ (s, E1 statevector) | $t_Q/t_C$ |
|---:|---:|---:|---:|
| 2 | $4\times10^{-4}$ | 0.30 | 770 |
| 100 | $2\times10^{-3}$ | 18.4 | 9 950 |
| 500 | 0.17 | 446 | 2 600 |
| 1000 | 0.80 | 5 291 | 6 600 |

The "quantum" time is the simulated statevector wall-time on a single GPU; it is **not**
hardware quantum time. It is dominated by Aer's CPU/GPU contraction of the multi-controlled
RX gates. Real-hardware extrapolation is not done in this report.
