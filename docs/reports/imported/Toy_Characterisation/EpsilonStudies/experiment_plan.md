# Epsilon Parameter Optimisation Study

## Objective

Determine the optimal values of the Hamiltonian control parameters — **scale factor $k$**,
**ERF smoothing width $\theta_d$**, and their interaction with detector physics conditions
(**measurement error $\sigma_{\text{res}}$**, **multiple scattering $\sigma_{\text{scatt}}$**) —
that maximise reconstruction efficiency while minimising the ghost rate.

The angular acceptance threshold $\varepsilon$ is derived from the paper formula:

$$
\varepsilon = \sqrt{2(k \cdot \sigma_s)^2 + 12 \arctan\!\left(\frac{k \cdot \sigma_{\text{res}}}{\Delta z}\right)^{\!2} + 2\,\theta_{\min}^2}
$$

where $\theta_{\min} = 1.5 \times 10^{-5}\;\text{rad}$ is a constant floor.

---

## Background

Previous studies in this workspace have characterised the toy detector at fixed
Hamiltonian parameters ($k = 3$, step-function acceptance, $\sigma_{\text{res}} = 0$,
$\sigma_{\text{scatt}} = 10^{-4}\;\text{rad}$). Key findings motivating this sweep:

| Study | Finding |
|-------|---------|
| **Baseline** | 100% efficiency at low density with default parameters |
| **Hit competition** | Performance degrades at high track density ($n > 75$) for narrow cones |
| **Recovery & separation** | Near-coincident track degeneracy causes recovery dips; separation collapses to $\sim 0.06\sigma$ |
| **Segment grass** | Geometric clipping at $\pm 0.2\;\text{rad}$ affects 38.5% of events |
| **OLD_TOY_CODE sweeps** | Earlier grid over $k \times \theta_d \times$ density $\times \sigma_{\text{res}}$ explored the parameter space but used a different codebase |

The current study uses the `lhcb_velo_toy` v2.0.0 package with `SimpleHamiltonianFast`
and **ERF-smoothed acceptance exclusively** (`convolution=True`). The step-function mode
is not used; previous studies showed ERF provides a strictly better efficiency–ghost tradeoff.

---

## Swept Parameters

### Hamiltonian hyperparameters (primary axes)

| Parameter | Symbol | Values | Role |
|-----------|--------|--------|------|
| Scale factor | $k$ | 1, 2, 3, 5, 10, 20 | Multiplier in $\varepsilon$ formula; controls width of angular acceptance |
| ERF smoothing width | $\theta_d$ | $10^{-4}$, $10^{-3}$, $10^{-2}$, $10^{-1}\;\text{rad}$ | Standard deviation of the error-function transition |
| Convolution mode | — | **ERF only** | Soft acceptance via $1 + \text{erf}\!\left(\frac{\varepsilon - \alpha}{\theta_d \sqrt{2}}\right)$; `convolution=True` for all jobs |

### Physics conditions (secondary axes)

| Parameter | Symbol | Values | Role |
|-----------|--------|--------|------|
| Measurement error | $\sigma_{\text{res}}$ | 0.0, 0.01, 0.05 mm | Hit position smearing (Gaussian) |
| Collision noise | $\sigma_{\text{scatt}}$ | $5 \times 10^{-5}$, $10^{-4}$, $4 \times 10^{-4}\;\text{rad}$ | Multiple scattering angle per module |

### Evaluation conditions (fixed per job, varied across jobs)

| Parameter | Values | Role |
|-----------|--------|------|
| Angular cone $\phi_{\max} = \theta_{\max}$ | $\pm 0.2$, $\pm 0.1$, $\pm 0.04\;\text{rad}$ | Track generation opening angle |
| Track multiplicity $n_{\text{tracks}}$ | 10, 50, 100 | Number of particles per event |
| Independent repeats | 3 | Events generated per parameter combination |

---

## Fixed Parameters

### Hamiltonian

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Self-interaction penalty | $\gamma$ | 1.5 | `helpers.py:GAMMA` |
| Bias weight | $\delta$ | 1.0 | `helpers.py:DELTA` |
| Baseline | $b$ | 0.4 | $= \delta / (\delta + \gamma)$ |
| Activation threshold | $\tau$ | 0.7 | $= (1 + b) / 2 = (1 + \delta/(\delta + \gamma))/2$ |
| Angular floor | $\theta_{\min}$ | $1.5 \times 10^{-5}\;\text{rad}$ | Constant in `compute_epsilon()` |

### Detector geometry

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Number of modules | $N$ | 5 | `helpers.py:N_MODULES` |
| Module spacing | $\Delta z$ | 33 mm | `helpers.py:DZ_MM` |
| First module $z$-position | $z_0$ | 100 mm | `helpers.py:Z_FIRST` |
| Module half-width $x$ | $L_x$ | 50 mm | `make_geometry(half_x=50)` |
| Module half-width $y$ | $L_y$ | 50 mm | `make_geometry(half_y=50)` |
| Module $z$-positions | — | [100, 133, 166, 199, 232] mm | $z_0 + i \cdot \Delta z$ |

### Event generation

| Parameter | Value | Source |
|-----------|-------|--------|
| Primary vertex spread $(\sigma_x, \sigma_y, \sigma_z)$ | (0.1, 0.1, 50.0) mm | Hardcoded in `generate_event()` |
| Particle type | pion ($m = 139.6\;\text{MeV}/c^2$, $q = +1$) | Hardcoded in `generate_event()` |
| Angular range | symmetric: $\phi \in [-\phi_{\max}, +\phi_{\max}]$, $\theta \in [-\theta_{\max}, +\theta_{\max}]$ | Swept (see above) |

> **Note on parameter choices:** The `segment_level_analysis.ipynb` validation notebook
> uses different values ($\gamma = 3.0$, $z_0 = 20\;\text{mm}$, angles $= [0.02, 0.2]$)
> matching an alternative configuration (Xeno). The condor pipeline values above match
> the OneBQF paper defaults.

---

## Job Counts

All jobs use ERF convolution (`convolution=True`):

$$
N_{\text{total}} = \underbrace{6}_{k} \times \underbrace{4}_{\theta_d} \times \underbrace{3}_{\sigma_{\text{res}}} \times \underbrace{3}_{\sigma_{\text{scatt}}} \times \underbrace{3}_{\text{angle}} \times \underbrace{3}_{n_{\text{tracks}}} = \boxed{1944 \text{ jobs}}
$$

Each job runs 3 independent events internally, giving **5,832** event-level measurements.

---

## Metrics

Each job produces per-event reconstruction metrics via `EventValidator.compute_metrics()`,
aggregated across repeats as mean $\pm$ standard error. The full LHCb-style metric set
is captured, following the conventions of the OneBQF paper and LHCb TrackChecker.

### Track-level metrics

| Metric | Key in CSV | Definition | Target |
|--------|-----------|-----------|--------|
| **Reconstruction efficiency** | `m_reconstruction_efficiency` | $N_{\text{matched truth}} / N_{\text{reconstructible truth}}$ | Maximise |
| **Ghost rate** | `m_ghost_rate` | $N_{\text{ghost}} / N_{\text{candidates}}$ (purity $< 0.7$, excl. clones) | Minimise |
| **Clone fraction (total)** | `m_clone_fraction_total` | $N_{\text{clones}} / N_{\text{candidates}}$ | Minimise |
| **Clone fraction (among matched)** | `m_clone_fraction_among_matched` | $N_{\text{clones}} / (N_{\text{good}} + N_{\text{clones}})$ | Minimise |

### Hit-level quality metrics (over GOOD primary tracks)

| Metric | Key in CSV | Definition | Target |
|--------|-----------|-----------|--------|
| **Hit purity** | `m_purity_all_matched` / `hit_purity_mean_primary` | $\langle |R_i \cap T_j| / |R_i| \rangle$ — fraction of reco hits from the matched truth | Maximise |
| **Hit efficiency (mean)** | `m_hit_efficiency_mean` / `hit_efficiency_mean_primary` | $\langle |R_i \cap T_j| / |T_j| \rangle$ — completeness of truth track recovery | Maximise |
| **Hit efficiency (weighted)** | `m_hit_efficiency_weighted` / `hit_efficiency_weighted_primary` | Same, weighted by truth track length $|T_j|$ | Maximise |
| **Purity (primary only)** | `m_purity_primary_only` | As above, restricted to primary reco per truth | Maximise |

### Segment-level metrics (from `segement_analysis.ipynb` reference)

| Metric | Definition |
|--------|-----------|
| **Segment efficiency** | $N_{\text{true pairs accepted}} / N_{\text{true pairs total}}$ |
| **Segment false rate** | $N_{\text{false pairs accepted}} / N_{\text{all accepted pairs}}$ |
| **Kink angle distributions** | Histograms of $\alpha$ for true vs false segment pairs |

> **Note:** Segment-level metrics are not computed per-job in the sweep (too expensive
> to store per point). They will be computed in the analysis notebook for selected
> optimal configurations using `collect_segment_pair_angles()` from `helpers.py`.

### Combined figures of merit

For ranking parameter combinations:

$$
\text{Score} = \text{efficiency} \times (1 - \text{ghost rate})
$$

A Pareto frontier analysis (efficiency vs ghost rate) will also be performed to identify
the set of non-dominated parameter configurations.

---

## Experimental Design

### Phase 1 — Coarse Scan (this experiment)

**Goal:** Map the full $k \times \theta_d \times \sigma_{\text{res}} \times \sigma_{\text{scatt}}$ landscape
at 3 angular cones and 3 track densities. Identify the region of parameter space with the
best efficiency–ghost tradeoff.

**Grid:** 1,944 HTCondor jobs (ERF-only) as described above.

**Execution:**

```
cd Toy_Characterisation/EpsilonStudies
./submit.sh              # generates params, submits to HTCondor
```

**Aggregation:**

```
python scripts/aggregate.py --results-dir results
```

Produces `results/aggregated/param_opt.csv` — a flat table with one row per
parameter combination and columns for all swept parameters + metrics.

### Phase 2 — Fine Scan (planned, contingent on Phase 1)

**Goal:** Zoom into the top-performing $(k, \theta_d)$ region identified in Phase 1
with finer grid spacing and increased repeats ($n = 5$).

**Grid:** To be determined after Phase 1 analysis. Expected $\sim$500–1,000 jobs.

---

## Pipeline Architecture

```
submit.sh
 │
 ├─ gen_params_opt.py   → results/params/job_XXXXX.json  (1,944 files)
 │                       → results/batches.txt
 │                       → results/run_summary.json
 │
 ├─ condor_submit scan.sub
 │   └─ run_worker.sh (per job on HTCondor)
 │       └─ run_worker.py --params-json job_XXXXX.json
 │           └─ run_param_opt()
 │               ├─ generate_event(geo, n_tracks, measurement_error, collision_noise, ...)
 │               ├─ SimpleHamiltonianFast(epsilon, gamma, delta, theta_d)
 │               ├─ construct_hamiltonian(event, convolution=True)
 │               ├─ solve_classicaly()
 │               ├─ get_tracks() + EventValidator
 │               └─ → results/job_XXXXX/results.json
 │                  → results/job_XXXXX/repeat_data.pkl.gz
 │
 └─ aggregate.py --results-dir results
     └─ → results/aggregated/param_opt.json
        → results/aggregated/param_opt.csv
```

### Data Persistence

Each job saves two output files for full reproducibility:

| File | Contents | Format |
|------|----------|--------|
| `results.json` | Aggregated metrics (mean ± SE), per-repeat metric rows, job parameters | JSON |
| `repeat_data.pkl.gz` | Full per-repeat data for offline reanalysis | gzip-compressed pickle |

The `repeat_data.pkl.gz` archive contains a list of dicts (one per repeat), each storing:

| Key | Type | Description |
|-----|------|-------------|
| `event` | dict | Hit positions (x, y, z, module_id, track_id) and truth tracks (track_id, hit_ids) |
| `A_data`, `A_row`, `A_col`, `A_shape` | numpy arrays + tuple | Hamiltonian matrix $A$ in COO sparse format |
| `b` | numpy array | Bias vector |
| `solution` | numpy array | Classical solution vector $x$ |
| `segments` | list of (int, int) | Segment hit-ID pairs (from_hit, to_hit) |
| `reco_tracks` | list of dicts | Reconstructed tracks with hit_ids |
| `full_metrics` | dict | Complete `EventValidator.compute_metrics()` output (28+ keys) |

This allows any analysis to be rerun offline without re-generating events or re-solving the Hamiltonian.

### Key files

All files are under `Toy_Characterisation/EpsilonStudies/`:

| File | Purpose |
|------|---------||
| `scripts/gen_params_opt.py` | Sweep grid definition and JSON generation |
| `scripts/run_worker.py` | `run_param_opt()` task handler |
| `scripts/aggregate.py` | `_agg_param_opt()` merges results |
| `scripts/helpers.py` | `compute_epsilon()`, event generation, validation |
| `submit.sh` | Submission orchestrator |
| `condor/scan.sub` | HTCondor job description |
| `condor/run_worker.sh` | Conda wrapper for condor nodes |
| `results/` | Symlink to `condor_pipeline/results_opt/` (19 GB) |
| `experiment_plan.md` | This document |

---

## Planned Analysis

After aggregation, the following analyses will be performed
(notebook: `EpsilonStudies/analysis.ipynb` — to be created), following
the structure and style of `Recreating_paper_results/from_OBQF/segement_analysis.ipynb`.

### A. Track-Level Performance vs Scale Factor $k$

*Reference: segement\_analysis.ipynb Parts 1–3 (resolution and scattering scans)*

1. **Efficiency vs $k$** — line plots (markers + error bars) at each
   $(\sigma_{\text{res}}, \sigma_{\text{scatt}})$ combination, faceted by angular cone.
   Separate curves per $\theta_d$.

2. **Ghost rate vs $k$** — same layout. Shows the cost of widening $\varepsilon$.

3. **Clone fraction vs $k$** — identifies the onset of track duplication
   as the acceptance window opens.

### B. Hit-Level Quality vs Scale Factor $k$

*Reference: segement\_analysis.ipynb Part 7 (optimal reconstruction metrics)*

4. **Hit purity vs $k$** — mean purity of GOOD primary tracks.
   Expect degradation at large $k$ as false segments contaminate tracks.

5. **Hit efficiency (completeness) vs $k$** — fraction of truth hits recovered.
   Expected to increase with $k$ up to a plateau.

6. **Hit efficiency (weighted) vs $k$** — same, weighted by truth track length.
   Emphasises completeness for longer tracks.

### C. ERF Smoothing Width Analysis

*Reference: segement\_analysis.ipynb Part 6 (epsilon formula comparison)*

7. **$\theta_d$ comparison** — efficiency + ghost rate vs $k$ at each $\theta_d$
   value. Shows how the smoothing width affects the efficiency–ghost tradeoff.

8. **Heatmaps: Score vs $(k, \theta_d)$** — 2D colour maps for each
   $(\sigma_{\text{res}}, \sigma_{\text{scatt}})$ and track density, faceted by angular cone.

### D. Physics Sensitivity

*Reference: segement\_analysis.ipynb Parts 2–3 (resolution and scattering scans)*

9. **Optimal $k$ vs $\sigma_{\text{res}}$** — does the formula correctly
   adapt $\varepsilon$ as measurement resolution degrades?

10. **Optimal $k$ vs $\sigma_{\text{scatt}}$** — same for multiple scattering.
    Plot the computed $\varepsilon$ at the optimum to check consistency.

11. **Physics-regime stability table** — for each $(\sigma_{\text{res}}, \sigma_{\text{scatt}})$
    pair, report the optimal $(k, \theta_d)$ and all metrics at that optimum.

### E. Density Dependence

*Reference: segement\_analysis.ipynb Part 4 (track density analysis)*

12. **All metrics vs $n_{\text{tracks}}$** — at the optimal parameters for each
    angular cone. Includes efficiency, ghost rate, clone fraction, hit purity,
    hit efficiency. Shows whether the optimum is stable across densities.

13. **Efficiency vs ghost rate scatter** — coloured by $k$, with the
    Pareto-optimal set highlighted. One panel per angular cone.

### F. Segment-Level Deep Dives (selected configurations)

*Reference: segement\_analysis.ipynb Part 8 (fixed epsilon experiment)*

14. **Kink angle histograms** — true vs false segment-pair angle distributions
    at the optimal $(k, \theta_d)$ vs a suboptimal choice. Overlay the
    $\varepsilon$ threshold to show how the acceptance window sits relative
    to the angle distributions.

15. **Segment efficiency and false rate vs $n_{\text{tracks}}$** — at the
    optimal parameters, showing how segment-level acceptance scales with
    density (true pairs $\propto n$, false pairs $\propto n^2$).

16. **Segment pair count scaling** — log-log plots of true/false segment
    counts vs $n_{\text{tracks}}$, confirming the expected $O(n)$ vs $O(n^2)$ scaling.

### G. Epsilon Landscape

17. **All metrics vs computed $\varepsilon$** — scatter plot using $\varepsilon$ as
    the x-axis (regardless of how it was derived from $k$, $\sigma_{\text{res}}$, $\sigma_{\text{scatt}}$).
    Determines whether there is a universal optimal $\varepsilon$ range.

### H. Summary Dashboard

18. **Best parameters table** — for each (angular cone, track density) condition:
    optimal $(k, \theta_d)$, achieved efficiency, ghost rate, clone
    fraction, hit purity, hit efficiency, and the derived $\varepsilon$.

---

## Verification Checklist

- [x] `gen_params_opt.py --dry-run` produces correct job count (1,944)
- [x] ERF-mode test job completes (k=10, θ_d=0.01, 50 tracks: eff=100%, ghost=0%)
- [x] Aggregation produces valid CSV with all parameter columns
- [x] `repeat_data.pkl.gz` saved per job with full event/Hamiltonian/solution data
- [x] Full Phase 1 submission via `submit_opt.sh`
- [x] All 1,944 jobs complete without failure (0 failures)
- [x] Aggregated `param_opt.csv` has 1,944 rows × 31 columns
- [x] Phase 1 analysis notebook completed (`analysis.ipynb`, sections A–H)
- [ ] Phase 2 grid defined and submitted (if warranted)

---

## Notes

- The `--no-banner` flag has been removed from `condor/run_worker.sh` (unsupported by installed conda).
- At $\pm 0.04\;\text{rad}$ with $n = 100$ tracks, track density is very high
  ($\sim 20{,}000\;\text{tracks/mrad}^2$) and recovery dips are expected.
  Results at this setting should be interpreted with caution.
- $\gamma$ and $\delta$ are held fixed at values from the baseline characterisation
  (1.5, 1.0). A follow-up study sweeping $(\gamma, \delta)$ at the optimal
  $(k, \theta_d)$ could further improve performance.
