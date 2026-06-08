# Epsilon_study_2 — calculated epsilon and sensitivity

## Motivation

In the original segment-level study (`docs/reports/segment_analysis/segment_level_report.pdf`)
the angular acceptance threshold $\varepsilon$ was held fixed at a single
hand-tuned value. The supervisor's task 2 asks: *replace this with the
calculated formula and check the sensitivity of the result to the choice
of inputs that feed the formula.*

The formula used is the one derived in §2 of the segment-level report
and implemented in `lhcb_velo_toy.analysis.compute_epsilon`:

$$
\varepsilon(\sigma_{\rm res},\,\sigma_{\rm scatt}) =
\sqrt{2(\,s\,\sigma_{\rm scatt})^2 + 12\,\arctan^2\!\bigl(\tfrac{s\,\sigma_{\rm res}}{\Delta z}\bigr) + 2\,\theta_{\min}^2}
$$

with $\Delta z = 33$ mm, $s = 3$ ("scale" — number of sigma we are willing
to accept), $\theta_{\min} = 1.5\!\times\!10^{-5}$ rad.

## Validation gate

Before running the sensitivity grid we re-run the canonical
Verify_new_results point ($\sigma_{\rm res}=0$, $\sigma_{\rm scatt}=10^{-4}$,
$\phi_{\max}=0.2$, $T \in \{10,20,50,100,200,400,700,1000\}$) but with
$\varepsilon$ supplied by `compute_epsilon` instead of the previously
hand-tuned scalar. The segment-level metrics — efficiency, purity, false
rate, and the classical/quantum cosine — must agree with the
Verify_new_results numbers within $\pm 1$ standard error per $T$.

## Sensitivity grid

Once the validation gate is passed we sweep:

| axis              | values                              |
|-------------------|-------------------------------------|
| $\sigma_{\rm res}$ (mm)  | $\{0.0,\, 0.01,\, 0.02,\, 0.05\}$ |
| $\sigma_{\rm scatt}$ (rad) | $\{1\!\times\!10^{-4},\, 3\!\times\!10^{-4},\, 5\!\times\!10^{-4}\}$ |
| $T$ (tracks)      | $\{10, 20, 50, 100, 200, 400, 700, 1000\}$ |

Repetitions taper with $T$: 10 reps at $T\le 50$, 5 at $T\le 200$,
3 at $T\ge 400$.

Per grid point we compute the calculated $\varepsilon$ from
$(\sigma_{\rm res},\sigma_{\rm scatt})$ and pass it to the worker. The
worker runs both the classical and statevector 1BQF (OneBitHHL) solvers
and saves the raw per-pair angle arrays so that histogram-based threshold
diagnostics are available without re-running.

## Fixed parameters

- Geometry: 5 planes, $\Delta z = 33$ mm, $\pm 40$ mm transverse extent.
- $\gamma = 3$, $\delta = 1$, $\theta_{\min} = 1.5\!\times\!10^{-5}$.
- $\phi_{\max} = \theta_{\max} = 0.2$ rad, $\sigma_{PV} = (0,0,1)$ mm.
- Convolution: **off** (step Hamiltonian).
- Hit inefficiency: 0.
- Solver readout: statevector (sparse $A$ via lhcb_velo_toy's OneBQF
  sparse path).

## Files

| file | purpose |
|------|---------|
| `theory.md` | Derivation of the $\varepsilon$ formula. |
| `gen_params.py` | Emit `params/eps2.csv` with one row per (T, rep, $\sigma_{\rm res}$, $\sigma_{\rm scatt}$). |
| `submit_eps2.sub` | Thin wrapper over `_shared/submit_base.sub` selecting memory tier. |
| `submit.sh` | One-shot submitter (`condor_submit -append PARAMS_CSV=... LOGDIR=... MEM_GB=...`). |
| `aggregate.py` | Calls `_shared/aggregate.py` with the right group-keys. |
| `analysis.ipynb` | Validation plots and the sensitivity-grid summary. |

All Python invocations use `/data/bfys/gscriven/conda/envs/Q_env/bin/python`.
