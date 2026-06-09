# Run-3 Verification — false-segment types & solver metrics on real VeLo events

Applies the [Segment_level_studies](../Segment_level_studies/) framework to **real
LHCb Run-3 events** (`events_bsphiphi_mag_down_run3`, Bs→φφ, magnet-down; ~1000
events, ~340 tracks / ~2400 hits each, 52 modules). It builds the segment
Hamiltonian on each event, solves classically and with the 1BQF, classifies every
**false** segment by cluster topology, and reports metrics — to check whether the
toy-model taxonomy holds on real data.

## Geometry fix (important)
The VeLo's left/right modules **interleave in z** (even z-rank = left `x<0`, odd =
right `x>0`); a track stays on one side, so its hits step by **2** in z-rank. The
toy's `idx,idx+1` adjacent-module segment model therefore builds almost no true
segments (n_true ≈ 4/2164). `run3_loader.py` fixes this by building segments
**within each half-detector** — the left and right module chains ordered by z,
separated by an empty module (`construct_segments` skips empty modules, so no
spurious cross-half segment is built). After the fix, n_true jumps to ~240 on the
smallest event and the classical solver gives eff ≈ 0.88 / purity ≈ 0.94.

## Settings
`A = (γ+δ)I − C`, solve `A x = δ·1`, active iff `x_i > τ = 0.35` (γ=3, δ=1).
Acceptance **ε = 5 mrad** (scanned: 1→100 mrad; 5 mrad balances efficiency/purity).
Quantum = 1BQF statevector (signal-support rescale for the metric).

## Files
- `run3_loader.py` — Allen-JSON → duck-typed event for the toy segment Hamiltonian.
- `Run3_segment_characterisation.ipynb` — classical (80 events) + quantum (12
  smallest) + the false-type census; writes `outputs/`.
- `outputs/` — figures + CSVs (`false_type_census`, `classical_metrics`,
  `quantum_vs_classical`).

## Findings (the toy taxonomy holds on real data)
**False-segment census** (over 80 events, 1.02 M false segments):

| type | % of false | active = false positive |
|---|---|---|
| isolated (size 1) | **99.72 %** | 0 % |
| pair (size 2) | 0.16 % | 0 % |
| chain ≥3 (bridge) | 0.05 % | **100 %** |
| hub (star) | 0.069 % | **100 %** |

→ all 1208 false positives come from the **coupled (chain≥3 / hub)** clusters —
the cross-track and dense-region confusions. The isolated bulk is rejected.

**Solver metrics** (mean over events):
- **Classical:** efficiency ≈ 0.87, purity ≈ 0.97, false rate ≈ 0.03.
- **1BQF (statevector):** efficiency ≈ 0.71 (endpoint-halving — the 1-bit filter
  drops the weakly-coupled first/last segment of each track), purity ≈ 0.98,
  false rate ≈ 0.02, `cos θ_QC ≈ 0.98`. The quantum solver even suppresses some
  hubs (48 % active vs the classical 100 %). Same trade-off as the toy: lower
  efficiency, comparable/higher purity.

## Note
The event JSONs (`events_bsphiphi_mag_down_run3/`) and the source `.zip` are
git-ignored (too large). Place the unzipped `events_bsphiphi_mag_down_run3/`
alongside the notebook to re-run.
