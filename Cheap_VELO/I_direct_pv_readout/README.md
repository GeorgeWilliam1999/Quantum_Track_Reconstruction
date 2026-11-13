# Items I and J: direct primary-vertex readout with a quantum algorithm (13 Nov 2026)

Status: **provisional, not verified**. Research plan "cheap and effective VELO track finding"
(Notion 4155d544…8cb), items I and J, reordered by George on 13 Nov 2026: the goal is a direct
readout of the primary vertices (PVs) by a quantum algorithm; the comparison object is qubits
and queries against tracks and against the number of PVs.

## Method

**Quantum: amplitude-amplified vertex sampler** (package `solvers/vertexing/pv_sampler.py`,
branch `claude/pv-readout`).

- Index register over items. *Doublets*: every package segment (the paper 1 segment register).
  *Triplets*: a segment and its k-th forward coupled segment (sparse access to A, as the block
  encoding uses), `ceil(log2 n_seg) + ceil(log2 d_max)` qubits.
- Oracle: loads the item's hits, writes the z of closest approach (POCA) of its line to the
  beamline into a 12-qubit vertex register (0.05 mm bins on the toy, 0.1 mm on Run 3), and a
  flag: distance of closest approach ≤ dmax, |t| ≥ 0.02, z in the window.
- Amplitude amplification on the flag (r rounds at the padded flagged fraction), then measure
  the vertex register. Each accepted shot is the POCA z of a uniformly chosen flagged item.
- PVs = peaks of the shot histogram (Gaussian smoothing, minimum content in items, 1 mm
  separation; `find_vertex_peaks`).
- **Measured vs model.** The output distribution is exact (the oracle truth table; no circuit is
  simulated); a statevector test checks this shortcut and the success probability
  (`tests/test_vertexing.py::test_amplification_matches_statevector`). Shots are drawn from it.
  Index, vertex and flag qubits are exact. The arithmetic workspace (192 qubits for doublets,
  240 for triplets at 16 bits) is a **model** (`workspace_qubits_model`). Queries assume the
  flagged fraction is known; exponential search for unknown P costs a constant factor more.

**Classical baselines on the same events.**
- Focusing scan as specified, and scan + refit (Run 3 README rules).
- Item J: tracks from the classical solver, the one-bit filter and the QSVT comb (reading I,
  coupled readout with hit uniqueness, experiment A), straight-line fit, z clustering, then a
  3D vertex fit with outlier removal (package `solvers/vertexing/track_vertices.py`). Also the
  older z0-median rule (08_pv_baseline) and truth tracks + fit as a bound.

**Events.** Toy: T = 100, 400, 1000 × 1 … 20 PVs, 3 events each (180, the section 8 seeds).
Run 3: 103 Gauss Sim10 events. Scoring: 1 mm greedy match to all truth PVs; efficiency over PVs
with ≥ 4 reconstructible tracks.

**Tuning.** The sampler's dmax, smoothing, minimum content and shoulder are tuned at K = ∞ on
separate events (toy: 18 events, seeds 70000+; Run 3: job_5001…5008, 24 events), objective
matched − fakes. All numbers below are on hold-out events (toy 180, Run 3 79). Tuned values in
`outputs/tuned.json`.

## Results

### PV finding (exact sampler output, K = ∞)

Run 3, 79 hold-out events (`outputs/summary.txt:3-13`):

| method | efficiency | fakes / event | merges / event | z rms (robust σ) |
|---|---|---|---|---|
| vertex sampler, doublets | **0.825** | 0.60 | 0.38 | 172 µm (155) |
| vertex sampler, triplets | 0.800 | **0.35** | 0.34 | 216 µm (188) |
| focusing scan, as specified | 0.793 | 75.8 | 0.30 | 446 µm |
| focusing scan + refit | 0.690 | 0.82 | 0.32 | 299 µm (240) |
| tracks (classical) + vertex fit (J) | 0.767 | 0.95 | 0.33 | 151 µm (102) |
| tracks (one-bit filter) + vertex fit | 0.760 | 0.99 | 0.33 | 157 µm |
| tracks (QSVT) + vertex fit | 0.751 | 1.24 | 0.28 | 204 µm |
| tracks (classical) + z0 median (old rule) | 0.765 | 3.13 | 0.30 | 223 µm |
| truth tracks + vertex fit (bound for J) | 0.765 | 0.87 | 0.35 | 133 µm |

Toy, efficiency at 16–20 PVs, T = 100 / 400 / 1000 (`outputs/summary.txt:18-44`):
triplet sampler 0.870 / 0.841 / 0.848; focusing scan + refit 0.542 / 0.770 / 0.722; tracks
(classical) + fit 0.819 / 0.774 / 0.700. At 1–5 PVs all methods are 0.93–1.0.

- **The triplet sampler is the best PV finder on the toy at every T and PV count, with no fakes
  (≤ 0.07 per event) and the smallest residuals** (figure I01).
- The doublet sampler is good on Run 3 but weak on the toy: at T = 1000 it gives about 7 fakes
  per event (accidental pointing pairs), and at T = 100 with many PVs it loses efficiency (0.56).
- **Item J.** The 3D vertex fit cuts Run 3 fakes from 3.1 to 0.95 and the rms from 223 to
  151 µm against the z0 median; efficiency 0.767 against the scan refit 0.690 (kill criterion
  "beat 0.69 efficiency, 0.82 fakes": efficiency passes, fakes are 0.13 higher). On the toy the
  fit makes the residual worse (beam at x = y = 0, and close PVs in one cluster pull the fit).
- **The track solver does not matter for PVs.** Classical, one-bit filter, QSVT and truth tracks
  give the same toy efficiency to ±0.02 and the same Run 3 efficiency to ±0.016. The limit is the
  clustering of track z0 (1 mm gap, ≥ 3 tracks), not the tracks.

### Shots (figure I03)

The sampled readout reaches the exact efficiency at K ≈ 1000–4000 accepted shots; the fakes
settle at about the same K. Per event, the smallest K from which the readout reproduces the
exact one (`outputs/summary_resources.csv`, K_rep; median):

| | triplets | doublets |
|---|---|---|
| toy T = 100 / 400 / 1000 | 128 / 256 / 1024 | 384 / 1024 / 8192 |
| Run 3 | 1024 | 1024 |

The 90th percentile is 4–42× the median (680× for doublets at T = 100); 0–18 of 60 toy events
per T never settle on the grid up to 2^18 (18 are doublets at T = 100; column `unsettled`).

### Qubits against tracks and PVs (figure I04; `outputs/summary_resources.csv`)

| | index | vertex | flag | data total | segment register 1BQF / QSVT |
|---|---|---|---|---|---|
| toy T = 100, triplets | 16 | 12 | 1 | 29 | 18 / 20 |
| toy T = 1000, triplets | 23 | 12 | 1 | 36 | 24 / 26 |
| Run 3, doublets / triplets | 17 / 19 | 12 | 1 | 30 / 32 | 19 / 21 |

- The data registers grow as log2 of the segment count (+1 for triplets), the same law as the
  paper 1 register, plus a fixed 13 qubits. **They do not depend on the number of PVs.**
- With the modelled arithmetic workspace the total is about 220–280 qubits.

### Queries against tracks and PVs (figure I05)

Median per event (`outputs/summary_resources.csv`):

| | quantum queries | classical sampling, same oracle | every segment read once | coupled-pair list read once |
|---|---|---|---|---|
| toy T = 100, triplets | 3.0e3 | 3.4e4 | 3.8e4 | 287 |
| toy T = 1000, triplets | 9.1e4 | 3.5e6 | 3.9e6 | 7.5e3 |
| toy T = 1000, doublets | 5.2e5 | 1.4e7 | 3.9e6 | — |
| Run 3, doublets | 1.6e4 | 1.2e5 | 7.0e4 | — |
| Run 3, triplets | 3.0e4 | 4.7e5 | 7.0e4 | 1.3e3 |

- **Square-root gain over the same oracle.** Against classical rejection sampling with the same
  index register, the quantum sampler needs 7.5–38× fewer queries (amplitude amplification:
  queries per shot ∝ 1/√P, with P = 3e-4 to 7e-3 padded).
- **Against reading every segment once** (the exhaustive histogram, which is also the K = ∞
  result) it needs 2.3–48× fewer queries for triplets and 4–11× for doublets.
- **But a classical code that has the coupled-pair list wins.** The triplet items that the
  oracle flags are 31–80 % of the coupled pairs, so reading that list once (287 to 7.5e3 items)
  costs 9–23× fewer reads than the quantum queries. The list needs the Hamiltonian built
  (every segment examined), which is the cost the sparse-access oracle avoids. So the quantum
  query advantage is real only against a classical method with the same oracle access.
- **Queries against PVs** (toy, triplets; `outputs/summary.txt`, last table): at T = 100 they grow
  from 380 (1 PV) to 6.3e3 (16–20 PVs); at T = 400 from 1.1e3 to 2.7e4; at T = 1000 they stay at
  1e4–2e5 with no clear trend. More PVs mean smaller peaks, which need more shots. The per-point
  medians are over 3 events and are noisy (figure I05 right).

## Verdict

- A quantum direct PV readout works: on Run 3 the vertex sampler at exact output (and from about
  1000 shots) is the best PV finder of all the methods here, and the triplet version is the best
  on the toy at every T and 1–20 PVs.
- Its advantage is in the **estimator**, not in the quantum step: the same histogram of
  triplet POCA z computed classically (from the coupled-pair list) gives the same PVs with fewer
  reads. The quantum gain is the square root of amplitude amplification against same-oracle
  sampling.
- The qubit count is the paper 1 segment register + 13 data qubits, flat in the PV count; the
  modelled arithmetic dominates the total.

## What is not done

- No circuit for the oracle (POCA arithmetic, division); its gate count is not priced.
- The vertex fit of J is not yet used by the sampler (a transverse fit per peak may cut the Run 3
  residual, which is set by the 24 µm beam spot).
- Clusters (Boole + VP clustering) are still missing on Run 3, as for every Run 3 number.
- Peak content and shots-per-PV scaling (why queries grow with PVs at small T) are not modelled.

## Files

- Package (`LHCb_VeLo_Toy_Model`, branch `claude/pv-readout`): `src/lhcb_velo_toy/solvers/vertexing/`
  (`pv_sampler.py`, `track_vertices.py`), `tests/test_vertexing.py` (5 tests).
- `scripts/01_events.py` — per event: segments and the three solvers (07_full_sweep), oracle
  tables, focusing scan, J baselines; `cache/` (not in git). Run 3 caching 64 s, toy ~4 min on
  16 processes.
- `scripts/02_analyse.py` — tuning, scoring, shot sweep → `outputs/tune_grid.csv`, `tuned.json`,
  `pv_events.csv`, `shots_events.csv`, `resources_events.csv`.
- `scripts/03_summary.py` — `outputs/summary.txt`, `summary_pv_*.csv`, `summary_resources*.csv`,
  `krep_events.csv`, `summary_queries_vs_npv.csv`; figures `I01`–`I05`.
- Imports read-only: `Vertexing/Focusing_scan/scripts` (fs_lib, 07_full_sweep, 08_pv_baseline;
  uncommitted, main checkout) and `Vertexing/Real_events/scripts/real_lib.py` (main checkout,
  which holds the event dumps). Interpreter: `/data/bfys/gscriven/conda/envs/Q_env/bin/python`.

To rerun: `01_events.py tune|real|toy 16`, then `02_analyse.py`, then `03_summary.py`.
