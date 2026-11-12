# Experiment A: coupling-following readout with hit uniqueness

Research plan candidate A (Notion "Research plan: cheap and effective VELO track finding",
approved 2026-11-11). Question: on Run 3 events, is the track merging (groups of hundreds of
hits) made by the readout rule `get_tracks` (active segments sharing a hit form one track)
rather than by the filter? If so, a readout that only follows the Hamiltonian couplings, plus
hit uniqueness, should fix it without touching the solver, the qubits or the gates.

New package code (LHCb_VeLo_Toy_Model, branch `claude/coupled-readout`):
`solvers/reconstruction/coupled_readout.py` (`get_tracks_coupled`, `coupled_hit_groups`),
`tests/test_coupled_readout.py`.

## Scripts -> outputs

| Script | Writes |
|---|---|
| `scripts/run_readouts.py real` | `outputs/real_readouts_events.csv`, `outputs/real_readouts_tracks.csv` (103 Run 3 Gauss Sim10 events, ~8 min on 12 cores) |
| `scripts/run_readouts.py toy` | `outputs/toy_readouts_events.csv`, `outputs/toy_readouts_tracks.csv` (T = 100, 400, 1000 x n_pv = 1, 2, 5, 10, 15, 20; 78 events, ~30 min) |
| `scripts/control_no_solver.py real\|toy` | `outputs/<mode>_control_events.csv` (no solve: every coupled segment active, uniqueness by a random or a coupling-count score) |
| `scripts/summarize.py` | `outputs/summary.txt`, `outputs/run3_table.csv`, `outputs/toy_table.csv`, `figures/A1`-`A6` |

Solvers, readings and readouts are defined in the `run_readouts.py` docstring. Everything is
scored by the package EventValidator (purity >= 0.7, reconstructible = >= 3 hits). Run 3
inputs: `../Real_events` (branch `claude/project-thread-4d3krr`), event conversion
`Real_events/scripts/real_lib.py`; solvers from `Focusing_scan/scripts/07_full_sweep.py`.

## What is measured and what is a model

* Track efficiency, ghost rate, clone fraction, group sizes: measured (exact solves per
  connected component, classical readout, EventValidator).
* Qubits and two-qubit gates (figure A5): the Givens cost model of paper 1 / section 4.7, as
  in `Real_events`; the readout is classical and does not change them.
* Run 3 hits are smeared Geant4 points, not clusters: every Run 3 number is optimistic until
  Boole/Moore clustering is in.
