# QSVT programme — where everything lives

Written by the 2026-09-09 audit. The QSVT programme's code is **not** all under
`QSVT/`: four studies that the paper depends on live elsewhere in this repo, and
one lives in the package repo. Nothing has been moved — moving it would break the
commit-pinned provenance in a dozen Notion pages — so this file is the map.

`N:` = the Notion write-up of record. All under the Write Up for PhDs database.

## In this folder

| Folder | What it is | Notion |
|---|---|---|
| `Initial/` | Feasibility steps A–E and the first circuit build (band-limited design) | N: QSVT I |
| `qsvt_store_campaign.py` | The production store campaign, solver `qsvt`, line-comb inverse | N: QSVT II |
| `Robustness/` | Comb noise robustness (store study QSVT_robustness) | N: QSVT III |
| `Codesign/` | Operator–response co-design: fork window scan, DP spectral atlas, fork at heavy noise, fit-to-measured-spectrum, tuned notch, twin count, occupancy refit | N: QSVT IV, IX, XI (superseded), XII, XIII |
| `Segment_level_studies/` | The seven segment-level notebooks: efficiency, amplitudes, hit drop, per-solver threshold, depth/qubits, activation spectrum, perfecting the comb | N: QSVT VII, VIII, Resources compendium, wp99 convention |
| `Resource_reduction/` | Degree and cluster reduction (D3/D4) | N: QSVT Resources compendium |
| `DSS/` | Direct structural synthesis and the QETU feasibility study | N: QSVT X, DSS analysis |
| `Run3/` | Spectral filters on real Run-3 VELO geometry | N: QSVT V, QSVT VII |
| `Efficiency_Frontier/` | The degree/operator/loss-budget campaign, 15 scripts | N: QSVT XIV, modified Hamiltonians |
| `Paper_planning/` | Readout and dequantisation, codesign deep dive, and the paper itself | N: QSVT VI, paper companion |
| `Paper_planning/section2_graphs/` | Cluster census, two-tooth comb, Gershgorin data | N: QSVT XV |
| `Paper_planning/section3_onebit/` | Finite-clock HHL, surviving false spectrum | N: QSVT XV |
| `Paper_planning/section4_qsvt/` | Phase factors, convention check, resource model, PREP operator | N: QSVT XV |
| `Paper_planning/qsvt_velo_paper/` | The LaTeX paper, figure factory, Overleaf bundle | N: paper companion, two-paper plan |

## Outside this folder — and load-bearing for the paper

| Path | What it is | Paper use | Notion |
|---|---|---|---|
| `../Block_Encoding/` | Four literature block encodings benchmarked; resource vs accuracy; the hit-level geometry oracle | §11.6, table 7, the `103·T^0.85` per-call price, subnormalisation 2–4, coverage 1.0 | N: Block Encoding I–IV (own Notion project) |
| `../Toy_Characterisation/Bifurification/` | Denby–Peterson penalty terms: fork and occupancy, and the hit-uniqueness gate | §7, §8, the occupancy proof figure | N: Bifurcation term, DP penalties, hit-uniqueness gate |
| `../Toy_Characterisation/ERF/` | The erf angular-cost kernel and its exact-evolution arbiter | §8.3 Trotter comparison | N: ERF (Task 5), ERF exact-evolution arbiter |
| `../Toy_Characterisation/Run3_Verification/` | Real Run-3 event loader and segment characterisation | §11.9 | N: Run3_Verification |
| `../Toy_Characterisation/_shared/qtrk_pipeline/` | Events, Hamiltonians, solvers, metrics — the data layer every study uses | all measured numbers | N: data-pipeline pages |
| `LHCb_VeLo_Toy_Model` (separate repo) | The package: `QSVT`, response families, phase sequence, refit, resources | the solver itself | N: package centralisation pages |

## Data

Solutions, events and metrics live in the store at `$QTRK_STORE`
(`/data/bfys/gscriven/qtrk_store`), not in this repo. Store studies for this
programme: `Verify_new_results` (qsvt rows), `QSVT_robustness`, `QSVT_codesign`,
`QSVT_fork_noisy`.

## Reading conventions

Every solver, classical included, is read at the **efficiency-first working
point**: the lowest threshold that keeps 99% of the true segments. The segment
false rate is an impurity of the selected sample. A fixed threshold is not used
anywhere.
