# Run 3 simulated events through the vertexing and tracking methods (20 Oct 2026)

Status: **provisional**. First test of the section 8 methods of the vertexing write-up
(https://app.notion.com/p/3fe5d544b9d9812d8a77e7f9d2c3d9a5) on full LHCb simulation instead of
the toy. The toy-side code is imported read-only from `Vertexing/Focusing_scan/scripts/`.

## Events

- **Simulation:** Gauss v56r9 (Sim10) from CVMFS, run on stbc through `lb-run` (apptainer).
  2024 production conditions: `Beam6800GeV-md100-2024.W40.42-nu7.6`, `Run3-detector`,
  `DataType-2024`, DDDB `dddb-20240427`, SIMCOND `sim10-2024.W40.42-v00.00-md100`, minimum
  bias (DecFile 30000000), Pythia 8, `G4PL_FTFP_BERT_EmOpt2`. Spill-over crossings are off.
  Options: `sim/gauss_prod.py`; 8 jobs of 3 events (runs 5001–5008) and 8 jobs of 10 events
  (runs 5011–5018). 104 crossings; one has no VELO hits, so **103 events**.
- **Export:** `scripts/dump_sim.py` (GaudiPython in the Gauss environment; `sim/dump_job.sh`)
  unpacks the `.sim` file and writes, per event, every VELO (VP) MC hit with its MC particle,
  each particle's origin vertex, mother and primary vertex, and every primary vertex.
- **Conversion:** `scripts/real_lib.py` builds the focusing-scan event and a package
  `lhcb_velo_toy` Event from it, so the package segment Hamiltonian and `EventValidator` run
  unchanged. Three changes are needed for real geometry:
  1. x and y are taken relative to the beam spot (0.450, 0.059) mm, so "beamline at x = y = 0"
     holds on average.
  2. The package Event orders the modules as two chains, one per VELO half (even module =
     x < 0 side, odd = x > 0), each in z, with an empty module between them. A track stays in
     one half, so its consecutive hits are on consecutive modules of a chain. This is the fix
     from `Toy_Characterisation/Run3_Verification`.
  3. The focusing scan's coarse range is widened from [−150, 28) mm to [−250, 250) mm, because
     the luminous region is centred at z ≈ +6 mm with σ ≈ 43 mm.
- **Hits are Geant4 points, not clusters.** The hit is the mid-point of the Geant4 step in the
  sensor, smeared by 12 µm in x and y. There is no Boole digitisation, no clustering, no
  cluster merging, no pixel inefficiency and no noise hits. This is the main limit of the test
  (see "What is still missing").

## Event content against the toy point used for comparison (T = 200, 5 vertices)

| | Run 3 simulation (103 events) | toy (section 8) |
|---|---|---|
| VELO hits per event | median 1862 (177–5723) | ≈ 1000 |
| reconstructible tracks (≥ 3 hits) | median 235 | 200 |
| hits per track | 3–13, median 6 | 4–5 |
| primary vertices | 7.7 per event (1–16); 5.5 with ≥ 4 tracks | 5 |
| tracks not from the PV (decays, material) | 22 % | 0 |
| electrons | 13 % | 0 |
| momentum | 22 % below 1 GeV | no momentum spread |
| slope \|t\| | 21 % above 0.25 rad | cone 0.2 rad |
| vertex position | inside the VELO, z = 6 ± 43 mm; transverse spread 24 µm | upstream of plane 1, x = y = 0 |
| segments (package Hamiltonian, ε = 5 mrad) | median 66 k, max 621 k | 154 k |

## Results (`outputs/real_summary.txt`; `scripts/run_real.py`, then `scripts/summarize.py`)

### Track reconstruction (EventValidator, purity ≥ 0.7, reconstructible = ≥ 3 hits)

| method | efficiency | ghost rate | clones | toy efficiency / ghost |
|---|---|---|---|---|
| classical, 99 % true segments (paper reading) | 0.000 | 1.000 | 0 | 0.48 / 0.26 |
| classical, ≤ 1 % false segments | 0.017 | 0.035 | 0.020 | 0.94 / 0.006 |
| classical, every coupled segment (no truth used) | 0.327 | 0.114 | 0.023 | — |
| one-bit filter, ≤ 1 % false segments | 0.029 | 0.011 | 0.069 | 0.94 / 0.006 |
| QSVT comb, ≤ 1 % false segments (paper reading) | 0.128 | 0.009 | 0.053 | 0.94 / 0.006 |
| QSVT comb, above the isolated value (no truth used) | 0.575 | 0.066 | 0.139 | — |
| hit register at the scan peaks (as specified) | 0.585 | 0.031 | 0.146 | 0.92 / 0.000 |
| hit register at the truth vertices | 0.581 | 0.025 | 0.154 | — |
| hit register at the truth vertices, 0.5 rad window | 0.594 | 0.025 | 0.152 | — |

On the tracks that look like the toy (prompt, not an electron, ≥ 4 hits, p > 2 GeV; 50 % of the
reconstructible tracks) the hit register finds **0.90** (0.93 at the truth vertices with the wide
window), the truth-free QSVT reading 0.77, and the classical solver 0.31.

The losses by category (per truth track, hit register at the scan peaks / QSVT truth-free):
3-hit tracks (16 %) 0.03 / 0.03, |t| > 0.25 (21 %) 0.01 / 0.30, p < 1 GeV (22 %) 0.13 / 0.29,
secondaries (22 %) 0.28 / 0.31, electrons (13 %) 0.36 / 0.19. Figure
`figures/02_track_eff_vs_kinematics.png`.

### Why the segment methods fail at their section 8 readings

- **No ≤ 1 % false working point in 91 % of the events.** About 18 % of the coupled segments
  are false (crossing tracks, secondaries, low-momentum scattering, two hits of one particle in
  the sensor overlap). They sit in the same components as true chains, so the threshold that
  gives ≤ 1 % false keeps only the top few segments (segment efficiency 0.04).
- **The 99 % reading switches on every isolated false segment**, as at high T in the toy
  (segment false rate 0.97).
- **At the truth-free reading the segments are good but the grouping fails.** Every coupled
  segment gives segment efficiency 0.90 at 18 % false, but the get_tracks rule (segments that
  share a hit form one track) joins tracks through the false segments: in a typical event one
  group holds 300–650 hits, and 50–75 % of the active hits sit in groups with more than one
  truth track. Track efficiency is 0.33.
- **The QSVT comb is designed for the toy's 5-hit chain lines.** Real chains have 2–12
  segments, so the comb lines do not match. Its truth-free reading still gives the best segment
  method (0.58).
- The segment solves are exact (eigen-decomposition per connected component, as in section 8),
  on the **full events**, up to 621 k segments; the largest component has 81 segments. No
  reduced event was needed. The one-bit filter and the QSVT are the exact filter functions,
  not circuit simulations, as in section 8.

### Primary vertices (truth PVs with ≥ 4 reconstructible tracks, 1 mm match)

| method | efficiency | fakes / event | merges / event | z residual rms (robust σ) |
|---|---|---|---|---|
| focusing scan, as specified | 0.80 | **75** | 0.33 | 466 µm (491 µm) |
| scan peaks → candidate fits → z clustering (hold-out, 79 events) | 0.69 | 0.82 | 0.32 | 299 µm (234 µm) |
| track baseline on the classical ≤ 1 % tracks | 0.03 | 0.03 | 0.02 | — |
| track baseline on the QSVT truth-free tracks | 0.75 | 2.2 | 0.36 | 282 µm (202 µm) |
| toy, T = 200, 5 vertices, focusing scan | 0.92 | 0.00 | 0 | 42 µm |

- **The focusing scan has no peaks on a real event.** N4(ζ) is a broad hump over the whole
  PV cluster with a high floor (`figures/01_scan_real_vs_toy.png`). The 0.80 efficiency only
  means there is a peak every few mm; no height cut separates the true ones (at N4 ≥ 30 there
  are 2.6 good and 6.5 fake peaks per event).
- **Cause 1, lever arm.** The focus width of a track is w/(|t|·range of 1/(z_k − z_v)). In the
  toy the planes are 33–165 mm from the vertex (range 0.024 /mm). In the VELO the nearest hit
  of a track is a median 88 mm away and the range is a median 0.0087 /mm, with a long tail of
  forward tracks whose hits are all far downstream. The predicted focus width at the 512 grid
  is a median 1.4 mm (toy 0.25 mm) and 46 mm at the 90th percentile.
- **Cause 2, PV density.** 7.7 PVs in σ_z ≈ 43 mm means a typical neighbour 5–15 mm away, about
  the width of the blurred peaks.
- **Cause 3, transverse beam spot.** The truth PVs scatter by 24 µm in x and y around the
  beam spot. That alone moves a track's z of closest approach by 24 µm/|t| ≈ 0.3 mm at the
  median |t| = 0.08, which sets the ~0.3 mm residuals of every method here, and it breaks the
  "beamline at x = y = 0" assumption of the scan (open issue 5 of the focusing-scan README).
- The adapted variant (every scan peak seeds candidates, each candidate is fitted by a straight
  line, and the z of closest approach is clustered with a 0.5 mm gap, ≥ 4 tracks, |t| ≥ 0.05) cuts
  the fakes from 75 to 0.8 per event. Its parameters were chosen on the first 24 events and the
  numbers above are on the other 79.

### Qubits and two-qubit gates (median per event, Givens price, as in section 6)

| | route | segment register | hit register, per vertex |
|---|---|---|---|
| real | 2-call filter | 19 qubits, P 1.0e-2, 5.6e6 gates | 13 qubits, P 0.18, 2.3e6 gates |
| real | QSVT phase sequence | 21 qubits, P 1.3e-3, 1.2e9 gates | 15 qubits, P 0.18, 1.8e8 gates |
| toy T = 200, 5 PV | 2-call filter | 20 qubits, P 1.7e-3, 6.8e6 gates | 12 qubits, P 0.94, 5.0e5 gates |
| toy T = 200, 5 PV | QSVT phase sequence | 22 qubits, P 7.6e-4, 8.2e8 gates | 14 qubits, P 0.94, 4.0e7 gates |

- The qubit saving of the hit register holds (6 qubits).
- **The hit-register success probability falls from 0.94 to 0.18.** The ideal projector is on
  the lines {3, 4} (5-hit and 4-hit cliques), and real tracks have 3–13 hits. On every integer
  line ≥ 2, P is 0.36 (1.6e6 / 1.3e8 gates). The rest is broken cliques (scattering, wrong
  vertex) and hits that are not on a track. The gain per event over the segment register drops
  from 14–20× in the toy to 2.5–7× here.

## What is still missing

1. **Clusters.** The hits are Geant4 points with 12 µm smearing. The next step is Boole
   digitisation plus the Moore VP clustering with the MC links, so that cluster merging,
   splitting, pixel inefficiency and noise are included. The LHCb stack on CVMFS has all of it
   (Boole v45–v48, Moore v58–v60); only the job options are missing.
2. **Spill-over** is off; the VELO timing makes it small, but it is not zero.
3. The segment Hamiltonian uses ε = 5 mrad (the Run3_Verification choice) and is not re-tuned.
4. The vertex finding uses the generator beam spot as the beamline. In data it comes from a
   calibration; a fit of x, y per vertex is needed in any case (cause 3).

## Files

- `sim/gauss_prod.py`, `sim/dump_job.sh` — simulation and export (the `.sim` files and JSON dumps
  stay on /data and are not in git).
- `scripts/dump_sim.py`, `scripts/real_lib.py`, `scripts/run_real.py`, `scripts/summarize.py`.
- `outputs/real_events.csv` (per event), `outputs/real_vertices.csv` (per truth PV and per scan
  peak), `outputs/real_tracks.csv` (per truth track), `outputs/real_track_summary.csv`,
  `outputs/real_resources.csv`, `outputs/real_summary.txt`.
- `figures/01_scan_real_vs_toy`, `figures/02_track_eff_vs_kinematics`, `figures/03_vertices`.

To rerun: `Q_env/bin/python scripts/run_real.py 12` (about 2 min for 103 events), then
`scripts/summarize.py`.
