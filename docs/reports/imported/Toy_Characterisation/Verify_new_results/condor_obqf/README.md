# §18 — 1BQF (`OneBitHHL`) characterisation

This folder contains the infrastructure for running the §18b/c
quantum solver sweep on HTCondor.  The local §18a sweep runs inside
the notebook directly.

## Contents

| file | purpose |
|------|---------|
| `run_event.py`     | single-event worker: builds Hamiltonian, runs classical solve, runs `OneBitHHL`, validates tracks for both trackers, dumps one pickle. |
| `run_worker.sh`    | condor wrapper — sets `PYTHONPATH`, activates `Q_env`, calls `run_event.py`. |
| `submit_cpu.sub`   | condor submit, 1 CPU slot, 6 GB RAM. |
| `submit_gpu.sub`   | condor submit, 1 GPU slot, 16 GB RAM (GPU-assisted `AerSimulator`). |
| `gen_params_obqf.py` | emits the (n_trk, rep, shots, device) CSV expected by the submit files. |

## Example — §18b CPU sweep

```bash
cd /data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Verify_new_results

mkdir -p outputs/segment_analysis/quantum_18b/results outputs/segment_analysis/quantum_18b/logs

python condor_obqf/gen_params_obqf.py \
    --device CPU \
    --out    outputs/segment_analysis/quantum_18b/params_cpu.csv \
    --n-trk-min 8 --n-trk-max 40 --n-trk-step 4 --reps 3

condor_submit \
  -append "PARAMS_CSV=$PWD/outputs/segment_analysis/quantum_18b/params_cpu.csv" \
  -append "RESULTSDIR=$PWD/outputs/segment_analysis/quantum_18b/results" \
  -append "LOGDIR=$PWD/outputs/segment_analysis/quantum_18b/logs" \
  condor_obqf/submit_cpu.sub
```

## Example — §18c GPU sweep

```bash
python condor_obqf/gen_params_obqf.py \
    --device GPU \
    --out    outputs/segment_analysis/quantum_18c/params_gpu.csv \
    --n-trk-min 40 --n-trk-max 100 --n-trk-step 10 --reps 3

condor_submit \
  -append "PARAMS_CSV=$PWD/outputs/segment_analysis/quantum_18c/params_gpu.csv" \
  -append "RESULTSDIR=$PWD/outputs/segment_analysis/quantum_18c/results" \
  -append "LOGDIR=$PWD/outputs/segment_analysis/quantum_18c/logs" \
  condor_obqf/submit_gpu.sub
```

## Aggregation

Each job produces a single `event_nNNNN_repRRR.pkl` file.  The notebook
aggregates all pickles in a results directory with a glob:

```python
from pathlib import Path
import pickle
results = [pickle.load(open(p, 'rb'))
           for p in sorted(Path('outputs/segment_analysis/quantum_18b/results').glob('*.pkl'))]
```

## Scaling notes & caveats

The `OneBitHHL` circuit compile time grows with the **number of
compatible segment pairs** (one multi-controlled `RX` per pair), not
with `n_seg` alone.  Empirically:

- `n_trk = 6`  → ~30 s on CPU.
- `n_trk = 10` → single digits of minutes on CPU.
- `n_trk = 40` → budget **many minutes per event** on CPU.
- `n_trk = 100` → likely tens of minutes per event even on GPU.

GPU speeds up the state-vector evolution step inside `AerSimulator`,
but **not** the Python-level Qiskit transpile.  If compile dominates,
move to a lower-level decomposition (deferred — §18d).

Memory: state vector uses `2^(n_q_sys + 2)` complex amplitudes.  At
`n_hits = 1000` (`n_seg ≈ 4·n_trk² ≈ 4·10⁴` → 16 qubits) this is
~1 GB per simulator; 16 GB on the GPU submit is sized for up to
~20 qubits safely.

## Verdict gate

A quantum run PASSES §18 only if, averaged over reps:

1. Cosine similarity `cos(s_Q, s_C) ≥ 0.95`.
2. Jaccard on the `τ = 0.35` active set `≥ 0.90`.
3. Layered-tracker efficiency within `5 %` of the classical efficiency
   at the same `n_trk`.
4. Post-selection success `P(ancilla=1)` stays above the 1/shots
   statistical-floor (i.e. non-zero in all reps).

Any violation → NEEDS INVESTIGATION, not FAIL, until the quantum/classical
delta can be attributed to shot noise vs. algorithmic error.
