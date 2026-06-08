#!/usr/bin/env bash
# ============================================================
# Submit the quantum §14e-mirror sweep + Appendix-A shot-scaling + Appendix-C
# large-T sampling sweep.
#
# Usage:
#   ./submit_quantum_seg14e.sh                # main + appendix A
#   ./submit_quantum_seg14e.sh main           # main sweep only
#   ./submit_quantum_seg14e.sh appA           # appendix A only
#   ./submit_quantum_seg14e.sh appC           # appendix C (sampling, T up to 2000)
#
# Outputs land under outputs/quantum_segment_analysis/{seg14e_T100, appendix_A_shots, seg14e_sampling_T500}/
# ============================================================
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
ROOT="$(pwd)"

SUB="${ROOT}/condor_obqf/submit_gpu.sub"
CSV_MAIN="${ROOT}/condor_obqf/params_seg14e_quantum_drop01.csv"
CSV_APPA="${ROOT}/condor_obqf/params_seg14e_quantum_appA.csv"
CSV_APPC="${ROOT}/condor_obqf/params_seg14e_quantum_appC.csv"

OUT_MAIN="${ROOT}/outputs/quantum_segment_analysis/seg14e_T100"
OUT_APPA="${ROOT}/outputs/quantum_segment_analysis/appendix_A_shots"
OUT_APPC="${ROOT}/outputs/quantum_segment_analysis/seg14e_sampling_T2000"

mode="${1:-both}"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
ensure_csv_main() {
    if [[ ! -f "${CSV_MAIN}" ]]; then
        echo "[gen]  ${CSV_MAIN}"
        python3 - <<'PY'
import csv
from pathlib import Path
TC   = [2, 5, 10, 15, 20, 30, 50, 75, 100]
REPS = {2:30, 5:30, 10:30, 15:30, 20:30, 30:20, 50:20, 75:10, 100:10}
p = Path('condor_obqf/params_seg14e_quantum_drop01.csv')
p.parent.mkdir(parents=True, exist_ok=True)
n = 0
with open(p, 'w', newline='') as f:
    w = csv.writer(f)
    for nt in TC:
        for r in range(REPS[nt]):
            w.writerow([nt, r, 1, 'GPU', 0.01, 3.0, 'statevector'])
            n += 1
print(f"wrote {n} rows to {p}")
PY
    fi
}

ensure_csv_appA() {
    if [[ ! -f "${CSV_APPA}" ]]; then
        echo "[gen]  ${CSV_APPA}"
        python3 - <<'PY'
import csv
from pathlib import Path
# Uniform log-spaced shot grid: 10, 100, ..., 1e6
SHOTS = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
TRK, REPS = 30, 5
p = Path('condor_obqf/params_seg14e_quantum_appA.csv')
p.parent.mkdir(parents=True, exist_ok=True)
n = 0
with open(p, 'w', newline='') as f:
    w = csv.writer(f)
    for s in SHOTS:
        for r in range(REPS):
            w.writerow([TRK, 10000+r, s, 'GPU', 0.01, 3.0, 'sampling'])
            n += 1
    for r in range(REPS):
        w.writerow([TRK, 20000+r, 1, 'GPU', 0.01, 3.0, 'statevector'])
        n += 1
print(f"wrote {n} rows to {p}")
PY
    fi
}

ensure_csv_appC() {
    if [[ ! -f "${CSV_APPC}" ]]; then
        echo "[gen]  ${CSV_APPC}"
        python3 - <<'PY'
import csv
from pathlib import Path
# Sampling sweep: 1e6 shots, track count extended to 2000 (sparse-A path).
TC   = [2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]
REPS = {2:30, 5:30, 10:30, 15:30, 20:30, 30:20, 50:20, 75:10, 100:10,
        150:5, 200:5, 300:3, 500:3, 750:2, 1000:2, 1500:1, 2000:1}
SHOTS = 1_000_000
p = Path('condor_obqf/params_seg14e_quantum_appC.csv')
p.parent.mkdir(parents=True, exist_ok=True)
n = 0
with open(p, 'w', newline='') as f:
    w = csv.writer(f)
    for nt in TC:
        for r in range(REPS[nt]):
            w.writerow([nt, 30000+r, SHOTS, 'GPU', 0.01, 3.0, 'sampling'])
            n += 1
print(f"wrote {n} rows to {p}")
PY
    fi
}

submit_one() {
    local tag="$1"   # main | appA
    local csv="$2"
    local outroot="$3"

    local resdir="${outroot}/pickles"
    local logdir="${outroot}/logs"
    mkdir -p "${resdir}" "${logdir}"

    local njobs
    njobs=$(wc -l < "${csv}")
    echo "[submit] ${tag}: ${njobs} jobs"
    echo "         CSV     = ${csv}"
    echo "         results = ${resdir}"
    echo "         logs    = ${logdir}"

    condor_submit "${SUB}" \
        -append "PARAMS_CSV = ${csv}" \
        -append "RESULTSDIR = ${resdir}" \
        -append "LOGDIR     = ${logdir}"
}

# ------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------
case "${mode}" in
    main)
        ensure_csv_main
        submit_one main "${CSV_MAIN}" "${OUT_MAIN}"
        ;;
    appA|appendixA|appendix_a)
        ensure_csv_appA
        submit_one appA "${CSV_APPA}" "${OUT_APPA}"
        ;;
    appC|appendixC|appendix_c)
        ensure_csv_appC
        submit_one appC "${CSV_APPC}" "${OUT_APPC}"
        ;;
    both|"")
        ensure_csv_main
        ensure_csv_appA
        submit_one main "${CSV_MAIN}" "${OUT_MAIN}"
        submit_one appA "${CSV_APPA}" "${OUT_APPA}"
        ;;
    *)
        echo "Unknown mode: ${mode}"
        echo "Usage: $0 [main|appA|appC|both]"
        exit 1
        ;;
esac

echo "[ok] Submitted. Track progress with:  condor_q $USER"
