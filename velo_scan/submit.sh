#!/bin/bash
# =============================================================================
# submit.sh – Generate parameters and submit the Velo scan to HTCondor
#
# Run from the Nikhef login node:
#   cd /data/bfys/gscriven/velo_scan
#   bash submit.sh
# =============================================================================
set -e

BASE_DIR="/data/bfys/gscriven/velo_scan"
RESULTS_DIR="$BASE_DIR/results"
CONDA_PREFIX="/data/bfys/gscriven/conda"
CONDA="$CONDA_PREFIX/bin/conda"
ENV_NAME="Velo_Toy_Tester"

echo "============================================"
echo "  Velo Scan – Full Cartesian Grid"
echo "============================================"
echo ""

# ── Step 1: Generate parameter grid ────────────────────────────────
if [ ! -f "$RESULTS_DIR/params.csv" ]; then
    echo "[1/3] Generating parameter grid ..."
    "$CONDA" run --no-banner -n "$ENV_NAME" python "$BASE_DIR/scripts/gen_params.py" \
        --outdir "$RESULTS_DIR"
    echo ""
else
    echo "[1/3] params.csv already exists – skipping generation."
    echo ""
fi

# Verify files exist
if [ ! -f "$RESULTS_DIR/params.csv" ] || [ ! -f "$RESULTS_DIR/batches.txt" ]; then
    echo "[ERROR] params.csv or batches.txt not found in $RESULTS_DIR" >&2
    exit 1
fi

N_BATCHES=$(wc -l < "$RESULTS_DIR/batches.txt")
echo "  Total batches : $N_BATCHES"
echo "  Events/batch  : 50"
echo "  Total events  : $((N_BATCHES * 50))"
echo ""

# ── Step 2: Create log directory ───────────────────────────────────
mkdir -p "$RESULTS_DIR/logs"

# ── Step 3: Submit to condor ───────────────────────────────────────
echo "[2/3] Submitting scan jobs to HTCondor ..."
condor_submit "$BASE_DIR/condor/scan.sub" \
    runsdir="$RESULTS_DIR" \
    batches_file="$RESULTS_DIR/batches.txt"

echo ""
echo "[3/3] Jobs submitted!"
echo ""
echo "Monitor with:"
echo "  condor_q"
echo "  condor_q -nobatch"
echo ""
echo "After all jobs finish, run aggregation:"
echo "  condor_submit $BASE_DIR/condor/aggregate.sub runsdir=$RESULTS_DIR"
echo ""
echo "Or run aggregation locally:"
echo "  $CONDA run -n $ENV_NAME python $BASE_DIR/scripts/aggregate.py --runs-dir $RESULTS_DIR"
echo "============================================"
