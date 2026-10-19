#!/bin/bash
# usage: dump_job.sh <job_dir>  -- dumps the job's .sim to <job_dir>/dump_<n>.json
S=/data/bfys/gscriven/Quantum_Track_Reconstruction/Vertexing/Real_events/sim
d=$1; cp $S/../scripts/dump_sim.py $S/$d/
f=$(ls $S/$d/*.sim | head -1)
env -i HOME=/data/bfys/gscriven/lbhome PATH=/usr/bin:/bin TERM=dumb bash -lc "source /cvmfs/lhcb.cern.ch/lib/LbEnv >/dev/null 2>&1; cd $S/$d; lb-run Gauss/v56r9 python dump_sim.py $(basename $f) dump" > $S/$d/dump.log 2>&1
grep "^event" $S/$d/dump.log
