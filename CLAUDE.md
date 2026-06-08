# CLAUDE.md — Quantum Track Reconstruction

## Memory

All persistent memory lives at `/data/bfys/gscriven/.claude/memory/`.
At the start of every session, read `/data/bfys/gscriven/.claude/memory/MEMORY.md` and then load any referenced memory files before doing anything else.
Never read from or write to `/user/gscriven` — that filesystem is not used here.

## Environment

- Working directory: `/data/bfys/gscriven/Quantum_Track_Reconstruction`
- Cluster: Nikhef HTCondor
- Conda env for notebooks: `Q_env`
- All data, results, and outputs: under `/data/bfys/gscriven/`
