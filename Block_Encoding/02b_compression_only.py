"""
02b -- redo ONLY the compression curves + all 02 figures (structure CSVs kept
from the interrupted 02 run; the FWHT is now vectorized).
"""

import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as sp
import pandas as pd

import be_lib as bl

# sanity: vectorized transforms still verify against the circuit-checked path
rng = np.random.default_rng(0)
M = rng.uniform(-1, 1, (6, 6)); M = (M + M.T) / 2
enc, kept = bl.fable_classical(M, 0.0)
assert np.abs(enc - M).max() < 1e-12, "vectorized FWHT broke fable_classical"
encL, _ = bl.lsfable_classical(M)

import importlib.util
spec = importlib.util.spec_from_file_location(
    "s02", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "02_structure_scaling.py"))
s02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module.__self__ if False else None
# load module without running main
import types
src = open(spec.origin).read()
mod = types.ModuleType("s02")
mod.__file__ = spec.origin
exec(compile(src, spec.origin, "exec"), mod.__dict__)

OUT = mod.OUT

comp_rows = []
thrs = np.geomspace(1e-6, 2.0, 25)
for T in mod.COMPRESS_T:
    mats = {"step": mod.build(T, "step", mod.CLEAN),
            "erf": mod.build(T, "erf", mod.CLEAN)}
    d = mats["step"].shape[0]
    rng = np.random.default_rng(3)
    R = rng.uniform(-1, 1, (d, d))
    R = sp.csr_matrix(np.where(rng.random((d, d)) < mats["step"].nnz / d ** 2,
                               (R + R.T) / 2, 0.0))
    mats["random_ctrl"] = R
    for name, Mx in mats.items():
        Ad = Mx.toarray()
        nrmA = np.linalg.norm(Ad)
        for thr in thrs:
            encS, keptS = bl.sfable_classical(Ad, thr)
            encF, keptF = bl.fable_classical(Ad, thr)
            comp_rows.append(dict(T=T, matrix=name, thr=thr,
                                  sfable_err=float(np.linalg.norm(encS - Ad) / nrmA),
                                  sfable_rot=keptS,
                                  fable_err=float(np.linalg.norm(encF - Ad) / nrmA),
                                  fable_rot=keptF))
        encL, keptL = bl.lsfable_classical(Ad)
        comp_rows.append(dict(T=T, matrix=name, thr=np.nan,
                              sfable_err=np.nan, sfable_rot=np.nan,
                              fable_err=float(np.linalg.norm(encL - Ad) / nrmA),
                              fable_rot=keptL, lsfable=True))
        print(f"compression T={T} {name}: LS-FABLE err "
              f"{np.linalg.norm(encL - Ad)/nrmA:.3f} rot {keptL}", flush=True)

dc = pd.DataFrame(comp_rows)
dc.to_csv(os.path.join(OUT, "02_compression_curves.csv"), index=False)

df = pd.read_csv(os.path.join(OUT, "02_structure.csv"))
dr = pd.read_csv(os.path.join(OUT, "02_method_resources.csv"))
mod.make_figures(df, dr, dc)
print("02b done")
