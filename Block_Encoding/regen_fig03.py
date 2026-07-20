"""Regenerate fig03_* from the final CSVs (wp99-aware figures())."""
import importlib.util
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("m03", os.path.join(HERE, "03_endtoend_filters.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

dfa = pd.read_csv(os.path.join(m.OUT, "03_degree_requirements.csv"))
dfb = pd.read_csv(os.path.join(m.OUT, "03_normalized_walk_metrics.csv"))
ds = pd.read_csv(os.path.join(m.OUT, "03_normalized_spectra.csv"))
m.figures(dfa, dfb, ds)
print("fig03 regenerated")
