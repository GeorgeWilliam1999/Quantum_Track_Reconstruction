"""
Shared bootstrap + plotting style for the mini-Q-workshop presentation figures.

Every figure/animation script imports this first.  It:
  * wires up the PYTHONPATH so ``lhcb_velo_toy``, ``helpers`` and
    ``qtrk_pipeline`` all import (the package source + the _shared dir);
  * points QTRK_STORE at the canonical store;
  * sets one consistent matplotlib style for the whole deck;
  * provides ``savefig`` (PNG, presentation dpi) and ``FIG`` / ``ANIM`` dirs.

All data is regenerated fresh from the store/package (user policy 2026-06-08:
"regenerate everything"), so figures never reuse pre-threshold-fix PNGs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# --- paths --------------------------------------------------------------
PKG_SRC = Path("/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")
SHARED = Path("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared")
PRES = Path("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Presentation_mini_q_workshop_UM")
FIG = PRES / "figures"
ANIM = PRES / "animations"
ASSETS = PRES / "assets"
STORE = Path(os.environ.get("QTRK_STORE", "/data/bfys/gscriven/qtrk_store"))
METRICS_CSV = STORE / "manifest" / "metrics.csv"

for p in (PKG_SRC, SHARED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("QTRK_STORE", str(STORE))
for d in (FIG, ANIM, ASSETS):
    d.mkdir(parents=True, exist_ok=True)

# --- matplotlib style ---------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# A clean, high-contrast palette for a balanced talk.
C = {
    "classical": "#1f77b4",   # blue
    "quantum":   "#d62728",   # red
    "true":      "#2ca02c",   # green
    "false":     "#7f7f7f",   # grey
    "step":      "#1f77b4",
    "erf":       "#ff7f0e",   # orange
    "accent":    "#9467bd",   # purple
    "hhl":       "#8c564b",   # brown
    "onebqf":    "#d62728",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "legend.fontsize": 11,
    "legend.framealpha": 0.9,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "figure.facecolor": "white",
    "axes.edgecolor": "#444444",
})


def savefig(fig, name: str, sub: str | None = None):
    """Save ``fig`` to figures/ (or figures/<sub>/) as a PNG; return the path."""
    out = FIG if sub is None else (FIG / sub)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (name if name.endswith(".png") else name + ".png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(PRES)}")
    return path
