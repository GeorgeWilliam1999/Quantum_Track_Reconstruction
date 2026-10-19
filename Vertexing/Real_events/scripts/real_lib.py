"""Real Run 3 simulated events (Gauss Sim10, 2024 conditions) in the focusing-scan format.

Input: the JSON written by dump_sim.py from a Gauss .sim file (one file per event): the
Geant4 VELO (VP) hits of the main bunch crossing with hit -> MC particle -> primary vertex
truth.  This module turns one into

  * the fs_lib.MultiEvent used by the focusing scan and the hit register
    (Vertexing/Focusing_scan/scripts/fs_lib.py, imported read-only), and
  * a package lhcb_velo_toy Event inside it (ev.pkg_event) for the segment Hamiltonian and
    the EventValidator.

What is done to the hits:
  * x and y are smeared by a Gaussian of sigma_xy (default 12 um; the Geant4 point is the
    mid-point of the step in the sensor, not a reconstructed cluster).
  * x and y are taken relative to the beamline (BEAM_X, BEAM_Y, the 2024 W40-42 Gauss beam
    spot), so the focusing scan's "beamline at x = y = 0" holds.
  * The package Event's modules are ordered as two chains, one per VELO half (even module
    numbers: x < 0 side, odd: x > 0), each sorted in z, with an empty separator module
    between them. A track stays in one half, so its consecutive hits are on consecutive
    modules of that chain (the left/right fix of Toy_Characterisation/Run3_Verification).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS = "/data/bfys/gscriven/Quantum_Track_Reconstruction/Vertexing/Focusing_scan/scripts"
if FS not in sys.path:
    sys.path.insert(0, FS)
import fs_lib as L  # noqa: E402

from lhcb_velo_toy.generation import PlaneGeometry  # noqa: E402
from lhcb_velo_toy.generation.entities.event import Event  # noqa: E402
from lhcb_velo_toy.generation.entities.hit import Hit  # noqa: E402
from lhcb_velo_toy.generation.entities.module import Module  # noqa: E402
from lhcb_velo_toy.generation.entities.primary_vertex import PrimaryVertex  # noqa: E402
from lhcb_velo_toy.generation.entities.track import Track  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
DUMPS = HERE / "sim"
BEAM_X, BEAM_Y = 0.450, 0.059      # mm, Beam6800GeV-md100-2024.W40.42-nu7.6.py
SIGMA_XY = 0.012                   # mm


@dataclass
class RealExtra:
    pv_key: np.ndarray            # MC key per pv index
    pv_n_reco: np.ndarray         # reconstructible tracks (>= 3 hits) per pv
    prompt: np.ndarray            # per track: origin vertex is its primary vertex
    electron: np.ndarray          # per track: |pid| == 11
    p: np.ndarray                 # per track momentum [MeV]
    n_modules: np.ndarray         # per track: distinct modules hit
    source: str


def dump_files(min_hits=50):
    """All dumped events with at least min_hits VELO hits (a few crossings have none)."""
    fs = sorted(DUMPS.glob("job_*/dump_*.json"), key=lambda p: (p.parent.name, int(p.stem.split("_")[1])))
    return [f for f in fs if len(json.load(open(f))["hits"]) >= min_hits]


def load_real(path, sigma_xy=SIGMA_XY, seed=0, zmax_track=None):
    d = json.load(open(path))
    H = np.asarray(d["hits"], float)
    rng = np.random.default_rng(seed)
    x = H[:, 0] - BEAM_X + (rng.normal(0, sigma_xy, len(H)) if sigma_xy > 0 else 0)
    y = H[:, 1] - BEAM_Y + (rng.normal(0, sigma_xy, len(H)) if sigma_xy > 0 else 0)
    z = H[:, 2]
    module = H[:, 4].astype(int)
    key = H[:, 5].astype(int)
    parts = d["particles"]
    pvs = d["pvs"]
    pv_index = {pv["key"]: i for i, pv in enumerate(pvs)}
    keys = sorted(set(key[key >= 0].tolist()))
    tid_of_key = {k: i for i, k in enumerate(keys)}
    track = np.array([tid_of_key.get(k, -1) for k in key])
    n_t = len(keys)
    pv_of_track = np.array([pv_index.get(parts[str(k)]["pv"], -1) for k in keys])
    t_of_track = np.array([[parts[str(k)]["tx"], parts[str(k)]["ty"]] for k in keys]).reshape(-1, 2)
    pv_xyz = np.array([[pv["xyz"][0] - BEAM_X, pv["xyz"][1] - BEAM_Y, pv["xyz"][2]] for pv in pvs])
    n_hits = np.bincount(track[track >= 0], minlength=n_t)
    n_mod = np.array([len(np.unique(module[track == t])) for t in range(n_t)])
    prompt = np.array([
        parts[str(k)]["origin"] is not None and parv >= 0 and
        np.linalg.norm(np.array(parts[str(k)]["origin"]) - np.array(pvs[parv]["xyz"])) < 1e-3
        for k, parv in zip(keys, pv_of_track)], bool)
    electron = np.array([abs(parts[str(k)]["pid"]) == 11 for k in keys], bool)
    pmom = np.array([parts[str(k)]["p"] for k in keys])
    pv_n_reco = np.array([int(((pv_of_track == i) & (n_hits >= 3)).sum()) for i in range(len(pvs))])

    # ── package Event ────────────────────────────────────────────────────────
    hits = [Hit(hit_id=i, x=float(x[i]), y=float(y[i]), z=float(z[i]), module_id=int(module[i]),
                track_id=int(track[i])) for i in range(len(x))]
    mods = sorted(set(module.tolist()))
    zmod = {m: float(np.mean(z[module == m])) for m in mods}
    chain = []
    for side in (0, 1):
        for m in sorted([m for m in mods if m % 2 == side], key=lambda m: zmod[m]):
            chain.append(Module(module_id=m, z=zmod[m], lx=50.0, ly=50.0,
                                hit_ids=[int(i) for i in np.nonzero(module == m)[0]]))
        if side == 0:
            chain.append(Module(module_id=1000, z=float("inf"), lx=50.0, ly=50.0, hit_ids=[]))
    geo = PlaneGeometry(module_id=[mm.module_id for mm in chain], lx=[50.0] * len(chain),
                        ly=[50.0] * len(chain), z=[mm.z for mm in chain])
    tracks = [Track(track_id=t, pv_id=int(pv_of_track[t]),
                    hit_ids=[int(i) for i in np.nonzero(track == t)[0]]) for t in range(n_t)]
    pvl = [PrimaryVertex(pv_id=i, x=float(pv_xyz[i, 0]), y=float(pv_xyz[i, 1]), z=float(pv_xyz[i, 2]),
                         track_ids=[int(t) for t in np.nonzero(pv_of_track == i)[0]])
           for i in range(len(pvs))]
    pkg = Event(detector_geometry=geo, primary_vertices=pvl, tracks=tracks, hits=hits, modules=chain,
                metadata=dict(source=str(path), sigma_xy=sigma_xy, beam=(BEAM_X, BEAM_Y)))
    ev = L.MultiEvent(x=x, y=y, z=z, module=module, track=track, pv_of_track=pv_of_track,
                      t_of_track=t_of_track, pv_xyz=pv_xyz, n_hits_of_track=n_hits, pkg_event=pkg)
    ex = RealExtra(pv_key=np.array([pv["key"] for pv in pvs]), pv_n_reco=pv_n_reco, prompt=prompt,
                   electron=electron, p=pmom, n_modules=n_mod, source=str(path))
    return ev, ex
