"""
Event generation, storage and retrieval for the qtrk data store.

An event is generated ONCE (deterministically, seed = ``seed_for(event_key)``)
and stored as compact float64 hit columns.  ``load_event`` rebuilds a fully
faithful :class:`Event` (hits, modules, truth tracks) so that regenerating the
matrix A from a stored event reproduces it bit-for-bit.

Positions are float64 on purpose: the ~4e-4 rad acceptance is finer than
float32 can resolve in a cos-of-near-parallel-vectors comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Make _shared importable so we reuse the single source of truth for geometry
# and generation (keeps every study in lockstep).
_SHARED = Path(__file__).resolve().parent.parent
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from .keys import event_key, seed_for  # noqa: E402
from .store import event_path  # noqa: E402

__all__ = ["generate_event", "save_event", "load_event", "ensure_event"]


def generate_event(
    n_trk: int,
    rep: int,
    sigma_scatt: float = 1e-4,
    sigma_res: float = 0.0,
    phi_max: float = 0.2,
    hit_ineff: float = 0.0,
    ghost_rate: float = 0.0,
    seed: Optional[int] = None,
):
    """Generate one event for the given physics point (deterministic).

    The seed is derived from the event key unless given explicitly, so the same
    physics point always yields the same event regardless of which study asks
    for it.  ``ghost_rate`` is accepted for forward-compatibility but injection
    is not yet implemented (only the four+Verify studies are in scope, all
    ghost_rate=0); a non-zero value raises so we never silently mislabel.
    """
    from helpers import make_geometry, safe_generate, apply_hit_inefficiency

    if ghost_rate and ghost_rate > 0.0:
        raise NotImplementedError(
            "ghost-hit injection not yet implemented (Segment_Grass only); "
            "the core studies all use ghost_rate=0."
        )

    ekey = event_key(n_trk, rep, sigma_scatt, sigma_res, phi_max,
                     hit_ineff, ghost_rate)
    if seed is None:
        seed = seed_for(ekey)

    geom = make_geometry()
    ev = safe_generate(
        int(n_trk), seed=int(seed), geom=geom,
        measurement_error=float(sigma_res),
        collision_noise=float(sigma_scatt),
        phi_max=float(phi_max), theta_max=float(phi_max),
    )
    if hit_ineff and hit_ineff > 0.0:
        rng_drop = np.random.default_rng(int(seed) + 31)
        ev = apply_hit_inefficiency(ev, float(hit_ineff), rng_drop)

    # Stash the physics + key on the event so save_event is self-describing.
    ev.metadata = dict(ev.metadata or {})
    ev.metadata.update(dict(
        event_key=ekey, n_trk=int(n_trk), rep=int(rep), seed=int(seed),
        sigma_scatt=float(sigma_scatt), sigma_res=float(sigma_res),
        phi_max=float(phi_max), hit_ineff=float(hit_ineff),
        ghost_rate=float(ghost_rate),
    ))
    return ev


def save_event(event, path: Optional[Path] = None) -> Path:
    """Persist an event via the package's canonical ``Event.to_dict`` schema.

    Stored as gzip-compressed JSON.  We reuse the package serializer (rather
    than a hand-rolled format) so events stay faithful to the library:
    geometry, tracks and primary vertices are preserved exactly, and JSON
    floats round-trip float64 so the regenerated matrix A is bit-identical.
    """
    import gzip
    import json

    ekey = event.metadata.get("event_key")
    if path is None:
        if ekey is None:
            raise ValueError("event has no 'event_key' in metadata; pass path explicitly")
        path = event_path(ekey)
    path = Path(path)

    from .store import atomic_write_bytes
    payload = gzip.compress(
        json.dumps(event.to_dict(), separators=(",", ":")).encode("utf-8"),
        compresslevel=6,
    )
    return atomic_write_bytes(path, payload)


def load_event(path):
    """Rebuild a faithful :class:`Event` via the package's ``Event.from_dict``.

    Geometry is embedded in the stored dict, so the event is self-contained and
    drives ``SimpleHamiltonianFast.construct_segments`` identically to the
    freshly generated one.
    """
    import gzip
    import json
    from lhcb_velo_toy.generation.entities.event import Event

    with gzip.open(path, "rb") as f:
        data = json.loads(f.read().decode("utf-8"))
    return Event.from_dict(data)


def ensure_event(
    n_trk: int, rep: int, sigma_scatt: float = 1e-4, sigma_res: float = 0.0,
    phi_max: float = 0.2, hit_ineff: float = 0.0, ghost_rate: float = 0.0,
    regenerate: bool = False,
):
    """Return (event, event_key), generating + caching to disk if absent."""
    ekey = event_key(n_trk, rep, sigma_scatt, sigma_res, phi_max,
                     hit_ineff, ghost_rate)
    path = event_path(ekey)
    if path.exists() and not regenerate:
        return load_event(path), ekey
    ev = generate_event(n_trk, rep, sigma_scatt, sigma_res, phi_max,
                        hit_ineff, ghost_rate)
    save_event(ev, path)
    return ev, ekey
