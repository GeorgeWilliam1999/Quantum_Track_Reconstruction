"""
Content-addressed keys for the qtrk data store.

A data point is decomposed into three independent layers, each with its own
deterministic key derived from the physics/config that defines it:

- ``event_key``  : the generated event  (T, noises, cone, drop/ghost, rep)
- ``ham_key``    : the Hamiltonian/matrix A built ON an event
                   (epsilon + provenance, kernel, erf_sigma, gamma, delta)
- ``sol_key``    : a solver run of a (event, ham) pair
                   (solver, device, readout)

Keys are short blake2b digests of a canonical string, so:
- regenerating the same physics point yields the SAME key (idempotent / dedup),
- the three layers compose (sol_key embeds event_key + ham_key),
- floats are canonicalised to 12 significant figures so 1e-4 and 0.0001 agree.

Nothing here imports the toy package; keys are pure functions of parameters.
"""
from __future__ import annotations

import hashlib

__all__ = [
    "fnum", "event_key", "ham_key", "sol_key", "seed_for",
]


def fnum(x: float) -> str:
    """Canonical float string: 12 significant figures, stable across 1e-4/0.0001."""
    return f"{float(x):.12g}"


def _digest(canonical: str, n: int = 12) -> str:
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()[:n]


def event_key(
    n_trk: int,
    rep: int,
    sigma_scatt: float,
    sigma_res: float,
    phi_max: float,
    hit_ineff: float = 0.0,
    ghost_rate: float = 0.0,
) -> str:
    """Stable key for one generated event (physics only — no solver/kernel)."""
    canonical = "|".join([
        "ev",
        f"T={int(n_trk)}",
        f"rep={int(rep)}",
        f"ss={fnum(sigma_scatt)}",
        f"sr={fnum(sigma_res)}",
        f"phi={fnum(phi_max)}",
        f"drop={fnum(hit_ineff)}",
        f"ghost={fnum(ghost_rate)}",
    ])
    return "ev_" + _digest(canonical)


def ham_key(
    epsilon: float,
    kernel: str = "step",
    erf_sigma: float = 1e-4,
    gamma: float = 3.0,
    delta: float = 1.0,
    eps_provenance: str = "formula",
    fork_beta: float = 0.0,
    fork_eps: float | None = None,
) -> str:
    """Stable key for the matrix A built on an event.

    ``kernel`` in {"step", "erf"}.  ``erf_sigma`` only matters for "erf" but is
    always folded in so a key uniquely identifies the matrix.  ``eps_provenance``
    in {"formula", "set"} records how epsilon was chosen (it does not change A,
    but it changes how a point should be interpreted in analysis).

    ``fork_beta`` / ``fork_eps`` identify the epsilon-windowed bifurcation
    (Denby-Peterson fork) modification  A' = A0 + fork_beta * B_{fork_eps}
    (B = co-hit pairs with mutual angle < fork_eps; the fork window may differ
    from the acceptance epsilon).  ``fork_beta == 0`` is normalised out so all
    pre-existing keys are unchanged.
    """
    kern = str(kernel).lower()
    if kern not in ("step", "erf"):
        raise ValueError(f"kernel must be 'step' or 'erf', got {kernel!r}")
    # For the step kernel erf_sigma is irrelevant to A: normalise it out so
    # step points never fragment across erf_sigma values.
    erfs = fnum(erf_sigma) if kern == "erf" else "na"
    parts = [
        "hm",
        f"eps={fnum(epsilon)}",
        f"prov={eps_provenance}",
        f"kern={kern}",
        f"erfs={erfs}",
        f"g={fnum(gamma)}",
        f"d={fnum(delta)}",
    ]
    if float(fork_beta) != 0.0:
        fe = fork_eps if fork_eps is not None else epsilon
        parts += [f"fb={fnum(fork_beta)}", f"fe={fnum(fe)}"]
    canonical = "|".join(parts)
    return "hm_" + _digest(canonical)


def sol_key(
    event_key_: str,
    ham_key_: str,
    solver: str,
    device: str = "CPU",
    readout: str = "statevector",
) -> str:
    """Stable key for a solver run of a (event, ham) pair.

    ``solver`` in {"classical", "quantum", "qsvt"}.  For "classical",
    device/readout are normalised out so a classical solve never fragments
    across them.

    NOTE: device IS part of the quantum key (legacy).  An exact statevector solve
    is device-independent (CPU == GPU result), so re-routing a quantum solve
    between CPU and GPU changes its key and *looks* undone.  We handle that by
    copying existing GPU statevector solutions onto their CPU key (see the
    re-route recovery), rather than changing the key scheme on a live campaign.
    For "qsvt" (exact statevector semantics, device-independent) the device is
    normalised out from the start; only the readout is kept.
    """
    slv = str(solver).lower()
    if slv not in ("classical", "quantum", "qsvt"):
        raise ValueError(
            f"solver must be 'classical', 'quantum' or 'qsvt', got {solver!r}")
    if slv == "classical":
        dev, rdt = "na", "na"
    elif slv == "qsvt":
        dev, rdt = "na", str(readout).lower()
    else:
        dev, rdt = str(device).upper(), str(readout).lower()
    canonical = "|".join([event_key_, ham_key_, f"slv={slv}", f"dev={dev}", f"rdt={rdt}"])
    return "sol_" + _digest(canonical, n=16)


def seed_for(event_key_: str) -> int:
    """Deterministic 31-bit generation seed derived from an event key."""
    h = hashlib.blake2b(event_key_.encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) % (2 ** 31)
