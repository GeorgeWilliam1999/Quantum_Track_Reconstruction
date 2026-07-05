"""
On-disk layout and IO for the qtrk data store.

Layout (root configurable via ``QTRK_STORE`` env var)::

    $QTRK_STORE/
      events/<event_key>.npz          # 6 hit columns + generation metadata
      solutions/<sol_key>.npz         # solution vector (float32) + scalars
      manifest/events.csv             # one row per unique event
      manifest/solutions.csv          # one row per planned/completed solve
      manifest/metrics.csv            # derived view (regenerable from the above)

Events keep float64 positions (float32 corrupts angle classification near the
~4e-4 rad acceptance); solution vectors are float32 (ample at the 0.35
threshold).  Truth is NOT stored — it is recomputed from the event.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

__all__ = [
    "store_root", "events_dir", "solutions_dir", "manifest_dir",
    "event_path", "solution_path", "atomic_savez", "atomic_write_bytes",
    "save_solution", "load_solution", "solution_exists",
]


def atomic_savez(path: Path, **arrays) -> Path:
    """Atomically write a compressed .npz (write to temp, then os.replace).

    Uses a temp name that already ends in ``.npz`` so numpy does not silently
    append a second ``.npz`` and break the rename.
    """
    path = Path(path)
    fd, tmpname = tempfile.mkstemp(suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmpname, **arrays)
        os.replace(tmpname, path)
    except BaseException:
        if os.path.exists(tmpname):
            os.remove(tmpname)
        raise
    return path

_DEFAULT_ROOT = "/data/bfys/gscriven/qtrk_store"


def store_root() -> Path:
    return Path(os.environ.get("QTRK_STORE", _DEFAULT_ROOT))


def events_dir() -> Path:
    p = store_root() / "events"
    p.mkdir(parents=True, exist_ok=True)
    return p


def solutions_dir() -> Path:
    p = store_root() / "solutions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def manifest_dir() -> Path:
    p = store_root() / "manifest"
    p.mkdir(parents=True, exist_ok=True)
    return p


def event_path(event_key_: str) -> Path:
    # Events are persisted via the package's canonical Event.to_dict schema,
    # gzip-compressed (see events.save_event).
    return events_dir() / f"{event_key_}.json.gz"


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    """Atomically write raw bytes (write to temp in same dir, then os.replace)."""
    path = Path(path)
    fd, tmpname = tempfile.mkstemp(suffix=path.suffix, dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmpname, path)
    except BaseException:
        if os.path.exists(tmpname):
            os.remove(tmpname)
        raise
    return path


def solution_path(sol_key_: str) -> Path:
    return solutions_dir() / f"{sol_key_}.npz"


# ---------------------------------------------------------------------------
# Solution IO
# ---------------------------------------------------------------------------

def save_solution(
    sol_key_: str,
    sol: np.ndarray,
    *,
    event_key_: str,
    ham_key_: str,
    solver: str,
    n_seg: int,
    n_true: int,
    overwrite: bool = False,
    **scalars,
) -> Path:
    """Persist a solver result.

    The solution vector is stored as float32.  ``n_seg``/``n_true`` are stored
    as cheap integrity checks so the metrics stage can assert its recomputed
    truth aligns with the vector.  Extra ``scalars`` (P_anc, n_sys, t_solve,
    epsilon, kernel, ...) are stored verbatim for provenance.
    """
    path = solution_path(sol_key_)
    if path.exists() and not overwrite:
        return path
    meta = dict(
        sol_key=sol_key_, event_key=event_key_, ham_key=ham_key_,
        solver=str(solver), n_seg=int(n_seg), n_true=int(n_true),
    )
    meta.update(scalars)
    # np.savez stores python scalars/strings as 0-d object arrays fine.
    return atomic_savez(path, sol=np.asarray(sol, dtype=np.float32),
                        meta=np.array(meta, dtype=object))


def load_solution(sol_key_: str) -> dict:
    """Return ``{'sol': ndarray(float32), **meta}`` for a stored solve."""
    path = solution_path(sol_key_)
    with np.load(path, allow_pickle=True) as z:
        out = dict(z["meta"].item())
        out["sol"] = z["sol"]
    return out


def solution_exists(sol_key_: str) -> bool:
    return solution_path(sol_key_).exists()


def load_metrics(study: str | None = None):
    """Load the regenerable metrics view (manifest/metrics.csv) as a DataFrame.

    This is what analysis notebooks read instead of their old result pickles.
    ``study`` filters to one study; the 'studies' column lists every requester.
    Membership is exact-token (a substring test would let 'Larger_Scatter'
    swallow every 'Larger_Scatter_Density' row).
    """
    import pandas as pd
    df = pd.read_csv(manifest_dir() / "metrics.csv")
    if study is not None:
        member = df["studies"].fillna("").str.split(r"[|,;\s]+").apply(
            lambda toks: study in toks)
        df = df[member | (df.get("study") == study)]
    return df.reset_index(drop=True)


EXPLODE_MAX = 50.0  # classical max|x| above this = lambda_min->0 explosion, not physics


def add_validity(df, explode_max: float = EXPLODE_MAX):
    """Attach the standard validity gate to a metrics-view DataFrame.

    Adds two columns: ``cls_max_abs`` (the classical partner's max_abs_x for
    the same event_key+ham_key — a row's own max_abs_x for classical rows) and
    ``valid`` (cls_max_abs <= explode_max).  Quantum/qsvt rows are judged by
    their classical partner: a rescaled quantum vector hides the explosion its
    linear system suffered.  Rows with no classical partner get valid=False.
    Per-event/per-cell means from metrics.csv MUST be taken over valid rows.
    """
    import pandas as pd
    cls = df[df["solver"] == "classical"]
    partner = (cls.groupby(["event_key", "ham_key"])["max_abs_x"]
                  .max().rename("cls_max_abs"))
    out = df.merge(partner, on=["event_key", "ham_key"], how="left")
    out["valid"] = out["cls_max_abs"].notna() & (out["cls_max_abs"] <= explode_max)
    return out
