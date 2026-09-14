#!/usr/bin/env python3
"""check_figures.py — is every figure in the paper documented and fresh?

Two checks, both against the single ledger
``QSVT/Efficiency_Frontier/FIGURE_DATA_SOURCES.md``:

1. **Ledger coverage.** Every ``\\includegraphics`` and every TikZ ``\\input`` in
   ``main.tex`` must have a ``## `` heading in the ledger that names it.
2. **Hand-copy freshness.** The graphics the factory (``make_paper_figures.py``)
   does not draw are copied in by hand from a study folder. Each must hash-match
   its source; a cropped copy must not be older than its source.

Exit status 1 on any failure, so it can gate a commit or the /dayend pass.
Written 2026-09-14 after ``hamiltonian_spectrum.pdf`` turned out to be a silently
renamed copy of ``spectrum.pdf`` and five ``xiv_*`` copies were a day older than
their sources.  Usage: ``python check_figures.py [--json]``.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MAIN = HERE / "main.tex"
FIG = HERE / "figures"
LEDGER = REPO / "QSVT/Efficiency_Frontier/FIGURE_DATA_SOURCES.md"
FACTORY = HERE / "make_paper_figures.py"

EF = REPO / "QSVT/Efficiency_Frontier/figures"
TC_SEG = REPO / "Toy_Characterisation/Segment_level_studies/outputs"
Q_SEG = REPO / "QSVT/Segment_level_studies/outputs/depth_and_qubits"

# hand-copied graphics: paper stem -> source file (byte-identical expected)
HAND_COPIED = {
    "amplitude_atlas": TC_SEG / "amplitude_atlas/amplitude_atlas.pdf",
    "chain_amplitude_ladder": TC_SEG / "amplitude_atlas/chain_amplitude_ladder.pdf",
    "hamiltonian_spectrum": TC_SEG / "hamiltonian_spectrum/spectrum.pdf",   # renamed on copy
    "qsvt_width_and_P": Q_SEG / "qsvt_width_and_P.pdf",
    "qsvt_total_depth": Q_SEG / "qsvt_total_depth.pdf",
    **{s: EF / f"{s}.pdf" for s in (
        "xiv_degree_marginal", "xiv_degree_scan_heavy", "xiv_degree_scan_moderate",
        "xiv_fig6_efficiency_first", "xiv_loss_budget", "xiv_metrics_composite",
        "xiv_motif_lines_modified", "xiv_responses_modified")},
}
# processed by the factory (crops): only "source not newer than copy" is checkable
PROCESSED = {
    "dp_occupancy_proof": REPO / "Toy_Characterisation/Bifurification/figures/dp_occupancy_proof.png",
    "run3_activation_spectra": REPO / "QSVT/Run3/outputs/run3_activation_spectra.png",
}
# ledger headings that are notes, not figure entries
NOTE_HEADINGS = ("Paper-copy freshness", "CONVENTION NOTE", "Action items",
                 "Entries added", "Cluster census", "Two-tooth", "Section-4 tables",
                 "HHL segment-level", "seg_store", "NOT YET")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def paper_graphics() -> tuple[list[str], list[str]]:
    tex = MAIN.read_text()
    gfx = sorted({Path(m).stem for m in re.findall(r"\\includegraphics[^{]*\{([^}]*)\}", tex)})
    tikz = sorted({Path(m).stem for m in re.findall(r"\\input\{figures/([^}]*)\}", tex)}
                  - {"circuit_styles"})
    return gfx, tikz


def factory_stems() -> set[str]:
    src = FACTORY.read_text()
    return set(re.findall(r'save\(fig,\s*"([^"]+)"', src)) | set(PROCESSED)


def ledger_headings() -> list[str]:
    return [l[3:] for l in LEDGER.read_text().splitlines()
            if l.startswith("## ") and not l[3:].startswith(NOTE_HEADINGS)]


def main() -> int:
    gfx, tikz = paper_graphics()
    heads = ledger_headings()
    fac = factory_stems()
    rows, bad = [], 0

    for stem in gfx + tikz:
        entry = any(stem in h for h in heads)
        state = "ledger ok" if entry else "NO LEDGER ENTRY"
        if not entry:
            bad += 1
        rows.append({"figure": stem, "ledger": entry, "state": state})

    for stem, src in HAND_COPIED.items():
        copy = FIG / f"{stem}.pdf"
        r = next(x for x in rows if x["figure"] == stem)
        if not src.exists():
            r["state"], bad = "SOURCE MISSING", bad + 1
        elif md5(copy) != md5(src):
            r["state"], bad = f"STALE (differs from {src.relative_to(REPO)})", bad + 1
        else:
            r["state"] += ", hash matches source"
    for stem, src in PROCESSED.items():
        copy = FIG / f"{stem}.png"
        r = next(x for x in rows if x["figure"] == stem)
        if src.stat().st_mtime > copy.stat().st_mtime:
            r["state"], bad = "SOURCE NEWER THAN CROP", bad + 1
        else:
            r["state"] += ", crop not older than source"

    unaccounted = [s for s in gfx if s not in fac and s not in HAND_COPIED and s not in PROCESSED]
    for s in unaccounted:
        bad += 1
        next(x for x in rows if x["figure"] == s)["state"] += "; NOT FACTORY, NOT LISTED AS HAND-COPIED"

    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1))
    else:
        w = max(len(r["figure"]) for r in rows)
        for r in rows:
            print(f"{r['figure']:<{w}}  {r['state']}")
        print(f"\n{len(gfx)} graphics + {len(tikz)} TikZ; {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
