"""
Apply persistence load-guards to both notebooks.

For hit_competition_study.ipynb:
  - Modify imports cell: add load_json, find_latest_run, persistence imports
  - Insert load-checkpoint cell
  - Add load-or-compute guards to all 6 step compute cells

For characterisation.ipynb:
  - Add load-or-compute guards to all compute cells
  - Add conditionals to all save cells
"""

import json, textwrap, copy
from pathlib import Path

HERE = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _src(cell):
    return ''.join(cell['source'])

def _set_src(cell, text):
    cell['source'] = [line + "\n" for line in text.split("\n")]
    if cell['source']:
        cell['source'][-1] = cell['source'][-1].rstrip("\n")

def _make_code_cell(source_str, cell_id=None):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [],
    }
    _set_src(cell, source_str)
    if cell_id:
        cell["id"] = cell_id
    return cell

def _indent(text, n=4):
    """Indent every non-blank line by n spaces."""
    return textwrap.indent(text, " " * n)


# ═════════════════════════════════════════════════════════════════
#  Part 1 — hit_competition_study.ipynb
# ═════════════════════════════════════════════════════════════════

def apply_hit_competition():
    path = HERE / "hit_competition_study.ipynb"
    with open(path) as f:
        nb = json.load(f)

    # ── 1. Modify imports cell (Cell 1) ─────────────────────────
    src = _src(nb['cells'][1])

    # Add load_json and find_latest_run helpers after save_json
    load_helpers = '''
def load_json(path):
    """Load a JSON file and return the parsed object."""
    with open(path) as f:
        return json.load(f)

def find_latest_run(root: Path, tag=None):
    """Return the latest (or specified) run directory under *root*.
    If *tag* is a string, use it directly.  Otherwise pick the
    lexicographically last sub-directory (most recent timestamp).
    Returns None if nothing is found.
    """
    if tag is not None:
        d = root / tag
        return d if d.is_dir() else None
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    return candidates[-1] if candidates else None

'''
    marker = "def save_fig(fig, name):"
    idx = src.index(marker)
    src = src[:idx] + load_helpers + src[idx:]

    # Add persistence import
    old_imp = "from lhcb_velo_toy.analysis import EventValidator, Match"
    new_imp = (old_imp + "\n"
               "from lhcb_velo_toy.persistence import save_pipeline, load_pipeline")
    src = src.replace(old_imp, new_imp)
    _set_src(nb['cells'][1], src)

    # ── 2. Insert load-checkpoint cell after imports ────────────
    load_cell_src = """\
# ── Load previous run (set LOAD_RUN to skip expensive recomputation) ──
# Set LOAD_RUN = None to run fresh (default).
# Set LOAD_RUN = "20250101_120000" to load a specific previous run.
# Set LOAD_RUN = "latest" to auto-detect the most recent run.

LOAD_RUN = None   # ← change this to reload a previous run

_prev_data_dir = None
if LOAD_RUN is not None:
    _tag = None if LOAD_RUN == "latest" else LOAD_RUN
    _run_dir = find_latest_run(OUTPUT_ROOT, tag=_tag)
    if _run_dir is not None:
        _prev_data_dir = _run_dir / "data"
        print(f"Will load cached results from: {_prev_data_dir}")
    else:
        print(f"WARNING: requested run '{LOAD_RUN}' not found — running fresh")
else:
    print("LOAD_RUN is None → all steps will compute from scratch.")\
"""
    nb['cells'].insert(2, _make_code_cell(load_cell_src, "load_prev_run"))
    # After insert: original cell indices shift by 1
    # Step cells are now at 5, 8, 11, 14, 17, 20

    # ── 3. Add load guards to step cells ────────────────────────
    #
    # For each step: split cell into HEADER (comment + data init)
    # and BODY (computation starting from t0 = time.time()).
    # Wrap BODY in `if not _stepN_loaded:` with 4-space indent.

    step_configs = [
        # (cell_idx, step_num, split_marker, load_code)
        (5, 1, "\nt0 = time.time()\n", """\

_step1_loaded = False
if _prev_data_dir is not None:
    try:
        for angle in ANGLE_SETTINGS:
            raw = load_json(_prev_data_dir / f"step1_occupancy_angle{angle}.json")
            occ_data[angle] = {
                int(k): {
                    'all': np.array(v['all']),
                    'mean': v['mean'], 'median': v['median'],
                    'max': v['max'], 'std': v['std'],
                } for k, v in raw.items()
            }
        _step1_loaded = True
        print("Step 1: loaded from previous run.")
    except Exception as e:
        print(f"Step 1: load failed ({e}), recomputing...")

"""),

        (8, 2, "\nt0 = time.time()\n", """\

_step2_loaded = False
if _prev_data_dir is not None:
    try:
        for angle in ANGLE_SETTINGS:
            raw = load_json(_prev_data_dir / f"step2_activation_angle{angle}.json")
            act_data[angle] = {
                int(k): {
                    'true_x': np.array(v['true_x']),
                    'false_x': np.array(v['false_x']),
                } for k, v in raw.items()
            }
        _step2_loaded = True
        print("Step 2: loaded from previous run.")
    except Exception as e:
        print(f"Step 2: load failed ({e}), recomputing...")

"""),

        (11, 3, "\nt0 = time.time()\n", """\

_step3_loaded = False
if _prev_data_dir is not None:
    try:
        for angle in ANGLE_SETTINGS:
            raw = load_json(_prev_data_dir / f"step3_competition_angle{angle}.json")
            comp_data[angle] = {
                int(k): {
                    'true_act': np.array(v['true_act']),
                    'false_sum': np.array(v['false_sum']),
                    'n_competitors': np.array(v['n_competitors']),
                } for k, v in raw.items()
            }
        _step3_loaded = True
        print("Step 3: loaded from previous run.")
    except Exception as e:
        print(f"Step 3: load failed ({e}), recomputing...")

"""),

        (14, 4, "\nt0 = time.time()\n", """\

_step4_loaded = False
if _prev_data_dir is not None:
    try:
        reco_results = load_json(_prev_data_dir / "step4_reco_results.json")
        for r in reco_results:
            r['angle'] = float(r['angle'])
            r['n_tracks'] = int(r['n_tracks'])
        _step4_loaded = True
        print(f"Step 4: loaded {len(reco_results)} result rows from previous run.")
    except Exception as e:
        reco_results = []
        print(f"Step 4: load failed ({e}), recomputing...")

"""),

        (17, 5, "\nt0 = time.time()\n", """\

_step5_loaded = False
if _prev_data_dir is not None:
    try:
        for angle in ANGLE_SETTINGS:
            raw = load_json(_prev_data_dir / f"step5_roc_angle{angle}.json")
            roc_data[angle] = {
                int(k): {
                    'thresholds': np.array(v['thresholds']),
                    'eff': np.array(v['eff']),
                    'ghost': np.array(v['ghost']),
                } for k, v in raw.items()
            }
        _step5_loaded = True
        print("Step 5: loaded from previous run.")
    except Exception as e:
        print(f"Step 5: load failed ({e}), recomputing...")

"""),
    ]

    for cell_idx, step_num, split_marker, load_code in step_configs:
        src = _src(nb['cells'][cell_idx])
        # Split at `t0 = time.time()`
        parts = src.split(split_marker, 1)
        if len(parts) != 2:
            print(f"  WARNING: split marker not found in cell {cell_idx} (Step {step_num})")
            continue
        header = parts[0]   # comment + data init
        body = "t0 = time.time()\n" + parts[1]  # computation

        new_src = header + load_code + "if not _step{}_loaded:\n".format(step_num) + _indent(body)
        _set_src(nb['cells'][cell_idx], new_src)

    # ── Step 6 is special (uses t0_all, has pre-computation setup) ──
    src6 = _src(nb['cells'][20])
    # Split: everything before `t0_all = time.time()` is header (constants + epsilon calculation)
    # Everything from `t0_all = time.time()` onward is the computation body
    split6 = "\nt0_all = time.time()\n"
    parts6 = src6.split(split6, 1)
    if len(parts6) == 2:
        header6 = parts6[0]
        body6 = "t0_all = time.time()\n" + parts6[1]

        load6 = """\

_step6_loaded = False
if _prev_data_dir is not None:
    try:
        raw_scan = load_json(_prev_data_dir / "step6_scattering_scan.json")
        all_scan_results = {int(k): v for k, v in raw_scan.items()}
        raw_eps = load_json(_prev_data_dir / "step6_epsilons.json")
        all_epsilons = {int(k): float(v) for k, v in raw_eps.items()}
        _step6_loaded = True
        print("Step 6: loaded from previous run.")
    except Exception as e:
        print(f"Step 6: load failed ({e}), recomputing...")

"""
        new6 = header6 + load6 + "if not _step6_loaded:\n" + _indent(body6)
        _set_src(nb['cells'][20], new6)
    else:
        print("  WARNING: split marker not found for Step 6")

    # ── Write modified notebook ─────────────────────────────────
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

    # ── Validate all step cells compile ─────────────────────────
    ok = True
    for idx, label in [(5, "Step 1"), (8, "Step 2"), (11, "Step 3"),
                       (14, "Step 4"), (17, "Step 5"), (20, "Step 6")]:
        src = _src(nb['cells'][idx])
        try:
            compile(src, f"cell_{idx}", "exec")
            print(f"  {label} (cell {idx}): compiles OK")
        except SyntaxError as e:
            print(f"  {label} (cell {idx}): SYNTAX ERROR: {e}")
            ok = False
    return ok


# ═════════════════════════════════════════════════════════════════
#  Part 2 — characterisation.ipynb
# ═════════════════════════════════════════════════════════════════

def apply_characterisation():
    path = HERE / "characterisation.ipynb"
    with open(path) as f:
        nb = json.load(f)

    # Cell ID → index mapping (from attachment order)
    # Find cells by first-line content match
    def find_cell(start_text):
        for i, c in enumerate(nb['cells']):
            if _src(c).startswith(start_text):
                return i
        return None

    # ── Cell 4: generate clean event ────────────────────────────
    idx = find_cell("# ── generate a clean event")
    if idx is not None:
        _set_src(nb['cells'][idx], """\
# ── generate a clean event ──────────────────────────────────────
N_TRACKS_CLEAN = 3
N_MODULES_CLEAN = 5

geo_clean = make_geometry(N_MODULES_CLEAN, z_spacing=33.0)

if "clean" in _loaded:
    event_clean = _loaded["clean"].event
    print(f"Loaded clean event from cache: {len(event_clean.tracks)} tracks, "
          f"{len(event_clean.hits)} hits across {len(event_clean.modules)} modules")
else:
    for attempt in range(20):
        gen_clean, event_clean = generate_event(geo_clean, N_TRACKS_CLEAN)
        min_hits = min(len(t.hit_ids) for t in event_clean.tracks)
        if min_hits >= 3:
            break

    print(f"Clean event: {len(event_clean.tracks)} tracks, "
          f"{len(event_clean.hits)} hits across {len(event_clean.modules)} modules")
    for trk in event_clean.tracks:
        print(f"  Track {trk.track_id}: {len(trk.hit_ids)} hits")\
""")
        print(f"  Cell {idx} (clean event): updated")

    # ── Cell 6: Hamiltonian & solve ─────────────────────────────
    idx = find_cell("# ── build Hamiltonian & solve")
    if idx is not None:
        _set_src(nb['cells'][idx], """\
# ── build Hamiltonian & solve classically ────────────────────────
SIGMA_RES   = 0.0     # measurement resolution (mm) — 0 for clean event
SIGMA_SCATT = 1e-4    # collision noise (rad)
DZ_MM       = 33.0    # module spacing (mm)
SCALE       = 3.0     # accept segment pairs within 3 sigma

EPSILON = compute_epsilon(SIGMA_RES, SIGMA_SCATT, DZ_MM, scale=SCALE)
GAMMA   = 1.5    # self-interaction penalty
DELTA   = 1.0    # bias weight

print(f"Computed epsilon = {EPSILON:.6f} rad ({EPSILON*1e3:.3f} mrad)")

ham_clean = SimpleHamiltonianFast(epsilon=EPSILON, gamma=GAMMA, delta=DELTA)
A_clean, b_clean = ham_clean.construct_hamiltonian(event_clean)

BASELINE  = DELTA / (DELTA + GAMMA)
THRESHOLD = (1 + BASELINE) / 2

if "clean" in _loaded:
    x_clean = _loaded["clean"].solution
    print("Loaded solution from cache")
else:
    x_clean = ham_clean.solve_classicaly()

print(f"Segments:  {ham_clean.n_segments}")
print(f"Matrix A:  {A_clean.shape[0]}x{A_clean.shape[1]}, nnz={A_clean.nnz}")
print(f"Uncoupled baseline:  x ~ {BASELINE:.3f}")
print(f"Threshold:           {THRESHOLD:.3f}")
print(f"Active (above thr):  {np.sum(x_clean > THRESHOLD)} / {len(x_clean)} segments")
print(f"Min / Max activation: {x_clean.min():.4f} / {x_clean.max():.4f}")\
""")
        print(f"  Cell {idx} (Hamiltonian): updated")

    # ── Cell 7: reconstruct tracks ──────────────────────────────
    idx = find_cell("# ── reconstruct tracks")
    if idx is not None:
        _set_src(nb['cells'][idx], """\
# ── reconstruct tracks ──────────────────────────────────────────
if "clean" in _loaded and _loaded["clean"].reco_event is not None:
    reco_clean = list(_loaded["clean"].reco_event.tracks)
    print(f"Loaded {len(reco_clean)} reconstructed tracks from cache")
else:
    reco_clean = get_tracks(ham_clean, x_clean, event_clean, threshold=THRESHOLD)
    print(f"Reconstructed {len(reco_clean)} tracks")
for trk in reco_clean:
    print(f"  Reco track {trk.track_id}: {len(trk.hit_ids)} hits")\
""")
        print(f"  Cell {idx} (reconstruct): updated")

    # ── Cell 8: validate ────────────────────────────────────────
    idx = find_cell("# ── validate: match reco -> truth")
    if idx is not None:
        _set_src(nb['cells'][idx], """\
# ── validate: match reco -> truth ────────────────────────────
if "clean" in _loaded and _loaded["clean"].matches is not None:
    matches_clean = _loaded["clean"].matches
    metrics_clean = _loaded["clean"].metrics or {}
    print("Loaded validation results from cache")
else:
    val_clean = EventValidator(event_clean, reco_clean)
    matches_clean, metrics_clean = val_clean.match_tracks(purity_min=0.7)

print("=== Clean Event Metrics ===")
for k, v in metrics_clean.items():
    print(f"  {k:25s}: {v}")

print("\\nPer-track matches:")
print(f"  {'Truth':>6}  {'RecHits':>7}  {'Purity':>7}  {'HitEff':>7}  {'Type'}")
for m in matches_clean:
    label = "PRIMARY" if m.is_primary else ("GHOST" if m.is_ghost else "CLONE")
    truth_str = str(m.truth_id) if m.truth_id is not None else "-"
    print(f"  {truth_str:>6}  {m.rec_hits:7d}  "
          f"{m.purity:7.3f}  {m.hit_efficiency:7.3f}  {label}")\
""")
        print(f"  Cell {idx} (validate): updated")

    # ── Cell 9: save clean event (conditional) ──────────────────
    idx = find_cell("# ── Save clean-event pipeline")
    if idx is not None:
        _set_src(nb['cells'][idx], """\
# ── Save clean-event pipeline ────────────────────────────────────
# Only save if data was freshly computed (not loaded from cache).

if "clean" not in _loaded:
    save_pipeline(
        OUTPUT_ROOT / "clean_event",
        event=event_clean,
        ham=ham_clean,
        solution=x_clean,
        threshold=THRESHOLD,
        reco_tracks=reco_clean,
        matches=matches_clean,
        metrics=metrics_clean,
        extra_config={
            "sigma_res": SIGMA_RES,
            "sigma_scatt": SIGMA_SCATT,
            "dz_mm": DZ_MM,
            "scale": SCALE,
            "baseline": BASELINE,
        },
    )
    print(f"Clean event saved to {OUTPUT_ROOT / 'clean_event'}")
else:
    print("Clean event already cached — skipping save.")\
""")
        print(f"  Cell {idx} (save clean): updated")

    # ── Cell 11: 10-event batch ─────────────────────────────────
    idx = find_cell("# ── generate 10 events with varying track counts")
    if idx is not None:
        src = _src(nb['cells'][idx])
        # Split at "events_data = []" line — everything before is header
        header_end = src.index("events_data = []")
        header = src[:header_end]
        body = src[header_end:]

        load_block = """\
if "ten_events" in _loaded:
    _batch = _loaded["ten_events"]
    events_data = []
    for i, pr in enumerate(_batch):
        ev = pr.event
        reco_trks = list(pr.reco_event.tracks) if pr.reco_event else []
        true_segments = []
        for trk in ev.tracks:
            hids = trk.hit_ids
            for j in range(len(hids) - 1):
                true_segments.append((hids[j], hids[j+1]))
        reco_segments = []
        for trk in reco_trks:
            hids = trk.hit_ids
            for j in range(len(hids) - 1):
                reco_segments.append((hids[j], hids[j+1]))
        events_data.append({
            'event_id': i,
            'n_tracks_true': len(ev.tracks),
            'n_tracks_reco': len(reco_trks),
            'event': ev,
            'reco_tracks': reco_trks,
            'true_segments': true_segments,
            'reco_segments': reco_segments,
            'matches': pr.matches or [],
            'metrics': pr.metrics or {},
            'ham': None,
            'x': pr.solution,
        })
    print(f"Loaded {len(events_data)} events from cache")
    print(f"\\n=== Summary ===")
    print(f"{'Event':>6}  {'True':>5}  {'Reco':>5}  {'Eff':>6}  {'Ghost':>6}  {'TrueSeg':>8}  {'RecoSeg':>8}")
    for ed in events_data:
        eff = ed['metrics'].get('efficiency', 0)
        gr = ed['metrics'].get('ghost_rate', 0)
        print(f"{ed['event_id']:6d}  {ed['n_tracks_true']:5d}  {ed['n_tracks_reco']:5d}  "
              f"{eff:6.3f}  {gr:6.3f}  "
              f"{len(ed['true_segments']):8d}  {len(ed['reco_segments']):8d}")
else:
"""
        new_src = header + load_block + _indent(body)
        _set_src(nb['cells'][idx], new_src)
        print(f"  Cell {idx} (10 events): updated")

    # ── Cell 12: save 10 events (conditional) ───────────────────
    idx = find_cell("# ── Save 10-event batch")
    if idx is not None:
        src = _src(nb['cells'][idx])
        new_src = '# ── Save 10-event batch ──────────────────────────────────────────\n'
        new_src += 'if "ten_events" not in _loaded:\n'
        # Find the save_events_batch call
        save_start = src.index("save_events_batch(")
        save_block = src[save_start:]
        new_src += _indent(save_block)
        new_src += '\nelse:\n    print("10-event batch already cached — skipping save.")'
        _set_src(nb['cells'][idx], new_src)
        print(f"  Cell {idx} (save 10 events): updated")

    # ── Cell 15: 100×50 angle distributions ─────────────────────
    idx = find_cell("# ── Generate 100 events x 50 tracks")
    if idx is not None:
        src = _src(nb['cells'][idx])
        # Split: keep N_EVENTS, N_TRACKS_STUDY as header, rest is computation
        split_at = src.index("\nall_true_angles = []\n")
        header = src[:split_at]
        body = src[split_at + 1:]  # skip leading newline

        load_block = """\

if "angles_100x50" in _loaded:
    all_true_angles = _loaded["angles_100x50"]["true"]
    all_false_angles = _loaded["angles_100x50"]["false"]
    print(f"Loaded 100×50 angle distributions from cache "
          f"({len(all_true_angles):,} true, {len(all_false_angles):,} false)")
else:
"""
        new_src = header + load_block + _indent(body)
        _set_src(nb['cells'][idx], new_src)
        print(f"  Cell {idx} (100×50): updated")

    # ── Cell 16: save 100×50 (conditional) ──────────────────────
    idx = find_cell("# ── Save 100×50 angle distributions")
    if idx is not None:
        src = _src(nb['cells'][idx])
        new_src = '# ── Save 100×50 angle distributions ─────────────────────────────\n'
        new_src += 'if "angles_100x50" not in _loaded:\n'
        save_start = src.index("np.savez(")
        save_block = src[save_start:]
        new_src += _indent(save_block)
        new_src += '\nelse:\n    print("100×50 angle distributions already cached — skipping save.")'
        _set_src(nb['cells'][idx], new_src)
        print(f"  Cell {idx} (save 100×50): updated")

    # ── Cell 19: parametric scan ────────────────────────────────
    idx = find_cell("# ── Scan: segment metrics vs event size")
    if idx is not None:
        src = _src(nb['cells'][idx])
        # Split: constants (TRACK_SIZES..ANGLE_SETTINGS) are header
        # Computation starts at "all_scan_results = {}"
        split_at = src.index("\n# Store results keyed by (angle, mult)")
        header = src[:split_at]
        body_start = src.index("all_scan_results = {}")
        body = src[body_start:]

        load_block = """\

if "scan" in _loaded:
    _study = _loaded["scan"]
    # Convert columnar format back to list-of-dicts
    all_scan_results = {}
    for (a_str, m_str), metrics_dict in _study.scan_results.items():
        angle = float(a_str)
        mult = int(m_str)
        n = len(next(iter(metrics_dict.values())))
        all_scan_results[(angle, mult)] = [
            {k: float(v[i]) for k, v in metrics_dict.items()}
            for i in range(n)
        ]
    all_epsilons = {int(k): float(v) for k, v in _study.config.get("epsilons", {}).items()}
    # Fill in any missing epsilons
    for mult in SCATT_MULTIPLIERS:
        if mult not in all_epsilons:
            sigma_s = SIGMA_SCATT_BASE * mult
            all_epsilons[mult] = compute_epsilon(SIGMA_RES, sigma_s, DZ_MM, scale=SCALE)
    print(f"Loaded parametric scan from cache ({len(all_scan_results)} groups)")
    for angle in ANGLE_SETTINGS:
        for mult in SCATT_MULTIPLIERS:
            eps = all_epsilons[mult]
            print(f'  angle=±{angle}, {mult}x scattering: epsilon = {eps*1e3:.3f} mrad')
else:
"""
        new_src = header + "\n" + load_block + _indent(body)
        _set_src(nb['cells'][idx], new_src)
        print(f"  Cell {idx} (parametric scan): updated")

    # ── Cell 20: save parametric scan (conditional) ─────────────
    idx = find_cell("# ── Save parametric scan study")
    if idx is not None:
        src = _src(nb['cells'][idx])
        new_src = '# ── Save parametric scan study ───────────────────────────────────\n'
        new_src += 'if "scan" not in _loaded:\n'
        save_start = src.index("save_study(")
        save_block = src[save_start:]
        new_src += _indent(save_block)
        new_src += '\nelse:\n    print("Parametric scan already cached — skipping save.")'
        _set_src(nb['cells'][idx], new_src)
        print(f"  Cell {idx} (save scan): updated")

    # ── Cell 22: acceptance histograms ──────────────────────────
    idx = find_cell("# ── Acceptance histograms: true vs false")
    if idx is not None:
        src = _src(nb['cells'][idx])
        # Split: constants are header, computation fills hist_data,
        # then plotting code starts at "# ── Plot:"
        plot_marker = "# ── Plot: one 3"  # Plot section marker
        plot_idx = src.index(plot_marker)
        compute_part = src[:plot_idx]
        plot_part = src[plot_idx:]

        # In compute_part, find where computation starts
        comp_split = compute_part.index("\nhist_data = {}")
        comp_header = compute_part[:comp_split]
        comp_body = compute_part[comp_split + 1:]  # skip leading newline

        load_block = """\

if "hist" in _loaded:
    _hstudy = _loaded["hist"]
    hist_data = {}
    for (a_str, m_str), v in (_hstudy.hist_data or {}).items():
        hist_data[(float(a_str), int(m_str))] = v
    print(f"Loaded acceptance histograms from cache ({len(hist_data)} groups)")
else:
"""
        new_src = comp_header + load_block + _indent(comp_body) + "\n" + plot_part
        _set_src(nb['cells'][idx], new_src)
        print(f"  Cell {idx} (acceptance histograms): updated")

    # ── Cell 23: save acceptance histograms (conditional) ───────
    idx = find_cell("# ── Save acceptance histogram data")
    if idx is not None:
        src = _src(nb['cells'][idx])
        new_src = '# ── Save acceptance histogram data ─────────────────────────────\n'
        new_src += 'if "hist" not in _loaded:\n'
        save_start = src.index("save_study(")
        save_block = src[save_start:]
        new_src += _indent(save_block)
        new_src += '\nelse:\n    print("Acceptance histograms already cached — skipping save.")'
        _set_src(nb['cells'][idx], new_src)
        print(f"  Cell {idx} (save histograms): updated")

    # ── Write modified notebook ─────────────────────────────────
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

    # ── Validate all edited cells compile ───────────────────────
    ok = True
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = _src(c)
        # Skip cells with IPython magics
        if '%matplotlib' in src or 'get_ipython' in src:
            continue
        try:
            compile(src, f"cell_{i}", "exec")
        except SyntaxError as e:
            first_line = src.split('\n')[0][:60]
            print(f"  Cell {i}: SYNTAX ERROR: {e}  ({first_line})")
            ok = False
    if ok:
        print("  All cells compile OK")
    return ok


# ═════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Applying persistence to hit_competition_study.ipynb")
    print("=" * 60)
    ok1 = apply_hit_competition()

    print()
    print("=" * 60)
    print("Applying persistence to characterisation.ipynb")
    print("=" * 60)
    ok2 = apply_characterisation()

    print()
    if ok1 and ok2:
        print("✓ Both notebooks updated successfully!")
    else:
        print("✗ Some cells have errors — check output above")
