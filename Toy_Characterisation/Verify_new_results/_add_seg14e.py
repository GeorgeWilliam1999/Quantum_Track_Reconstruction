"""Add §7d — direct §14e mirror (2x2 panels: Efficiency, False Rate,
Segment Pair Counts, Active Segment Pairs) with classical + E1 + E2 overlaid.
Inserted before §11 output artefacts."""
import json
from pathlib import Path

P = Path('/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation'
         '/Verify_new_results/Quantum_segment_level_analysis.ipynb')
nb = json.loads(P.read_text())

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}

MD = r"""## §7d Figure 5 — direct §14e mirror (classical + quantum overlay)

Same four panels as `fig14_solver_segment_efficiency_drop1pct` in
`segment_level_analysis.ipynb` §14e, but with the **quantum** (`OneBitHHL`)
solver overlaid on top of the classical solver at every point on the
$T \in [2, 1000]$ grid, for both readouts (E1 statevector, E2 sampling).

Panels
- **i)** Segment Efficiency $= N_\mathrm{true\,act}\,/\,N_\mathrm{true\,all}$
- **ii)** Segment False Rate $= N_\mathrm{false\,act}\,/\,N_\mathrm{accepted}$
- **iii)** Segment Pair Counts (true / false) — combinatoric, identical for C and Q
- **iv)** Active Segment Pairs (true active / false active) — solver-dependent
"""

CODE = r"""# §7d Figure 5 — §14e mirror: classical + quantum on the same 2x2 layout
def _pull_seg14(pdir):
    rows = []
    for fp in sorted(Path(pdir).glob('event_*.pkl')):
        try:    r = pickle.load(open(fp, 'rb'))
        except Exception: continue
        sC, sQ = r.get('seg_C'), r.get('seg_Q')
        if not sC or not sQ: continue
        n_true_clean = float(sC.get('n_true_clean') or sC.get('n_true_all') or np.nan)
        n_true_all   = float(sC.get('n_true_all',   np.nan))
        n_false_all  = float(sC.get('n_false_all',  np.nan))
        def _eff(s):
            n = s.get('n_true_clean') or s.get('n_true_all')
            return 100.0 * s['n_true_active'] / n if n else np.nan
        def _fr(s):
            return 100.0 * s['n_false_active'] / s['n_active'] if s.get('n_active') else np.nan
        rows.append(dict(
            n_trk=r['n_trk'], rep=r['rep'],
            n_true_all=n_true_all, n_false_all=n_false_all,
            true_act_C=float(sC.get('n_true_active', np.nan)),
            false_act_C=float(sC.get('n_false_active', np.nan)),
            true_act_Q=float(sQ.get('n_true_active', np.nan)),
            false_act_Q=float(sQ.get('n_false_active', np.nan)),
            eff_C_pct=_eff(sC), eff_Q_pct=_eff(sQ),
            fr_C_pct=_fr(sC),   fr_Q_pct=_fr(sQ),
        ))
    return pd.DataFrame(rows)

def _agg14(df):
    if df.empty: return df
    cols = ['n_true_all','n_false_all','true_act_C','false_act_C',
            'true_act_Q','false_act_Q','eff_C_pct','eff_Q_pct',
            'fr_C_pct','fr_Q_pct']
    sem = lambda x: (x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 1 else 0.0
    spec = {f'{c}_mean': (c, 'mean') for c in cols}
    spec.update({f'{c}_sem': (c, sem) for c in cols})
    spec['n_reps'] = ('rep', 'count')
    return df.groupby('n_trk').agg(**spec).reset_index().sort_values('n_trk')

df14_sv = _pull_seg14(PIK_SV)
df14_sm = _pull_seg14(PIK_SM)
g_sv = _agg14(df14_sv)
g_sm = _agg14(df14_sm)
print(f'E1 events: {len(df14_sv)}    E2 events: {len(df14_sm)}')

def _logclip_yerr(y, e, floor):
    y = np.asarray(y, dtype=float); e = np.asarray(e, dtype=float)
    lo = np.minimum(e, np.maximum(y - floor, 0.0))
    return np.vstack([lo, e])

# §14e palette (classical = blue, quantum = warm)
c14_C, c14_Qsv, c14_Qsm = '#2166ac', '#d62728', '#e08214'
c14_true, c14_false = '#1b7837', '#c51b7d'

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# ----------------------------------------------------------------------------
# i) Segment Efficiency  (linear x)
# ----------------------------------------------------------------------------
ax = axes[0, 0]
def _eb(ax, g, ycol, fmt, color, label, mfc=None, capsize=5, capthick=1.5):
    if g.empty: return
    kw = dict(fmt=fmt, color=color, capsize=capsize, capthick=capthick,
              markeredgecolor='black', markeredgewidth=0.8, zorder=3, label=label)
    if mfc is not None: kw['mfc'] = mfc
    ax.errorbar(g['n_trk'], g[f'{ycol}_mean'], yerr=g[f'{ycol}_sem'], **kw)

_eb(ax, g_sv, 'eff_C_pct', 'D--', c14_C,   'Classical solver', mfc='white')
_eb(ax, g_sv, 'eff_Q_pct', 'o-',  c14_Qsv, 'OneBitHHL (E1 statevector)')
_eb(ax, g_sm, 'eff_Q_pct', 's-',  c14_Qsm, 'OneBitHHL (E2 sampling)')
ax.axhline(100, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
ax.set_ylabel('Segment Efficiency (%)')
ax.set_title(r'i) Segment Efficiency ($N_{\mathrm{true\,act}} / N_{\mathrm{true\,all}}$)',
             fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax.legend(loc='lower left', framealpha=0.95); _style_axes(ax)

# ----------------------------------------------------------------------------
# ii) Segment False Rate  (linear x, log y so we can see the classical floor)
# ----------------------------------------------------------------------------
ax = axes[0, 1]
_eb(ax, g_sv, 'fr_C_pct', 'D--', c14_C,   'Classical solver', mfc='white')
_eb(ax, g_sv, 'fr_Q_pct', 'o-',  c14_Qsv, 'OneBitHHL (E1 statevector)')
_eb(ax, g_sm, 'fr_Q_pct', 's-',  c14_Qsm, 'OneBitHHL (E2 sampling)')
ax.set_ylabel('Segment False Rate (%)')
ax.set_title(r'ii) Segment False Rate ($N_{\mathrm{false\,act}} / N_{\mathrm{accepted}}$)',
             fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax.legend(loc='upper left', framealpha=0.95); _style_axes(ax)

# ----------------------------------------------------------------------------
# iii) Segment Pair Counts (log x, log y) — solver-independent (use E1)
# ----------------------------------------------------------------------------
_FLOOR_III = 0.5
ax = axes[1, 0]
if not g_sv.empty:
    ax.errorbar(g_sv['n_trk'], g_sv['n_true_all_mean'],
                yerr=_logclip_yerr(g_sv['n_true_all_mean'],
                                   g_sv['n_true_all_sem'], _FLOOR_III),
                fmt='o-', color=c14_true,
                capsize=4, capthick=1.2,
                markeredgecolor='black', markeredgewidth=0.8,
                label='True segments', zorder=3)
    ax.errorbar(g_sv['n_trk'], g_sv['n_false_all_mean'],
                yerr=_logclip_yerr(g_sv['n_false_all_mean'],
                                   g_sv['n_false_all_sem'], _FLOOR_III),
                fmt='s-', color=c14_false,
                capsize=4, capthick=1.2,
                markeredgecolor='black', markeredgewidth=0.8,
                label='False segments', zorder=3)
ax.set_yscale('log')
ax.set_ylabel('Number of Segment Pairs')
ax.set_title('iii) Segment Pair Counts', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.95); _style_axes(ax)

# ----------------------------------------------------------------------------
# iv) Active Segment Pairs (log x, log y) — classical vs quantum
# ----------------------------------------------------------------------------
_FLOOR_IV = 0.5
ax = axes[1, 1]
def _eb_act(ax, g, ycol, fmt, color, label, mfc=None):
    if g.empty: return
    y  = g[f'{ycol}_mean'].values
    e  = g[f'{ycol}_sem'].values
    yp = np.maximum(y, _FLOOR_IV)
    ep = np.where(y >= _FLOOR_IV, e, 0.0)
    kw = dict(fmt=fmt, color=color, capsize=4, capthick=1.2,
              markeredgecolor='black', markeredgewidth=0.8, zorder=3, label=label)
    if mfc is not None: kw['mfc'] = mfc
    ax.errorbar(g['n_trk'], yp, yerr=_logclip_yerr(yp, ep, _FLOOR_IV), **kw)

# True-active
_eb_act(ax, g_sv, 'true_act_C',  'D--', c14_C,   'True active — classical', mfc='white')
_eb_act(ax, g_sv, 'true_act_Q',  'o-',  c14_Qsv, 'True active — E1')
_eb_act(ax, g_sm, 'true_act_Q',  's-',  c14_Qsm, 'True active — E2')
# False-active
_eb_act(ax, g_sv, 'false_act_C', 'D:',  c14_false, 'False active — classical', mfc='white')
_eb_act(ax, g_sv, 'false_act_Q', 'o:',  '#7b3294', 'False active — E1')
_eb_act(ax, g_sm, 'false_act_Q', 's:',  '#bf812d', 'False active — E2')
ax.set_yscale('log')
ax.set_ylim(_FLOOR_IV * 0.7, None)
ax.set_ylabel('Number of Active Segments')
ax.set_title('iv) Active Segment Pairs', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.95, fontsize=8, ncol=2); _style_axes(ax)

for ax in axes.flat:
    ax.set_xlabel('Number of Tracks')
# Bottom row: log x (matches §14e)
for ax in axes[1, :]:
    ax.set_xscale('log')
# Top row: linear x (matches §14e)
for ax in axes[0, :]:
    ax.set_xscale('linear')

fig.tight_layout()
_save_paper(fig, 'fig5_seg14e_mirror_classical_vs_quantum')
plt.show()
"""

# Insert at position 19 (before §11 markdown)
nb['cells'].insert(19, md(MD))
nb['cells'].insert(20, code(CODE))

P.write_text(json.dumps(nb, indent=1))
print(f'inserted §7d (Figure 5) — notebook now has {len(nb["cells"])} cells')
