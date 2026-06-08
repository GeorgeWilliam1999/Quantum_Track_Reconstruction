"""Regenerate fig14_solver_segment_efficiency_drop1pct without the 2000-track
data point and with larger tick-label fonts. Reads the cached per-event
solver records from outputs/segment_analysis/cache/ so it does not need to
re-run any simulation."""
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as _mt14s

HERE       = Path(__file__).resolve().parent
OUT_DIR    = HERE / "outputs" / "segment_analysis"
CACHE_DIR  = OUT_DIR / "cache"
PAPER_DIR  = OUT_DIR / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

# Track-count grid for the regenerated figure (drops the 2000-track point).
TRACK_COUNTS = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 75, 100,
                150, 200, 300, 500, 750, 1000, 1500]


def _agg_solver(raw, track_counts):
    tc   = np.array(track_counts, dtype=float)
    se_m, se_e = [], []
    fr_m, fr_e = [], []
    ntm, ntse  = [], []
    nfm, nfse  = [], []
    tam, tase  = [], []
    fam, fase  = [], []

    for k in track_counts:
        recs = raw[k]
        nr   = len(recs)
        eff  = np.array([r['n_true_active'] / max(r['n_true_clean'], 1) * 100
                         for r in recs])
        na_a = np.array([r['n_active'] for r in recs], dtype=float)
        fr   = np.array([r['n_false_active'] / max(na, 1) * 100
                         for r, na in zip(recs, na_a)])

        def sem14(a):
            return a.std(ddof=1) / np.sqrt(nr) if nr > 1 else 0.

        se_m.append(eff.mean());  se_e.append(sem14(eff))
        fr_m.append(fr.mean());   fr_e.append(sem14(fr))
        _nt = np.array([r['n_true_all']     for r in recs])
        _nf = np.array([r['n_false_all']    for r in recs])
        _ta = np.array([r['n_true_active']  for r in recs])
        _fa = np.array([r['n_false_active'] for r in recs])
        ntm.append(_nt.mean());  ntse.append(sem14(_nt))
        nfm.append(_nf.mean());  nfse.append(sem14(_nf))
        tam.append(_ta.mean());  tase.append(sem14(_ta))
        fam.append(_fa.mean());  fase.append(sem14(_fa))

    return dict(tc=tc,
                se_m=np.array(se_m),  se_e=np.array(se_e),
                fr_m=np.array(fr_m),  fr_e=np.array(fr_e),
                ntm=np.array(ntm),    ntse=np.array(ntse),
                nfm=np.array(nfm),    nfse=np.array(nfse),
                tam=np.array(tam),    tase=np.array(tase),
                fam=np.array(fam),    fase=np.array(fase))


def _logclip_yerr(y, e, floor):
    y = np.asarray(y, dtype=float)
    e = np.asarray(e, dtype=float)
    lo = np.minimum(e, np.maximum(y - floor, 0.0))
    return np.vstack([lo, e])


# Colour palette (matches the notebook).
_c14_drop  = '#e31a1c'
_c14_fr    = '#c51b7d'
_c14_true  = '#2166ac'
_c14_false = '#d6604d'

# Larger axis-number fontsize than the original (was matplotlib default ~10).
TICK_FS  = 13
LABEL_FS = 13

with open(CACHE_DIR / 'solver_segment_efficiency_ext_drop1pct.pkl', 'rb') as _f:
    _sol_drop = pickle.load(_f)

missing = [k for k in TRACK_COUNTS if k not in _sol_drop]
if missing:
    raise SystemExit(f"Cache is missing track counts: {missing}")

_agg_drop = _agg_solver(_sol_drop, TRACK_COUNTS)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
d1 = _agg_drop
tc = d1['tc']

# i) Segment efficiency  (linear x)
ax = axes[0, 0]
ax.errorbar(tc, d1['se_m'], yerr=d1['se_e'], fmt='o-', color=_c14_drop,
            capsize=5, capthick=1.5, markeredgecolor='black', markeredgewidth=0.8,
            zorder=3)
ax.axhline(100, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
_emin14s = max(0, np.floor((d1['se_m'] - d1['se_e']).min()) - 1)
ax.set_ylim(_emin14s, 100.5)
ax.set_yticks(np.arange(int(_emin14s), 101, 1))
ax.set_ylabel('Segment Efficiency (%)', fontsize=LABEL_FS)
ax.set_title(r'i) Segment Efficiency ($N_{\mathrm{true\,act}} / N_{\mathrm{true\,all}}$)',
             fontweight='bold')
ax.grid(True, alpha=0.3, zorder=1)
ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=TICK_FS)
ax.yaxis.set_major_formatter(_mt14s.FormatStrFormatter('%g%%'))

# ii) Segment false rate  (linear x)
ax = axes[0, 1]
ax.errorbar(tc, d1['fr_m'], yerr=d1['fr_e'], fmt='s-', color=_c14_fr,
            capsize=5, capthick=1.5, markeredgecolor='black', markeredgewidth=0.8,
            zorder=3)
ax.set_ylabel('Segment False Rate (%)', fontsize=LABEL_FS)
ax.set_title(r'ii) Segment False Rate ($N_{\mathrm{false\,act}} / N_{\mathrm{accepted}}$)',
             fontweight='bold')
ax.grid(True, alpha=0.3, zorder=1)
ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=TICK_FS)
ax.yaxis.set_major_formatter(_mt14s.FormatStrFormatter('%g%%'))

# iii) Total segment pairs  (log x, log y)
_FLOOR14_III = 0.5
ax = axes[1, 0]
ax.errorbar(tc, d1['ntm'], yerr=_logclip_yerr(d1['ntm'], d1['ntse'], _FLOOR14_III),
            fmt='o-', color=_c14_true,
            capsize=4, capthick=1.2, markeredgecolor='black', markeredgewidth=0.8,
            label='True segments', zorder=3)
ax.errorbar(tc, d1['nfm'], yerr=_logclip_yerr(d1['nfm'], d1['nfse'], _FLOOR14_III),
            fmt='s-', color=_c14_false,
            capsize=4, capthick=1.2, markeredgecolor='black', markeredgewidth=0.8,
            label='False segments', zorder=3)
ax.set_yscale('log')
ax.set_ylabel('Number of Segment Pairs', fontsize=LABEL_FS)
ax.set_title('iii) Segment Pair Counts', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
ax.grid(True, alpha=0.3, which='both', zorder=1)
ax.minorticks_on()
ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=TICK_FS)

# iv) Active segment pairs  (log x, log y)
_FLOOR14_IV = 0.5
ax = axes[1, 1]
_fa_p14s = np.maximum(d1['fam'], _FLOOR14_IV)
_fa_e14s = np.where(d1['fam'] >= _FLOOR14_IV, d1['fase'], 0.0)
ax.errorbar(tc, d1['tam'],
            yerr=_logclip_yerr(d1['tam'], d1['tase'], _FLOOR14_IV),
            fmt='o-', color=_c14_true,
            capsize=4, capthick=1.2, markeredgecolor='black', markeredgewidth=0.8,
            label='True active', zorder=3)
ax.errorbar(tc, _fa_p14s,
            yerr=_logclip_yerr(_fa_p14s, _fa_e14s, _FLOOR14_IV),
            fmt='s-', color=_c14_false,
            capsize=4, capthick=1.2, markeredgecolor='black', markeredgewidth=0.8,
            label='False active', zorder=3)
ax.set_yscale('log')
ax.set_ylim(_FLOOR14_IV * 0.7, None)
ax.set_ylabel('Number of Active Segments', fontsize=LABEL_FS)
ax.set_title('iv) Active Segment Pairs', fontweight='bold')
ax.legend(loc='lower right', framealpha=0.95, fontsize=10)
ax.grid(True, alpha=0.3, which='both', zorder=1)
ax.minorticks_on()
ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=TICK_FS)

for ax in axes.flat:
    ax.set_xlabel('Number of Tracks', fontsize=LABEL_FS)

# Bottom row: log x for panels iii) and iv)
for ax in axes[1, :]:
    ax.set_xscale('log')

fig.tight_layout()

_stem14s = 'fig14_solver_segment_efficiency_drop1pct'
fig.savefig(PAPER_DIR / f'{_stem14s}.pdf',
            format='pdf', dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(PAPER_DIR / f'{_stem14s}.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved {_stem14s} (track counts: {TRACK_COUNTS})")
