#!/usr/bin/env python
"""Merge fig14-extension Condor outputs into the §14 caches and regenerate
the fig14_solver_segment_efficiency_drop1pct figure (no display floor).
"""
import pickle, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as _mt

ROOT      = Path('/data/bfys/gscriven/Quantum_Track_Reconstruction/'
                 'Toy_Characterisation/Verify_new_results')
RESDIR    = ROOT / 'results_fig14_ext'
CACHE_DIR = ROOT / 'outputs/segment_analysis/cache'
PAPER_DIR = ROOT / 'outputs/segment_analysis/paper'
PAPER_DIR.mkdir(parents=True, exist_ok=True)

CACHE_BASE = CACHE_DIR / 'solver_segment_efficiency_ext.pkl'
CACHE_DROP = CACHE_DIR / 'solver_segment_efficiency_ext_drop1pct.pkl'


def merge():
    base = pickle.load(open(CACHE_BASE, 'rb'))
    drop = pickle.load(open(CACHE_DROP, 'rb'))
    files = sorted(RESDIR.glob('fig14_n*_rep*.pkl'))
    if not files:
        print(f"no per-event pickles in {RESDIR}"); sys.exit(1)
    grouped = {}
    for fp in files:
        with open(fp, 'rb') as f:
            d = pickle.load(f)
        grouped.setdefault(d['n'], []).append(d)
    for n in sorted(grouped):
        recs = sorted(grouped[n], key=lambda d: d['rep'])
        recs_base = [r['base'] for r in recs]
        recs_drop = [r['drop'] for r in recs]
        if n in base:
            print(f"  n={n}: cache already has {len(base[n])} reps; replacing with {len(recs_base)} new reps")
        base[n] = recs_base
        drop[n] = recs_drop
        print(f"  n={n}: {len(recs_base)} reps merged")
    pickle.dump(base, open(CACHE_BASE, 'wb'))
    pickle.dump(drop, open(CACHE_DROP, 'wb'))
    print(f"caches updated: keys = {sorted(base.keys())}")
    return drop


def replot(drop_cache):
    ns = sorted(drop_cache.keys())
    raw = drop_cache

    def agg(key):
        m  = np.array([np.mean([r[key] for r in raw[n]]) for n in ns], dtype=float)
        se = np.array([np.std([r[key] for r in raw[n]], ddof=1) /
                       np.sqrt(len(raw[n])) if len(raw[n]) > 1 else 0.0
                       for n in ns], dtype=float)
        return m, se
    ntm, ntse = agg('n_true_all')
    nfm, nfse = agg('n_false_all')
    tam, tase = agg('n_true_active')
    fam, fase = agg('n_false_active')
    se_per = [[100.0 * r['n_true_active'] / max(r['n_true_clean'], 1)
               for r in raw[n]] for n in ns]
    se_m = np.array([np.mean(v) for v in se_per])
    se_e = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                     for v in se_per])
    fr_per = [[100.0 * r['n_false_active'] / max(r['n_active'], 1)
               for r in raw[n]] for n in ns]
    fr_m = np.array([np.mean(v) for v in fr_per])
    fr_e = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                     for v in fr_per])
    tc = np.array(ns, dtype=float)

    def _logclip(y, e, floor):
        y = np.asarray(y, float); e = np.asarray(e, float)
        lo = np.minimum(e, np.maximum(y - floor, 0.0))
        return np.vstack([lo, e])

    _c_drop  = '#e31a1c'; _c_fr    = '#c51b7d'
    _c_true  = '#2166ac'; _c_false = '#d6604d'

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.errorbar(tc, se_m, yerr=se_e, fmt='o-', color=_c_drop,
                capsize=5, capthick=1.5, markeredgecolor='black',
                markeredgewidth=0.8, zorder=3)
    ax.axhline(100, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
    _emin = max(0, np.floor((se_m - se_e).min()) - 1)
    ax.set_ylim(_emin, 100.5)
    ax.set_yticks(np.arange(int(_emin), 101, 1))
    ax.set_ylabel('Segment Efficiency (%)', fontsize=11)
    ax.set_title(r'i) Segment Efficiency ($N_{\mathrm{true\,act}} / N_{\mathrm{true\,all}}$)',
                 fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=1)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.yaxis.set_major_formatter(_mt.FormatStrFormatter('%g%%'))

    ax = axes[0, 1]
    ax.errorbar(tc, fr_m, yerr=fr_e, fmt='s-', color=_c_fr,
                capsize=5, capthick=1.5, markeredgecolor='black',
                markeredgewidth=0.8, zorder=3)
    ax.set_ylabel('Segment False Rate (%)', fontsize=11)
    ax.set_title(r'ii) Segment False Rate ($N_{\mathrm{false\,act}} / N_{\mathrm{accepted}}$)',
                 fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=1)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.yaxis.set_major_formatter(_mt.FormatStrFormatter('%g%%'))

    ax = axes[1, 0]
    ax.errorbar(tc, ntm, yerr=_logclip(ntm, ntse, 0.5),
                fmt='o-', color=_c_true,
                capsize=4, capthick=1.2, markeredgecolor='black',
                markeredgewidth=0.8, label='True segments', zorder=3)
    ax.errorbar(tc, nfm, yerr=_logclip(nfm, nfse, 0.5),
                fmt='s-', color=_c_false,
                capsize=4, capthick=1.2, markeredgecolor='black',
                markeredgewidth=0.8, label='False segments', zorder=3)
    ax.set_yscale('log')
    ax.set_ylabel('Number of Segment Pairs', fontsize=11)
    ax.set_title('iii) Segment Pair Counts', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, which='both', zorder=1)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)

    ax = axes[1, 1]
    ax.errorbar(tc, tam, yerr=_logclip(tam, tase, 0.5),
                fmt='o-', color=_c_true,
                capsize=4, capthick=1.2, markeredgecolor='black',
                markeredgewidth=0.8, label='True active', zorder=3)
    mfa = fam > 0
    if mfa.any():
        ax.errorbar(tc[mfa], fam[mfa],
                    yerr=_logclip(fam[mfa], fase[mfa], 1e-3),
                    fmt='s-', color=_c_false,
                    capsize=4, capthick=1.2, markeredgecolor='black',
                    markeredgewidth=0.8, label=r'False active (where $>0$)',
                    zorder=3)
    ax.set_yscale('log')
    ax.set_ylabel('Number of Active Segments', fontsize=11)
    ax.set_title('iv) Active Segment Pairs', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, which='both', zorder=1)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)

    for ax in axes.flat:
        ax.set_xlabel('Number of Tracks', fontsize=11)
    for ax in axes[1, :]:
        ax.set_xscale('log')
    fig.tight_layout()

    stem = 'fig14_solver_segment_efficiency_drop1pct'
    fig.savefig(PAPER_DIR / f'{stem}.pdf', format='pdf', dpi=600,
                bbox_inches='tight', facecolor='white', pad_inches=0.15)
    fig.savefig(PAPER_DIR / f'{stem}.png', dpi=300,
                bbox_inches='tight', facecolor='white', pad_inches=0.15)
    plt.close(fig)
    print(f"saved {stem}.{{pdf,png}} -> {PAPER_DIR}")
    print(f"n grid: {ns}")
    print(f"  fa means: {dict(zip(ns, [round(float(x),3) for x in fam]))}")


if __name__ == '__main__':
    drop = merge()
    replot(drop)
