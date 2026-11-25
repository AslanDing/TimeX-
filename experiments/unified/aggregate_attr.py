"""Aggregate the attribution results (e0*) of TimeX_a++: mean over splits of the per-split AUPRC/AUP/AUR (paper
protocol: test series with ground-truth positives) and the standard error over test instances of that mean,
sqrt(sum_s se_s^2) / n_splits. Writes results*/attr_rows.json and prints a summary; also plots the r sensitivity."""
import os, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_D = os.environ.get('TIMEX_DIST', 'kl'); _W = float(os.environ.get('TIMEX_DIST_W', '0.1'))
RES = os.path.join(HERE, 'results' + ('' if _D == 'kl' else f'_{_D}' + ('' if _W == 0.1 else f'_w{_W:g}')))
DSS = ['freqshape', 'seqcomb_uv', 'seqcomb_mv', 'lowvar', 'mitecg']
M = ['auprc', 'aup', 'aur']


def agg(variant, ds, splits=None):
    fs = sorted(glob.glob(os.path.join(RES, 'e0' + variant, f'{ds}_split=*.json')))
    J = [json.load(open(f)) for f in fs]
    J = [j for j in J if 'auprc' in j and (splits is None or j['split'] in splits)]
    if not J:
        return None
    out = {'n_splits': len(J), 'splits': sorted(j['split'] for j in J)}
    for m in M:
        v = np.array([j[m] for j in J]); se = np.array([j[m + '_se'] for j in J])
        out[m] = (float(v.mean()), float(np.sqrt((se ** 2).sum()) / len(v)))
        out[m + '_split_sd'] = float(v.std())
    return out


if __name__ == '__main__':
    out = {}
    variants = [''] + sorted({os.path.basename(p)[2:] for p in glob.glob(os.path.join(RES, 'e0_*'))})
    for v in variants:
        for ds in DSS:
            st = agg(v, ds)
            if st:
                out[f'e0{v}/{ds}'] = st
    json.dump(out, open(os.path.join(RES, 'attr_rows.json'), 'w'), indent=1)
    for k, st in out.items():
        print(f"{k:32s} n={st['n_splits']} " + '  '.join(f"{m}={st[m][0]:.3f}±{st[m][1]:.3f} (sd {st[m + '_split_sd']:.3f})" for m in M))
    # r sensitivity (splits 1-3 for every r, r=0.5 from the default variant restricted to the same splits)
    rs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    have = {ds: {r: (agg('', ds, splits=[1, 2, 3]) if r == 0.5 else agg(f'_r{r:g}', ds)) for r in rs} for ds in ['freqshape', 'seqcomb_mv', 'mitecg']}
    if any(all(have[ds][r] for r in rs) for ds in have):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names = {'freqshape': 'FreqShapes', 'seqcomb_mv': 'SeqComb-MV', 'mitecg': 'ECG'}
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
        for ax, ds in zip(axes, have):
            for m, mk, lab in [('auprc', 'o', 'AUPRC'), ('aup', 's', 'AUP'), ('aur', '^', 'AUR')]:
                xs = [r for r in rs if have[ds][r]]
                ax.plot(xs, [have[ds][r][m][0] for r in xs], marker=mk, label=lab)
            ax.set_title(names[ds], fontsize=10); ax.set_xlabel('$r$'); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        os.makedirs(os.path.join(RES, 'fig'), exist_ok=True)
        fig.savefig(os.path.join(RES, 'fig', 'attr_r_sensitivity.pdf')); fig.savefig(os.path.join(RES, 'fig', 'attr_r_sensitivity.png'), dpi=130)
        json.dump({ds: {str(r): (have[ds][r] or None) for r in rs} for ds in have}, open(os.path.join(RES, 'attr_r_sweep.json'), 'w'), indent=1)
        print('r sweep plotted')
