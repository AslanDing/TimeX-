"""Noise robustness of counterfactuals (paper protocol: N(0,1) noise added to a random fraction of the CF
coordinates), recomputed with the final-protocol models of TimeX_cf++ and the saved baseline counterfactuals,
plus the confidence-stratified analysis. Writes results/robust_cf/<ds>.json and figure cf_robustness.pdf."""
import argparse, glob
from common import *
from score_cf import our_cfs, noisy_validity

BASE = {'freqshape': ('experiments/freqshape/results/{m}_transformer/cfs', ['abcf', 'comte', 'mcels', 'confetti']),
        'lowvar': ('experiments/lowvardetect/results/{m}/cfs', ['abcf', 'comte', 'mcels', 'confetti'])}
RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


@torch.no_grad()
def noisy_conf(enc, cf_btd, target, times, ratio, seed):
    g = torch.Generator().manual_seed(seed)
    cf = torch.tensor(np.ascontiguousarray(cf_btd), dtype=torch.float32)
    B, T_, D = cf.shape
    k = int(round(ratio * T_ * D))
    noise = torch.zeros(B, T_, D)
    if k > 0:
        idx = torch.rand(B, T_ * D, generator=g).argsort(dim=1)[:, :k]
        noise.view(B, -1).scatter_(1, idx, torch.randn(B, k, generator=g))
    x = (cf + noise).to(times.device).transpose(0, 1)
    p = bb_logits(enc, x, times[:, :1].repeat(1, B)).softmax(-1)
    return (p.argmax(-1) == target).float().cpu().numpy(), p[:, target].cpu().numpy()


def run(ds, tag, dev):
    pat, meths = BASE[ds]
    rows = {}
    for split in range(1, 6):
        data = load_split(ds, split, dev, main=True)
        enc = make_cf_model(ds, split, data, dev).encoder_main.eval()
        pl = bb_logits(enc, data['test'][0], data['test'][1]).argmax(-1)
        X, T = data['test'][0], data['test'][1]
        items = []
        for m in meths:
            for f in sorted(glob.glob(os.path.join(REPO, pat.format(m=m), f'*split={split}_target=*.npz'))):
                if 'untargeted' in f:
                    continue
                z = np.load(f)
                items.append((m, int(z['target_label'][0]), z['counterfactual']))
        for target in range(DS[ds]['n_classes']):
            ck = os.path.join(CKPT, ds, tag, f'split={split}_target={target}_learned.pt')
            if not os.path.exists(ck):
                continue
            sel = torch.where(pl != target)[0]
            mdl = make_cf_model(ds, split, data, dev)
            mdl.load_state_dict(torch.load(ck, map_location='cpu')); mdl.eval()
            items.append(('ours', target, our_cfs(mdl, X[:, sel], T[:, sel])))
        for meth, target, cf in items:
            r = rows.setdefault(meth, {})
            for ratio in RATIOS:
                v, c = noisy_conf(enc, cf, target, T, ratio, seed=1000 * split + 10 * target + int(ratio * 100))
                r.setdefault(f'v{ratio}', []).append(v)
                r.setdefault(f'c{ratio}', []).append(c)
        log(ds, 'split', split, {k: len(v.get('v0.0', [])) for k, v in rows.items()})
    out = {}
    bins = [0.0, 0.5, 0.7, 0.9, 0.99, 1.0001]
    for meth, r in rows.items():
        c = {k: np.concatenate(v) for k, v in r.items()}
        o = {'curve_validity': [float(c[f'v{q}'].mean()) for q in RATIOS], 'curve_confidence': [float(c[f'c{q}'].mean()) for q in RATIOS],
             'n': int(c['v0.0'].shape[0])}
        for lo, hi in zip(bins[:-1], bins[1:]):
            mk = (c['c0.0'] >= lo) & (c['c0.0'] < hi)
            o[f'conf[{lo},{min(hi, 1.0)})'] = {'n': int(mk.sum()), 'v0.3': float(c['v0.3'][mk].mean()) if mk.any() else None}
        out[meth] = o
        log(ds, meth, o['curve_validity'])
    os.makedirs(os.path.join(RES, 'robust_cf'), exist_ok=True)
    json.dump({'ratios': RATIOS, 'methods': out}, open(os.path.join(RES, 'robust_cf', f'{ds}.json'), 'w'), indent=1)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {'ours': 'TimeX$_{cf}$++', 'abcf': 'AB-CF', 'comte': 'CoMTE', 'mcels': 'M-CELS', 'confetti': 'CONFETTI'}
    cols = {'ours': '#1b9e77', 'abcf': '#d95f02', 'mcels': '#7570b3', 'confetti': '#e7298a', 'comte': '#66a61e'}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    for j, ds in enumerate(['freqshape', 'lowvar']):
        f = os.path.join(RES, 'robust_cf', f'{ds}.json')
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for i, key in enumerate(['curve_validity', 'curve_confidence']):
            ax = axes[2 * j + i]
            for m, o in d['methods'].items():
                ax.plot(d['ratios'], o[key], marker='o', ms=3, color=cols[m], label=names[m])
            ax.set_xlabel('Noise ratio'); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
            ax.set_title(f"{'Validity' if i == 0 else 'Confidence'} on {'FreqShapes' if ds == 'freqshape' else 'LowVar'}", fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(RES, 'robust_cf', 'cf_robustness.pdf'))
    fig.savefig(os.path.join(RES, 'robust_cf', 'cf_robustness.png'), dpi=150)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='freqshape,lowvar')
    ap.add_argument('--tag', default='main')
    ap.add_argument('--plot_only', action='store_true')
    a = ap.parse_args()
    if not a.plot_only:
        for ds in a.ds.split(','):
            run(ds, a.tag, torch.device('cuda'))
    plot()
