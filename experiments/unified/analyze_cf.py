"""Evaluation-only analyses with the temporal editor (ckpt/C1/main/<ds>/split=<s>_cond.pt) as TimeX_cf++.

--part robust : Fig. 'cf_robustness' (robust_cf.py protocol): one CF per eligible test instance and target (mask
                sample seed 0), N(0,1) noise on a random fraction of the coordinates with the same seeds
                (1000*split + 10*target + 100*ratio), validity/confidence curves and the confidence-stratified validity at
                30% noise; FreqShapes and LowVar, 5 splits. Baselines and the paper model are taken from
                results/robust_cf/<ds>.json (same code, same seeds). -> results/D2_temporal/robust/<ds>.json + figure
--part pi     : Fig. 'pi_hist' (leakage.pi_stats on the analysis subsets; FreqShapes, SeqComb-MV, ECG; 5 splits;
                all targets on the instances not predicted as the target). -> results/D2_temporal/pi/<ds>_split=<s>.json + figure
--part timing : Table 15 (split 1, FreqShapes / LowVar / Epilepsy, one GPU): per-instance generation
                time, targeted (batch 256, one mask sample) and untargeted (all |C|-1 targets + max-confidence
                selection), for the local per-target models AND the temporal cond model in the same session, plus the
                cond model with all targets stacked in one batch. -> results/D2_temporal/timing/timing.json
"""
import argparse, glob
from common import *
from train_cf_variants import make_d2_model, c1_pi
from score_cf import our_cfs
from robust_cf import noisy_conf, RATIOS

OUT = os.path.join(RES2, 'D2_temporal')


def load_c1(ds, split, data, dev):
    m = make_d2_model(ds, split, data, dev)
    m.load_state_dict(torch.load(os.path.join(CK2, 'C1', 'main', ds, f'split={split}_cond.pt'), map_location='cpu'))
    return m.eval()


def robust(ds, dev):
    rows = {}
    for split in range(1, 6):
        data = load_split(ds, split, dev, main=True)
        enc = make_cf_model(ds, split, data, dev).encoder_main.eval()
        pl = bb_logits(enc, data['test'][0], data['test'][1]).argmax(-1)
        X, T = data['test'][0], data['test'][1]
        mdl = load_c1(ds, split, data, dev)
        for target in range(DS[ds]['n_classes']):
            sel = torch.where(pl != target)[0]
            mdl.cur_target = target
            cf = our_cfs(mdl, X[:, sel], T[:, sel])
            r = rows.setdefault('ours_c1', {})
            for ratio in RATIOS:
                v, c = noisy_conf(enc, cf, target, T, ratio, seed=1000 * split + 10 * target + int(ratio * 100))
                r.setdefault(f'v{ratio}', []).append(v); r.setdefault(f'c{ratio}', []).append(c)
        log(ds, 'robust split', split)
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
    old = json.load(open(os.path.join(RES1, 'robust_cf', f'{ds}.json')))
    old['methods'] = {**old['methods'], **out}
    jdump(old, os.path.join(OUT, 'robust', f'{ds}.json'))


def plot_robust():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {'ours_c1': 'TimeX$_{cf}$++ (temporal $\\psi_{cf}$)', 'ours': 'TimeX$_{cf}$++ (paper)', 'abcf': 'AB-CF', 'comte': 'CoMTE',
             'mcels': 'M-CELS', 'confetti': 'CONFETTI'}
    cols = {'ours_c1': '#1b9e77', 'ours': '#1b9e77', 'abcf': '#d95f02', 'mcels': '#7570b3', 'confetti': '#e7298a', 'comte': '#66a61e'}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    for j, ds in enumerate(['freqshape', 'lowvar']):
        f = os.path.join(OUT, 'robust', f'{ds}.json')
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for i, key in enumerate(['curve_validity', 'curve_confidence']):
            ax = axes[2 * j + i]
            for m in ['ours_c1', 'ours', 'comte', 'abcf', 'mcels', 'confetti']:
                if m not in d['methods']:
                    continue
                ax.plot(d['ratios'], d['methods'][m][key], marker='o', ms=3, color=cols[m], label=names[m], ls='--' if m == 'ours' else '-')
            ax.set_xlabel('Noise ratio'); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
            ax.set_title(f"{'Validity' if i == 0 else 'Confidence'} on {'FreqShapes' if ds == 'freqshape' else 'LowVar'}", fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT, 'robust', 'd2_robustness.pdf')); fig.savefig(os.path.join(OUT, 'robust', 'd2_robustness.png'), dpi=150)


def pis(ds, dev):
    from leakage import pi_stats
    for split in range(1, 6):
        data = load_split(ds, split, dev, use_subset=True)
        pl = pseudo_labels(ds, split, data, dev)
        X, T = data['test'][0], data['test'][1]
        m = load_c1(ds, split, data, dev)
        res = {'pi_cf': {}}
        allp = []
        for t in range(DS[ds]['n_classes']):
            sel = torch.where(pl['test'] != t)[0]
            p = c1_pi(m, X[:, sel], T[:, sel], t)
            res['pi_cf'][t] = pi_stats(p); allp.append(p.flatten().cpu())
        res['pi_cf_pooled'] = pi_stats(torch.cat(allp))
        jdump(res, os.path.join(OUT, 'pi', f'{ds}_split={split}.json'))
        log(ds, split, {t: (round(v['frac_lt_0.1'], 3), round(v['frac_gt_0.9'], 3), round(v['mean'], 3)) for t, v in res['pi_cf'].items()})


def plot_pi():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    NAMES = {'freqshape': 'FreqShapes', 'seqcomb_mv': 'SeqComb-MV', 'mitecg': 'ECG'}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.6), sharey=True)
    summ = {}
    for ax, ds in zip(axes, NAMES):
        A, P, N, edges = [], [], [], None
        for f in sorted(glob.glob(os.path.join(RES1, 'e2', f'{ds}_split=*.json'))):
            d = json.load(open(f)); A.append(d['pi_attr']['hist']); edges = d['pi_attr']['edges']
            P += [v['hist'] for v in d['pi_cf'].values()]
        for f in sorted(glob.glob(os.path.join(OUT, 'pi', f'{ds}_split=*.json'))):
            d = json.load(open(f)); N += [v['hist'] for v in d['pi_cf'].values()]
        if not N:
            continue
        A, P, N = np.mean(A, 0), np.mean(P, 0), np.mean(N, 0)
        e = np.asarray(edges); c = (e[:-1] + e[1:]) / 2; w = e[1] - e[0]
        ax.bar(c - w / 3, A, width=w / 3, color='#1b9e77', label=r'TimeX$_a$++ ($r=0.5$)')
        ax.bar(c, P, width=w / 3, color='#bbbbbb', label=r'TimeX$_{cf}$++ (paper $\psi_{cf}$)')
        ax.bar(c + w / 3, N, width=w / 3, color='#d95f02', label=r'TimeX$_{cf}$++ (temporal $\psi_{cf}$)')
        ax.set_title(NAMES[ds], fontsize=10); ax.set_xlabel(r'$\pi$'); ax.set_xlim(0, 1); ax.grid(alpha=0.3, axis='y')
        summ[ds] = {'paper_cf_lt0.1': float(P[:2].sum()), 'temporal_cf_lt0.1': float(N[:2].sum()),
                    'paper_cf_gt0.9': float(P[-2:].sum()), 'temporal_cf_gt0.9': float(N[-2:].sum())}
    axes[0].set_ylabel('Fraction of coordinates'); axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'pi', 'd2_pi_hist.pdf')); fig.savefig(os.path.join(OUT, 'pi', 'd2_pi_hist.png'), dpi=120)
    jdump(summ, os.path.join(OUT, 'pi', 'summary.json'))


def timing(dev, reps=10, bs=256):
    def sync():
        torch.cuda.synchronize()

    def stats(v):
        v = np.asarray(v); q1, q3 = np.percentile(v, [25, 75])
        return {'median': float(np.median(v)), 'iqr': [float(q1), float(q3)]}

    @torch.no_grad()
    def gen(m, X, T, t=None):
        outs = []
        for i in range(0, X.shape[1], bs):
            Xb, Tb = X[:, i:i + bs], T[:, i:i + bs]
            o = m(Xb, Tb, captum_input=False, target_label=(None if t is None else t[i:i + bs]))
            outs.append((o['X_cf'], m.encoder_main(o['X_cf'], Tb, captum_input=False).softmax(-1)))
        return outs

    res = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    for ds in ['freqshape', 'lowvar', 'epilepsy']:
        data = load_split(ds, 1, dev, main=True)
        pl = pseudo_labels(ds, 1, data, dev)['test']
        X, T = data['test'][0], data['test'][1]
        C = DS[ds]['n_classes']
        paper = {}
        for t in range(C):
            m = make_cf_model(ds, 1, data, dev)
            m.load_state_dict(torch.load(os.path.join(CK1, ds, 'main', f'split=1_target={t}_learned.pt'), map_location='cpu'), strict=False)
            paper[t] = m.eval()
        c1 = load_c1(ds, 1, data, dev)
        r = {'n_test': int(X.shape[1]), 'n_classes': C, 'batch': bs}
        for name in ['paper', 'c1']:
            tg = {}
            for t in range(C):
                m = paper[t] if name == 'paper' else c1
                m.cur_target = t
                sel = torch.where(pl != t)[0]; Xs, Ts = X[:, sel], T[:, sel]
                gen(m, Xs[:, :bs], Ts[:, :bs])
                v = []
                for _ in range(reps):
                    sync(); t0 = time.time(); gen(m, Xs, Ts); sync(); v.append((time.time() - t0) / Xs.shape[1])
                tg[t] = {**stats(v), 'n': int(Xs.shape[1])}
            r[f'{name}_targeted_s_per_instance'] = float(np.average([tg[t]['median'] for t in tg], weights=[tg[t]['n'] for t in tg]))
            v = []
            for _ in range(reps + 1):
                sync(); t0 = time.time()
                best_conf = torch.full((X.shape[1],), -1.0, device=dev); best_cf = torch.zeros_like(X)
                for t in range(C):
                    m = paper[t] if name == 'paper' else c1
                    m.cur_target = t
                    sel = torch.where(pl != t)[0]
                    outs = gen(m, X[:, sel], T[:, sel])
                    cf = torch.cat([o[0] for o in outs], 1); conf = torch.cat([o[1][:, t] for o in outs])
                    better = conf > best_conf[sel]; idx = sel[better]
                    best_conf[idx] = conf[better]; best_cf[:, idx] = cf[:, better]
                sync(); v.append((time.time() - t0) / X.shape[1])
            r[f'{name}_untargeted'] = stats(v[1:])
        # cond model: all (instance, target != pred) pairs stacked in one pass
        pairs_i, pairs_t = [], []
        for t in range(C):
            sel = torch.where(pl != t)[0]; pairs_i.append(sel); pairs_t.append(torch.full_like(sel, t))
        pi_, pt_ = torch.cat(pairs_i), torch.cat(pairs_t)
        v = []
        for _ in range(reps + 1):
            sync(); t0 = time.time()
            outs = gen(c1, X[:, pi_], T[:, pi_], t=pt_)
            conf = torch.cat([o[1] for o in outs]).gather(1, pt_.view(-1, 1)).squeeze(1)
            best = torch.full((X.shape[1],), -1.0, device=dev).scatter_reduce(0, pi_, conf, reduce='amax')
            sync(); v.append((time.time() - t0) / X.shape[1])
        r['c1_untargeted_stacked'] = stats(v[1:])
        r['c1_n_params_editor'] = int(sum(p.numel() for p in c1.editor.parameters()))
        r['paper_n_params_psi_cf'] = int(sum(p.numel() for p in paper[0].perturbation_generator.parameters()))
        res[ds] = r
        log(ds, {k: (v if not isinstance(v, dict) else v.get('median')) for k, v in r.items()})
        jdump(res, os.path.join(OUT, 'timing', 'timing.json'))


def agree(ds, dev, splits=(1, 2, 3)):
    """Mask agreement of Section 6.4 (train_cf_local, --no_agree off): overlap at k between the top-k coordinates of the
    learned CF mask (C1 main cond, k = round(sum pi_cf)) and of pi_a, chance level k/(T d), and AUPRC of both vs GT;
    analysis subsets, test instances not predicted as the target, tie-break generator seed 7."""
    from train_cf_local import rank_topk
    from train_cf_variants import load_pi_a
    from txai.utils.evaluation import ground_truth_xai_eval
    out = []
    for split in splits:
        data = load_split(ds, split, dev, use_subset=True)
        pl = pseudo_labels(ds, split, data, dev)
        X, T = data['test'][0], data['test'][1]
        pi_a = load_pi_a(ds, split, data, dev)['test']
        m = load_c1(ds, split, data, dev)
        for t in range(DS[ds]['n_classes']):
            sel = torch.where(pl['test'] != t)[0]
            pic = c1_pi(m, X[:, sel], T[:, sel], t)
            kt = pic.sum(dim=(1, 2)).round().clamp(min=1).long()
            gen = torch.Generator().manual_seed(7)
            top_cf = rank_topk(pic, kt, gen); top_a = rank_topk(pi_a[sel], kt, gen)
            ov = ((top_cf * top_a).sum(dim=(1, 2)) / kt.float()).cpu().numpy()
            ch = (kt.float() / (pic.shape[1] * pic.shape[2])).cpu().numpy()
            gt = data['gt'][:, sel]
            r_cf = ground_truth_xai_eval(pic.transpose(0, 1).cpu(), gt.cpu())
            r_a = ground_truth_xai_eval(pi_a[sel].transpose(0, 1).cpu(), gt.cpu())
            out.append({'split': split, 'target': t, 'overlap_at_k': float(ov.mean()), 'overlap_chance': float(ch.mean()),
                        'auprc_cf_mask_vs_gt': float(np.nanmean(r_cf['auprc'])), 'auprc_attr_mask_vs_gt': float(np.nanmean(r_a['auprc']))})
        log(ds, split, out[-1])
    old = [json.load(open(f)) for f in glob.glob(os.path.join(RES1, 'e1', ds, 'split=*_agreement.json'))]
    res = {'temporal': {k: float(np.mean([o[k] for o in out])) for k in out[0] if k not in ('split', 'target')},
           'paper': {k: float(np.mean([o[k] for o in old])) for k in ('overlap_at_k', 'overlap_chance', 'auprc_cf_mask_vs_gt', 'auprc_attr_mask_vs_gt')} if old else None,
           'cells': out}
    jdump(res, os.path.join(OUT, 'agree', f'{ds}.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['robust', 'pi', 'timing', 'plots', 'agree'])
    ap.add_argument('--ds', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda') if a.part != 'plots' else None
    if a.part == 'robust':
        for ds in (a.ds or 'freqshape,lowvar').split(','):
            robust(ds, dev)
        plot_robust()
    elif a.part == 'pi':
        for ds in (a.ds or 'freqshape,seqcomb_mv,mitecg').split(','):
            pis(ds, dev)
        plot_pi()
    elif a.part == 'timing':
        timing(dev)
    elif a.part == 'agree':
        for ds in (a.ds or 'freqshape,seqcomb_mv,mitecg').split(','):
            agree(ds, dev)
    else:
        plot_robust(); plot_pi()


if __name__ == '__main__':
    main()
