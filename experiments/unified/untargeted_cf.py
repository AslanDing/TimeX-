"""Untargeted counterfactuals with the temporal editor (cond model of train_cf.py) as TimeX_cf++.

Candidate cache: for every test instance, every target != black-box prediction and
every mask sample (seeds 0..4, torch.manual_seed(s), batch 256, i.e. the seeds of eval_cf_k used for the C1 main JSONs),
the classifier output and metrics of the candidate of the ONE cond model of the split (cur_target = target).
Model: ckpt/C1/main/<ds>/split=<s>_cond.pt (C1 'main', paper selection rule). Then both rules on the same cache:
  rule A (highest confidence) and rule B (smallest L2 among valid), with
  'paper'  : Table 5/6 procedure of aggregate_cf_local.untargeted (candidate metrics averaged over the 5 samples first,
             candidate valid if mean validity >= 0.5),
  'sample' : rules applied within each mask sample (recommended).
Aggregation as the paper: mean over instances within split, mean +- SE over the 5 splits.
Writes results/D2_temporal/untargeted/{per_sample_<ds>.csv.gz, table56.csv, check.json}.
usage: python untargeted_cf.py --ds freqshape,seqcomb_uv,seqcomb_mv,lowvar
"""
import argparse, gzip, csv, glob
import pandas as pd
from common import *
from train_cf_variants import make_d2_model

OUT = os.path.join(RES2, 'D2_temporal', 'untargeted')
NAMES = {'freqshape': 'FreqShapes', 'seqcomb_uv': 'SeqComb-UV', 'seqcomb_mv': 'SeqComb-MV', 'lowvar': 'LowVar'}


@torch.no_grad()
def candidates(model, X, T, target, y0, seed, bs=256):
    model.eval(); model.cur_target = target
    g = torch.random.get_rng_state(); gc = torch.cuda.get_rng_state_all()
    torch.manual_seed(seed)
    out = {k: [] for k in ['p_tgt', 'pred', 'p_orig', 'l2', 'l1', 'spars']}
    for i in range(0, X.shape[1], bs):
        Xb, Tb = X[:, i:i + bs], T[:, i:i + bs]
        cf = model(Xb, Tb, captum_input=False)['X_cf']
        p = model.encoder_main(cf, Tb, captum_input=False).softmax(-1)
        out['p_tgt'].append(p[:, target]); out['pred'].append(p.argmax(-1))
        out['p_orig'].append(p.gather(1, y0[i:i + bs].view(-1, 1)).squeeze(1))
        d = (cf - Xb).abs()
        out['l2'].append(torch.sqrt((d ** 2).mean(dim=(0, 2)) + 1e-12)); out['l1'].append(d.mean(dim=(0, 2)))
        out['spars'].append((d > 1e-4).float().mean(dim=(0, 2)))
    torch.random.set_rng_state(g); torch.cuda.set_rng_state_all(gc)
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def build(ds, splits, nsamples, dev):
    C = DS[ds]['n_classes']; rows = []
    for split in splits:
        data = load_split(ds, split, dev, main=True)
        pl = pseudo_labels(ds, split, data, dev)
        X, T = data['test'][0], data['test'][1]; y0 = pl['test']; ytrue = data['test'][2]
        m = make_d2_model(ds, split, data, dev)
        m.load_state_dict(torch.load(os.path.join(CK2, 'C1', 'main', ds, f'split={split}_cond.pt'), map_location='cpu')); m.eval()
        for target in range(C):
            sel = torch.where(y0 != target)[0]
            for s in range(nsamples):
                r = candidates(m, X[:, sel], T[:, sel], target, y0[sel], seed=s)
                for j, i in enumerate(sel.tolist()):
                    rows.append((split, i, int(ytrue[i]), int(y0[i]), target, s, float(r['p_tgt'][j]), int(r['pred'][j]),
                                 float(r['p_orig'][j]), float(r['l2'][j]), float(r['l1'][j]), float(r['spars'][j])))
        log(ds, split, 'done', len(rows))
        del m; torch.cuda.empty_cache()
    hdr = ['split', 'idx', 'y_true', 'y_orig_pred', 'target', 'sample', 'p_target', 'pred', 'p_orig', 'l2', 'l1', 'spars']
    with gzip.open(os.path.join(OUT, f'per_sample_{ds}.csv.gz'), 'wt', newline='') as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)


def analyse(ds):
    D = pd.read_csv(os.path.join(OUT, f'per_sample_{ds}.csv.gz'))
    D['hit'] = (D['pred'] == D['target']).astype(float)
    D['leave'] = (D['pred'] != D['y_orig_pred']).astype(float)
    # check against the C1 main per-instance JSONs (mean over the 5 samples)
    chk = []
    for jf in glob.glob(os.path.join(RES2, 'C1_temporal', 'main', ds, 'split=*_target=*_cond.json')):
        j = json.load(open(jf)); s = j['summary']['split']; t = j['summary']['target']
        if 'per_instance' not in j or not j['per_instance']:
            continue
        ref = pd.DataFrame({'idx': j['idx'], 'valid_ref': j['per_instance']['valid'], 'l2_ref': j['per_instance']['prox_l2']})
        mine = D[(D.split == s) & (D.target == t)].groupby('idx')[['hit', 'l2']].mean().reset_index()
        mm = ref.merge(mine, on='idx')
        chk.append((float(np.abs(mm.valid_ref - mm.hit).max()), float(np.abs(mm.l2_ref - mm.l2).max()), len(mm), len(ref)))
    chk = np.array(chk) if chk else np.zeros((0, 4))
    check = {'max_abs_dvalid': float(chk[:, 0].max()) if len(chk) else None, 'max_abs_dl2': float(chk[:, 1].max()) if len(chk) else None,
             'matched_pairs': int(chk[:, 2].sum()) if len(chk) else 0, 'json_pairs': int(chk[:, 3].sum()) if len(chk) else 0}
    res = {}
    Cn = D.groupby(['split', 'idx', 'target']).agg(hit=('hit', 'mean'), leave=('leave', 'mean'), p=('p_target', 'mean'),
                                                   l2=('l2', 'mean'), l1=('l1', 'mean'), spars=('spars', 'mean')).reset_index()
    pick_A = Cn.loc[Cn.groupby(['split', 'idx'])['p'].idxmax()]
    okB = Cn[Cn.hit >= 0.5]
    pick_B = okB.loc[okB.groupby(['split', 'idx'])['l2'].idxmin()]
    noneB = Cn.groupby(['split', 'idx']).size().index.difference(pick_B.set_index(['split', 'idx']).index)
    if len(noneB):
        pick_B = pd.concat([pick_B, pick_A.set_index(['split', 'idx']).loc[noneB].reset_index()])
    res['paper_A'], res['paper_B'] = pick_A, pick_B
    pA = D.loc[D.groupby(['split', 'idx', 'sample'])['p_target'].idxmax()]
    vB = D[D.hit == 1]
    pB = vB.loc[vB.groupby(['split', 'idx', 'sample'])['l2'].idxmin()]
    miss = D.groupby(['split', 'idx', 'sample']).size().index.difference(pB.set_index(['split', 'idx', 'sample']).index)
    if len(miss):
        pB = pd.concat([pB, pA.set_index(['split', 'idx', 'sample']).loc[miss].reset_index()])
    res['sample_A'] = pA.rename(columns={'p_target': 'p'}); res['sample_B'] = pB.rename(columns={'p_target': 'p'})
    lab = {'paper_A': ('highest confidence', 'paper (Table 5/6 procedure)'), 'paper_B': ('smallest L2 among valid', 'paper (Table 5/6 procedure)'),
           'sample_A': ('highest confidence', 'per mask sample'), 'sample_B': ('smallest L2 among valid', 'per mask sample')}
    rows = []
    cols = ['hit', 'leave', 'p', 'spars', 'l1', 'l2']
    for k, P in res.items():
        g = P.groupby('split')[cols].mean()
        m, se = g.mean(), g.std(ddof=0) / np.sqrt(len(g))
        rows.append({'dataset': NAMES[ds], 'rule': lab[k][0], 'procedure': lab[k][1], 'n_splits': len(g),
                     **{c: round(float(m[c]), 4) for c in cols}, **{c + '_se': round(float(se[c]), 4) for c in cols}})
    return rows, check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='freqshape,seqcomb_uv,seqcomb_mv,lowvar')
    ap.add_argument('--splits', default='1,2,3,4,5')
    ap.add_argument('--nsamples', type=int, default=5)
    ap.add_argument('--analyse_only', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device('cuda') if not a.analyse_only else None
    allrows, checks = [], {}
    for ds in a.ds.split(','):
        if not a.analyse_only:
            build(ds, [int(s) for s in a.splits.split(',')], a.nsamples, dev)
        rows, chk = analyse(ds)
        allrows += rows; checks[ds] = chk
        log(ds, chk)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, 'table56.csv'), index=False)
    jdump(checks, os.path.join(OUT, 'check.json'))
    print(pd.DataFrame(allrows).to_string())


if __name__ == '__main__':
    main()
