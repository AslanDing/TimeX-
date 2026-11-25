"""Re-evaluate the attribution models of train_attr.py with the evaluation protocol of the ICML/paper scripts:
AUPRC is only defined for test series that contain ground-truth positives, so series without any
ground-truth coordinate (e.g. the null class 0 of SeqComb-MV) are excluded.

usage: python reeval_attr.py [--noleak]
"""
import argparse
from common import *
from txai.utils.evaluation import ground_truth_xai_eval

ap = argparse.ArgumentParser()
ap.add_argument('--noleak', action='store_true')
ap.add_argument('--ds', default='freqshape,seqcomb_mv,mitecg')
a = ap.parse_args()
SUF = '_noleak' if a.noleak else ''
dev = torch.device('cuda')
out = {}
for ds in a.ds.split(','):
    rows = []
    for split in range(1, 6):
        ck = os.path.join(CKPT, ds, f'attr{SUF}_split={split}.pt')
        if not os.path.exists(ck):
            continue
        data = load_split(ds, split, dev)
        cp = os.path.join(CACHE, ds, f'attr{SUF}_pi_split={split}.pt')
        if os.path.exists(cp):
            pi = torch.load(cp)['test'].float()
        else:
            m = make_attr_model(ds, split, data, dev, noleak=a.noleak)
            sd, _ = torch.load(ck, map_location='cpu', weights_only=False)
            m.load_state_dict(sd)
            set_seed(0)
            pi = attr_pi(m, data['test'][0], data['test'][1]).cpu()
        gt = data['gt'].cpu()                      # (T,N,d)
        has = gt.sum(dim=(0, 2)) > 0
        y = data['test'][2].cpu()
        r_all = ground_truth_xai_eval(pi.transpose(0, 1), gt)
        r_pos = ground_truth_xai_eval(pi[has].transpose(0, 1), gt[:, has])
        rows.append({'split': split, 'n_all': int(gt.shape[1]), 'n_with_gt': int(has.sum()),
                     'classes_without_gt': sorted(set(y[~has].tolist())),
                     'auprc_all': float(np.nanmean(r_all['auprc'])), 'auprc_gt_only': float(np.mean(r_pos['auprc'])),
                     'aup_gt_only': float(np.mean(r_pos['aup'])), 'aur_gt_only': float(np.mean(r_pos['aur']))})
        log(ds, rows[-1])
    if rows:
        out[ds] = {'per_split': rows, **{k: float(np.mean([r[k] for r in rows])) for k in ['auprc_all', 'auprc_gt_only', 'aup_gt_only', 'aur_gt_only']},
                   'auprc_gt_only_se': float(np.std([r['auprc_gt_only'] for r in rows]) / np.sqrt(len(rows)))}
os.makedirs(os.path.join(RES, 'e0' + SUF), exist_ok=True)
json.dump(out, open(os.path.join(RES, 'e0' + SUF, 'reeval_gt_only.json'), 'w'), indent=1)
for ds, v in out.items():
    print(ds, {k: round(x, 4) for k, x in v.items() if k != 'per_split'})
