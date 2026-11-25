"""Occlusion check of TimeX_a++ on real-world data without ground truth (Epilepsy, PAM), for the attribution models
of the current TIMEX_DIST configuration ('kl' = conference code, 'mmd' = corrected distribution term).

  delete10_{mean,zero}: replace the 10% most salient coordinates (per series, ranked by pi) by the training mean
                        of the coordinate or by zero; prediction AUROC (lower = more faithful explanation)
  keep_<q>            : keep only the coordinates above the q-quantile of pi, fill the rest with background draws
                        N(mu, sigma) and mask time steps without kept coordinates in attention, as in
                        experiments/evaluation/occlusion_exp.py; prediction AUROC (higher = more faithful)
A random explainer (uniform scores) is evaluated with the same code. Writes results*/occlusion_real/<ds>.json."""
import argparse
from common import *
from sklearn.metrics import roc_auc_score
from txai.utils.functional import transform_to_attn_mask

QS = [0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]


def auroc(enc, X, T, y, n_classes, attn_mask=None, bs=256):
    with torch.no_grad():
        ps = []
        for i in range(0, X.shape[1], bs):
            am = None if attn_mask is None else attn_mask[i:i + bs].float()
            ps.append(enc(X[:, i:i + bs], T[:, i:i + bs], attn_mask=am).softmax(-1).cpu())
        p = torch.cat(ps).numpy()
    yy = np.eye(n_classes)[y.cpu().numpy()]
    return float(roc_auc_score(yy, p, average='macro', multi_class='ovo'))


def evaluate(enc, X, T, y, n_classes, score, mu, sd, seed):
    out = {}
    N = X.shape[1]
    flat = score.transpose(0, 1).reshape(N, -1)                       # (N, T*d)
    thr = torch.quantile(flat.float(), 0.9, dim=1, keepdim=True)
    top = (flat > thr).float().reshape(N, X.shape[0], X.shape[2]).transpose(0, 1)  # (T, N, d)
    out['delete10_mean'] = auroc(enc, X * (1 - top) + top * mu.unsqueeze(1), T, y, n_classes)
    out['delete10_zero'] = auroc(enc, X * (1 - top), T, y, n_classes)
    g = torch.Generator(device=X.device).manual_seed(seed)
    base = mu.unsqueeze(1) + sd.unsqueeze(1) * torch.randn(X.shape, generator=g, device=X.device)
    for q in QS:
        tq = torch.quantile(flat.float(), q, dim=1, keepdim=True)
        keep = (flat > tq).float().reshape(N, X.shape[0], X.shape[2]).transpose(0, 1)
        Xp = X * keep + (1 - keep) * base
        seq = (keep.sum(-1) > 0).transpose(0, 1)
        out[f'keep_{q}'] = auroc(enc, Xp, T, y, n_classes, attn_mask=transform_to_attn_mask(seq))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='epilepsy,pam')
    a = ap.parse_args()
    dev = torch.device('cuda')
    os.makedirs(os.path.join(RES, 'occlusion_real'), exist_ok=True)
    for ds in a.ds.split(','):
        rows = {}
        for split in range(1, 6):
            ck = os.path.join(CKPT, ds, f'attr_split={split}.pt')
            if not os.path.exists(ck):
                continue
            data = load_split(ds, split, dev)
            X, T, y = data['test']
            Xtr = data['train'][0]
            mu, sd = Xtr.mean(1), Xtr.std(1)
            m = make_attr_model(ds, split, data, dev)
            st, _ = torch.load(ck, map_location='cpu', weights_only=False)
            m.load_state_dict(st); m.eval()
            set_seed(0)
            pi = attr_pi(m, X, T).transpose(0, 1).to(dev)                # (T, N, d)
            enc = m.encoder_main
            n_classes = DS[ds]['n_classes']
            g = torch.Generator().manual_seed(split)
            rnd = torch.rand(pi.shape, generator=g).to(dev)
            for name, sc in [('ours', pi), ('random', rnd)]:
                r = evaluate(enc, X, T, y, n_classes, sc, mu, sd, seed=100 + split)
                for k, v in r.items():
                    rows.setdefault(name, {}).setdefault(k, []).append(v)
            log(ds, split, {k: round(v[-1], 3) for k, v in rows['ours'].items()})
        summ = {n: {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))), len(v)) for k, v in r.items()} for n, r in rows.items()}
        json.dump({'dist': DIST, 'summary': summ}, open(os.path.join(RES, 'occlusion_real', f'{ds}.json'), 'w'), indent=1)
        for n, r in summ.items():
            log(ds, DIST, n, {k: round(v[0], 3) for k, v in r.items()})
