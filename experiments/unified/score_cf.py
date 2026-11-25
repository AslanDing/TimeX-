"""Scoring of saved counterfactuals (semantic validity, operating points, transfer).

All methods are scored by the same code from their CF arrays:
  validity / confidence on the explained transformer, sparsity at 3 tolerances, Prox-L1/L2 on all
  instances and on valid instances only, transfer validity on independently trained CNN and LSTM
  black boxes, and (FreqShapes) semantic validity from the rule-based oracle of the generator.

Modes
  --mode e4    : operating-point sweep (splits 1-3, first 100 eligible test instances per target)
                 baselines from results/e4_bl/<ds>/<cfg>/split=<s>/cfs, ours from ckpt/<ds>/e4_r<r>/
  --mode paper : FreqShapes, the paper's main-table runs (5 splits, all eligible instances):
                 baselines from experiments/freqshape/results/*_transformer/cfs, ours = paper ckpts
"""
import argparse, glob
from common import *
from oracles import freqshape_oracle, freqshape_factors, FS_FACTORS


@torch.no_grad()
def score(ds, cf_btd, x_btd, target, encs, times):
    """cf_btd, x_btd: numpy (B,T,D). encs: dict name->encoder. Returns per-instance dict."""
    dev = times.device
    cf = torch.tensor(cf_btd, dtype=torch.float32, device=dev).transpose(0, 1)
    x = torch.tensor(x_btd, dtype=torch.float32, device=dev).transpose(0, 1)
    tb = times[:, :1].repeat(1, cf.shape[1])
    out = {}
    for name, enc in encs.items():
        logits = bb_logits(enc, cf, tb)
        out['valid' if name == 'bb' else 'valid_' + name] = (logits.argmax(-1) == target).float().cpu().numpy()
        if name == 'bb':
            out['conf'] = logits.softmax(-1)[:, target].cpu().numpy()
    delta = (cf - x).abs()
    for t in TOLS:
        out[f'spars_{t}'] = (delta > t).float().mean(dim=(0, 2)).cpu().numpy()
    out['prox_l1'] = delta.mean(dim=(0, 2)).cpu().numpy()
    out['prox_l2'] = torch.sqrt((delta ** 2).mean(dim=(0, 2)) + 1e-12).cpu().numpy()
    if ds == 'freqshape':
        ts_, tp_ = FS_FACTORS[target]
        facs = [freqshape_factors(cf_btd[i, :, 0]) for i in range(cf_btd.shape[0])]
        out['oracle_valid'] = np.array([float(f == (ts_, tp_)) for f in facs])
        out['oracle_shape_ok'] = np.array([float(f[0] == ts_) for f in facs])
        out['oracle_period_ok'] = np.array([float(f[1] == tp_) for f in facs])
        out['oracle_src_ok'] = np.array([float(freqshape_oracle(x_btd[i, :, 0]) != -1) for i in range(cf_btd.shape[0])])
    return out


def summarize_pi(recs):
    """recs: list of (split, per-instance dict). Mean over instances within split, then mean+-SE over splits."""
    keys = [k for k in recs[0][1].keys()]
    by_split = {}
    for s, r in recs:
        by_split.setdefault(s, []).append(r)
    per_split = {}
    for s, rs in by_split.items():
        cat = {k: np.concatenate([r[k] for r in rs]) for k in keys}
        v = cat['valid'].astype(bool)
        m = {k: float(np.mean(cat[k])) for k in keys}
        m['prox_l1_valid'] = float(cat['prox_l1'][v].mean()) if v.any() else float('nan')
        m['prox_l2_valid'] = float(cat['prox_l2'][v].mean()) if v.any() else float('nan')
        for k in keys:
            if k.startswith('valid_') or k.startswith('oracle_'):
                m[k + '_given_valid'] = float(cat[k][v].mean()) if v.any() else float('nan')
        m['n'] = int(v.shape[0])
        per_split[s] = m
    agg = {}
    for k in per_split[next(iter(per_split))].keys():
        vals = np.array([per_split[s][k] for s in per_split], dtype=float)
        agg[k] = float(np.nanmean(vals))
        agg[k + '_se'] = float(np.nanstd(vals) / np.sqrt(max(1, np.sum(~np.isnan(vals)))))
    agg['n_splits'] = len(per_split)
    return agg, per_split


def get_encoders(ds, split, d_inp, dev):
    enc = make_cf_model(ds, split, load_split(ds, split, dev, use_subset=True), dev).encoder_main.eval()
    return {'bb': enc, 'cnn': load_aux_bb(ds, split, 'cnn', d_inp, dev), 'lstm': load_aux_bb(ds, split, 'lstm', d_inp, dev)}


@torch.no_grad()
def our_cfs(model, X, T, seed=0, bs=256, ext=None):
    torch.manual_seed(seed)
    out = []
    for i in range(0, X.shape[1], bs):
        eb = ext[i:i + bs] if ext is not None else None
        out.append(model(X[:, i:i + bs], T[:, i:i + bs], captum_input=False, ext_mask=eb)['X_cf'])
    return torch.cat(out, 1).transpose(0, 1).cpu().numpy()


def run_e1_sem(dev, ds='freqshape', variants=('orig', 'learned', 'attr', 'nested', 'full', 'rand', 'shift', 'shuffle')):
    """Semantic (oracle) and transfer validity of the mask-source variants (train_cf_local.py) on FreqShapes."""
    from train_cf_local import build_masks
    res = {}
    for split in range(1, 6):
        data = load_split(ds, split, dev, use_subset=True)
        pl = pseudo_labels(ds, split, data, dev)
        X, T = data['test'][0], data['test'][1]
        cache = torch.load(os.path.join(CACHE, ds, f'attr_pi_split={split}.pt'))
        pi_a = {s: cache[s].float().to(dev) for s in ['train', 'val', 'test']}
        encs = get_encoders(ds, split, X.shape[-1], dev)
        for target in range(DS[ds]['n_classes']):
            te_sel = torch.where(pl['test'] != target)[0]
            m0 = load_orig_cf(ds, split, target, data, dev)
            pic = {s: cf_pi(m0, data[s][0], data[s][1]) for s in ['train', 'val', 'test']}
            ks = {s: pic[s].sum(dim=(1, 2)).round().clamp(min=1).long() for s in pic}
            xb = X[:, te_sel].transpose(0, 1).cpu().numpy()
            for v in variants:
                if v == 'orig':
                    cf = our_cfs(m0, X[:, te_sel], T[:, te_sel])
                else:
                    ck = os.path.join(CKPT, ds, 'e1', f'split={split}_target={target}_{v}.pt')
                    if not os.path.exists(ck):
                        continue
                    gen = torch.Generator().manual_seed(1000 * split + 10 * target + sum(map(ord, v)))
                    masks = None
                    if v not in ('learned', 'learned_noeps'):
                        masks = {s: build_masks(v, pi_a[s], ks[s], gen) for s in ['train', 'val', 'test']}
                    mode = {'learned': 'learned', 'learned_noeps': 'learned', 'nested': 'nested'}.get(v, 'fixed')
                    m = make_cf_model(ds, split, data, dev, mask_mode=mode, use_noise=(v != 'learned_noeps'))
                    m.load_state_dict(torch.load(ck, map_location='cpu'))
                    m.eval()
                    cf = our_cfs(m, X[:, te_sel], T[:, te_sel], ext=None if masks is None else masks['test'][te_sel])
                res.setdefault(v, []).append((split, score(ds, cf, xb, target, encs, T)))
        log('e1sem split', split, {k: len(v) for k, v in res.items()})
    out = {k: summarize_pi(v)[0] for k, v in res.items()}
    os.makedirs(os.path.join(RES, 'e3'), exist_ok=True)
    json.dump(out, open(os.path.join(RES, 'e3', f'{ds}_e1_variants.json'), 'w'), indent=1)
    for k, v in out.items():
        log(k, {kk: round(vv, 4) for kk, vv in v.items() if not kk.endswith('_se')})


def run_e4(ds, dev, n_per_target=100, splits=(1, 2, 3)):
    res = {}
    cfg_dirs = sorted(glob.glob(os.path.join(RES, 'e4_bl', ds, '*')))
    for split in splits:
        data = load_split(ds, split, dev, use_subset=True)
        pl = pseudo_labels(ds, split, data, dev)
        X, T = data['test'][0], data['test'][1]
        encs = get_encoders(ds, split, X.shape[-1], dev)
        # baselines
        for cd in cfg_dirs:
            name = os.path.basename(cd)
            for f in sorted(glob.glob(os.path.join(cd, f'split={split}', 'cfs', '*.npz'))):
                if 'untargeted' in f:
                    continue
                z = np.load(f)
                tgt = int(z['target_label'][0])
                r = score(ds, z['counterfactual'], z['original'], tgt, encs, T)
                res.setdefault(name, []).append((split, r))
        # ours (regenerate CFs from the operating-point checkpoints, same instances as the baselines)
        for rdir in sorted(glob.glob(os.path.join(CKPT, ds, 'e4_r*'))):
            name = 'ours_' + os.path.basename(rdir).replace('e4_', '')
            for target in range(DS[ds]['n_classes']):
                ck = os.path.join(rdir, f'split={split}_target={target}_learned.pt')
                if not os.path.exists(ck):
                    continue
                sel = torch.where(pl['test'] != target)[0][:n_per_target]
                r_ = float(os.path.basename(rdir).replace('e4_r', ''))
                m = make_cf_model(ds, split, data, dev, mask_mode='learned', r=r_)
                m.load_state_dict(torch.load(ck, map_location='cpu'))
                m.eval()
                cf = our_cfs(m, X[:, sel], T[:, sel])
                xb = X[:, sel].transpose(0, 1).cpu().numpy()
                res.setdefault(name, []).append((split, score(ds, cf, xb, target, encs, T)))
        log(ds, 'split', split, 'done', sorted(res.keys()))
    out = {k: summarize_pi(v)[0] for k, v in res.items()}
    os.makedirs(os.path.join(RES, 'e4'), exist_ok=True)
    json.dump(out, open(os.path.join(RES, 'e4', f'{ds}_summary.json'), 'w'), indent=1)
    for k, v in out.items():
        log(ds, k, {kk: round(vv, 4) for kk, vv in v.items() if not kk.endswith('_se')})


def run_paper_freqshape(dev):
    ds = 'freqshape'
    res = {}
    base = os.path.join(REPO, 'experiments', 'freqshape', 'results')
    for split in range(1, 6):
        data = load_split(ds, split, dev)
        pl = pseudo_labels(ds, split, data, dev)
        X, T = data['test'][0], data['test'][1]
        encs = get_encoders(ds, split, X.shape[-1], dev)
        for meth in ['abcf', 'comte', 'mcels', 'confetti']:
            pat = os.path.join(base, f'{meth}_transformer', 'cfs', f'*split={split}_target=*.npz')
            for f in sorted(glob.glob(pat)):
                if 'untargeted' in f:
                    continue
                z = np.load(f)
                tgt = int(z['target_label'][0])
                res.setdefault(meth, []).append((split, score(ds, z['counterfactual'], z['original'], tgt, encs, T)))
        for target in range(4):
            sel = torch.where(pl['test'] != target)[0]
            m = load_orig_cf(ds, split, target, data, dev)
            cf = our_cfs(m, X[:, sel], T[:, sel])
            xb = X[:, sel].transpose(0, 1).cpu().numpy()
            res.setdefault('ours', []).append((split, score(ds, cf, xb, target, encs, T)))
        log('paper split', split, {k: len(v) for k, v in res.items()})
    out = {k: summarize_pi(v)[0] for k, v in res.items()}
    os.makedirs(os.path.join(RES, 'e3'), exist_ok=True)
    json.dump(out, open(os.path.join(RES, 'e3', 'freqshape_paper_runs.json'), 'w'), indent=1)
    for k, v in out.items():
        log(k, {kk: round(vv, 4) for kk, vv in v.items() if not kk.endswith('_se')})


def run_e7(dev, n_per_target=50):
    """PAM: baselines (first 50 eligible test instances per target) vs ours on the same instances."""
    ds = 'pam'
    res = {}
    for split in range(1, 6):
        data = load_split(ds, split, dev)
        pl = pseudo_labels(ds, split, data, dev)
        X, T = data['test'][0], data['test'][1]
        encs = {'bb': make_cf_model(ds, split, data, dev).encoder_main.eval()}
        for meth in ['comte', 'abcf', 'mcels', 'confetti']:
            for f in sorted(glob.glob(os.path.join(RES, 'e7_bl', ds, meth, f'split={split}', 'cfs', '*.npz'))):
                if 'untargeted' in f:
                    continue
                z = np.load(f)
                res.setdefault(meth, []).append((split, score(ds, z['counterfactual'], z['original'], int(z['target_label'][0]), encs, T)))
        for target in range(DS[ds]['n_classes']):
            sel = torch.where(pl['test'] != target)[0][:n_per_target]
            xb = X[:, sel].transpose(0, 1).cpu().numpy()
            m0 = load_orig_cf(ds, split, target, data, dev)
            res.setdefault('ours_paper_ckpt', []).append((split, score(ds, our_cfs(m0, X[:, sel], T[:, sel]), xb, target, encs, T)))
            for tag, r_ in [('e7', None), ('e7_long', None), ('e7_long_r0.1', 0.1)]:
                ck = os.path.join(CKPT, ds, tag, f'split={split}_target={target}_learned.pt')
                if not os.path.exists(ck):
                    continue
                m = make_cf_model(ds, split, data, dev, mask_mode='learned', r=r_)
                m.load_state_dict(torch.load(ck, map_location='cpu'))
                m.eval()
                res.setdefault('ours_' + tag, []).append((split, score(ds, our_cfs(m, X[:, sel], T[:, sel]), xb, target, encs, T)))
        log('e7 split', split, {k: len(v) for k, v in res.items()})
    out = {k: summarize_pi(v)[0] for k, v in res.items()}
    os.makedirs(os.path.join(RES, 'e7'), exist_ok=True)
    json.dump(out, open(os.path.join(RES, 'e7', 'pam_summary.json'), 'w'), indent=1)
    for k, v in out.items():
        log(k, {kk: round(vv, 4) for kk, vv in v.items() if not kk.endswith('_se')})


def mmd2_rbf(a, b, gamma=None):
    """Unbiased MMD^2 with an RBF kernel; a,b: (n, p) numpy. Median-heuristic bandwidth on b."""
    a = torch.tensor(a, dtype=torch.float64); b = torch.tensor(b, dtype=torch.float64)
    if gamma is None:
        d = torch.cdist(b, b) ** 2
        gamma = 1.0 / torch.median(d[d > 0]).item()
    k = lambda x, y: torch.exp(-gamma * torch.cdist(x, y) ** 2)
    kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
    n, m = a.shape[0], b.shape[0]
    return float((kaa.sum() - kaa.diag().sum()) / (n * (n - 1)) + (kbb.sum() - kbb.diag().sum()) / (m * (m - 1)) - 2 * kab.mean()), gamma


def run_e8(dev, n_ref=500, n_cf=300):
    """In-distribution-ness of FreqShapes CFs w.r.t. target-class training data (paper runs)."""
    from sklearn.neighbors import LocalOutlierFactor
    ds = 'freqshape'
    base = os.path.join(REPO, 'experiments', 'freqshape', 'results')
    out = {}
    rng = np.random.default_rng(0)
    for split in range(1, 6):
        data = load_split(ds, split, dev)
        pl = pseudo_labels(ds, split, data, dev)
        X, T = data['test'][0], data['test'][1]
        Xtr = data['train'][0].transpose(0, 1).cpu().numpy()
        ytr = pl['train'].cpu().numpy()
        cfs = {}
        for meth in ['abcf', 'comte', 'mcels', 'confetti']:
            for f in sorted(glob.glob(os.path.join(base, f'{meth}_transformer', 'cfs', f'*split={split}_target=*.npz'))):
                if 'untargeted' in f:
                    continue
                z = np.load(f)
                cfs[(meth, int(z['target_label'][0]))] = z['counterfactual']
        for target in range(4):
            sel = torch.where(pl['test'] != target)[0]
            cfs[('ours', target)] = our_cfs(load_orig_cf(ds, split, target, data, dev), X[:, sel], T[:, sel])
            cfs[('original inputs', target)] = X[:, sel].transpose(0, 1).cpu().numpy()
        for target in range(4):
            ref_all = Xtr[ytr == target].reshape((ytr == target).sum(), -1)
            perm = rng.permutation(ref_all.shape[0])
            held = ref_all[perm[:n_cf]]           # held-out target-class training series (reference row)
            rest = ref_all[perm[n_cf:]]
            ref = rest[:n_ref]
            lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(rest)
            _, gamma = mmd2_rbf(ref[:10], ref)
            entries = dict(cfs)
            entries[('target-class train (held)', target)] = held.reshape(held.shape[0], -1, 1)
            for (meth, t), cf in entries.items():
                if t != target:
                    continue
                c = cf.reshape(cf.shape[0], -1)
                c = c[rng.choice(c.shape[0], min(n_cf, c.shape[0]), replace=False)]
                mmd, _ = mmd2_rbf(c, ref, gamma)
                o = out.setdefault(meth, {'mmd2': [], 'lof_inlier': []})
                o['mmd2'].append(mmd)
                o['lof_inlier'].append(float((lof.predict(c) == 1).mean()))
        log('e8 split', split)
    summ = {m: {k: float(np.mean(v)) for k, v in o.items()} | {k + '_se': float(np.std(v) / np.sqrt(len(v))) for k, v in o.items()} for m, o in out.items()}
    os.makedirs(os.path.join(RES, 'e8'), exist_ok=True)
    json.dump(summ, open(os.path.join(RES, 'e8', 'freqshape_ood.json'), 'w'), indent=1)
    for m, v in summ.items():
        log(m, v)


@torch.no_grad()
def noisy_validity(enc, cf_btd, target, times, ratio, seed):
    """Paper's robustness protocol: add N(0,1) noise to a random `ratio` of the CF coordinates."""
    g = torch.Generator().manual_seed(seed)
    cf = torch.tensor(np.ascontiguousarray(cf_btd), dtype=torch.float32)
    B, T_, D = cf.shape
    k = int(round(ratio * T_ * D))
    noise = torch.zeros(B, T_, D)
    if k > 0:
        scores = torch.rand(B, T_ * D, generator=g)
        idx = scores.argsort(dim=1)[:, :k]
        vals = torch.randn(B, k, generator=g)
        noise.view(B, -1).scatter_(1, idx, vals)
    x = (cf + noise).to(times.device).transpose(0, 1)
    tb = times[:, :1].repeat(1, B)
    return (bb_logits(enc, x, tb).argmax(-1) == target).float().cpu().numpy()


def run_e6(dev, ratios=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Noise robustness on FreqShapes stratified by the CF's initial confidence."""
    ds = 'freqshape'
    base = os.path.join(REPO, 'experiments', 'freqshape', 'results')
    rows = {}
    for split in range(1, 6):
        data = load_split(ds, split, dev)
        pl = pseudo_labels(ds, split, data, dev)
        X, T = data['test'][0], data['test'][1]
        enc = make_cf_model(ds, split, data, dev).encoder_main.eval()
        items = []
        for meth in ['abcf', 'comte', 'mcels', 'confetti']:
            for f in sorted(glob.glob(os.path.join(base, f'{meth}_transformer', 'cfs', f'*split={split}_target=*.npz'))):
                if 'untargeted' in f:
                    continue
                z = np.load(f)
                items.append((meth, int(z['target_label'][0]), z['counterfactual'], z['original']))
        for target in range(4):
            sel = torch.where(pl['test'] != target)[0]
            m = load_orig_cf(ds, split, target, data, dev)
            items.append(('ours', target, our_cfs(m, X[:, sel], T[:, sel]), X[:, sel].transpose(0, 1).cpu().numpy()))
        for meth, target, cf, x in items:
            cft = torch.tensor(cf, dtype=torch.float32, device=dev).transpose(0, 1)
            p = bb_logits(enc, cft, T[:, :1].repeat(1, cf.shape[0])).softmax(-1)
            r = rows.setdefault(meth, {'conf': [], 'valid0': [], 'l2': []})
            r['conf'].append(p[:, target].cpu().numpy())
            r['valid0'].append((p.argmax(-1) == target).float().cpu().numpy())
            r['l2'].append(np.sqrt(((cf - x) ** 2).mean(axis=(1, 2))))
            for ratio in ratios:
                r.setdefault(f'v{ratio}', []).append(noisy_validity(enc, cf, target, T, ratio, seed=1000 * split + 10 * target + int(ratio * 100)))
        log('e6 split', split)
    bins = [0.0, 0.5, 0.7, 0.9, 0.99, 1.0001]
    out = {}
    for meth, r in rows.items():
        c = {k: np.concatenate(v) for k, v in r.items()}
        o = {'overall': {f'v{ratio}': float(c[f'v{ratio}'].mean()) for ratio in ratios}}
        o['overall']['valid0'] = float(c['valid0'].mean()); o['overall']['n'] = int(c['conf'].shape[0])
        for lo, hi in zip(bins[:-1], bins[1:]):
            mk = (c['conf'] >= lo) & (c['conf'] < hi)
            o[f'conf[{lo},{min(hi, 1.0)})'] = {'n': int(mk.sum()), **({f'v{ratio}': float(c[f'v{ratio}'][mk].mean()) for ratio in ratios} if mk.any() else {})}
        # among instances valid before noise, regress robustness on confidence and L2 (per-method Spearman)
        from scipy.stats import spearmanr
        v0 = c['valid0'].astype(bool)
        if v0.sum() > 10:
            o['spearman_conf_vs_v0.3'] = float(spearmanr(c['conf'][v0], c['v0.3'][v0])[0])
            o['spearman_l2_vs_v0.3'] = float(spearmanr(c['l2'][v0], c['v0.3'][v0])[0])
        out[meth] = o
        log(meth, json.dumps(o)[:600])
    os.makedirs(os.path.join(RES, 'e6'), exist_ok=True)
    json.dump(out, open(os.path.join(RES, 'e6', 'freqshape_noise_stratified.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True)
    ap.add_argument('--ds', default='freqshape')
    a = ap.parse_args()
    dev = torch.device('cuda')
    if a.mode == 'e4':
        run_e4(a.ds, dev)
    elif a.mode == 'e6':
        run_e6(dev)
    elif a.mode == 'e1sem':
        run_e1_sem(dev)
    elif a.mode == 'e7':
        run_e7(dev)
    elif a.mode == 'e8':
        run_e8(dev)
    else:
        run_paper_freqshape(dev)
