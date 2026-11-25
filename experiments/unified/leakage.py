"""Does the attribution generator psi_a leak information through unselected coordinates?
Also: distribution of the soft masks pi (attribution r=0.5 vs counterfactual r<0.5).

psi_a is applied coordinate-wise: X_a[t,d] = h(X[t,d], M[t,d]). Hence on unselected coordinates
the generated value is h0(x) = h(x, 0), a fixed scalar function of the original value. We
  (i)  characterise h0 and h1 directly (correlation with x, variance ratio, curves), and
  (ii) measure how often f(psi_a(X, M)) keeps the original prediction when M is the learned
       mask, a size-matched random / shifted / shuffled mask, the empty mask, or the full mask,
       and compare with the generator-free reference X_r = X*M + (1-M)*b.
"""
import argparse
from common import *
from train_cf_local import rank_topk, shift_mask, derangement
from scipy.stats import spearmanr, pearsonr


@torch.no_grad()
def gen_attr(model, X, M, b=None):
    if isinstance(model, AttrNoLeak):
        X = X * M + (1 - M) * b
    return model.mask_connection_src(torch.stack([X, M], dim=-1)).squeeze(-1)


@torch.no_grad()
def eval_masks(model, X, T, y0, p0, M, bs=256):
    enc = model.encoder_main
    agree_g, agree_r, js_g, js_r, py_g, py_r = [], [], [], [], [], []
    for i in range(0, X.shape[1], bs):
        Xb, Tb, Mb = X[:, i:i + bs], T[:, i:i + bs], M[i:i + bs].transpose(0, 1)
        yb, pb = y0[i:i + bs], p0[i:i + bs]
        b = model._get_baseline(B=Xb.shape[1])
        xg = gen_attr(model, Xb, Mb, b)
        xr = Xb * Mb + (1 - Mb) * b
        for x, ag, js, py in [(xg, agree_g, js_g, py_g), (xr, agree_r, js_r, py_r)]:
            p = enc(x, Tb, captum_input=False).softmax(-1)
            ag.append((p.argmax(-1) == yb).float())
            m = 0.5 * (p + pb)
            js.append(0.5 * (pb * (pb.clamp_min(1e-12) / m.clamp_min(1e-12)).log()).sum(-1)
                      + 0.5 * (p * (p.clamp_min(1e-12) / m.clamp_min(1e-12)).log()).sum(-1))
            py.append(p.gather(1, yb.view(-1, 1)).squeeze(1))
    c = lambda v: float(torch.cat(v).mean())
    return {'agree_gen': c(agree_g), 'agree_ref': c(agree_r), 'js_gen': c(js_g), 'js_ref': c(js_r),
            'p_orig_gen': c(py_g), 'p_orig_ref': c(py_r), 'mask_frac': float(M.float().mean())}


def pi_stats(pi):
    p = pi.flatten().float().cpu().numpy()
    h, edges = np.histogram(p, bins=20, range=(0, 1))
    ent = -(p * np.log2(np.clip(p, 1e-9, 1)) + (1 - p) * np.log2(np.clip(1 - p, 1e-9, 1)))
    return {'hist': (h / h.sum()).round(5).tolist(), 'edges': edges.round(3).tolist(), 'mean': float(p.mean()),
            'frac_lt_0.1': float((p < 0.1).mean()), 'frac_gt_0.9': float((p > 0.9).mean()),
            'frac_0.4_0.6': float(((p >= 0.4) & (p <= 0.6)).mean()), 'mean_entropy_bits': float(ent.mean()),
            'frac_gt_0.5': float((p > 0.5).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True)
    ap.add_argument('--split', type=int, required=True)
    ap.add_argument('--noleak', action='store_true')
    a = ap.parse_args()
    SUF = '_noleak' if a.noleak else ''
    dev = torch.device('cuda')
    ds = a.ds
    os.makedirs(os.path.join(RES, 'e2' + SUF), exist_ok=True)
    data = load_split(ds, a.split, dev, use_subset=True)
    model = make_attr_model(ds, a.split, data, dev, noleak=a.noleak)
    sd, _ = torch.load(os.path.join(CKPT, ds, f'attr{SUF}_split={a.split}.pt'), map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    X, T = data['test'][0], data['test'][1]
    p0 = bb_logits(model.encoder_main, X, T).softmax(-1)
    y0 = p0.argmax(-1)
    set_seed(0)
    pi = attr_pi(model, X, T)
    gen = torch.Generator().manual_seed(11)
    M_s = torch.bernoulli(pi.clamp(0, 1))
    k = M_s.sum(dim=(1, 2)).long().clamp(min=1)
    masks = {
        'learned_sample': M_s,
        'learned_det': (pi > 0.5).float(),
        'rand': rank_topk(torch.rand(pi.shape, generator=gen).to(dev), k, gen),
        'shift': shift_mask(M_s, gen),
        'shuffle': M_s[derangement(M_s.shape[0], gen).to(dev)],
        'empty': torch.zeros_like(pi),
        'full': torch.ones_like(pi),
    }
    res = {'ds': ds, 'split': a.split, 'n': int(X.shape[1]),
           'majority_rate': float(torch.bincount(y0).max() / y0.shape[0])}
    for name, M in masks.items():
        set_seed(1)
        res[name] = eval_masks(model, X, T, y0, p0, M)
        log(ds, a.split, name, res[name])

    # characterise the coordinate-wise generator
    x = X.flatten()
    with torch.no_grad():
        bflat = model._get_baseline(B=X.shape[1]).flatten().view(-1, 1, 1)
        h0 = gen_attr(model, x.view(-1, 1, 1), torch.zeros_like(x).view(-1, 1, 1), bflat).flatten()
        h1 = gen_attr(model, x.view(-1, 1, 1), torch.ones_like(x).view(-1, 1, 1), bflat).flatten()
    xs, h0s, h1s = x.cpu().numpy(), h0.cpu().numpy(), h1.cpu().numpy()
    rng = np.random.default_rng(0)
    sub = rng.choice(xs.shape[0], size=min(200000, xs.shape[0]), replace=False)
    mu = model.masktoken_stats[0].to(dev)
    res['h'] = {
        'pearson_h0_x': float(pearsonr(h0s[sub], xs[sub])[0]),
        'spearman_h0_x': float(spearmanr(h0s[sub], xs[sub])[0]),
        'var_ratio_h0': float(h0s.var() / xs.var()),
        'pearson_h1_x': float(pearsonr(h1s[sub], xs[sub])[0]),
        'var_ratio_h1': float(h1s.var() / xs.var()),
        'mse_h1_x_over_varx': float(((h1s - xs) ** 2).mean() / xs.var()),
        'mse_h0_x_over_varx': float(((h0s - xs) ** 2).mean() / xs.var()),
        'mse_mu_x_over_varx': float(((mu.unsqueeze(1).expand_as(X).flatten() - x) ** 2).mean().item() / xs.var()),
    }
    lo, hi = np.quantile(xs, [0.005, 0.995])
    grid = torch.linspace(float(lo), float(hi), 101, device=dev)
    with torch.no_grad():
        g0 = gen_attr(model, grid.view(-1, 1, 1), torch.zeros_like(grid).view(-1, 1, 1), grid.view(-1, 1, 1)).flatten()
        g1 = gen_attr(model, grid.view(-1, 1, 1), torch.ones_like(grid).view(-1, 1, 1), grid.view(-1, 1, 1)).flatten()
    res['h_curve'] = {'x': grid.cpu().numpy().round(4).tolist(), 'h0': g0.cpu().numpy().round(4).tolist(),
                      'h1': g1.cpu().numpy().round(4).tolist()}
    log(ds, a.split, 'h', res['h'])

    # pi statistics for attribution and for the paper's CF checkpoints
    res['pi_attr'] = pi_stats(pi)
    pl = pseudo_labels(ds, a.split, data, dev)
    res['pi_cf'] = {}
    for target in range(DS[ds]['n_classes']):
        sel = torch.where(pl['test'] != target)[0]
        m0 = load_orig_cf(ds, a.split, target, data, dev)
        res['pi_cf'][target] = pi_stats(cf_pi(m0, X[:, sel], T[:, sel]))
    log(ds, a.split, 'pi_attr', {k: v for k, v in res['pi_attr'].items() if k not in ('hist', 'edges')})
    json.dump(res, open(os.path.join(RES, 'e2' + SUF, f'{ds}_split={a.split}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
