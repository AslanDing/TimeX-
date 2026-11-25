"""How far from the data distribution are the instances on which the classifier is queried for
attribution? Recomputes the KDE / MMD comparison of the OOD section with non-learned statistics and adds the
generator-free reference and the leakage-free generator.

Instances (all with the same Bernoulli(pi) mask sample of TimeX_a++):
  zero      X*M
  mean      X*M + (1-M)*mu
  reference X*M + (1-M)*b,   b ~ N(mu, sigma)  (the generator-free query)
  gen       psi_a(X, M)                         (TimeX_a++ as published; can pass x through h(x,0))
  gen_noleak psi_a(X*M+(1-M)*b, M)              (leakage-free variant, own mask), if trained
  test      X                                   (sanity check)
Metrics against held-out training series: Gaussian-KDE log-likelihood per series, unbiased RBF-MMD^2,
LOF inlier rate.
"""
import argparse, glob
from common import *
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from score_cf import mmd2_rbf
from scipy.spatial.distance import cdist


def energy_dist(x, y):
    """Unbiased energy distance 2E|X-Y| - E|X-X'| - E|Y-Y'| (Szekely & Rizzo); no kernel or bandwidth."""
    n, m = len(x), len(y)
    return float(2 * cdist(x, y).mean() - cdist(x, x).sum() / (n * (n - 1)) - cdist(y, y).sum() / (m * (m - 1)))


@torch.no_grad()
def build(model, X, M, b, noleak=False):
    Mr = M.transpose(0, 1)
    inp = X * Mr + (1 - Mr) * b if noleak else X
    return model.mask_connection_src(torch.stack([inp, Mr], dim=-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='freqshape,seqcomb_uv,seqcomb_mv,lowvar')
    ap.add_argument('--n_ref', type=int, default=2000)
    ap.add_argument('--n_eval', type=int, default=500)
    a = ap.parse_args()
    dev = torch.device('cuda')
    os.makedirs(os.path.join(RES, 'ood'), exist_ok=True)
    for ds in a.ds.split(','):
        rows = {}
        for split in range(1, 6):
            ck = os.path.join(CKPT, ds, f'attr_split={split}.pt')
            if not os.path.exists(ck):
                continue
            data = load_split(ds, split, dev)
            X, T = data['test'][0], data['test'][1]
            rng = np.random.default_rng(split)
            ev = torch.tensor(rng.choice(X.shape[1], min(a.n_eval, X.shape[1]), replace=False), device=dev)
            X, T = X[:, ev], T[:, ev]
            Xtr = data['train'][0]
            perm = rng.permutation(Xtr.shape[1])
            ref = Xtr[:, torch.tensor(perm[:a.n_ref], device=dev)].transpose(0, 1).reshape(min(a.n_ref, Xtr.shape[1]), -1).cpu().numpy()
            held = Xtr[:, torch.tensor(perm[a.n_ref:a.n_ref + a.n_eval], device=dev)].transpose(0, 1).reshape(-1, ref.shape[1]).cpu().numpy()
            inst = {}
            for tag, noleak in [('gen', False), ('gen_noleak', True)]:
                ckp = os.path.join(CKPT, ds, f'attr{"_noleak" if noleak else ""}_split={split}.pt')
                if not os.path.exists(ckp):
                    continue
                m = make_attr_model(ds, split, data, dev, noleak=noleak)
                sd, _ = torch.load(ckp, map_location='cpu', weights_only=False)
                m.load_state_dict(sd); m.eval()
                set_seed(0)
                pi = attr_pi(m, X, T)
                M = torch.bernoulli(pi.clamp(0, 1))
                set_seed(1)
                b = m._get_baseline(B=X.shape[1])
                inst[tag] = build(m, X, M, b, noleak=noleak)
                if not noleak:
                    Mr = M.transpose(0, 1)
                    mu = m.masktoken_stats[0].unsqueeze(1)
                    inst['zero'] = X * Mr
                    inst['mean'] = X * Mr + (1 - Mr) * mu
                    inst['reference'] = X * Mr + (1 - Mr) * b
                    inst['test'] = X
            if not inst:
                continue
            # KDE bandwidth: Scott's rule with the average per-dimension std of the reference set
            n, p = ref.shape
            h = float(ref.std(0).mean() * n ** (-1.0 / (p + 4)))
            kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(ref)
            lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(ref)
            _, gamma = mmd2_rbf(held[:10], ref[:1000])
            inst_np = {k: v.transpose(0, 1).reshape(v.shape[1], -1).cpu().numpy() for k, v in inst.items()}
            inst_np['heldout_train'] = held
            for k, v in inst_np.items():
                r = rows.setdefault(k, {'kde': [], 'mmd2': [], 'energy': [], 'lof_inlier': []})
                r['kde'].append(float(kde.score_samples(v).mean()))
                r['mmd2'].append(mmd2_rbf(v, ref[:1000], gamma)[0])
                r['energy'].append(energy_dist(v, ref[:1000]))
                r['lof_inlier'].append(float((lof.predict(v) == 1).mean()))
            log(ds, split, {k: round(v['mmd2'][-1], 4) for k, v in rows.items()})
        summ = {k: {**{m: float(np.mean(x)) for m, x in v.items()}, **{m + '_se': float(np.std(x) / np.sqrt(len(x))) for m, x in v.items()},
                    'n_splits': len(v['kde'])} for k, v in rows.items()}
        json.dump(summ, open(os.path.join(RES, 'ood', f'{ds}.json'), 'w'), indent=1)
        for k, v in summ.items():
            log(ds, k, {m: round(x, 4) for m, x in v.items() if not m.endswith('_se')})


if __name__ == '__main__':
    main()
