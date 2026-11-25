"""Local (coordinate-wise) editor of TimeX_cf++, and the mask-source control: does the attribution mask support
counterfactuals?

For each (dataset, split, target) we train psi_cf (+psi_n) with the SAME architecture, losses
and optimiser as TimeX_cf++, but change where the mask comes from:

  orig        : the paper's checkpoint (learned CF mask), evaluated with our evaluator
  learned     : the paper's method retrained under this protocol (sanity check)
  learned_noeps : learned, with the auxiliary variation epsilon disabled (multivariate only)
  attr        : frozen TimeX_a++ mask, top-k coordinates of pi_a
  rand        : k coordinates chosen uniformly at random
  shift       : the attr top-k mask circularly shifted in time (and channels permuted if d>1)
  shuffle     : the attr top-k ranking of a different instance
  full        : M = 1 (no localisation)
  nested      : learned CF mask restricted to the 2k most-attributed coordinates

k is matched per instance and per target to the expected size of the paper's learned CF mask,
k = round(sum pi_cf), so every fixed-mask variant has the same budget as `orig`.
"""
import argparse
from common import *
from txai.utils.evaluation import ground_truth_xai_eval


def rank_topk(score, k, gen):
    N = score.shape[0]
    flat = score.reshape(N, -1).float()
    flat = flat + 1e-6 * torch.rand(flat.shape, generator=gen, device='cpu').to(flat.device)
    order = flat.argsort(dim=1, descending=True)
    D = flat.shape[1]
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(D, device=flat.device).expand(N, D).contiguous())
    return (ranks < k.view(-1, 1)).float().reshape(score.shape)


def shift_mask(m, gen):
    N, T, d = m.shape
    off = torch.randint(T // 4, 3 * T // 4 + 1, (N,), generator=gen).to(m.device)
    idx = (torch.arange(T, device=m.device).view(1, T) - off.view(N, 1)) % T
    out = torch.gather(m, 1, idx.unsqueeze(-1).expand(N, T, d))
    if d > 1:
        perm = torch.stack([torch.randperm(d, generator=gen) for _ in range(N)]).to(m.device)
        out = torch.gather(out, 2, perm.unsqueeze(1).expand(N, T, d))
    return out


def derangement(N, gen):
    p = torch.randperm(N, generator=gen)
    fixed = torch.where(p == torch.arange(N))[0]
    if len(fixed) > 0:
        p[fixed] = p[torch.roll(fixed, 1)] if len(fixed) > 1 else p[(fixed + 1) % N]
    return p


def build_masks(variant, pi_a, k, gen):
    N, T, d = pi_a.shape
    if variant == 'attr':
        return rank_topk(pi_a, k, gen)
    if variant == 'rand':
        return rank_topk(torch.rand(pi_a.shape, generator=gen).to(pi_a.device), k, gen)
    if variant == 'shift':
        return shift_mask(rank_topk(pi_a, k, gen), gen)
    if variant == 'shuffle':
        return rank_topk(pi_a[derangement(N, gen).to(pi_a.device)], k, gen)
    if variant == 'full':
        return torch.ones_like(pi_a, dtype=torch.float)
    if variant == 'nested':
        return rank_topk(pi_a, torch.clamp(2 * k, max=T * d), gen)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True)
    ap.add_argument('--split', type=int, required=True)
    ap.add_argument('--variants', default='orig,learned,attr,rand,shift,shuffle,full,nested')
    ap.add_argument('--targets', default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--max_epochs', type=int, default=None)
    ap.add_argument('--tag', default='e1')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--r', type=float, default=None, help='override gsat r of the CF model (operating-point sweep)')
    ap.add_argument('--restarts', type=int, default=1, help='training restarts per variant, best on validation kept')
    # main-table protocol
    ap.add_argument('--main', action='store_true', help='full data (ECG: 2k validation subset only), no analysis subsets')
    ap.add_argument('--nsamples', type=int, default=1, help='average test metrics over this many mask samples')
    ap.add_argument('--no_agree', action='store_true', help='skip the CF/attribution mask-agreement analysis')
    ap.add_argument('--nested_rule', default='k', choices=['k', 'prior'],
                    help="support of the nested mask: top-2k with k from the paper CF mask ('k') or k = r*T*d ('prior')")
    ap.add_argument('--arch', default='transformer', choices=['transformer', 'cnn', 'lstm'])
    ap.add_argument('--abl', default='', help='comma list of no_lc,no_bound,no_kl,no_con,no_ste')
    ap.add_argument('--kref', default='orig', choices=['orig', 'learned'],
                    help="model defining the per-instance budget k: the paper checkpoint ('orig') or the learned model "
                         "of this run with the same tag ('learned', trained by a separate job)")
    ap.add_argument('--lr', type=float, default=None, help='override the CF learning rate')
    ap.add_argument('--wd', type=float, default=None, help='override the CF weight decay')
    a = ap.parse_args()
    dev = torch.device('cuda')
    ds = a.ds
    cfg = DS[ds]
    variants = a.variants.split(',')
    abl = set(x for x in a.abl.split(',') if x)
    outdir = os.path.join(RES, a.tag, ds)
    ckdir = os.path.join(CKPT, ds, a.tag)
    os.makedirs(outdir, exist_ok=True); os.makedirs(ckdir, exist_ok=True)
    hp = CF_ARCH.get(ds, {}) if a.arch != 'transformer' else None
    if a.lr is not None or a.wd is not None:  # explicit override (learning-rate selection for CNN/LSTM black boxes)
        hp = dict(hp or {})
        if a.lr is not None:
            hp['lr'] = a.lr
        if a.wd is not None:
            hp['wd'] = a.wd

    data = load_split(ds, a.split, dev, use_subset=not a.main, main=a.main)
    if a.arch == 'transformer':
        pl = pseudo_labels(ds, a.split, data, dev)
    else:  # pseudo-labels of the explained black box
        enc = make_cf_model(ds, a.split, data, dev, arch=a.arch).encoder_main
        pl = {k: bb_logits(enc, data[k][0], data[k][1]).argmax(-1) for k in ['train', 'val', 'test']}
        del enc
    train = (data['train'][0], data['train'][1], pl['train'])
    val = (data['val'][0], data['val'][1], pl['val'])
    Xte, Tte = data['test'][0], data['test'][1]
    gt_all = data['gt']
    cpath = os.path.join(CACHE, ds, f'attr_pi_split={a.split}.pt')
    pi_a = None
    if os.path.exists(cpath):
        cache = torch.load(cpath)
        pi_a = {}
        for s in ['train', 'val', 'test']:
            p = cache[s].float()
            if data.get('sub') is not None:
                p = p[torch.tensor(data['sub'][s])]
            if data.get('sub_main') and s in data['sub_main']:
                p = p[torch.tensor(data['sub_main'][s])]
            pi_a[s] = p.to(dev)
    d_inp = Xte.shape[-1]
    aux = {w: load_aux_bb(ds, a.split, w, d_inp, dev) for w in ['cnn', 'lstm'] if cfg.get(w)} if a.arch == 'transformer' else {}
    targets = range(cfg['n_classes']) if a.targets is None else [int(t) for t in a.targets.split(',')]
    need_k = any(v in ('attr', 'rand', 'shift', 'shuffle', 'full') for v in variants) or \
        ('nested' in variants and a.nested_rule == 'k') or any(v.startswith('orig') for v in variants) or not a.no_agree
    assert not (a.kref == 'learned' and need_k and 'learned' in variants), 'run the learned variant in a separate job'

    for target in targets:
        te_sel = torch.where(pl['test'] != target)[0]
        if te_sel.numel() == 0:
            continue
        gt = gt_all[:, te_sel] if gt_all is not None else None
        m0, ks, pic = None, None, None
        if need_k and a.arch == 'transformer':
            if a.kref == 'orig':
                m0 = load_orig_cf(ds, a.split, target, data, dev)
            else:  # learned CF model of this run (same tag), trained by a separate job: wait for it
                pk = os.path.join(ckdir, f'split={a.split}_target={target}_learned.pt')
                for _ in range(720):
                    if os.path.exists(pk) and os.path.exists(os.path.join(outdir, f'split={a.split}_target={target}_learned.json')):
                        break
                    time.sleep(30)
                m0 = make_cf_model(ds, a.split, data, dev, r=a.r)
                m0.load_state_dict(torch.load(pk, map_location='cpu'))
                m0.eval()
            pic = {s: cf_pi(m0, data[s][0], data[s][1]) for s in ['train', 'val', 'test']}
            ks = {s: pic[s].sum(dim=(1, 2)).round().clamp(min=1).long() for s in pic}
        if 'nested' in variants and a.nested_rule == 'prior':
            Td = Xte.shape[0] * Xte.shape[2]
            kp = int(np.ceil((cfg['cf']['r'] if a.r is None else a.r) * Td))
            ks = {s: torch.full((data[s][0].shape[1],), kp, dtype=torch.long, device=dev) for s in ['train', 'val', 'test']}

        # ---- mask agreement between the learned CF mask and the attribution mask (test, te_sel)
        if not a.no_agree and pic is not None and pi_a is not None and gt is not None:
            gen = torch.Generator().manual_seed(7)
            kt = ks['test'][te_sel]
            top_cf = rank_topk(pic['test'][te_sel], kt, gen)
            top_a = rank_topk(pi_a['test'][te_sel], kt, gen)
            overlap = ((top_cf * top_a).sum(dim=(1, 2)) / kt.float()).cpu().numpy()
            chance = (kt.float() / (top_cf.shape[1] * top_cf.shape[2])).cpu().numpy()
            r_cf = ground_truth_xai_eval(pic['test'][te_sel].transpose(0, 1).cpu(), gt.cpu())
            r_a = ground_truth_xai_eval(pi_a['test'][te_sel].transpose(0, 1).cpu(), gt.cpu())
            agree = {'overlap_at_k': float(overlap.mean()), 'overlap_chance': float(chance.mean()),
                     'auprc_cf_mask_vs_gt': float(np.nanmean(r_cf['auprc'])),
                     'auprc_attr_mask_vs_gt': float(np.nanmean(r_a['auprc'])),
                     'k_mean_frac': float(chance.mean())}
            json.dump(agree, open(os.path.join(outdir, f'split={a.split}_target={target}_agreement.json'), 'w'), indent=1)
            log(ds, a.split, target, 'agreement', agree)

        for v in variants:
            fout = os.path.join(outdir, f'split={a.split}_target={target}_{v}.json')
            if os.path.exists(fout) and not a.force:
                log('skip existing', fout)
                continue
            t0 = time.time()
            info = {}
            if v in ('orig', 'orig_det'):
                m0.det_eval = (v == 'orig_det')
                r = eval_cf_k(m0, Xte[:, te_sel], Tte[:, te_sel], target, k=a.nsamples, aux_bbs=aux, gt=gt)
                m0.det_eval = False
            else:
                gen = torch.Generator().manual_seed(1000 * a.split + 10 * target + sum(map(ord, v)))
                if v in ('learned', 'learned_noeps'):
                    masks = {s: None for s in ['train', 'val', 'test']}
                else:
                    masks = {s: build_masks(v, pi_a[s], ks[s], gen) for s in ['train', 'val', 'test']}
                mode = {'learned': 'learned', 'learned_noeps': 'learned', 'nested': 'nested'}.get(v, 'fixed')
                mask_fn = None if masks['train'] is None else (lambda s, idx, M=masks: M[s][idx.to(dev)])
                # psi_cf is a tiny coordinate-wise MLP and its training is sensitive to initialisation, so every
                # variant gets the same number of restarts; the restart with the best VALIDATION confidence is kept.
                ck_final = os.path.join(ckdir, f'split={a.split}_target={target}_{v}.pt')
                best, val_confs = None, []
                for rs in range(a.restarts):
                    set_seed(cfg['cf']['seed'] + 1000 * rs)
                    model = make_cf_model(ds, a.split, data, dev, mask_mode=mode, use_noise=(v != 'learned_noeps'), r=a.r,
                                          arch=a.arch, use_ste=('no_ste' not in abl),
                                          connect=(0.0 if 'no_con' in abl else None))
                    ck_rs = ck_final[:-3] + f'_rs{rs}.pt'
                    inf = train_cf(model, ds, train, val, target, ck_rs, mask_fn=mask_fn, max_epochs=a.max_epochs,
                                   patience=a.patience, verbose=True, hp=hp, abl=abl)
                    val_confs.append(inf['best_val_conf'])
                    if best is None or inf['best_val_conf'] > best[0]['best_val_conf']:
                        best = (inf, rs, model)
                    else:
                        del model
                info, model = dict(best[0]), best[2]
                info.update({'restart_selected': best[1], 'val_conf_restarts': val_confs})
                torch.save(model.state_dict(), ck_final)
                for rs in range(a.restarts):  # keep only the selected model
                    p_ = ck_final[:-3] + f'_rs{rs}.pt'
                    if os.path.exists(p_):
                        os.remove(p_)
                ext = None if masks['test'] is None else masks['test'][te_sel]
                r = eval_cf_k(model, Xte[:, te_sel], Tte[:, te_sel], target, k=a.nsamples, ext_mask=ext, aux_bbs=aux, gt=gt)
                del model
            s = summarize(r)
            s.update({'variant': v, 'ds': ds, 'split': a.split, 'target': target, 'time_s': time.time() - t0, 'r': a.r,
                      'arch': a.arch, 'abl': sorted(abl), 'nsamples': a.nsamples, 'main': a.main, **info})
            log(ds, a.split, target, v, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in s.items()}))
            json.dump({'summary': s, 'idx': te_sel.cpu().tolist(), 'per_instance': to_jsonable(r)}, open(fout, 'w'))
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
