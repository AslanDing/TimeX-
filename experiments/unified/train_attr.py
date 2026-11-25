"""Train TimeX_a++ attribution models with the paper's configuration and cache pi.

The distribution term follows common.DIST (env TIMEX_DIST: 'kl' = conference code, 'mmd' = fixed-kernel MMD).
Output (default variant):
  ckpt/<ds>/attr_split=<s>.pt
  cache/<ds>/attr_pi_split=<s>.pt     {'train','val','test'}: (N,T,d) float16
  results/e0/<ds>_split=<s>.json       AUPRC/AUP/AUR (paper protocol: test series with ground-truth positives)
Variants (--noleak, --abl, --arch, --r) write to their own e0_<variant> folders; only the default and --noleak
variants cache pi.
"""
import argparse
from common import *
from txai.utils.evaluation import ground_truth_xai_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True)
    ap.add_argument('--split', type=int, required=True)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--noleak', action='store_true', help='generator input = masked reference (no leakage channel)')
    ap.add_argument('--cache_only', action='store_true', help='checkpoint is complete: just (re)build the pi cache')
    ap.add_argument('--abl', default='', help='one of no_lc, no_bound, no_kl, no_ste (attribution ablation)')
    ap.add_argument('--arch', default='transformer', choices=['transformer', 'cnn', 'lstm'])
    ap.add_argument('--r', type=float, default=None, help='override the prior r (sensitivity analysis)')
    a = ap.parse_args()
    SUF = '_noleak' if a.noleak else ''
    if a.abl:
        SUF += f'_abl_{a.abl}'
    if a.arch != 'transformer':
        SUF += f'_{a.arch}'
    if a.r is not None:
        SUF += f'_r{a.r:g}'
    do_cache = SUF in ('', '_noleak')
    cpath = os.path.join(CACHE, a.ds, f'attr{SUF}_pi_split={a.split}.pt')
    path = os.path.join(CKPT, a.ds, f'attr{SUF}_split={a.split}.pt')
    fout = os.path.join(RES, 'e0' + SUF, f'{a.ds}_split={a.split}.json')
    if not a.force and not a.cache_only:
        if (do_cache and os.path.exists(cpath)) or (not do_cache and os.path.exists(fout)):
            log('done, skip', a.ds, a.split, SUF)
            return
        if do_cache and os.path.exists(path):  # another host is training this model: wait for its cache
            for _ in range(120):
                if os.path.exists(cpath):
                    log('cache appeared, skip', cpath)
                    return
                time.sleep(15)
    dev = torch.device('cuda')
    set_seed(42)
    data = load_split(a.ds, a.split, dev)
    ac = DS[a.ds]['attr']
    for d in [os.path.join(CKPT, a.ds), os.path.join(CACHE, a.ds), os.path.dirname(fout)]:
        os.makedirs(d, exist_ok=True)
    model = make_attr_model(a.ds, a.split, data, dev, noleak=a.noleak, arch=a.arch, use_ste=(a.abl != 'no_ste'), r=a.r)
    t0 = time.time()
    info = {}
    if a.force or not os.path.exists(path):
        Xtr, Ttr, ytr = data['train']
        loader = torch.utils.data.DataLoader(DatasetwInds(Xtr, Ttr, ytr), batch_size=ac['bs'], shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=ac['lr'], weight_decay=ac['wd'])
        info = train_attr(model, opt, loader, data['val'], ac['epochs'], ac['beta_exp'], ac['beta_sim'], data['train'], path,
                          use_scheduler=ac['scheduler'], batch_forward_size=64 if a.ds == 'mitecg' else None,
                          abl=(a.abl,) if a.abl and a.abl != 'no_ste' else ())
    train_time = time.time() - t0
    sd, _ = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    set_seed(0)
    pis = {k: attr_pi(model, data[k][0], data[k][1]) for k in (['train', 'val', 'test'] if do_cache else ['test'])}
    if do_cache:
        torch.save({k: v.half().cpu() for k, v in pis.items()}, cpath)
    gt = data['gt']
    out = {'ds': a.ds, 'split': a.split, 'variant': SUF or 'default', 'dist': DIST, 'dist_w': DIST_W,
           'train_time_s': train_time, 'pi_test_mean': float(pis['test'].float().mean()), **info}
    if gt is not None:
        # paper protocol: AUPRC is defined only for series with ground-truth positives (e.g. not the null class)
        has = (gt.sum(dim=(0, 2)) > 0).cpu()
        r = ground_truth_xai_eval(pis['test'][has].transpose(0, 1).cpu(), gt.cpu()[:, has])
        se = lambda v: float(np.std(v) / np.sqrt(len(v)))
        out.update({'auprc': float(np.mean(r['auprc'])), 'auprc_se': se(r['auprc']),
                    'aup': float(np.mean(r['aup'])), 'aup_se': se(r['aup']),
                    'aur': float(np.mean(r['aur'])), 'aur_se': se(r['aur']),
                    'n_test': int(gt.shape[1]), 'n_with_gt': int(has.sum()), 'protocol': 'gt_only'})
    log(json.dumps(out))
    json.dump(out, open(fout, 'w'), indent=1)


if __name__ == '__main__':
    main()
