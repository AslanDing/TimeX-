"""The temporal, target-conditioned psi_cf of train_cf.py ('cond' model: ONE model per split for all target classes,
paper selection rule) as the default TimeX_cf++, retrained for the remaining counterfactual tables/figures.

Model = train_cf.CFTemporal(cond=True) (identical to the C1 'main' cond model); trainer = train_cf.train_cf_cond
generalised with (i) an external mask per (instance, target) for the mask-source / nested variants of
train_cf_local.py, (ii) the loss ablations of common.train_cf, (iii) optimiser overrides for CNN/LSTM black boxes,
(iv) an r override. Everything else as in C1: random target != predicted class per instance and epoch, reference of
the distribution term drawn from training instances predicted as that target, validation criterion = mean confidence
over all (validation instance, target != predicted) pairs, early stop when it exceeds 0.99, restarts with seeds
cfg.cf.seed + 1000*rs, best restart on validation confidence. Budgets (max_epochs, patience) are passed already
multiplied by (n_classes - 1), as for C1.

Mask sources (as train_cf_local.build_masks, same tie-break generator seeds 1000*split + 10*target + sum(ord(v))):
  learned            : g_phi (default method)
  nested  (rule prior): learned mask restricted to the top-ceil(2 r T d) coordinates of pi_a          (Table 10)
  nested  (rule k)    : learned mask restricted to the top-2k coordinates of pi_a                    (Table 9)
  attr / shift / shuffle / rand / full : fixed masks with k coordinates                               (Table 9)
k (rule k / fixed masks) = round(sum pi_cf) of the C1 main cond model of this split and target (kref='c1main'),
i.e. the learned CF mask of the default model, as the paper's k came from the default (paper) model.

Data: --data main  : Table-4 protocol (ECG: 12k/2k training/validation subsets, full test set)
      --data subset: analysis subsets (ECG 12k/2k/4k), Table 9 / Fig. operating points
Outputs results/D2_temporal/<tag>/<ds>/split=<s>_target=<t>_<mask>.json (same format as train_cf.py / train_cf_local.py);
checkpoints ckpt/D2/<tag>/<ds>/split=<s>_<mask>.pt.
usage: python train_cf_variants.py --ds freqshape --split 1 --tag nested --mask nested --nested_rule prior --data main \
         --max_epochs 300 --patience 45 --restarts 2 --nsamples 5
"""
import argparse
from common import *
from train_cf import CFTemporal
from train_cf_local import build_masks
from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.utils.data.datasets import DatasetwInds
from txai.trainers.train_mv6_consistency_cf import kl_divergence_gaussian, _apply_warmup_cosine_lr


def make_d2_model(ds, split, data, dev, mask_mode='learned', r=None, arch='transformer', use_ste=True, connect=None):
    """make_c1_model(cond=True) with the options of common.make_cf_model (r, arch, use_ste, connect)."""
    cfg = DS[ds]; cc = cfg['cf']
    X = data['train'][0]
    mu = X.mean(dim=1); std = X.std(unbiased=True, dim=1)
    m = CFTemporal(d_inp=X.shape[-1], max_len=X.shape[0], n_classes=cfg['n_classes'], n_prototypes=50,
                   gsat_r=cc['r'] if r is None else r, transformer_args=targs_for(ds, RC.cf_mod),
                   ablation_parameters=RC._abl(RC.cf_mod, arch=arch, use_ste=use_ste),
                   loss_weight_dict={'gsat': cc['gsat'], 'connect': cc['connect'] if connect is None else connect},
                   masktoken_stats=(mu, std), tau=cc.get('tau', 1.0), mask_mode=mask_mode, use_noise=True, cond=True)
    m.encoder_main.load_state_dict(load_bb_state(ds, split, 'bb' if arch == 'transformer' else arch))
    for p in m.encoder_main.parameters():
        p.requires_grad = False
    return m.to(dev)


@torch.no_grad()
def c1_pi(model, X, T, t, bs=256):
    """pi_cf (N,T,d) of a cond model for target t (common.cf_pi + the target embedding)."""
    model.eval()
    out = []
    for i in range(0, X.shape[1], bs):
        Xb, Tb = X[:, i:i + bs], T[:, i:i + bs]
        z = model.encoder_pret.embed(Xb, Tb, captum_input=False, aggregate=False)
        z = z + model.tgt_emb(model._targets(t, Xb.shape[1], Xb.device)).unsqueeze(0)
        mask_in, _ = model.mask_generator(Xb, Tb, z_seq=z, training=False)
        out.append(mask_in)
    return torch.cat(out, 0)


def val_pairs_m(model, Xv, Tv, yv, C, vmask=None, bs=256):
    """train_cf.val_pairs (with sparsity) + optional external masks vmask (C, N, T, d)."""
    confs, vals, sps = [], [], []
    for c in range(C):
        sel = torch.where(yv != c)[0]
        if sel.numel() == 0:
            continue
        model.cur_target = c
        r = eval_cf(model, Xv[:, sel], Tv[:, sel], c, ext_mask=None if vmask is None else vmask[c][sel], bs=bs)
        confs.append(r['conf']); vals.append(r['valid']); sps.append(r['spars'][1e-4])
    return float(np.concatenate(confs).mean()), float(np.concatenate(vals).mean()), float(np.concatenate(sps).mean())


def train_d2(model, ds, train, val, save_path, max_epochs, patience, masks=None, hp=None, abl=(), verbose=True):
    """train_cf.train_cf_cond (budget=None) + masks / hp / abl (loss weights exactly as common.train_cf)."""
    cc = dict(DS[ds]['cf'])
    if hp:
        cc.update(hp)
    C = DS[ds]['n_classes']
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cc['lr'], weight_decay=cc['wd'])
    cosine = cc['scheduler'] == 'cosine'
    sched = None
    if cc['scheduler'] and not cosine:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=5, threshold=1e-5,
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-8)
    init_lrs = [float(pg['lr']) for pg in opt.param_groups]
    clf_crit = Poly1CrossEntropyLoss(num_classes=C, epsilon=1.0, weight=None, reduction='mean')
    Xtr, Ttr, ytr = train
    loader = torch.utils.data.DataLoader(DatasetwInds(Xtr, Ttr, ytr), batch_size=cc['bs'], shuffle=True)
    Xv, Tv, yv = val
    pools = [torch.where(ytr == c)[0] for c in range(C)]
    avail = torch.tensor([p.numel() > 0 for p in pools], device=ytr.device)
    w_kl = 0.0 if 'no_kl' in abl else RC.DIST_W
    w_bd = 0.0 if 'no_bound' in abl else 1.0
    w_lc = 0.0 if 'no_lc' in abl else 1.0
    best, best_ep, since = -1e9, -1, 0
    for ep in range(max_epochs):
        if cosine:
            _apply_warmup_cosine_lr(opt, init_lrs, ep, max_epochs, min(10, max_epochs), 1e-6, 0.0)
        model.train()
        stats = []
        for X, times, y, ids in loader:
            B = X.shape[0]
            sc = torch.rand(B, C, device=X.device)
            sc[:, ~avail] = -1.0
            sc[torch.arange(B), y] = -1.0
            t = sc.argmax(1)
            ok = sc.max(1).values >= 0
            if ok.sum() == 0:
                continue
            X, times, y, t, ids = X[ok], times[ok], y[ok], t[ok], ids.to(X.device)[ok]
            ref_idx = torch.empty_like(t)
            for c in t.unique().tolist():
                w = torch.where(t == c)[0]
                ref_idx[w] = pools[c][torch.randint(pools[c].numel(), (w.numel(),), device=X.device)]
            ref = Xtr[:, ref_idx].transpose(0, 1)
            ext = None if masks is None else masks['train'][t, ids]
            opt.zero_grad()
            out = model(X, times, captum_input=True, reference_x=ref, target_label=t, ext_mask=ext)
            clf_loss = clf_crit(out['pred_cf'], t)
            xp = out['X_prime_cf'].transpose(0, 1)
            (rm, rlv), (cm, clv) = out['all_z']
            kl = kl_divergence_gaussian(cm, clv, rm, rlv)
            inv = 1.0 - out['ste_mask']
            rec = F.mse_loss(xp * inv, X * inv)
            l2 = F.mse_loss(xp, X)
            loss = w_lc * clf_loss + cc['beta_sim'] * (w_kl * kl + w_bd * (rec + l2))
            if out['mask_logits'] is not None:
                loss = loss + cc['beta_exp'] * model.compute_loss(out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats.append([clf_loss.item(), kl.item(), rec.item(), l2.item(), out['ste_mask'].float().mean().item()])
        st = np.mean(stats, 0) if stats else [np.nan] * 5
        met, vval, sp = val_pairs_m(model, Xv, Tv, yv, C, None if masks is None else masks['val'])
        if verbose:
            log(f'ep {ep} CE {st[0]:.4f} KL {st[1]:.4f} Rec {st[2]:.4f} L2 {st[3]:.4f} |M| {st[4]:.3f} | val conf {met:.4f} valid {vval:.4f} spars {sp:.3f}')
        if met > best:
            best, best_ep, since = met, ep, 0
            torch.save(model.state_dict(), save_path)
            if met > 0.99 or vval > 0.99:
                break
        else:
            since += 1
        if sched is not None and ep > 20:
            sched.step(met)
        if patience is not None and since >= patience:
            break
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    model.eval()
    return dict(best_val_conf=best, best_epoch=best_ep, epochs_run=ep + 1)


def load_pi_a(ds, split, data, dev):
    cpath = os.path.join(CA1, ds, f'attr_pi_split={split}.pt')
    cache = torch.load(cpath)
    pi_a = {}
    for s in ['train', 'val', 'test']:  # same indexing as train_cf_local
        p = cache[s].float()
        if data.get('sub') is not None:
            p = p[torch.tensor(data['sub'][s])]
        if data.get('sub_main') and s in data['sub_main']:
            p = p[torch.tensor(data['sub_main'][s])]
        pi_a[s] = p.to(dev)
    return pi_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True)
    ap.add_argument('--split', type=int, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--mask', default='learned', choices=['learned', 'nested', 'attr', 'rand', 'shift', 'shuffle', 'full'])
    ap.add_argument('--nested_rule', default='prior', choices=['prior', 'k'])
    ap.add_argument('--data', default='main', choices=['main', 'subset'])
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--max_epochs', type=int, required=True)
    ap.add_argument('--patience', type=int, required=True)
    ap.add_argument('--nsamples', type=int, default=5)
    ap.add_argument('--r', type=float, default=None)
    ap.add_argument('--arch', default='transformer', choices=['transformer', 'cnn', 'lstm'])
    ap.add_argument('--abl', default='', help='comma list of no_lc,no_bound,no_kl,no_con,no_ste')
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--first100', action='store_true', help='also score the first 100 eligible test instances per target with score_cf.score (operating-point figure)')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    dev = torch.device('cuda')
    ds = a.ds; cfg = DS[ds]; C = cfg['n_classes']
    abl = set(x for x in a.abl.split(',') if x)
    outdir = os.path.join(RES2, 'D2_temporal', a.tag, ds); ckdir = os.path.join(CK2, 'D2', a.tag, ds)
    os.makedirs(outdir, exist_ok=True); os.makedirs(ckdir, exist_ok=True)
    fouts = {t: os.path.join(outdir, f'split={a.split}_target={t}_{a.mask}.json') for t in range(C)}
    if all(os.path.exists(f) for f in fouts.values()) and not a.force:
        log('all outputs exist'); return
    if os.path.exists(os.path.join(ckdir, f'split={a.split}_{a.mask}.remote')) and not a.force:
        log('job assigned to an AutoDL machine (.remote marker); results are copied back from there'); return
    lock = os.path.join(ckdir, f'split={a.split}_{a.mask}.lock')  # two queue runners may hold the same job
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        if not a.force:
            log('job is running elsewhere (lock exists):', lock); return
    import atexit
    atexit.register(lambda: os.path.exists(lock) and os.remove(lock))
    hp = dict(CF_ARCH.get(ds, {})) if a.arch != 'transformer' else {}
    if a.lr is not None:
        hp['lr'] = a.lr
    hp.pop('epochs', None)  # the budget is given explicitly

    data = load_split(ds, a.split, dev, use_subset=(a.data == 'subset'), main=(a.data == 'main'))
    if a.arch == 'transformer':
        pl = pseudo_labels(ds, a.split, data, dev)
    else:  # pseudo-labels of the explained black box (as train_cf_local --arch)
        enc = make_d2_model(ds, a.split, data, dev, arch=a.arch).encoder_main
        pl = {k: bb_logits(enc, data[k][0], data[k][1]).argmax(-1) for k in ['train', 'val', 'test']}
        del enc
    train = (data['train'][0], data['train'][1], pl['train'])
    val = (data['val'][0], data['val'][1], pl['val'])
    Xte, Tte = data['test'][0], data['test'][1]
    gt_all = data['gt']
    aux = {w: load_aux_bb(ds, a.split, w, Xte.shape[-1], dev) for w in ['cnn', 'lstm'] if cfg.get(w)} if a.arch == 'transformer' else {}

    # ---- external masks (C, N, T, d) per part
    masks, kinfo = None, {}
    if a.mask != 'learned':
        pi_a = load_pi_a(ds, a.split, data, dev)
        Td = Xte.shape[0] * Xte.shape[2]
        need_k = a.mask in ('attr', 'rand', 'shift', 'shuffle') or (a.mask == 'nested' and a.nested_rule == 'k')
        ks = {}
        if need_k:  # k = round(sum pi_cf) of the C1 main cond model (the default learned CF mask)
            ckr = os.path.join(CK2, 'C1', 'main', ds, f'split={a.split}_cond.pt')
            m0 = make_d2_model(ds, a.split, data, dev)
            m0.load_state_dict(torch.load(ckr, map_location='cpu')); m0.eval()
            for t in range(C):
                ks[t] = {s: c1_pi(m0, data[s][0], data[s][1], t).sum(dim=(1, 2)).round().clamp(min=1).long() for s in ['train', 'val', 'test']}
            kinfo = {'kref': ckr, 'k_frac_test': {str(t): float(ks[t]['test'].float().mean() / Td) for t in range(C)}}
            del m0
        else:
            kp = int(np.ceil((cfg['cf']['r'] if a.r is None else a.r) * Td))
            for t in range(C):
                ks[t] = {s: torch.full((data[s][0].shape[1],), kp, dtype=torch.long, device=dev) for s in ['train', 'val', 'test']}
            kinfo = {'nested_rule': 'prior', 'k_prior': kp, 'support_frac': float(min(2 * kp, Td) / Td)}
        masks = {s: [] for s in ['train', 'val', 'test']}
        for t in range(C):  # one generator per target, parts in the order train/val/test (as train_cf_local)
            gen = torch.Generator().manual_seed(1000 * a.split + 10 * t + sum(map(ord, a.mask)))
            for s in ['train', 'val', 'test']:
                masks[s].append(build_masks(a.mask, pi_a[s], ks[t][s], gen).float())
        masks = {s: torch.stack(v, 0) for s, v in masks.items()}
        del pi_a
    mode = {'learned': 'learned', 'nested': 'nested'}.get(a.mask, 'fixed')

    # ---- training (restarts, best on validation confidence)
    ck_final = os.path.join(ckdir, f'split={a.split}_{a.mask}.pt')
    best, val_confs, t0 = None, [], time.time()
    for rs in range(a.restarts):
        set_seed(cfg['cf']['seed'] + 1000 * rs)
        model = make_d2_model(ds, a.split, data, dev, mask_mode=mode, r=a.r, arch=a.arch, use_ste=('no_ste' not in abl),
                              connect=(0.0 if 'no_con' in abl else None))
        ck_rs = ck_final[:-3] + f'_rs{rs}.pt'
        inf = train_d2(model, ds, train, val, ck_rs, a.max_epochs, a.patience, masks=masks, hp=hp, abl=abl)
        log(f'restart {rs}: {inf}')
        val_confs.append(inf['best_val_conf'])
        if best is None or inf['best_val_conf'] > best[0]['best_val_conf']:
            best = (inf, rs, model)
        else:
            del model
    info, model = dict(best[0]), best[2]
    info.update({'restart_selected': best[1], 'val_conf_restarts': val_confs, 'train_time_s': time.time() - t0})
    torch.save(model.state_dict(), ck_final)
    for rs in range(a.restarts):
        p_ = ck_final[:-3] + f'_rs{rs}.pt'
        if os.path.exists(p_):
            os.remove(p_)

    # ---- evaluation
    encs = None
    if a.first100:
        from score_cf import score, our_cfs
        encs = {'bb': model.encoder_main, **{w: load_aux_bb(ds, a.split, w, Xte.shape[-1], dev) for w in ['cnn', 'lstm'] if cfg.get(w)}}
    for t in range(C):
        te_sel = torch.where(pl['test'] != t)[0]
        if te_sel.numel() == 0:
            continue
        model.cur_target = t
        gt = gt_all[:, te_sel] if gt_all is not None else None
        ext = None if masks is None else masks['test'][t][te_sel]
        t1 = time.time()
        r = eval_cf_k(model, Xte[:, te_sel], Tte[:, te_sel], t, k=a.nsamples, ext_mask=ext, aux_bbs=aux, gt=gt)
        s = summarize(r)
        s.update({'variant': a.mask, 'ds': ds, 'split': a.split, 'target': t, 'eval_time_s': time.time() - t1,
                  'nsamples': a.nsamples, 'data': a.data, 'main': a.data == 'main', 'max_epochs': a.max_epochs,
                  'patience': a.patience, 'restarts': a.restarts, 'r': a.r if a.r is not None else cfg['cf']['r'],
                  'arch': a.arch, 'abl': sorted(abl), 'hp': hp, 'nested_rule': a.nested_rule if a.mask == 'nested' else None,
                  'ckpt': ck_final, 'model': 'C1 cond (TemporalEditor)', **kinfo, **info})
        rec = {'summary': s, 'idx': te_sel.cpu().tolist(), 'per_instance': to_jsonable(r)}
        if a.first100:
            sel = te_sel[:100]
            cf = our_cfs(model, Xte[:, sel], Tte[:, sel], ext=None if masks is None else masks['test'][t][sel])
            sc = score(ds, cf, Xte[:, sel].transpose(0, 1).cpu().numpy(), t, encs, Tte)
            rec['first100'] = {'idx': sel.cpu().tolist(), 'scores': {k: np.asarray(v).round(5).tolist() for k, v in sc.items()}}
        log(ds, a.split, t, a.tag, a.mask, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in s.items() if k not in ('hp', 'k_frac_test')}))
        json.dump(rec, open(fouts[t], 'w'))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
