"""TimeX_cf++ with a temporal, target-conditioned editing generator psi_cf (the default counterfactual explainer).

The local editor (train_cf_local.py) is a coordinate-wise MLP, X'' = X + M * e(X[t,d], M[t,d]): it cannot use the context of a coordinate
and every target class needs its own model. Here psi_cf is replaced by a dilated 1D-CNN over time
(TemporalEditor; input channels [X, M, one-hot(target)], 6 residual blocks, kernel 5, dilations 1..32, receptive field
253 steps), and the edit is still restricted to the mask: X_cf = X + M * e(X, M, t).
Everything else is the paper's model and protocol (mask generator g_phi with Gumbel-STE and Bernoulli-KL prior r,
auxiliary variation psi_n during training, losses, optimiser, frozen black box, data protocol of Table 4,
best of 2 restarts on validation confidence, 5 mask samples at test).

Two variants
  cond      : ONE model per split for all target classes. The target enters psi_cf (one-hot channels) and the mask
              generator (learned target embedding added to the encoder memory, zero-initialised). Each training
              instance gets a target drawn uniformly from the classes != its predicted class (new draw every epoch);
              the reference of the distribution term is drawn from training instances predicted as that target.
              Validation criterion: mean confidence over all (validation instance, target != predicted class) pairs.
              Budget: the same number of (instance, target) training examples as the per-target models, i.e.
              max_epochs and patience x (n_classes - 1) (set in the job file).
  pertarget : one model per target class as in the paper (common.train_cf unchanged), only psi_cf replaced.

Outputs results/C1_temporal/<tag>/<ds>/split=<s>_target=<t>_<variant>.json (same format as the local-editor runs, results/main);
checkpoints ckpt/C1/<tag>/<ds>/.
PAM (Table 11 protocol, --pam): evaluation on the first 50 eligible test instances per target, one mask sample (seed 0).
usage:
  python train_cf.py --ds freqshape --split 1 --variant cond --max_epochs 300 --patience 45
  python train_cf.py --ds freqshape --split 1 --variant pertarget --targets 0 --max_epochs 100 --patience 15
"""
import argparse
import torch.nn as nn
from common import *
from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.utils.data.datasets import DatasetwInds
from txai.trainers.train_mv6_consistency_cf import kl_divergence_gaussian, _apply_warmup_cosine_lr


class TemporalEditor(nn.Module):
    def __init__(self, d, n_classes, cond=True, hidden=64, n_layers=6, k=5):
        super().__init__()
        self.cond, self.n_classes = cond, n_classes
        self.inp = nn.Conv1d(2 * d + (n_classes if cond else 0), hidden, 1)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dil = 2 ** i
            self.blocks.append(nn.Sequential(nn.GroupNorm(1, hidden), nn.GELU(),
                                             nn.Conv1d(hidden, hidden, k, padding=dil * (k - 1) // 2, dilation=dil)))
        self.out = nn.Conv1d(hidden, d, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # starts at X_cf = X

    def forward(self, X, M, t):
        """X, M: (B,T,d); t: (B,) long. Returns e: (B,T,d)."""
        h = [X, M]
        if self.cond:
            h.append(F.one_hot(t, self.n_classes).float().unsqueeze(1).expand(-1, X.shape[1], -1))
        h = self.inp(torch.cat(h, -1).transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        return self.out(F.gelu(h)).transpose(1, 2)


class CFTemporal(RC.CFVariant):
    def __init__(self, *a, cond=True, hidden=64, n_layers=6, ksize=5, **k):
        super().__init__(*a, **k)
        self.cond = cond
        self.cur_target = 0
        self.editor = TemporalEditor(self.d_inp, self.n_classes, cond, hidden, n_layers, ksize)
        if cond:
            self.tgt_emb = nn.Embedding(self.n_classes, self.d_inp + self.d_pe)
            nn.init.zeros_(self.tgt_emb.weight)
        self.perturbation_generator = None  # coordinate-wise psi_cf of the paper: not used

    def _targets(self, target_label, B, dev):
        t = self.cur_target if target_label is None else target_label
        if torch.is_tensor(t):
            return t.to(dev).long()
        return torch.full((B,), int(t), dtype=torch.long, device=dev)

    def forward(self, src, times, captum_input=False, reference_x=None, target_label=None, ext_mask=None):
        if captum_input:
            src = src.transpose(0, 1); times = times.transpose(0, 1)
            if reference_x is not None:
                reference_x = reference_x.transpose(0, 1)
        T, B, d = src.shape
        t = self._targets(target_label, B, src.device)
        mask_in = None
        if self.mask_mode in ('learned', 'nested'):
            z_seq = self.encoder_pret.embed(src, times, captum_input=False, aggregate=False)
            if self.cond:
                z_seq = z_seq + self.tgt_emb(t).unsqueeze(0)
            mask_in, ste_mask = self.mask_generator(src, times, z_seq=z_seq, training=self.training)
            if ste_mask.ndim == 2:
                ste_mask = ste_mask.unsqueeze(-1)
            if (not self.training) and self.det_eval:
                ste_mask = (mask_in > 0.5).float()
            if self.mask_mode == 'nested':
                ste_mask = ste_mask * ext_mask
        else:
            ste_mask = ext_mask
        ste_rs = ste_mask.transpose(0, 1)  # (T,B,d)
        e_raw = self.editor(src.transpose(0, 1), ste_mask, t).transpose(0, 1)
        e_cf = e_raw * ste_rs
        if reference_x is None:
            reference_x = src
        if self.training and self.use_noise:
            noise = self.noise_generator(torch.stack([src, ste_rs, reference_x], dim=-1)).squeeze(-1)
            noise = noise - noise.mean(dim=-1, keepdim=True)
        else:
            noise = torch.zeros_like(src)
        x_prime = src + e_cf + noise
        pred_cf = self.encoder_main(x_prime, times, captum_input=False)
        ref_distr = self.src_distribution(reference_x.transpose(0, 1).reshape(-1, self.max_len * self.d_inp))
        cf_distr = self.mask_src_distribution(x_prime.transpose(0, 1).reshape(-1, self.max_len * self.d_inp))
        return {'pred_cf': pred_cf, 'mask_logits': mask_in, 'ste_mask': ste_mask, 'smooth_src': src,
                'all_z': (ref_distr, cf_distr), 'X_prime_cf': x_prime, 'E_cf': e_cf, 'X_cf': src + e_cf, 'noise': noise}


def make_c1_model(ds, split, data, dev, cond, hidden=64, n_layers=6, mask_mode='learned'):
    cfg = DS[ds]; cc = cfg['cf']
    X = data['train'][0]
    mu = X.mean(dim=1); std = X.std(unbiased=True, dim=1)
    m = CFTemporal(d_inp=X.shape[-1], max_len=X.shape[0], n_classes=cfg['n_classes'], n_prototypes=50, gsat_r=cc['r'],
                   transformer_args=targs_for(ds, RC.cf_mod), ablation_parameters=RC._abl(RC.cf_mod),
                   loss_weight_dict={'gsat': cc['gsat'], 'connect': cc['connect']}, masktoken_stats=(mu, std),
                   tau=cc.get('tau', 1.0), mask_mode=mask_mode, use_noise=True, cond=cond, hidden=hidden, n_layers=n_layers)
    m.encoder_main.load_state_dict(load_bb_state(ds, split, 'bb'))
    for p in m.encoder_main.parameters():
        p.requires_grad = False
    return m.to(dev)


def val_pairs(model, Xv, Tv, yv, C, bs=256, with_spars=False):
    """Mean confidence / validity (and sparsity) over all (validation instance, target != predicted class) pairs (1 mask sample)."""
    confs, vals, sps = [], [], []
    for c in range(C):
        sel = torch.where(yv != c)[0]
        if sel.numel() == 0:
            continue
        model.cur_target = c
        r = eval_cf(model, Xv[:, sel], Tv[:, sel], c, bs=bs)
        confs.append(r['conf']); vals.append(r['valid']); sps.append(r['spars'][1e-4])
    out = (float(np.concatenate(confs).mean()), float(np.concatenate(vals).mean()))
    return out + (float(np.concatenate(sps).mean()),) if with_spars else out


def ref_val_sparsity(ds, split, data, dev, yv, target, pam):
    """Validation sparsity (1 sample, seed 0) of the local-editor model of this (split, target): main protocol (Table 4) or,
    on PAM, the 30-epoch r=0.3 model of Table 11 (tag e7_long)."""
    ck = os.path.join(CK1, ds, 'e7_long' if pam else 'main', f'split={split}_target={target}_learned.pt')
    m = make_cf_model(ds, split, data, dev, mask_mode='learned')
    # copies on the AutoDL machines may lack the two DistributionParams networks (training-only; X_cf does not use them)
    miss, unexp = m.load_state_dict(torch.load(ck, map_location='cpu'), strict=False)
    assert not unexp and all(k.startswith(('src_distribution.', 'mask_src_distribution.')) for k in miss), (miss, unexp)
    m.eval()
    sel = torch.where(yv != target)[0]
    r = eval_cf(m, data['val'][0][:, sel], data['val'][1][:, sel], target)
    del m
    return float(np.mean(r['spars'][1e-4])), int(sel.numel())


def train_cf_budget(model, ds, train, val, target, save_path, max_epochs, patience, budget, verbose=True):
    """common.train_cf (same losses, optimiser, schedulers, patience) with a sparsity-constrained selection:
    only epochs whose validation sparsity is <= budget can be selected (and trigger the early stop); patience counts
    from the first feasible epoch. If no epoch is feasible, the epoch with the lowest validation sparsity is kept."""
    cc = dict(DS[ds]['cf']); n_classes = DS[ds]['n_classes']
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cc['lr'], weight_decay=cc['wd'])
    cosine = cc['scheduler'] == 'cosine'
    sched = None
    if cc['scheduler'] and not cosine:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=5, threshold=1e-5,
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-8)
    init_lrs = [float(pg['lr']) for pg in opt.param_groups]
    clf_crit = Poly1CrossEntropyLoss(num_classes=n_classes, epsilon=1.0, weight=None, reduction='mean')
    Xtr, Ttr, ytr = train
    loader = torch.utils.data.DataLoader(DatasetwInds(Xtr, Ttr, ytr), batch_size=cc['bs'], shuffle=True)
    Xv, Tv, yv = val
    vsel = torch.where(yv != target)[0]
    best, best_ep, since, feasible_seen = -1e9, -1, 0, False
    fb_sp, fb_path = 1e9, save_path[:-3] + '_fallback.pt'
    for ep in range(max_epochs):
        if cosine:
            _apply_warmup_cosine_lr(opt, init_lrs, ep, max_epochs, min(10, max_epochs), 1e-6, 0.0)
        model.train()
        stats = []
        for X, times, y, ids in loader:
            valid = (y != target)
            if valid.sum() == 0:
                continue
            opt.zero_grad()
            ref = RC._sample_reference_batch(X, y, target, Xtr, ytr)
            out = model(X, times, captum_input=True, reference_x=ref, target_label=target)
            tv = torch.full((int(valid.sum()),), int(target), device=y.device, dtype=torch.long)
            clf_loss = clf_crit(out['pred_cf'][valid], tv)
            xp = out['X_prime_cf'].transpose(0, 1)
            (rm, rlv), (cm, clv) = out['all_z']
            kl = kl_divergence_gaussian(cm[valid], clv[valid], rm[valid], rlv[valid])
            inv = 1.0 - out['ste_mask']
            rec = F.mse_loss(xp[valid] * inv[valid], X[valid] * inv[valid])
            l2 = F.mse_loss(xp[valid], X[valid])
            loss = clf_loss + cc['beta_sim'] * (RC.DIST_W * kl + rec + l2) + cc['beta_exp'] * model.compute_loss(out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats.append([clf_loss.item(), out['ste_mask'].float().mean().item()])
        r = eval_cf(model, Xv[:, vsel], Tv[:, vsel], target)
        met, vval, sp = float(np.mean(r['conf'])), float(np.mean(r['valid'])), float(np.mean(r['spars'][1e-4]))
        if verbose:
            st = np.mean(stats, 0) if stats else [np.nan] * 2
            log(f'ep {ep} tgt {target} CE {st[0]:.4f} |M| {st[1]:.3f} | val conf {met:.4f} valid {vval:.4f} spars {sp:.3f} (budget {budget:.3f})')
        if sp < fb_sp:
            fb_sp = sp; torch.save(model.state_dict(), fb_path)
        if sp <= budget:
            feasible_seen = True
            if met > best:
                best, best_ep, since = met, ep, 0
                torch.save(model.state_dict(), save_path)
                if met > 0.99 or vval > 0.99:
                    break
            else:
                since += 1
        elif feasible_seen:
            since += 1
        if sched is not None and ep > 20:
            sched.step(met)
        if patience is not None and feasible_seen and since >= patience:
            break
    if not feasible_seen:
        os.replace(fb_path, save_path)
    elif os.path.exists(fb_path):
        os.remove(fb_path)
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    model.eval()
    return dict(best_val_conf=best if feasible_seen else -1.0, best_epoch=best_ep, epochs_run=ep + 1,
                budget=budget, budget_met=feasible_seen, fallback_val_spars=None if feasible_seen else fb_sp)


def train_cf_cond(model, ds, train, val, save_path, max_epochs, patience, verbose=True, budget=None):
    """common.train_cf with one random target per instance (classes != predicted class) and pair-averaged
    validation confidence as the selection criterion. Losses, optimiser and schedulers unchanged.
    budget: optional sparsity constraint on the selected epoch (as train_cf_budget)."""
    cc = dict(DS[ds]['cf']); C = DS[ds]['n_classes']
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
    best, best_ep, since, feasible_seen = -1e9, -1, 0, False
    fb_sp, fb_path = 1e9, save_path[:-3] + '_fallback.pt'
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
            X, times, y, t = X[ok], times[ok], y[ok], t[ok]
            ref_idx = torch.empty_like(t)
            for c in t.unique().tolist():
                w = torch.where(t == c)[0]
                ref_idx[w] = pools[c][torch.randint(pools[c].numel(), (w.numel(),), device=X.device)]
            ref = Xtr[:, ref_idx].transpose(0, 1)
            opt.zero_grad()
            out = model(X, times, captum_input=True, reference_x=ref, target_label=t)
            clf_loss = clf_crit(out['pred_cf'], t)
            xp = out['X_prime_cf'].transpose(0, 1)
            (rm, rlv), (cm, clv) = out['all_z']
            kl = kl_divergence_gaussian(cm, clv, rm, rlv)
            inv = 1.0 - out['ste_mask']
            rec = F.mse_loss(xp * inv, X * inv)
            l2 = F.mse_loss(xp, X)
            loss = clf_loss + cc['beta_sim'] * (RC.DIST_W * kl + rec + l2) + cc['beta_exp'] * model.compute_loss(out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats.append([clf_loss.item(), kl.item(), rec.item(), l2.item(), out['ste_mask'].float().mean().item()])
        st = np.mean(stats, 0) if stats else [np.nan] * 5
        met, vval, sp = val_pairs(model, Xv, Tv, yv, C, with_spars=True)
        if verbose:
            log(f'ep {ep} CE {st[0]:.4f} KL {st[1]:.4f} Rec {st[2]:.4f} L2 {st[3]:.4f} |M| {st[4]:.3f} | val conf {met:.4f} valid {vval:.4f} spars {sp:.3f}'
                + ('' if budget is None else f' (budget {budget:.3f})'))
        feas = budget is None or sp <= budget
        if budget is not None and sp < fb_sp:
            fb_sp = sp; torch.save(model.state_dict(), fb_path)
        if feas:
            feasible_seen = True
            if met > best:
                best, best_ep, since = met, ep, 0
                torch.save(model.state_dict(), save_path)
                if met > 0.99 or vval > 0.99:
                    break
            else:
                since += 1
        elif feasible_seen:
            since += 1
        if sched is not None and ep > 20:
            sched.step(met)
        if patience is not None and feasible_seen and since >= patience:
            break
    if not feasible_seen:
        os.replace(fb_path, save_path)
    elif budget is not None and os.path.exists(fb_path):
        os.remove(fb_path)
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    model.eval()
    out = dict(best_val_conf=best if feasible_seen else -1.0, best_epoch=best_ep, epochs_run=ep + 1)
    if budget is not None:
        out.update(budget=budget, budget_met=feasible_seen, fallback_val_spars=None if feasible_seen else fb_sp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True)
    ap.add_argument('--split', type=int, required=True)
    ap.add_argument('--variant', required=True, choices=['cond', 'pertarget'])
    ap.add_argument('--targets', default=None)
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--max_epochs', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--nsamples', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--pam', action='store_true', help='Table-11 protocol: first 50 eligible test instances per target, 1 sample')
    ap.add_argument('--tag', default='main')
    ap.add_argument('--budget', default='none', choices=['none', 'main'],
                    help="'main': sparsity-matched selection, validation sparsity <= tol x that of the local-editor model")
    ap.add_argument('--budget_tol', type=float, default=1.1)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    dev = torch.device('cuda')
    ds = a.ds; cfg = DS[ds]; C = cfg['n_classes']
    outdir = os.path.join(RES2, 'C1_temporal', a.tag, ds); ckdir = os.path.join(CK2, 'C1', a.tag, ds)
    os.makedirs(outdir, exist_ok=True); os.makedirs(ckdir, exist_ok=True)
    data = load_split(ds, a.split, dev, main=not a.pam)
    pl = pseudo_labels(ds, a.split, data, dev)
    train = (data['train'][0], data['train'][1], pl['train'])
    val = (data['val'][0], data['val'][1], pl['val'])
    Xte, Tte = data['test'][0], data['test'][1]
    aux = {w: load_aux_bb(ds, a.split, w, Xte.shape[-1], dev) for w in ['cnn', 'lstm'] if cfg.get(w)}
    targets = list(range(C)) if a.targets is None else [int(t) for t in a.targets.split(',')]
    cond = a.variant == 'cond'
    fouts = {t: os.path.join(outdir, f'split={a.split}_target={t}_{a.variant}.json') for t in targets}
    if all(os.path.exists(f) for f in fouts.values()) and not a.force:
        log('all outputs exist'); return

    budget, ref_sp = None, None
    if a.budget == 'main':
        ref = {t: ref_val_sparsity(ds, a.split, data, dev, pl['val'], t, a.pam) for t in (range(C) if cond else targets)}
        ref_sp = {str(t): v[0] for t, v in ref.items()}
        if cond:  # pair-weighted mean over targets, the quantity monitored by val_pairs
            budget = a.budget_tol * sum(v[0] * v[1] for v in ref.values()) / sum(v[1] for v in ref.values())
        log('reference validation sparsity', ref_sp, 'budget', budget)

    def fit(target):
        ck_final = os.path.join(ckdir, f'split={a.split}_{a.variant}' + ('' if cond else f'_target={target}') + '.pt')
        best, val_confs, t0 = None, [], time.time()
        for rs in range(a.restarts):
            set_seed(cfg['cf']['seed'] + 1000 * rs)
            model = make_c1_model(ds, a.split, data, dev, cond, a.hidden, a.layers)
            ck_rs = ck_final[:-3] + f'_rs{rs}.pt'
            if cond:
                inf = train_cf_cond(model, ds, train, val, ck_rs, a.max_epochs, a.patience, budget=budget)
            elif a.budget == 'main':
                model.cur_target = target
                inf = train_cf_budget(model, ds, train, val, target, ck_rs, a.max_epochs, a.patience,
                                      budget=a.budget_tol * ref_sp[str(target)])
            else:
                model.cur_target = target
                inf = train_cf(model, ds, train, val, target, ck_rs, max_epochs=a.max_epochs, patience=a.patience, verbose=True)
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
        return model, info, ck_final

    model, info, ck = (fit(None) if cond else (None, None, None))
    for t in targets:
        if os.path.exists(fouts[t]) and not a.force:
            log('skip existing', fouts[t]); continue
        if not cond:
            model, info, ck = fit(t)
        te_sel = torch.where(pl['test'] != t)[0]
        if a.pam:
            te_sel = te_sel[:50]
        if te_sel.numel() == 0:
            continue
        model.cur_target = t
        t0 = time.time()
        r = eval_cf_k(model, Xte[:, te_sel], Tte[:, te_sel], t, k=1 if a.pam else a.nsamples, aux_bbs=aux)
        s = summarize(r)
        s.update({'variant': a.variant, 'ds': ds, 'split': a.split, 'target': t, 'eval_time_s': time.time() - t0,
                  'nsamples': 1 if a.pam else a.nsamples, 'main': not a.pam, 'max_epochs': a.max_epochs, 'patience': a.patience,
                  'restarts': a.restarts, 'hidden': a.hidden, 'layers': a.layers, 'ckpt': ck, 'budget_mode': a.budget,
                  'budget_tol': a.budget_tol, 'ref_val_spars': ref_sp,
                  'n_params_editor': sum(p.numel() for p in model.editor.parameters()), **info})
        log(ds, a.split, t, a.variant, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in s.items()}))
        json.dump({'summary': s, 'idx': te_sel.cpu().tolist(), 'per_instance': to_jsonable(r)}, open(fouts[t], 'w'))
        if not cond:
            del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
