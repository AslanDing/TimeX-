"""Shared utilities of the experiments in this folder: dataset configurations, data loading, model builders,
the counterfactual trainer and the evaluators. Outputs go to experiments/unified/{ckpt,cache,results,logs};
nothing in the per-dataset experiment folders is modified.
"""
import sys, os, json, time, math, random
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
for _p in [REPO, os.path.join(REPO, 'timesynth-0.2.4')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F

from txai.synth_data.simple_spike import SpikeTrainDataset  # noqa: F401
import __main__
__main__.SpikeTrainDataset = SpikeTrainDataset  # synthetic splits were pickled from a __main__ script

from txai.utils.data import process_Synth
from txai.utils.data.preprocess import process_MITECG
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.utils.functional import js_divergence
from txai.models.encoders.simple import CNN, LSTM
import txai.models.bc_model as attr_mod
import txai.models.bc_model_cf as cf_mod
from txai.trainers.train_mv6_consistency_cf import kl_divergence_gaussian, _sample_reference_batch, _apply_warmup_cosine_lr

REV = os.path.dirname(os.path.abspath(__file__))
# Distribution term of the objective:
#   'kl'  : learned Gaussian KL surrogate of the conference code (two trainable networks; collapses to ~0)
#   'mmd' : biased MMD^2 with a fixed Gaussian kernel (bandwidth from the training data, median heuristic)
DIST = os.environ.get('TIMEX_DIST', 'kl')
DIST_W = float(os.environ.get('TIMEX_DIST_W', '0.1'))
_SFX = '' if DIST == 'kl' else f'_{DIST}' + ('' if DIST_W == 0.1 else f'_w{DIST_W:g}')
CKPT = os.path.join(REV, 'ckpt' + _SFX)
CACHE = os.path.join(REV, 'cache' + _SFX)
RES = os.path.join(REV, 'results' + _SFX)
RES_BASE = os.path.join(REV, 'results')  # baseline counterfactuals: independent of our distribution term
for _d in [CKPT, CACHE, RES]:
    os.makedirs(_d, exist_ok=True)


def _sqdist(a, b):
    return (a.pow(2).sum(1, keepdim=True) + b.pow(2).sum(1).unsqueeze(0) - 2.0 * a @ b.t()).clamp_min(0.0)


def mmd2_rbf(x, y, sigma):
    """Biased (V-statistic) MMD^2 between two samples with a fixed Gaussian kernel of bandwidth sigma.
    x: (n, ...), y: (m, ...). Equals ||mean_i phi(x_i) - mean_j phi(y_j)||^2 in the RKHS, hence >= 0, and has no
    trainable parameters, so it can only be reduced by moving the samples."""
    x = x.reshape(x.shape[0], -1).float(); y = y.reshape(y.shape[0], -1).float()
    g = 0.5 / (sigma ** 2)
    v = torch.exp(-g * _sqdist(x, x)).mean() + torch.exp(-g * _sqdist(y, y)).mean() - 2.0 * torch.exp(-g * _sqdist(x, y)).mean()
    return v.clamp_min(0.0)


def median_sigma(X, n=1000, seed=0):
    """Median pairwise Euclidean distance of up to n training series (X: (T, N, d)); fixed for the whole training."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[1], generator=g)[:n].to(X.device)
    Z = X[:, idx].transpose(0, 1).reshape(len(idx), -1).float()
    D = _sqdist(Z, Z).sqrt()
    off = ~torch.eye(len(idx), dtype=torch.bool, device=D.device)
    return float(D[off].median())

# ---------------------------------------------------------------------------
# Dataset configurations: copied from the scripts that produced the paper's
# numbers (experiments/<ds>/bc_model_ptype.py for attribution and
# experiments/<ds>/bc_model_ptype_cf.py for counterfactuals).
# ---------------------------------------------------------------------------
DS = {
    'freqshape': dict(
        n_classes=4, kind='synth', data='datasets/FreqShape',
        bb='experiments/freqshape/models/Scomb_transformer_split={}.pt',
        cnn='experiments/freqshape/models/Freqshape_cnn_split={}.pt',
        lstm='experiments/freqshape/models/Freqshape_lstm_split={}.pt',
        targs=dict(trans_dim_feedforward=16, trans_dropout=0.1, nlayers=1, norm_embedding=False),
        attr=dict(r=0.5, gsat=1.0, connect=1.0, beta_exp=2.0, beta_sim=1.0, lr=1e-3, wd=1e-3,
                  epochs=50, scheduler=True, bs=64),
        cf=dict(r=0.1, gsat=1.0, connect=1.0, beta_exp=1.0, beta_sim=0.5, lr=1e-3, wd=1e-3,
                epochs=100, scheduler=True, bs=64, seed=42),
        cf_orig='experiments/freqshape/models/our_bc_full_split_cf={}_target={}.pt',
        subset=None,
    ),
    'seqcomb_mv': dict(
        n_classes=4, kind='synth', data='datasets/SeqCombMV2',
        bb='experiments/seqcomb_mv/models/transformer_split={}.pt',
        cnn='experiments/seqcomb_mv/models/ScombMV_cnn_split={}.pt',
        lstm='experiments/seqcomb_mv/models/ScombMV_lstm_split={}.pt',
        targs=dict(trans_dim_feedforward=128, trans_dropout=0.25, nlayers=2, norm_embedding=False),
        attr=dict(r=0.5, gsat=1.0, connect=2.0, beta_exp=2.0, beta_sim=1.0, lr=2e-3, wd=1e-3,
                  epochs=100, scheduler=False, bs=64),
        cf=dict(r=0.1, gsat=1.0, connect=2.0, beta_exp=2.0, beta_sim=1.0, lr=2e-3, wd=5e-2,
                epochs=500, scheduler=False, bs=64, seed=42),
        cf_orig='experiments/seqcomb_mv/models/our_bc_full_split_cf={}_target={}.pt',
        subset=None,
    ),
    'mitecg': dict(
        n_classes=2, kind='ecg', data='datasets/MITECG/',
        bb='experiments/mitecg_hard/models/transformer_exc_split={}.pt',
        cnn='experiments/mitecg_hard/models/MITECG-Hard_cnn_split={}.pt',
        lstm='experiments/mitecg_hard/models/MITECG-Hard_lstm_split={}.pt',
        targs=dict(trans_dim_feedforward=64, trans_dropout=0.1, nlayers=1, stronger_clf_head=False,
                   norm_embedding=True),
        attr=dict(r=0.5, gsat=1.0, connect=2.0, beta_exp=2.0, beta_sim=1.0, lr=5e-4, wd=1e-4,
                  epochs=5, scheduler=False, bs=64),
        cf=dict(r=0.2, gsat=1.0, connect=0.5, beta_exp=1.0, beta_sim=0.5, lr=1e-3, wd=1e-6,
                epochs=50, scheduler=False, bs=32, seed=42),
        cf_orig='experiments/mitecg_hard/models/our_bc_full_split_cf={}_target={}.pt',
        # ECG is large (60k/20k/20k). Analyses that retrain CF models use fixed
        # stratified subsets; the same subsets are used for every variant.
        subset=dict(train=12000, val=2000, test=4000),
    ),
    # PAM (multivariate, 8 classes): configuration of experiments/PAM/bc_model_ptype_cf.py
    'pam': dict(
        n_classes=8, kind='pam', data='datasets/PAM/',
        bb='experiments/PAM/models/transformer_split={}.pt',
        cnn=None, lstm=None,
        targs=dict(norm_embedding=False),
        # attribution config of experiments/PAM/bc_model_ptype.py (non-TimeX branch)
        attr=dict(r=0.1, gsat=1.0, connect=0.0, beta_exp=2.0, beta_sim=1.0, lr=5e-4, wd=1e-2,
                  epochs=100, scheduler=False, bs=32),
        cf=dict(r=0.3, gsat=1.0, connect=0.0, beta_exp=2.0, beta_sim=1.0, lr=1e-3, wd=1e-5,
                epochs=5, scheduler=True, bs=256, seed=45, tau=5.0),
        cf_orig='experiments/PAM/models/our_bc_full_split_cf={}_target={}.pt',
        subset=None,
    ),
    # Remaining datasets of the main counterfactual table (configs from experiments/<ds>/bc_model_ptype*.py)
    'seqcomb_uv': dict(
        n_classes=4, kind='synth', data='datasets/SeqCombSingle',
        bb='experiments/scs_better/models/Scomb_transformer_split={}.pt', cnn=None, lstm=None,
        targs=dict(trans_dim_feedforward=64, trans_dropout=0.25, nlayers=2, norm_embedding=False),
        attr=dict(r=0.5, gsat=1.0, connect=2.0, beta_exp=2.0, beta_sim=1.0, lr=1e-3, wd=1e-3,
                  epochs=50, scheduler=True, bs=64),
        cf=dict(r=0.3, gsat=2.0, connect=1.0, beta_exp=2.0, beta_sim=1.0, lr=1e-2, wd=1e-4,
                epochs=150, scheduler='cosine', bs=128, seed=42),
        cf_orig='experiments/scs_better/models/our_bc_full_split_cf={}_target={}.pt',
        subset=None,
    ),
    'lowvar': dict(
        n_classes=4, kind='synth', data='datasets/LowVarDetect',
        bb='experiments/lowvardetect/models/transformer_new2_split={}.pt', cnn=None, lstm=None,
        targs=dict(trans_dim_feedforward=32, trans_dropout=0.1, nlayers=1, norm_embedding=True),
        attr=dict(r=0.5, gsat=1.0, connect=2.0, beta_exp=2.0, beta_sim=1.0, lr=5e-3, wd=1e-4,
                  epochs=100, scheduler=False, bs=64),
        cf=dict(r=0.1, gsat=1.0, connect=2.0, beta_exp=1.0, beta_sim=0.5, lr=1e-3, wd=5e-2,
                epochs=150, scheduler='cosine', bs=64, seed=42),
        cf_orig='experiments/lowvardetect/models/our_bc_full_split_cf={}_target={}.pt',
        subset=None,
    ),
    'epilepsy': dict(
        n_classes=2, kind='epilepsy', data='datasets/Epilepsy/',
        bb='experiments/epilepsy/models/transformer_split={}.pt', cnn=None, lstm=None,
        targs=dict(trans_dim_feedforward=16, trans_dropout=0.1, norm_embedding=False),
        attr=dict(r=0.5, gsat=1.0, connect=2.0, beta_exp=2.0, beta_sim=1.0, lr=5e-4, wd=1e-3,
                  epochs=50, scheduler=False, bs=32),
        cf=dict(r=0.1, gsat=1.0, connect=2.0, beta_exp=1.0, beta_sim=0.5, lr=1e-3, wd=1e-4,
                epochs=100, scheduler=True, bs=32, seed=42),
        cf_orig='experiments/epilepsy/models/our_bc_full_split_cf={}_target={}.pt',
        subset=None,
    ),
}

# Optimiser settings of the CF scripts when the black box is a CNN or an LSTM (--cnn / --lstm branches)
CF_ARCH = {
    'freqshape': dict(lr=5e-3, wd=5e-3),
    'seqcomb_mv': dict(lr=1e-4),
    'mitecg': dict(lr=1e-3, epochs=10),
}
# ECG main-table protocol: 12k stratified training subset (full data is ~10 min per epoch on shared GPUs), 2k validation subset, FULL test set
DS['mitecg']['subset_main'] = dict(train=12000, val=2000)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def _subset_idx(y, n, seed):
    g = torch.Generator().manual_seed(seed)
    y = y.cpu()
    if n is None or n >= y.shape[0]:
        return torch.arange(y.shape[0])
    idx = []
    classes = torch.unique(y)
    for c in classes:
        ci = torch.where(y == c)[0]
        nc = int(round(n * ci.shape[0] / y.shape[0]))
        idx.append(ci[torch.randperm(ci.shape[0], generator=g)[:nc]])
    idx = torch.cat(idx)
    return idx[torch.randperm(idx.shape[0], generator=g)]


def load_split(ds, split, device, use_subset=False, main=False):
    """use_subset: analysis subsets (ECG 12k/2k/4k). main: main-table protocol (ECG: full train/test,
    2k validation subset used only for model selection)."""
    cfg = DS[ds]
    if cfg['kind'] == 'epilepsy':
        from txai.utils.data.preprocess import process_Epilepsy
        trE, va, te = process_Epilepsy(split_no=split, device=device, base_path=os.path.join(REPO, cfg['data']))
        tr = (trE.X, trE.time, trE.y)
        va = (va.X, va.time, va.y)
        te = (te.X, te.time, te.y)
        gt = None
    elif cfg['kind'] == 'synth':
        D = process_Synth(split_no=split, device=device, base_path=os.path.join(REPO, cfg['data']))
        tr = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y)
        va, te, gt = D['val'], D['test'], D['gt_exps']
    elif cfg['kind'] == 'pam':
        from txai.utils.data.preprocess import process_PAM
        trP, va, te = process_PAM(split_no=split, device=device, base_path=os.path.join(REPO, cfg['data']), gethalf=True)
        tr = (trP.X, trP.time, trP.y)
        va = (va.X, va.time, va.y)
        te = (te.X, te.time, te.y)
        gt = None
    else:
        trE, va, te, gt = process_MITECG(split_no=split, device=device, hard_split=True, need_binarize=True,
                                         base_path=os.path.join(REPO, cfg['data']))
        tr = (trE.X, trE.time, trE.y)
        va = (va.X, va.time, va.y)
        te = (te.X, te.time, te.y)
    tr = tuple(t.to(device) for t in tr)
    va = tuple(t.to(device) for t in va)
    te = tuple(t.to(device) for t in te)
    gt = gt.to(device) if gt is not None else None
    out = dict(train=tr, val=va, test=te, gt=gt, sub=None)
    if main and cfg.get('subset_main'):
        sub = {}
        for j, (k, n) in enumerate(cfg['subset_main'].items()):
            i = _subset_idx(out[k][2], n, 2000 + split + j).to(device)
            X, T, y = out[k]
            out[k] = (X[:, i], T[:, i], y[i])
            sub[k] = i.tolist()
        out['sub_main'] = sub
        return out
    if use_subset and cfg['subset'] is not None:
        s = cfg['subset']
        idx = {k: _subset_idx(out[k][2], s[k], 1000 + split + j) for j, k in enumerate(['train', 'val', 'test'])}
        for k in ['train', 'val', 'test']:
            i = idx[k].to(device)
            X, T, y = out[k]
            out[k] = (X[:, i], T[:, i], y[i])
        out['gt'] = gt[:, idx['test'].to(device)] if gt is not None else None
        out['sub'] = {k: v.tolist() for k, v in idx.items()}
    return out


def targs_for(ds, mod):
    t = dict(mod.transformer_default_args)
    t.update(DS[ds]['targs'])
    return t


def _abl(mod, arch='transformer', use_ste=True):
    return mod.AblationParameters(equal_g_gt=False, g_pret_equals_g=False, label_based_on_mask=True,
                                  ptype_assimilation=True, side_assimilation=True, use_ste=use_ste,
                                  archtype=arch, cnn_dim=128)


def load_bb_state(ds, split, which='bb'):
    return torch.load(os.path.join(REPO, DS[ds][which].format(split)), map_location='cpu', weights_only=False)


class AttrNoLeak(attr_mod.TimeXModel):
    """TimeX_a++ whose generator only sees the masked reference X*M + (1-M)*b (plus M), so that
    unselected coordinates cannot pass information about X to the generated instance."""

    def multivariate_mask(self, src, ste_mask):
        baseline = self._get_baseline(B=src.shape[1])
        m = ste_mask.transpose(0, 1)
        if m.ndim == 2:
            m = m.unsqueeze(-1)
        ref = src * m + (1 - m) * baseline
        return self.mask_connection_src(torch.stack([ref, m], dim=-1)).squeeze(-1), ref


def make_attr_model(ds, split, data, device, noleak=False, arch='transformer', use_ste=True, r=None):
    cfg = DS[ds]; ac = cfg['attr']
    X = data['train'][0]
    mu = X.mean(dim=1); std = X.std(unbiased=True, dim=1)
    cls = AttrNoLeak if noleak else attr_mod.TimeXModel
    m = cls(d_inp=X.shape[-1], max_len=X.shape[0], n_classes=cfg['n_classes'], n_prototypes=50,
                            gsat_r=ac['r'] if r is None else r, transformer_args=targs_for(ds, attr_mod),
                            ablation_parameters=_abl(attr_mod, arch=arch, use_ste=use_ste),
                            loss_weight_dict={'gsat': ac['gsat'], 'connect': ac['connect']},
                            masktoken_stats=(mu, std), tau=1.0)
    m.encoder_main.load_state_dict(load_bb_state(ds, split, 'bb' if arch == 'transformer' else arch))
    for p in m.encoder_main.parameters():
        p.requires_grad = False
    return m.to(device)


def train_attr(model, optimizer, train_loader, val_tuple, num_epochs, beta_exp, beta_sim, train_tuple, save_path,
               use_scheduler=False, batch_forward_size=None, bias_weight=None, abl=(), verbose=True):
    """Port of txai/trainers/train_mv6_consistency.train_mv6_consistency (attribution) with the same losses,
    selection rule and scheduler, and a switchable distribution term (DIST): 'kl' reproduces the conference
    code exactly; 'mmd' replaces the learned Gaussian KL surrogate by mmd2_rbf(X_tilde_a, X) with a fixed kernel.
    abl: subset of {'no_lc', 'no_bound', 'no_kl'} (loss terms removed for the ablation study)."""
    from txai.utils.predictors.eval import eval_mv4
    from txai.models.run_model_utils import batch_forwards
    from txai.trainers.train_mv6_consistency import kl_divergence_gaussian as kl_attr, default_scheduler_args
    w = (DIST_W if bias_weight is None else bias_weight) if 'no_kl' not in abl else 0.0
    sigma = median_sigma(train_tuple[0]) if DIST == 'mmd' else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **{k: v for k, v in default_scheduler_args.items() if k != 'verbose'}) if use_scheduler else None
    best_val_metric, best_epoch = -1e9, 0
    for epoch in range(num_epochs):
        model.train()
        cum = []
        for X, times, y, ids in train_loader:
            optimizer.zero_grad()
            out = model(X, times, captum_input=True)
            clf_loss = js_divergence(out['pred_mask'].softmax(dim=-1), out['pred'].softmax(dim=-1))
            reference_src, exp_src = out['reference_z']
            if DIST == 'mmd':
                dist = mmd2_rbf(exp_src.transpose(0, 1), out['smooth_src'].transpose(0, 1), sigma)
            else:
                org_div, emb_div = out['all_z']
                dist = kl_attr(emb_div[0], emb_div[1], org_div[0].mean(0, keepdim=True), org_div[1].mean(0, keepdim=True))
            bound = F.mse_loss(reference_src, exp_src)
            sim_loss = beta_sim * (w * dist + (0.0 if 'no_bound' in abl else 1.0) * bound)
            exp_loss = beta_exp * model.compute_loss(out)
            loss = (0.0 if 'no_lc' in abl else 1.0) * clf_loss + exp_loss + sim_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # (called before backward, as in the original)
            loss.backward()
            optimizer.step()
            cum.append([clf_loss.item(), float(dist), bound.item(), out['ste_mask'].float().mean().item()])
        st = np.mean(cum, 0)
        model.eval()
        if batch_forward_size is None:
            eval_mv4(val_tuple, model)
        else:
            batch_forwards(model, val_tuple[0], val_tuple[1], batch_size=64)
        met = -1.0 * float(loss)  # original selection rule: negative loss of the last training batch
        if met > best_val_metric:
            best_val_metric, best_epoch = met, epoch
            model.save_state(save_path)
        if scheduler is not None and epoch > 20:
            scheduler.step(met)
        if verbose:
            log(f'attr ep {epoch} JS {st[0]:.4f} {DIST.upper()} {st[1]:.5f} bound {st[2]:.4f} |M| {st[3]:.3f} sel {met:.4f}')
    return dict(best_epoch=best_epoch, sigma=sigma)


class CFVariant(cf_mod.TimeXModel):
    """TimeX_cf++ with a configurable mask source.

    mask_mode:
      'learned' : original method (mask from g_phi, Gumbel-STE)
      'fixed'   : mask supplied externally (ext_mask), g_phi unused
      'nested'  : learned mask multiplied by an external support (ext_mask)
    use_noise : if False, the auxiliary variation term epsilon = psi_n(.) is disabled
    det_eval  : if True, at eval time use the deterministic mask 1[pi > 0.5] instead of a Gumbel sample
    """

    def __init__(self, *a, mask_mode='learned', use_noise=True, **k):
        super().__init__(*a, **k)
        self.mask_mode = mask_mode
        self.use_noise = use_noise
        self.det_eval = False

    def forward(self, src, times, captum_input=False, reference_x=None, target_label=None, ext_mask=None):
        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)
            if reference_x is not None:
                reference_x = reference_x.transpose(0, 1)
        T, B, d = src.shape
        mask_in = None
        if self.mask_mode in ('learned', 'nested'):
            z_seq = self.encoder_pret.embed(src, times, captum_input=False, aggregate=False)
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
        e_raw = self.perturbation_generator(torch.stack([src, ste_rs], dim=-1)).squeeze(-1)
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
                'all_z': (ref_distr, cf_distr), 'X_prime_cf': x_prime, 'E_cf': e_cf, 'X_cf': src + e_cf,
                'noise': noise}


def make_cf_model(ds, split, data, device, mask_mode='learned', use_noise=True, r=None, arch='transformer',
                  use_ste=True, connect=None):
    cfg = DS[ds]; cc = cfg['cf']
    X = data['train'][0]
    mu = X.mean(dim=1); std = X.std(unbiased=True, dim=1)
    m = CFVariant(d_inp=X.shape[-1], max_len=X.shape[0], n_classes=cfg['n_classes'], n_prototypes=50,
                  gsat_r=cc['r'] if r is None else r, transformer_args=targs_for(ds, cf_mod),
                  ablation_parameters=_abl(cf_mod, arch=arch, use_ste=use_ste),
                  loss_weight_dict={'gsat': cc['gsat'], 'connect': cc['connect'] if connect is None else connect},
                  masktoken_stats=(mu, std), tau=cc.get('tau', 1.0), mask_mode=mask_mode, use_noise=use_noise)
    m.encoder_main.load_state_dict(load_bb_state(ds, split, 'bb' if arch == 'transformer' else arch))
    for p in m.encoder_main.parameters():
        p.requires_grad = False
    return m.to(device)


def load_orig_cf(ds, split, target, data, device):
    if DIST != 'kl':
        # corrected distribution term: the paper checkpoints were trained with the old surrogate, so the reference
        # model of our method is the one retrained under the main protocol (PAM: tag 'e7')
        for tag in ('main', 'e7'):
            p = os.path.join(CKPT, ds, tag, f'split={split}_target={target}_learned.pt')
            if os.path.exists(p):
                m = make_cf_model(ds, split, data, device, mask_mode='learned')
                m.load_state_dict(torch.load(p, map_location='cpu'))
                m.eval()
                return m
        raise FileNotFoundError(f'no retrained CF model for {ds} split={split} target={target} under {CKPT}')
    m = make_cf_model(ds, split, data, device, mask_mode='learned')
    sd, _ = torch.load(os.path.join(REPO, DS[ds]['cf_orig'].format(split, target)), map_location='cpu', weights_only=False)
    m.load_state_dict(sd)
    m.eval()
    return m


def load_aux_bb(ds, split, which, d_inp, device):
    sd = load_bb_state(ds, split, which)
    n_classes = DS[ds]['n_classes']
    for dim in [128, 64, 32, 256, 16]:
        m = CNN(d_inp=d_inp, n_classes=n_classes, dim=dim) if which == 'cnn' else LSTM(d_inp=d_inp, n_classes=n_classes, dim=dim)
        try:
            m.load_state_dict(sd)
            return m.to(device).eval()
        except Exception:
            continue
    raise RuntimeError(f'cannot load {which} for {ds} split {split}')


@torch.no_grad()
def bb_logits(enc, X, T, bs=512):
    was = enc.training
    enc.eval()
    out = [enc(X[:, i:i + bs], T[:, i:i + bs], captum_input=False) for i in range(0, X.shape[1], bs)]
    enc.train(was)
    return torch.cat(out, 0)


@torch.no_grad()
def attr_pi(model, X, T, bs=256):
    """Attribution soft mask pi, returned as (N, T, d)."""
    model.eval()
    out = []
    for i in range(0, X.shape[1], bs):
        o = model.get_saliency_explanation(X[:, i:i + bs], T[:, i:i + bs])
        out.append(o['mask_in'].transpose(0, 1))  # mask_in is (T,B,d) -> (B,T,d)
    return torch.cat(out, 0)


@torch.no_grad()
def cf_pi(model, X, T, bs=256):
    model.eval()
    out = []
    for i in range(0, X.shape[1], bs):
        Xb, Tb = X[:, i:i + bs], T[:, i:i + bs]
        z_seq = model.encoder_pret.embed(Xb, Tb, captum_input=False, aggregate=False)
        mask_in, _ = model.mask_generator(Xb, Tb, z_seq=z_seq, training=False)
        out.append(mask_in)
    return torch.cat(out, 0)  # (N,T,d)


# ---------------------------------------------------------------------------
# Counterfactual training (same losses as txai/trainers/train_mv6_consistency_cf.py)
# ---------------------------------------------------------------------------

def train_cf(model, ds, train, val, target, save_path, mask_fn=None, max_epochs=None, patience=None,
             eval_bs=256, verbose=True, hp=None, abl=()):
    """train/val: (X, T, y_pred). mask_fn(split_name, idx_tensor) -> (B,T,d) float mask or None.
    hp: overrides of the dataset's cf config (e.g. lr/wd/epochs for CNN/LSTM black boxes).
    abl: subset of {'no_lc', 'no_bound', 'no_kl'} -- loss terms removed for the ablation study."""
    cc = dict(DS[ds]['cf'])
    if hp:
        cc.update(hp)
    n_classes = DS[ds]['n_classes']
    epochs = cc['epochs'] if max_epochs is None else max_epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cc['lr'], weight_decay=cc['wd'])
    sched = None
    cosine = cc['scheduler'] == 'cosine'
    if cc['scheduler'] and not cosine:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=5, threshold=1e-5,
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-8)
    init_lrs = [float(pg['lr']) for pg in opt.param_groups]
    clf_crit = Poly1CrossEntropyLoss(num_classes=n_classes, epsilon=1.0, weight=None, reduction='mean')
    Xtr, Ttr, ytr = train
    loader = torch.utils.data.DataLoader(DatasetwInds(Xtr, Ttr, ytr), batch_size=cc['bs'], shuffle=True)
    Xv, Tv, yv = val
    vsel = torch.where(yv != target)[0]
    sigma = median_sigma(Xtr) if DIST == 'mmd' else None
    best, best_ep, since = -1e9, -1, 0
    for ep in range(epochs):
        if cosine:  # linear warm-up (10 epochs) then cosine decay, as in the original scripts
            _apply_warmup_cosine_lr(opt, init_lrs, ep, epochs, min(10, epochs), 1e-6, 0.0)
        model.train()
        stats = []
        for X, times, y, ids in loader:
            valid = (y != target)
            if valid.sum() == 0:
                continue
            opt.zero_grad()
            ref = _sample_reference_batch(X, y, target, Xtr, ytr)
            ext = mask_fn('train', ids) if mask_fn is not None else None
            out = model(X, times, captum_input=True, reference_x=ref, target_label=target, ext_mask=ext)
            tv = torch.full((int(valid.sum()),), int(target), device=y.device, dtype=torch.long)
            clf_loss = clf_crit(out['pred_cf'][valid], tv)
            xp = out['X_prime_cf'].transpose(0, 1)
            if DIST == 'mmd':  # counterfactuals vs. target-class reference instances, fixed kernel
                kl = mmd2_rbf(xp[valid], ref[valid], sigma)
            else:
                (rm, rlv), (cm, clv) = out['all_z']
                kl = kl_divergence_gaussian(cm[valid], clv[valid], rm[valid], rlv[valid])
            inv = 1.0 - out['ste_mask']
            rec = F.mse_loss(xp[valid] * inv[valid], X[valid] * inv[valid])
            l2 = F.mse_loss(xp[valid], X[valid])
            sim = cc['beta_sim'] * ((0.0 if 'no_kl' in abl else DIST_W) * kl + (0.0 if 'no_bound' in abl else 1.0) * (rec + l2))
            loss = (0.0 if 'no_lc' in abl else 1.0) * clf_loss + sim
            if out['mask_logits'] is not None:
                loss = loss + cc['beta_exp'] * model.compute_loss(out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats.append([clf_loss.item(), kl.item(), rec.item(), l2.item(), out['ste_mask'].float().mean().item()])
        st = np.mean(stats, 0) if stats else [np.nan] * 5
        ext_v = mask_fn('val', vsel) if mask_fn is not None else None
        r = eval_cf(model, Xv[:, vsel], Tv[:, vsel], target, ext_mask=ext_v, bs=eval_bs)
        met = float(np.mean(r['conf']))
        vval = float(np.mean(r['valid']))
        if verbose:
            log(f'ep {ep} tgt {target} CE {st[0]:.4f} {"MMD" if DIST == "mmd" else "KL"} {st[1]:.4f} Rec {st[2]:.4f} L2 {st[3]:.4f} '
                f'|M| {st[4]:.3f} | val conf {met:.4f} valid {vval:.4f} spars {np.mean(r["spars"][1e-4]):.3f}')
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


TOLS = (1e-4, 1e-3, 1e-2)


@torch.no_grad()
def eval_cf(model, X, T, target, ext_mask=None, bs=256, aux_bbs=None, gt=None, seed=0):
    """Per-instance CF metrics (same definitions as eval_mv4_cf), plus extras.

    Returns dict of numpy arrays: valid, conf, prox_l1, prox_l2, mask_size, spars[tol],
    and optionally valid_<aux> for transfer validity, gt_frac_in / gt_cover for GT overlap.
    """
    model.eval()
    g = torch.random.get_rng_state(); gc = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    res = {k: [] for k in ['valid', 'conf', 'prox_l1', 'prox_l2', 'mask_size', 'n_changed']}
    res['spars'] = {t: [] for t in TOLS}
    if aux_bbs:
        for k in aux_bbs:
            res['valid_' + k] = []
    if gt is not None:
        res['gt_frac_in'] = []; res['gt_cover'] = []
    for i in range(0, X.shape[1], bs):
        Xb, Tb = X[:, i:i + bs], T[:, i:i + bs]
        eb = ext_mask[i:i + bs] if ext_mask is not None else None
        out = model(Xb, Tb, captum_input=False, ext_mask=eb)
        cf = out['X_cf']
        logits = model.encoder_main(cf, Tb, captum_input=False)
        p = logits.softmax(-1)
        res['valid'].append((logits.argmax(-1) == target).float())
        res['conf'].append(p[:, target])
        delta = (cf - Xb).abs()
        for t in TOLS:
            res['spars'][t].append((delta > t).float().mean(dim=(0, 2)))
        changed = (delta > 1e-4).float()
        res['n_changed'].append(changed.sum(dim=(0, 2)))
        res['prox_l1'].append(delta.mean(dim=(0, 2)))
        res['prox_l2'].append(torch.sqrt((delta ** 2).mean(dim=(0, 2)) + 1e-12))
        res['mask_size'].append(out['ste_mask'].float().mean(dim=(1, 2)))
        if aux_bbs:
            for k, enc in aux_bbs.items():
                res['valid_' + k].append((enc(cf, Tb, captum_input=False).argmax(-1) == target).float())
        if gt is not None:
            gb = gt[:, i:i + bs].float()
            inter = (changed * gb).sum(dim=(0, 2))
            res['gt_frac_in'].append(inter / changed.sum(dim=(0, 2)).clamp(min=1))
            res['gt_cover'].append(inter / gb.sum(dim=(0, 2)).clamp(min=1))
    torch.random.set_rng_state(g)
    if gc is not None:
        torch.cuda.set_rng_state_all(gc)
    out = {}
    for k, v in res.items():
        if k == 'spars':
            out[k] = {t: torch.cat(vv).cpu().numpy() for t, vv in v.items()}
        else:
            out[k] = torch.cat(v).cpu().numpy()
    return out


def eval_cf_k(model, X, T, target, k=1, **kw):
    """Average the per-instance metrics over k independent samples of the stochastic mask (seeds 0..k-1).
    With k>1, 'valid' becomes the fraction of samples that are valid."""
    if k <= 1:
        return eval_cf(model, X, T, target, **kw)
    rs = [eval_cf(model, X, T, target, seed=s, **kw) for s in range(k)]
    out = {}
    for key, v in rs[0].items():
        if isinstance(v, dict):
            out[key] = {t: np.mean([r[key][t] for r in rs], 0) for t in v}
        else:
            out[key] = np.mean([r[key] for r in rs], 0)
    return out


def summarize(r):
    """Aggregate per-instance results into scalar metrics (success-only proximity included)."""
    v = np.asarray(r['valid']) >= 0.5
    s = {
        'n': int(v.shape[0]),
        'validity': float(r['valid'].mean()),
        'confidence': float(r['conf'].mean()),
        'sparsity': float(r['spars'][1e-4].mean()),
        'sparsity_1e-3': float(r['spars'][1e-3].mean()),
        'sparsity_1e-2': float(r['spars'][1e-2].mean()),
        'prox_l1': float(r['prox_l1'].mean()),
        'prox_l2': float(r['prox_l2'].mean()),
        'prox_l1_valid': float(r['prox_l1'][v].mean()) if v.any() else float('nan'),
        'prox_l2_valid': float(r['prox_l2'][v].mean()) if v.any() else float('nan'),
        'mask_size': float(r['mask_size'].mean()),
    }
    for k in list(r.keys()):
        if k.startswith('valid_'):
            s[k] = float(r[k].mean())
            # transfer validity among instances that are valid on the source black box
            s[k + '_given_valid'] = float(r[k][v].mean()) if v.any() else float('nan')
        if k in ('gt_frac_in', 'gt_cover'):
            s[k] = float(r[k].mean())
    return s


def to_jsonable(r):
    out = {}
    for k, v in r.items():
        if isinstance(v, dict):
            out[k] = {str(kk): np.asarray(vv).round(5).tolist() for kk, vv in v.items()}
        else:
            out[k] = np.asarray(v).round(5).tolist()
    return out


def pseudo_labels(ds, split, data, device):
    enc = make_cf_model(ds, split, data, device).encoder_main
    return {k: bb_logits(enc, data[k][0], data[k][1]).argmax(-1) for k in ['train', 'val', 'test']}


# ---------------------------------------------------------------------------
# Paths and helpers of the temporal-editor scripts (train_cf.py, train_cf_variants.py, ...).
# They write to the same folder: ckpt/C1, ckpt/D2, results/C1_temporal, results/D2_temporal, ...
# CK1 / CA1 / RES1 are the folders of the attribution and local-editor runs (with the TIMEX_DIST suffix).
# ---------------------------------------------------------------------------
RC = sys.modules[__name__]   # `from common import *` also exports this module as RC (used as RC.DS, RC.CFVariant, ...)
R2 = REV
CK2 = os.path.join(R2, 'ckpt')
CA2 = os.path.join(R2, 'cache')
BIG2 = os.path.join(R2, 'big')
RES2 = os.path.join(R2, 'results')
for _d in (CK2, CA2, BIG2, RES2):
    os.makedirs(_d, exist_ok=True)
CK1, CA1, RES1 = CKPT, CACHE, RES
PY = os.environ.get('PY', sys.executable)

# Baseline attribution cache (optional):
#   cache/attr_baselines/<ds>/<method>_split=<s>.pt = {part: {'idx': LongTensor, 'attr': HalfTensor (n,T,d)}}
#   idx indexes the FULL arrays of load_split(ds, split, dev) (use_subset=False, main=False) for that part.
ATTR_BL = os.path.join(CA2, 'attr_baselines')


def load_attr_baseline(ds, method, split, part='test'):
    p = os.path.join(ATTR_BL, ds, f'{method}_split={split}.pt')
    if not os.path.exists(p):
        return None
    d = torch.load(p, map_location='cpu')
    return d.get(part)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)
