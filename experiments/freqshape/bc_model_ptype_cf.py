import sys
import os
# Add the parent directory of TimeXplusplus to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../timesynth-0.2.4')))

import torch
import argparse, os
from typing import Tuple
from sklearn import metrics
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
torch.cuda.set_device(2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4_cf
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import *
from txai.utils.predictors.select_models import *
from txai.models.run_model_utils import batch_forwards

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


is_timex = False

DEBUG_LOG_PATH = os.environ.get('TIMEX_DEBUG_LOG', os.devnull)
DEBUG_SESSION_ID = '79f5d2'

def _debug_log(run_id, hypothesis_id, location, message, data):
    payload = {
        'sessionId': DEBUG_SESSION_ID,
        'runId': run_id,
        'hypothesisId': hypothesis_id,
        'location': location,
        'message': message,
        'data': data,
        'timestamp': int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=True) + '\n')
    except Exception:
        pass

if is_timex:
    from txai.models.bc_model4 import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv4_consistency import train_mv6_consistency
else:  # timex++
    from txai.models.bc_model_cf import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv6_consistency_cf import train_mv6_consistency


def naming_convention(args):
    if args.eq_ge:
        name = "bc_eqge_split_cf={}.pt"
    elif args.eq_pret:
        name = "bc_eqpret_split_cf={}.pt"
    elif args.ge_rand_init:
        name = "bc_gerand_split_cf={}.pt"
    elif args.no_ste:
        name = "bc_noste_split_cf={}.pt"
    elif args.simclr:
        name = "bc_simclr_split_cf={}.pt"
    elif args.no_la:
        name = "bc_nola_split_cf={}.pt"
    elif args.no_con:
        name = "bc_nocon_split_cf={}.pt"
    elif args.cnn:
        name = "bc_cnn_split_cf={}.pt"
    elif args.lstm:
        name = "bc_lstm_split_cf={}.pt"
    else:
        name = 'bc_full_split_cf={}.pt'
    
    if not is_timex:
        name = 'our_'+name

    if args.lam != 1.0:
        # Not included in ablation parameters or other, so name it;
        name = name[:-3] + '_lam_cf={}'.format(args.lam) + '.pt'
    
    return name


def _tensor_tbd_to_numpy_samples(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu()
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if x.ndim != 3:
        raise ValueError(f'Expected time series tensor with 2 or 3 dims, got shape={tuple(x.shape)}')
    return x.permute(1, 0, 2).reshape(x.shape[1], -1).numpy().astype(np.float32)


def _extract_cf_tensor(out_dict, src=None):
    if out_dict.get('X_cf') is not None:
        return out_dict['X_cf']
    if out_dict.get('reference_z') is not None:
        return out_dict['reference_z'][1]
    if src is not None and out_dict.get('constraint_src') is not None:
        return src + out_dict['constraint_src']
    return None


def compute_lof_plausibility(reference_x, cf_x, n_neighbors=20, max_ref=1000, seed=42):
    ref_np = _tensor_tbd_to_numpy_samples(reference_x)
    cf_np = _tensor_tbd_to_numpy_samples(cf_x)
    if ref_np is None or cf_np is None or ref_np.shape[0] < 3 or cf_np.shape[0] == 0:
        return float('nan')

    if max_ref is not None and max_ref > 0 and ref_np.shape[0] > max_ref:
        rng = np.random.default_rng(seed)
        ref_idx = rng.choice(ref_np.shape[0], size=max_ref, replace=False)
        ref_np = ref_np[ref_idx]

    n_neighbors = min(int(n_neighbors), max(2, ref_np.shape[0] - 1))
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,
        metric='euclidean',
        n_jobs=-1,
    )
    lof.fit(ref_np)
    cf_pred = lof.predict(cf_np)
    return float((cf_pred == 1).mean())


def main(args):

    model_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
    if args.lstm:
        arch = 'lstm'
        tencoder_path = model_dir_path+"/models/Freqshape_lstm_split={}.pt"
    elif args.cnn:
        arch = 'cnn'
        tencoder_path = model_dir_path+"/models/Freqshape_cnn_split={}.pt"
    else:
        arch = 'transformer'
        tencoder_path = model_dir_path+"/models/Scomb_transformer_split={}.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf_criterion = Poly1CrossEntropyLoss(
        num_classes = 4,
        epsilon = 1.0,
        weight = None,
        reduction = 'mean'
    )

    sim_criterion_label = LabelConsistencyLoss()
    if args.simclr:
        sim_criterion_cons = SimCLRLoss()
        sc_expand_args = {'simclr_training':True, 'num_negatives_simclr':32}
    else:
        sim_criterion_cons = EmbedConsistencyLoss()
        sc_expand_args = {'simclr_training':False, 'num_negatives_simclr':32}

    if args.no_la:
        sim_criterion = sim_criterion_cons
        selection_criterion = simloss_on_val_cononly(sim_criterion)
        label_matching = False
        embedding_matching = True
    elif args.no_con:
        sim_criterion = sim_criterion_label
        selection_criterion = simloss_on_val_laonly(sim_criterion)
        label_matching = True
        embedding_matching = False
    else: # Regular
        sim_criterion = [sim_criterion_cons, sim_criterion_label]
        if args.simclr:
            selection_criterion = simloss_on_val_wboth([cosine_sim_for_simclr, sim_criterion_label], lam = 1.0)
        else:
            selection_criterion = simloss_on_val_wboth(sim_criterion, lam = 1.0)
        label_matching = True
        embedding_matching = True

    targs = transformer_default_args

    cf_metric_keys = [
        "f1_macro",
        "validity",
        "confidence",
        "sparsity",
        "sparsity_count",
        "proximity_l1",
        "proximity_l2",
        "plausibility",
    ]
    all_results = {
        k: {
            'values': [],
            'weights': [],
            'weighted_sum': 0.0,
            'total_weight': 0.0,
        } for k in cf_metric_keys
    }
    n_classes = 4
    test_range =  range(1, 6)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/FreqShape/'))
    for i in test_range:
    # for i in [1]:
        D = process_Synth(split_no = i, device = device, base_path = base_path)

        val, test = D['val'], D['test']
        gt_exps = D['gt_exps']

        # Calc statistics for baseline:
        mu = D['train_loader'].X.mean(dim=1)
        std = D['train_loader'].X.std(unbiased = True, dim = 1)

        # Change transformer args:
        targs['trans_dim_feedforward'] = 16
        targs['trans_dropout'] = 0.1
        targs['nlayers'] = 1
        targs['norm_embedding'] = False

        abl_params = AblationParameters(
            equal_g_gt = args.eq_ge,
            g_pret_equals_g = args.eq_pret, 
            label_based_on_mask = True,
            ptype_assimilation = True, 
            side_assimilation = True,
            use_ste = (not args.no_ste),
            archtype = arch
        )
        loss_weight_dict = {
                    'gsat': 1.0,
                    'connect': 1.0
                }

        # Pseudo-labels from frozen pretrained encoder (not dataset GT), for train/val/test splits.
        probe = TimeXModel(
            d_inp = val[0].shape[-1],
            max_len = val[0].shape[0],
            n_classes = n_classes,
            n_prototypes = 50,
            gsat_r = 0.1,
            transformer_args = targs,
            ablation_parameters = abl_params,
            loss_weight_dict = loss_weight_dict,
            masktoken_stats = (mu, std),
        )
        probe.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
        probe.to(device)
        probe.eval()
        with torch.inference_mode():
            train_out = batch_forwards(probe, D['train_loader'].X, D['train_loader'].times, batch_size = 64)
            train_y_pred = train_out['pred'].argmax(dim = 1).to(device)
            val_out = batch_forwards(probe, val[0], val[1], batch_size = 64)
            val_y_pred = val_out['pred'].argmax(dim = 1).to(device)
            test_out = batch_forwards(probe, test[0], test[1], batch_size = 64)
            test_y_pred = test_out['pred'].argmax(dim = 1).to(device)
        del probe

        if args.train:
            dset = DatasetwInds(D['train_loader'].X.to(device), D['train_loader'].times.to(device), train_y_pred)
            train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

        val_pred_tuple = (val[0], val[1], val_y_pred)
        val_reference_tuple = (val[0], val[1], val[2])

        for target_label in [0,1,2,3]:
            
            my_seed=42
            random.seed(my_seed)
            np.random.seed(my_seed)
            torch.manual_seed(my_seed)
            torch.cuda.manual_seed_all(my_seed)
        # for target_label in range(n_classes):
            # Calculate the target_label accuracy on test
            test_target_mask = (test_y_pred != target_label)
            if test_target_mask.sum() > 0:
                true_target = (D['test'][2].to(device) == target_label)
                acc = (test_y_pred[test_target_mask] == D['test'][2].to(device)[test_target_mask]).float().mean().item()
                print(f"Test accuracy for target_label={target_label}: {acc:.4f} ({int(test_target_mask.sum().item())} samples)")
            else:
                print(f"No test samples predicted as target_label={target_label}")
     
            model = TimeXModel(
                d_inp = val[0].shape[-1],
                max_len = val[0].shape[0],
                n_classes = n_classes,
                n_prototypes = 50,
                gsat_r = 0.1,
                transformer_args = targs,
                ablation_parameters = abl_params,
                loss_weight_dict = loss_weight_dict,
                masktoken_stats = (mu, std),
            )

            model.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
            model.to(device)
            if is_timex:
                model.init_prototypes(train = (D['train_loader'].X.to(device), D['train_loader'].times.to(device), train_y_pred))

                if not args.ge_rand_init: # Copies if not running this ablation
                    model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

            for param in model.encoder_main.parameters():
                param.requires_grad = False

            model_suffix = naming_convention(args)
            model_suffix = model_suffix[:-3] + f'_target={target_label}.pt'
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
            spath = os.path.join(model_path, 'models', model_suffix)
            spath = spath.format(i)
            print(f'split={i}, target={target_label}, checkpoint at {spath}')

            if args.train:
                if args.lstm or args.cnn:
                    optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-3, weight_decay = 0.005)
                else:
                    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.001)

                # #region agent log
                _debug_log(
                    run_id='pre-fix-run',
                    hypothesis_id='H5',
                    location='bc_model_ptype_cf.py:train_start',
                    message='start train target run',
                    data={'split': int(i), 'target_label': int(target_label), 'save_path': spath}
                )
                # #endregion

                train_mv6_consistency(
                    model,
                    optimizer = optimizer,
                    train_loader = train_loader,
                    clf_criterion = clf_criterion,
                    sim_criterion = sim_criterion,
                    beta_exp = 1.0, # 2
                    beta_sim = 0.5,  # 1
                    val_tuple = val_pred_tuple,
                    val_reference_tuple = val_reference_tuple,
                    num_epochs = 100, # 20
                    save_path = spath,
                    train_tuple = (D['train_loader'].X, D['train_loader'].times, train_y_pred),
                    early_stopping = True, # False, #
                    selection_criterion = selection_criterion,
                    label_matching = label_matching,
                    embedding_matching = embedding_matching,
                    use_scheduler = True,
                    target_label = target_label,
                    **sc_expand_args
                )

                # # #endregion
            sdict, config = torch.load(spath, weights_only=False)
            model.load_state_dict(sdict)
            model.eval()

            test_mask = (test_y_pred != target_label)
            if int(test_mask.sum().item()) == 0:
                print(f'split={i}, target={target_label}: no valid test samples (all y_pred==target), skip.')
                continue
            test_filtered = (test[0][:, test_mask, :], test[1][:, test_mask], test_y_pred[test_mask])
            target_labels = torch.full((test_filtered[2].shape[0],), int(target_label), device=test_filtered[2].device, dtype=torch.long)
            test_X, test_T = test[0], test[1]
            reference_mask = (test[2] == target_label)
            reference_X = test_X[:,reference_mask]
            reference_T = test_T[:,reference_mask]
            reference_Y = test[2][reference_mask]
            reference_tuple = (reference_X, reference_T, reference_Y)
            f1, out_dict, cf_metrics = eval_mv4_cf(test_filtered, model, target_labels=target_labels, reference_tuple=reference_tuple)
            cf_tensor = _extract_cf_tensor(out_dict, src=out_dict.get('smooth_src', test_filtered[0]))
            lof_reference_mask = (D['train_loader'].y.to(D['train_loader'].X.device) == target_label)
            lof_reference_X = D['train_loader'].X[:, lof_reference_mask, :]
            lof_plausibility = compute_lof_plausibility(
                lof_reference_X,
                cf_tensor,
                n_neighbors=args.lof_n_neighbors,
                max_ref=args.lof_max_ref,
                seed=42 + i * 10 + target_label,
            )
            cf_metrics['plausibility'] = lof_plausibility
            n_eval = int(test_filtered[2].shape[0])
            print(f'split={i}, target={target_label}, n_eval={n_eval}, test_f1={f1:.4f}, lof_plausibility={lof_plausibility:.4f}, cf_metrics={cf_metrics}')

            # if args.num_vis_cf > 0: 
            #     vis_dir = os.path.join(model_path, 'cf_visualizations', f'split={i}', f'target={target_label}')
            #     visualize_counterfactual_examples(
            #         out_dict = out_dict,
            #         times = test_filtered[1],
            #         y_true = test_filtered[2],
            #         target_label = target_label,
            #         save_dir = vis_dir,
            #         max_examples = args.num_vis_cf,
            #     )

            if args.vis_split is not None and int(i) == int(args.vis_split):
                label_pair_vis_dir = os.path.join(
                    model_path,
                    'cf_label_pair_visualizations',
                    f'split={i}',
                    f'target={target_label}',
                )
                visualize_label_to_target_pairs(
                    out_dict=out_dict,
                    times=test_filtered[1],
                    source_labels=test_filtered[2],
                    target_label=target_label,
                    save_dir=label_pair_vis_dir,
                    n_classes=n_classes,
                    max_per_pair=args.vis_per_pair,
                    change_eps=args.vis_change_eps,
                    seed=args.vis_seed + i * 100 + target_label,
                )

            metric_values = {"f1_macro": f1, **cf_metrics}
            for k in cf_metric_keys:
                v = float(metric_values[k])
                all_results[k]['values'].append(v)
                all_results[k]['weights'].append(float(n_eval))
                all_results[k]['weighted_sum'] += v * float(n_eval)
                all_results[k]['total_weight'] += float(n_eval)

    print('============================================================================================')
    for k in cf_metric_keys:
        rec = all_results[k]
        if rec['total_weight'] <= 0:
            print(f'{k}: no valid samples aggregated.')
            continue
        vals = np.array(rec['values'], dtype=np.float64)
        weights = np.array(rec['weights'], dtype=np.float64)
        wmean = rec['weighted_sum'] / rec['total_weight']
        wvar = np.average((vals - wmean) ** 2, weights=weights)
        wstd = np.sqrt(max(wvar, 0.0))
        eff_n = (weights.sum() ** 2) / (np.square(weights).sum() + 1e-12)
        wsem = wstd / np.sqrt(max(eff_n, 1.0))
        print(f'{k}: weighted_mean = {wmean:.4f} +- {wsem:.4f} (total_n={int(rec["total_weight"])})')


def print_results(mask_labelss, true_labelss):
    mask_labelss = normalize_exp(mask_labelss)

    if torch.is_tensor(mask_labelss):
        mask_labelss = mask_labelss.cpu().numpy()
    if torch.is_tensor(true_labelss):
        true_labelss = true_labelss.cpu().numpy()

    all_aupoc, all_auprc, all_aup, all_aur = [], [], [], []

    for i in range(mask_labelss.shape[1]):
        mask_label = mask_labelss[:, i, :]
        true_label = true_labelss[:, i, :]

        mask_prec, mask_rec, mask_thres = metrics.precision_recall_curve(
            true_label.flatten().astype(int), mask_label.flatten())
        AUROC = metrics.roc_auc_score(true_label.flatten(), mask_label.flatten())
        all_aupoc.append(AUROC)
        AUPRC = metrics.auc(mask_rec, mask_prec)
        all_auprc.append(AUPRC)
        AUP = metrics.auc(mask_thres, mask_prec[:-1])
        all_aup.append(AUP)
        AUR = metrics.auc(mask_thres, mask_rec[:-1])
        all_aur.append(AUR)
    print('Saliency AUROC: = {:.4f} +- {:.4f}'.format(np.mean(all_aupoc), np.std(all_aupoc) / np.sqrt(len(all_aupoc))))
    print('Saliency AUPRC: = {:.4f} +- {:.4f}'.format(np.mean(all_auprc), np.std(all_auprc) / np.sqrt(len(all_auprc))))
    print('Saliency AUP: = {:.4f} +- {:.4f}'.format(np.mean(all_aup), np.std(all_aup) / np.sqrt(len(all_aup))))
    print('Saliency AUR: = {:.4f} +- {:.4f}'.format(np.mean(all_aur), np.std(all_aur) / np.sqrt(len(all_aur))))

    resutlt_cur = {"AUROC": all_aupoc,
                   "AUPRC": all_auprc,
                   "AUP": all_aup,
                   "AUR": all_aur}
    return resutlt_cur

def normalize_exp(exps):
    norm_exps = torch.empty_like(exps)
    for i in range(exps.shape[1]):
        norm_exps[:,i,:] = (exps[:,i,:] - exps[:,i,:].min()) / (exps[:,i,:].max() - exps[:,i,:].min() + 1e-9)
    return norm_exps


def _batch_mask_at_index(mask, batch_idx, time_len):
    if mask is None:
        return None
    if mask.ndim == 1:
        return mask
    if mask.shape[0] == time_len:
        return mask[:, batch_idx]
    return mask[batch_idx]


def _changed_timestep_mask(src_t, cf_t, ste_mask_t, change_eps=1e-4):
    delta = (cf_t - src_t).abs()
    # Match eval_mv4_cf sparsity: only count timestamps whose CF value changed.
    if delta.ndim == 1:
        return delta > change_eps
    return delta.max(dim=-1).values > change_eps


def _select_pair_indices(source_labels, source_label, max_examples, seed, target_label):
    if torch.is_tensor(source_labels):
        src_np = source_labels.detach().cpu().numpy()
    else:
        src_np = np.asarray(source_labels)
    pair_idx = np.flatnonzero(src_np == int(source_label))
    if pair_idx.size == 0:
        return []
    rng = np.random.default_rng(int(seed) + int(target_label) * 1009 + int(source_label) * 9176)
    if pair_idx.size > max_examples:
        chosen = rng.choice(pair_idx, size=max_examples, replace=False)
    else:
        chosen = pair_idx
    return sorted(chosen.tolist())


def _timestep_span_bounds(t_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return left/right x bounds for a vertical band at each timestep."""
    t_np = np.asarray(t_np, dtype=np.float64)
    n = int(t_np.size)
    if n == 0:
        return t_np.copy(), t_np.copy()
    if n == 1:
        half = 0.5
        return np.array([t_np[0] - half]), np.array([t_np[0] + half])
    left = np.empty(n, dtype=np.float64)
    right = np.empty(n, dtype=np.float64)
    left[0] = t_np[0] - (t_np[1] - t_np[0]) * 0.5
    right[-1] = t_np[-1] + (t_np[-1] - t_np[-2]) * 0.5
    mids = (t_np[:-1] + t_np[1:]) * 0.5
    left[1:] = mids
    right[:-1] = mids
    return left, right


def _plot_cf_with_changed_highlight(
    ax,
    t_np: np.ndarray,
    cf_np: np.ndarray,
    changed_np: np.ndarray,
    changed_color: str = "#2d6a4f", #"#7EC8E3",
    base_color: str = "black", #"tab:green",
) -> None:
    """Draw counterfactual output first, then overlay vertical bands on changed timesteps."""
    ax.plot(t_np, cf_np, color=base_color, linewidth=1, label="counterfactual", zorder=2)

    changed_np = np.asarray(changed_np, dtype=bool)
    if not np.any(changed_np):
        return

    left, right = _timestep_span_bounds(t_np)
    labeled_changed = False
    n = len(changed_np)
    i = 0
    while i < n:
        if not changed_np[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and changed_np[i + 1]:
            i += 1
        ax.axvspan(
            left[start],
            right[i],
            color=changed_color,
            alpha=0.5,
            linewidth=0,
            label="changed" if not labeled_changed else None,
            zorder=3,
        )
        labeled_changed = True
        i += 1


def _vis_save_paths(save_path: str) -> Tuple[str, str]:
    base, ext = os.path.splitext(save_path)
    return f"{base}_input{ext}", f"{base}_cf{ext}"


def _save_label_pair_figure(
    t,
    src,
    cf,
    changed,
    source_label,
    target_label,
    cf_pred_label,
    sample_rank,
    batch_idx,
    save_path,
):
    t_np = t.numpy() if torch.is_tensor(t) else np.asarray(t)
    src_np = src.numpy() if torch.is_tensor(src) else np.asarray(src)
    cf_np = cf.numpy() if torch.is_tensor(cf) else np.asarray(cf)
    changed_np = changed.numpy() if torch.is_tensor(changed) else np.asarray(changed)
    changed_idx = np.flatnonzero(changed_np)
    ts_text = ", ".join(f"{t_np[i]:.2f}" for i in changed_idx[:12])
    if changed_idx.size > 12:
        ts_text += ", ..."
    input_path, cf_path = _vis_save_paths(save_path)

    fig_in, ax_in = plt.subplots(1, 1, figsize=(4, 3))
    # ax_in.plot(t_np, src_np, color="tab:blue", linewidth=1, label="original")
    ax_in.plot(t_np, src_np, color="black", linewidth=1, label="original")
    ax_in.set_ylabel("value")
    ax_in.set_xlabel("time")
    ax_in.legend(loc="upper right")
    # ax_in.set_title(
        # f"Original input | source={int(source_label)} | sample={sample_rank} (idx={batch_idx})"
    # )
    fig_in.tight_layout()
    fig_in.savefig(input_path, dpi=160)
    plt.close(fig_in)

    valid = int(cf_pred_label) == int(target_label)
    fig_cf, ax_cf = plt.subplots(1, 1, figsize=(4, 3))
    _plot_cf_with_changed_highlight(ax_cf, t_np, cf_np, changed_np)
    ax_cf.set_ylabel("value")
    ax_cf.set_xlabel("time")
    ax_cf.legend(loc="upper right")
    # ax_cf.set_title(
        # f"Counterfactual explanation | pred={int(cf_pred_label)} "
        # f"(target={int(target_label)}, valid={valid})"
        # + f" | changed n={int(changed_idx.size)}"
        # + (f" | t=[{ts_text}]" if changed_idx.size > 0 else "")
    # )
    fig_cf.tight_layout()
    fig_cf.savefig(cf_path, dpi=160)
    plt.close(fig_cf)


def visualize_label_to_target_pairs(
    out_dict,
    times,
    source_labels,
    target_label,
    save_dir,
    n_classes=4,
    max_per_pair=20,
    change_eps=1e-4,
    seed=42,
):
    """Save up to ``max_per_pair`` figures for each (source_label -> target_label) pair."""
    os.makedirs(save_dir, exist_ok=True)

    src = out_dict.get('smooth_src', None)
    cf = out_dict.get('X_cf', None)
    if cf is None and out_dict.get('reference_z') is not None:
        cf = out_dict['reference_z'][1]
    if cf is None and src is not None and out_dict.get('constraint_src') is not None:
        cf = src + out_dict['constraint_src']
    if src is None or cf is None:
        print(f'[vis] skip label-pair vis for target={target_label}: missing smooth_src/CF tensor')
        return

    src = src.detach().cpu()
    cf = cf.detach().cpu()
    times_cpu = times.detach().cpu() if torch.is_tensor(times) else None
    mask = out_dict.get('ste_mask', None)
    if mask is not None:
        mask = mask.detach().cpu()
    pred_cf = out_dict.get('pred_cf', None)
    if pred_cf is not None:
        pred_cf = pred_cf.detach().cpu().argmax(dim=-1)

    n_saved = 0
    n_skipped_pairs = 0
    for source_label in range(n_classes):
        if int(source_label) == int(target_label):
            continue
        pair_dir = os.path.join(
            save_dir,
            f'source={int(source_label)}_target={int(target_label)}',
        )
        chosen = _select_pair_indices(
            source_labels,
            source_label,
            max_per_pair,
            seed,
            target_label,
        )
        if len(chosen) == 0:
            n_skipped_pairs += 1
            print(
                f'[vis] no samples for source={source_label} -> target={target_label}, skip.'
            )
            continue
        os.makedirs(pair_dir, exist_ok=True)
        for rank, batch_idx in enumerate(chosen):
            t = (
                times_cpu[:, batch_idx]
                if times_cpu is not None
                else torch.arange(src.shape[0], dtype=torch.float32)
            )
            src_t = src[:, batch_idx, 0]
            cf_t = cf[:, batch_idx, 0]
            ste_mask_t = _batch_mask_at_index(mask, batch_idx, src.shape[0])
            changed = _changed_timestep_mask(src_t, cf_t, ste_mask_t, change_eps=change_eps)
            cf_pred_label = int(pred_cf[batch_idx]) if pred_cf is not None else -1
            save_path = os.path.join(pair_dir, f'example_{rank:02d}_idx={batch_idx}.png')
            _save_label_pair_figure(
                t=t,
                src=src_t,
                cf=cf_t,
                changed=changed,
                source_label=source_label,
                target_label=target_label,
                cf_pred_label=cf_pred_label,
                sample_rank=rank,
                batch_idx=batch_idx,
                save_path=save_path,
            )
            n_saved += 1
    print(
        f'[vis] target={target_label}: saved {n_saved} example(s) ({n_saved * 2} files) under {save_dir} '
        f'({n_skipped_pairs} empty source->target pair(s))'
    )


def visualize_counterfactual_examples(out_dict, times, y_true, target_label, save_dir, max_examples = 5):
    os.makedirs(save_dir, exist_ok = True)

    src = out_dict.get('smooth_src', None)
    cf = out_dict.get('X_cf', None)
    if src is None or cf is None:
        print(f'[vis] skip: missing smooth_src/X_cf for target={target_label}')
        return

    src = src.detach().cpu()  # (T, B, d)
    cf = cf.detach().cpu()    # (T, B, d)
    times_cpu = times.detach().cpu() if torch.is_tensor(times) else None
    y_cpu = y_true.detach().cpu() if torch.is_tensor(y_true) else None
    mask = out_dict.get('ste_mask', None)
    if mask is not None:
        mask = mask.detach().cpu()

    num_examples = min(int(src.shape[1]), int(max_examples))
    for b in range(num_examples):
        t = times_cpu[:, b] if times_cpu is not None else torch.arange(src.shape[0], dtype=torch.float32)
        x = src[:, b, 0]
        xcf = cf[:, b, 0]
        delta = xcf - x
        m = None
        if mask is not None:
            m = mask[b] if mask.shape[0] == src.shape[1] else mask[:, b]

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

        axes[0].plot(t.numpy(), x.numpy(), label='original', linewidth=1.4)
        axes[0].plot(t.numpy(), xcf.numpy(), label='counterfactual', linewidth=1.4)
        axes[0].set_ylabel('value')
        axes[0].legend(loc='upper right')
        axes[0].set_title(f'CF vs Original | y={int(y_cpu[b]) if y_cpu is not None else "?"} -> target={int(target_label)} | sample={b}')

        axes[1].plot(t.numpy(), delta.numpy(), color='tab:red', linewidth=1.4)
        axes[1].axhline(0.0, color='gray', linewidth=0.8, linestyle='--')
        axes[1].set_ylabel('cf - orig')

        if m is not None:
            axes[2].plot(t.numpy(), m.numpy(), color='tab:green', linewidth=1.2, label='mask')
            axes[2].set_ylabel('mask')
            axes[2].set_ylim(-0.05, 1.05)
            axes[2].legend(loc='upper right')
        else:
            axes[2].plot(t.numpy(), np.abs(delta.numpy()), color='tab:purple', linewidth=1.2)
            axes[2].set_ylabel('|cf-orig|')

        axes[2].set_xlabel('time')
        fig.tight_layout()

        save_path = os.path.join(save_dir, f'cf_example_{b}.png')
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    print(f'[vis] saved {num_examples} CF figure(s) to {save_dir}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ablations = parser.add_mutually_exclusive_group()
    ablations.add_argument('--eq_ge', action = 'store_true', help = 'G = G_E')
    ablations.add_argument('--eq_pret', action = 'store_true', help = 'G_pret = G')
    ablations.add_argument('--ge_rand_init', action = 'store_true', help = "Randomly initialized G_E, i.e. don't copy")
    ablations.add_argument('--no_ste', action = 'store_true', help = 'Does not use STE')
    ablations.add_argument('--simclr', action = 'store_true', help = 'Uses SimCLR loss instead of consistency loss')
    ablations.add_argument('--no_la', action = 'store_true', help = 'No label alignment - just consistency loss')
    ablations.add_argument('--no_con', action = 'store_true', help = 'No consistency loss - just label')
    ablations.add_argument('--lstm', action = 'store_true') # default=True, 
    ablations.add_argument('--cnn',  action = 'store_true') # default=False, 
    # Note if you don't activate any of them, it just trains the normal method

    parser.add_argument('--r', type = float, default = 0.5, help = 'r for GSAT loss')
    parser.add_argument('--lam', type = float, default = 1.0, help = 'lambda between label alignment and consistency loss')
    parser.add_argument('--num_vis_cf', type = int, default = 0, help = 'number of counterfactual visualization examples per split/target')
    parser.add_argument('--vis_split', type = int, default = 1, help = 'if set, save label-pair CF visualizations for this split only (e.g. 1)')
    parser.add_argument('--vis_per_pair', type = int, default = 20, help = 'max examples per (source_label, target_label) pair when --vis_split is set')
    parser.add_argument('--vis_change_eps', type = float, default = 1e-4, help = 'threshold for marking changed timestamps in label-pair visualizations')
    parser.add_argument('--vis_seed', type = int, default = 42, help = 'RNG seed for sampling label-pair visualization examples')
    parser.add_argument('--train', action = 'store_true', help = 'train target-specific CF models before evaluation; default only loads saved checkpoints')
    parser.add_argument('--lof_n_neighbors', type = int, default = 5, help = 'number of neighbors for LOF plausibility')
    parser.add_argument('--lof_max_ref', type = int, default = 100000, help = 'maximum target-class training samples used to fit each LOF model')

    args = parser.parse_args()

    main(args)

"""

"""

