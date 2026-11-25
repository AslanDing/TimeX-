import math
import torch
from torch.nn import MSELoss
import torch.nn.functional as F
import numpy as np
import os
import json
import time

from txai.utils.predictors.loss_smoother_stats import exp_criterion_eval_smoothers
from txai.utils.predictors.eval import eval_mv4_cf
from txai.utils.cl import in_batch_triplet_sampling
from txai.models.run_model_utils import batch_forwards, batch_forwards_TransformerMVTS
from txai.utils.cl import basic_negative_sampling

from txai.utils.functional import js_divergence

default_scheduler_args = {
    'mode': 'max', 
    'factor': 0.1, 
    'patience': 5,
    'threshold': 0.00001, 
    'threshold_mode': 'rel',
    'cooldown': 0, 
    'min_lr': 1e-8, 
    'eps': 1e-08, 
    'verbose': True
}

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

def kl_divergence_gaussian(mean1, logvar1, mean2, logvar2):
    logvar1 = torch.clamp(logvar1, min=-10.0, max=10.0)
    logvar2 = torch.clamp(logvar2, min=-10.0, max=10.0)
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    kl = 0.5 * (logvar2 - logvar1 - 1 + (var1 + (mean1 - mean2).pow(2)) / (var2 + 1e-6))
    return kl.mean()

# def kl_divergence_gaussian(mean1, logvar1, mean2, logvar2):  
#     kl_div = 0.5 * torch.mean((logvar2 - logvar1 - 1 + (logvar1.exp() + (mean1 - mean2).pow(2)) / logvar2.exp()))  
#     return kl_div  
  
def _sample_reference_batch(X, y, target_label, dataX, dataY):
    # X is batch-first (B, T, d); dataX is train tuple tensor (T, N, d)
    # Reference samples are drawn from the full training pool using reference labels
    # (e.g. ground truth), instead of the batch labels used for target filtering.
    bsz = X.shape[0]
    batch_pool = torch.where(y == target_label)[0]
    if batch_pool.numel() > 0:
        picks = batch_pool[torch.randint(batch_pool.numel(), (bsz,), device=X.device)]
        return X[picks]

    global_pool = torch.where(dataY == target_label)[0]
    if global_pool.numel() == 0:
        global_pool = torch.arange(dataY.shape[0], device=dataY.device)
    picks = global_pool[torch.randint(global_pool.numel(), (bsz,), device=dataY.device)]
    return dataX[:, picks, :].transpose(0, 1).to(X.device)

def _filter_eval_tuple(eval_tuple, target_label):
    X, times, y = eval_tuple
    valid = (y != target_label)
    if valid.sum() == 0:
        return None, 0
    return (X[:, valid, :], times[:, valid], y[valid]), int(valid.sum().item())

def _apply_warmup_cosine_lr(optimizer, initial_lrs, epoch, num_epochs, warmup_eff, lr_warmup_start, eta_min):
    """Set LR per param group: linear warmup from lr_warmup_start to initial lr, then cosine decay to eta_min.
    warmup_eff is the number of warmup epochs (0 = cosine decay over all epochs; already clamped to num_epochs)."""
    for pg, base in zip(optimizer.param_groups, initial_lrs):
        if warmup_eff > 0 and epoch < warmup_eff:
            if warmup_eff <= 1:
                lr = lr_warmup_start + (base - lr_warmup_start) * ((epoch + 1) / max(warmup_eff, 1))
            else:
                lr = lr_warmup_start + (base - lr_warmup_start) * (epoch / (warmup_eff - 1))
        else:
            n_cos = num_epochs - warmup_eff
            if n_cos <= 1:
                lr = base
            else:
                t = epoch - warmup_eff
                cos_scale = 0.5 * (1.0 + math.cos(math.pi * t / (n_cos - 1)))
                lr = eta_min + (base - eta_min) * cos_scale
        pg['lr'] = lr

def train_mv6_consistency(
        model,
        optimizer,
        train_loader,
        val_tuple,
        num_epochs,
        # Criterions:
        clf_criterion,
        sim_criterion,
        beta_exp,
        beta_sim,
        train_tuple,
        lam_label = 1.0,
        clip_norm = True,
        use_scheduler = False,
        wait_for_scheduler = 20,
        scheduler_args = default_scheduler_args,
        selection_criterion = None,
        save_path = None,
        early_stopping = True,
        label_matching = False,
        embedding_matching = True,
        opt_pred_mask = False, # If true, optimizes based on clf_criterion
        opt_pred_mask_to_full_pred = False,
        batch_forward_size = None,
        simclr_training = False,
        num_negatives_simclr = 64,
        max_batch_size_simclr_negs = None,
        bias_weight = 0.1,
        target_label = 0,
        cf_eval_minibatch = True,
        cf_eval_batch_size = 64,
        use_clf_loss = True,
        use_kl_loss = True,
        use_rec_l2_loss = True,
        val_reference_tuple = None,
        use_warmup_cosine_scheduler = False,
        warmup_epochs = 10,
        lr_warmup_start = 1e-6,
        lr_cosine_eta_min = 0.0,
        weight_lc_loss = 1.0
    ):
    '''
    Args:
        selection_criterion: function w signature f(out, val_tuple)

        if both label_matching and embedding_matching are true, then sim_criterion must be a list of length 2 
            with [embedding_sim, label_sim] functions

        use_clf_loss / use_kl_loss / use_rec_l2_loss: set False to ablate the corresponding term in the CF objective
            (see clf_loss, KL term in sim_loss, and rec_loss+l2_loss in train loop).

        use_warmup_cosine_scheduler: if True, LR is linearly warmed from lr_warmup_start to each param group's
            initial lr over warmup_epochs (capped by num_epochs), then cosine-annealed to lr_cosine_eta_min.
            Set warmup_epochs=0 to skip warmup (cosine over all epochs). Mutually exclusive with use_scheduler.

    '''
    # TODO: Add weights and biases logging

    best_epoch = 0
    best_val_metric = -1e9
    best_val_f1 = -1e9

    if use_warmup_cosine_scheduler and use_scheduler:
        raise ValueError("use_warmup_cosine_scheduler and use_scheduler are mutually exclusive.")

    initial_lrs = [float(pg['lr']) for pg in optimizer.param_groups]
    scheduler = None
    if not use_warmup_cosine_scheduler and use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)

    warmup_eff = min(max(warmup_epochs, 0), num_epochs) if use_warmup_cosine_scheduler else 0

    dataX, dataT, dataY = train_tuple # Unpack training variables
    mse_loss_fn = MSELoss()

    for epoch in range(num_epochs):

        if use_warmup_cosine_scheduler:
            _apply_warmup_cosine_lr(
                optimizer,
                initial_lrs,
                epoch,
                num_epochs,
                warmup_eff,
                lr_warmup_start,
                lr_cosine_eta_min,
            )
        
        model.train()
        cum_sparse, cum_exp_loss, cum_clf_loss, cum_sim_loss = [], [], [], []
        cum_kl_loss, cum_rec_loss, cum_l1_loss, cum_l2_loss, cum_valid_ratio = [], [], [], [], []
        cum_cf_acc = []
        label_sim_list, emb_sim_list = [], []
        count = 0
        for X, times, y, ids in train_loader: # Need negative sampling here
            # if count >= 1281:
            #     print('count', count)

            optimizer.zero_grad()

            # if detect_irreg:
            #     src_mask = (X < 1e-7)
            #     out_dict = model(X, times, captum_input = True)

            valid_mask = (y != target_label)
            if valid_mask.sum() == 0:
                continue
            reference_batch = _sample_reference_batch(X, y, target_label, dataX, dataY)
            out_dict = model(X, times, captum_input = True, reference_x = reference_batch, target_label = target_label)
            out = out_dict['pred']
            ste_mask = out_dict['ste_mask']

            # #endregion
            if out.isnan().sum() > 0:
                # Exits if nan's are found
                print('out', out.isnan().sum())
                exit()

            pred_cf = out_dict['pred_cf']
            target_vec = torch.full((int(valid_mask.sum().item()),), int(target_label), device = y.device, dtype = torch.long)
            clf_loss = clf_criterion(pred_cf[valid_mask], target_vec)
            with torch.no_grad():
                pred_cf_vm = pred_cf[valid_mask]
                cf_acc_batch = (pred_cf_vm.argmax(dim=-1) == target_vec).float().mean().item()
     
            if not use_clf_loss:
                clf_loss = clf_loss.detach() * 0.0

            ref_div, cf_div = out_dict['all_z']
            cf_mean, cf_logvar = cf_div
            ref_mean, ref_logvar = ref_div
            kl_loss =  kl_divergence_gaussian(
                cf_mean[valid_mask], cf_logvar[valid_mask], ref_mean[valid_mask], ref_logvar[valid_mask]
            )
            if not use_kl_loss:
                kl_loss = kl_loss.detach() * 0.0

            x_prime_cf = out_dict['X_prime_cf'].transpose(0, 1)
            e_cf = out_dict['E_cf'].transpose(0, 1)

            if ste_mask.ndim == 2:
                ste_mask_rs = ste_mask.unsqueeze(-1)
            else:
                ste_mask_rs = ste_mask
            inv_mask = 1.0 - ste_mask_rs

            rec_loss = F.mse_loss(
                (x_prime_cf[valid_mask] * inv_mask[valid_mask]),
                (X[valid_mask] * inv_mask[valid_mask])
            )
            # rec_loss = F.mse_loss( x_prime_cf[valid_mask] , X[valid_mask] )
            l1_loss = mse_loss_fn(e_cf[valid_mask], X[valid_mask])
            # l2_loss = torch.sqrt((e_cf[valid_mask] ** 2).mean() + 1e-12)
            l2_loss = mse_loss_fn(x_prime_cf[valid_mask], X[valid_mask])
            if not use_rec_l2_loss:
                rec_loss = rec_loss.detach() * 0.0
                l2_loss = l2_loss.detach() * 0.0

            sim_loss = bias_weight * kl_loss + rec_loss + l2_loss # l1_loss + 

            sim_loss = beta_sim * sim_loss
            exp_loss = beta_exp * model.compute_loss(out_dict)
            # conc_emb_loss = conc_embeddings.abs().mean()

            loss = weight_lc_loss*clf_loss + exp_loss + sim_loss# + conc_emb_loss
            # #region agent log
            # #endregion
            # print('---------')
            # print('clf', clf_loss)
            # print('exp', exp_loss)
            # print('sim', sim_loss)
            # print('loss', loss)

            #import ipdb; ipdb.set_trace()

            loss.backward()
            if clip_norm:
                #print('Clip')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            cum_sparse.append(((ste_mask).sum() / ste_mask.flatten().shape[0]).item())
            cum_clf_loss.append(clf_loss.detach().item())
            cum_exp_loss.append(exp_loss.detach().item())
            cum_sim_loss.append(sim_loss.detach().item())
            cum_kl_loss.append(kl_loss.detach().item())
            cum_rec_loss.append(rec_loss.detach().item())
            cum_l1_loss.append(l1_loss.detach().item())
            cum_l2_loss.append(l2_loss.detach().item())
            cum_valid_ratio.append(valid_mask.float().mean().detach().item())
            cum_cf_acc.append(cf_acc_batch)
            count += 1

        if len(cum_clf_loss) == 0:
            print(f'Epoch: {epoch}: no valid samples for target={target_label}, skip optimization stats.')
            # #region agent log
            _debug_log(
                run_id='pre-fix-run',
                hypothesis_id='H2',
                location='train_mv6_consistency_cf.py:epoch_skip',
                message='skip epoch because all batches invalid',
                data={'epoch': int(epoch), 'target_label': int(target_label)}
            )
            # #endregion
            continue

        # Print all stats:
        # Convert to np:
        sparse = np.array(cum_sparse) # Should be size (B, M)
        sparse = sparse.mean()
        clf = sum(cum_clf_loss) / len(cum_clf_loss)
        exp = float(np.mean(cum_exp_loss))
        sim = np.mean(cum_sim_loss)
        kl = np.mean(cum_kl_loss)
        rec = np.mean(cum_rec_loss)
        l1 = np.mean(cum_l1_loss)
        l2 = np.mean(cum_l2_loss)
        valid_ratio = np.mean(cum_valid_ratio)
        cf_acc = np.mean(cum_cf_acc)

        sim_s = f'{sim:.4f}'

        print(
            f'Epoch: {epoch}: target={target_label} valid={valid_ratio:.4f} '
            f'Sparsity={sparse:.4f} Exp={exp:.4f} CE={clf:.4f} CF_Acc={cf_acc:.4f} Sim={sim_s} '
            f'KL={kl:.4f} Rec={rec:.4f} L1={l1:.4f} L2={l2:.4f} rec_loss={rec_loss.detach().item():.4f}'
        )

        # Eval after every epoch (inference only — no autograd graph)
        model.eval()
        with torch.inference_mode():
            val_filtered, n_val = _filter_eval_tuple(val_tuple, target_label)
            ref_source = val_tuple if val_tuple is not None else val_reference_tuple
            val_X, val_T, val_Y = ref_source
            reference_mask = (val_Y == target_label)
            reference_X = val_X[:,reference_mask]
            reference_T = val_T[:,reference_mask]
            reference_Y = val_Y[reference_mask]
            reference_tuple = (reference_X, reference_T, reference_Y)
            if val_filtered is not None and batch_forward_size is None:
                target_labels = torch.full((val_filtered[2].shape[0],), int(target_label), device=val_filtered[2].device, dtype=torch.long)
                eval_batch_size = cf_eval_batch_size if cf_eval_minibatch else None
                f1, out, cf_metrics = eval_mv4_cf(
                    val_filtered,
                    model,
                    target_labels=target_labels,
                    reference_tuple=reference_tuple,
                    batch_size=eval_batch_size,
                )
                # met = -1.0 * loss.detach().item()
                # sparsity =  cf_metrics['sparsity']
                # met =  cf_metrics['validity'] + cf_metrics['confidence'] - sparsity -  abs(cf_metrics['proximity_l1']) - abs(cf_metrics['proximity_l2'])
                met =  cf_metrics['confidence']
                # #endregion
            else:
                out = batch_forwards(model, val_tuple[0], val_tuple[1], batch_size = 64)
                f1 = 0
                met = -1.0 * loss.detach().item()
                cf_metrics = {
                    'validity': float('nan'),
                    'confidence': float('nan'),
                    'proximity_l1': float('nan'),
                    'proximity_l2': float('nan'),
                }
            print(
                f"Val F1 = {f1:.4f}, Val CF metrics = {cf_metrics['validity']:.4f}, "
                f"{cf_metrics['confidence']:.4f},  {cf_metrics['sparsity']:.4f}, {cf_metrics['proximity_l1']:.4f},"
                f"{cf_metrics['proximity_l2']:.4f},Val loss = {loss.detach().item()}"
            )

        ste_mask = out['ste_mask']
        sparse = ste_mask.mean().item()
        cond = not early_stopping
        if early_stopping:
            cond = (met > best_val_metric)
        if cond:
            best_val_metric = met
            best_val_f1 = f1
            if save_path is not None:
                model.save_state(save_path)
            best_epoch = epoch
            print('Save at epoch {}: Metric={:.4f}'.format(epoch, met))
            if cf_metrics['confidence']>0.99 or cf_metrics['validity']>0.99 :
                break

        if scheduler is not None and (epoch > wait_for_scheduler):
            scheduler.step(met)

        if (epoch + 1) % 10 == 0:
            valsparse = '{:.4f}'.format(sparse)
            print(f'Epoch {epoch + 1}, Val F1 = {f1:.4f}, Val Sparsity = {valsparse}')

    print(f'Best Epoch: {best_epoch + 1} \t Val Metric = {best_val_metric:.4f} \t Val F1 = {best_val_f1:.4f}')