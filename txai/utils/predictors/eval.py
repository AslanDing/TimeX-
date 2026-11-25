import os
import torch
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from txai.models.run_model_utils import batch_forwards_TransformerMVTS
from txai.models.encoders.simple import CNN, LSTM
from txai.utils.evaluation import ground_truth_xai_eval, ground_truth_IoU

@torch.no_grad()
def eval_on_tuple(test_tuple, model, n_classes, mask = None):
    '''
    Returns f1 score
    '''

    model.eval()

    X, times, y = test_tuple

    pred, mask = model(X, times, mask = mask)
    print(pred.shape)

    f1 = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro')

    return f1, mask

@torch.no_grad()
def eval_cbmv1(test_tuple, model):
    model.eval()
    X, times, y = test_tuple
    pred, concept_scores, masks, logits = model(X, times, captum_input = False)

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, (pred, concept_scores, masks, logits)

@torch.no_grad()
def eval_filter(test_tuple, model):
    model.eval()
    X, times, y = test_tuple
    pred, _, masks, logits = model(X, times, captum_input = False)
    # print('masks', masks.shape)
    # print('mask', masks)

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, (pred, masks, logits)

@torch.no_grad()
def eval_mv2(test_tuple, model):
    model.eval()
    X, times, y = test_tuple
    pred, mask_in, ste_mask, smoother_stats, smooth_src = model(X, times, captum_input = False)

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, (pred, mask_in, ste_mask, smoother_stats, smooth_src)

@torch.no_grad()
def eval_mv3(test_tuple, model):
    model.eval()
    X, times, y = test_tuple
    pred, pred_tilde, mask_in, ste_mask, smoother_stats, smooth_src = model(X, times, captum_input = False)

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, (pred, pred_tilde, mask_in, ste_mask, smoother_stats, smooth_src)

@torch.no_grad()
def eval_mv3_sim(test_tuple, model):
    model.eval()
    X, times, y = test_tuple
    pred, pred_tilde, mask_in, ste_mask, smoother_stats, smooth_src, zs = model(X, times, captum_input = False)

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, (pred, pred_tilde, mask_in, ste_mask, smoother_stats, smooth_src, zs)

@torch.inference_mode()
def eval_mv4_cfa(
    test_tuple,
    model,
    masked = True,
    reference_tuple = None,
    target_labels = None,
    change_eps = 1e-4,
    plausibility_z = 3.0,
    untargeted = False,
):
    # Evaluation for the actual output schema of bc_model_0421_cf.TimeXModel:
    # {
    #   'pred', 'pred_mask', 'mask_logits', 'ste_mask', 'smooth_src',
    #   'all_z', 'reference_z'=(reference_src, exp_src), ...
    # }
    def _to_tbd(x, batch_size):
        # Convert to (T, B, d).
        if x is None:
            return None
        if x.ndim == 2:
            return x.unsqueeze(-1)
        if x.ndim != 3:
            return x
        if x.shape[0] == batch_size and x.shape[1] != batch_size:
            return x.transpose(0, 1)
        return x

    def _get_reference_stats(ref_x, src_x, batch_size):
        x_ref = src_x if ref_x is None else ref_x
        x_ref = _to_tbd(x_ref, batch_size)
        mu = x_ref.mean(dim=1, keepdim=True)
        std = x_ref.std(dim=1, keepdim=True, unbiased=False)
        std = torch.clamp(std, min=1e-6)
        return mu, std

    model.eval()
    X, times, y = test_tuple
    out = model(X, times, captum_input=False)

    if masked:
        pred = out['pred_mask']
    else:
        pred = out['pred']
    # pred_cf = out['pred']
    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1aa = f1_score(ynp, prednp, average='macro')



    if 'pred' not in out or 'pred_mask' not in out or 'reference_z' not in out:
        raise KeyError("eval_mv4_cfa expects out keys: 'pred', 'pred_mask', 'reference_z'.")

    pred_orig = out['pred']
    pred_cf = out['pred_cf']

    batch_size = y.shape[0]
    src = _to_tbd(out.get('smooth_src', X), batch_size)
    reference_z = out['reference_z']
    if (not isinstance(reference_z, (tuple, list))) or len(reference_z) < 2:
        raise ValueError("out['reference_z'] must be a tuple/list: (reference_src, exp_src).")
    cf = _to_tbd(reference_z[-1], batch_size)  # exp_src from bc_model_0421_cf

    orig_cls = pred_orig.argmax(dim=-1)
    cf_cls = pred_cf.argmax(dim=-1)

    use_untargeted_metrics = untargeted or (target_labels is None)
    if use_untargeted_metrics:
        f1 = f1_score(orig_cls.detach().cpu().numpy(), cf_cls.detach().cpu().numpy(), average='macro')
    else:
        tgt = target_labels.to(cf_cls.device).long().view(-1)
        if tgt.shape[0] != cf_cls.shape[0]:
            raise ValueError(f"target_labels size mismatch: got {tgt.shape[0]}, expected {cf_cls.shape[0]}.")
        f1 = f1_score(tgt.detach().cpu().numpy(), cf_cls.detach().cpu().numpy(), average='macro')

    cf_prob = pred_cf.softmax(dim=-1)
    if use_untargeted_metrics:
        validity_vec = (cf_cls != orig_cls).float()
        alt_conf = cf_prob.clone()
        idx = torch.arange(cf_prob.shape[0], device=cf_prob.device, dtype=torch.long)
        alt_conf[idx, orig_cls.long()] = -1e9
        confidence_vec = alt_conf.max(dim=1).values
    else:
        tgt = target_labels.to(cf_cls.device).long().view(-1)
        validity_vec = (cf_cls == tgt).float()
        confidence_vec = cf_prob.gather(1, tgt.view(-1, 1)).squeeze(1)

    cf_metrics = {
        'validity': validity_vec.mean().item(),
        'confidence': confidence_vec.mean().item(),
        'sparsity': 0.0,
        'sparsity_count': 0.0,
        'proximity_l1': 0.0,
        'proximity_l2': 0.0,
        'plausibility': 0.0,
    }

    if (src is not None) and (cf is not None):
        delta = (cf - src).abs()
        changed = (delta > change_eps).float()
        sparsity_ratio = changed.mean(dim=(0, 2))
        sparsity_count = changed.sum(dim=(0, 2))
        proximity_l1 = delta.mean(dim=(0, 2))
        proximity_l2 = torch.sqrt((delta ** 2).mean(dim=(0, 2)) + 1e-12)

        ref_x = reference_tuple[0] if (reference_tuple is not None) else X
        mu_ref, std_ref = _get_reference_stats(ref_x, src, batch_size)
        plausibility_vec = ((cf >= (mu_ref - plausibility_z * std_ref)) & (cf <= (mu_ref + plausibility_z * std_ref))).float()
        plausibility_vec = plausibility_vec.mean(dim=(0, 2))

        cf_metrics.update({
            'sparsity': sparsity_ratio.mean().item(),
            'sparsity_count': sparsity_count.mean().item(),
            'proximity_l1': proximity_l1.mean().item(),
            'proximity_l2': proximity_l2.mean().item(),
            'plausibility': plausibility_vec.mean().item(),
        })

    return f1aa, out, cf_metrics


@torch.no_grad()
def eval_mv4(test_tuple, model, masked = False, gt_exps=None, epoch = -1 ):
    # Also evaluates models above v4
    model.eval()
    X, times, y = test_tuple
    out = model(X, times, captum_input = False)

    # INSERT_YOUR_CODE
    # Visualization of smooth_src, X_prime_sal, X_prime_cf, ste_mask for the first sample in the batch
    if False:
        import matplotlib.pyplot as plt

        out_dict = out    
        for sample_idx in range(min(out_dict['smooth_src'].shape[0], 1)):
            # sample_idx = 0  # visualize the first example in the batch

            # Move tensors to cpu if necessary and detach for plotting
            src = out_dict['smooth_src'][:,sample_idx].detach().cpu()
            x_prime_sal = out_dict['X_prime_sal'][:,sample_idx].detach().cpu()
            ste_mask = out_dict['ste_mask'][sample_idx].detach().cpu()
            # X_prime_cf may be None if not using counterfactuals
            x_prime_cf = out_dict['X_prime_cf'][:,sample_idx].detach().cpu() if out_dict.get('X_prime_cf') is not None else None
            
            X_recon = out_dict['X_recon'][:,sample_idx].detach().cpu()  # d_psi(g_theta(X)) for L_AE reconstruction
            
            # If input is multivariate, we’ll plot each feature dimension
            num_features = src.shape[-1] if src.ndim > 1 else 1
            time_axis = np.arange(src.shape[0])

            fig, axs = plt.subplots(num_features, 5 if x_prime_cf is not None else 4, figsize=(36, 4 * num_features), squeeze=False)
            plot_titles = ['smooth_src', 'X_prime_sal', 'ste_mask', 'X_prime_cf'] if x_prime_cf is not None else ['smooth_src', 'X_prime_sal', 'ste_mask']

            for f in range(num_features):
                # Prepare y values
                src_y = src[:, f] if num_features > 1 else src
                sal_y = x_prime_sal[:, f] if num_features > 1 else x_prime_sal
                mask_y = ste_mask[:, f] if ste_mask.ndim > 1 else ste_mask
                if x_prime_cf is not None:
                    cf_y = x_prime_cf[:, f] if num_features > 1 else x_prime_cf

                axs[f,0].plot(time_axis, src_y, label='smooth_src')
                axs[f,0].set_title(f'Feature {f}: smooth_src')
                axs[f,0].legend()

                axs[f,1].plot(time_axis, X_recon[:, f], label='X_recon')
                axs[f,1].set_title(f'Feature {f}: X_recon')
                axs[f,1].legend()

                axs[f,2].plot(time_axis, sal_y, label='X_prime_sal')
                axs[f,2].set_title(f'Feature {f}: X_prime_sal')
                axs[f,2].legend()

                axs[f,3].plot(time_axis, mask_y, label='ste_mask')
                axs[f,3].set_title(f'Feature {f}: ste_mask')
                axs[f,3].legend()

                if x_prime_cf is not None:
                    axs[f,4].plot(time_axis, cf_y, label='X_prime_cf')
                    axs[f,4].set_title(f'Feature {f}: X_prime_cf')
                    axs[f,4].legend()

            plt.tight_layout()
            # plt.show()
            os.makedirs('visualization', exist_ok=True)
            plt.savefig(os.path.join('visualization', f'epoch_{epoch}_sample_{sample_idx}.png'))
            plt.close()



    if masked:
        pred = out['pred_mask']
    else:
        pred = out['pred']

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')

    if gt_exps!=None:
        out = model(X, times, captum_input = False)
        generated_exps = out['mask_logits'].transpose(0,1).clone().detach().cpu()
        print(generated_exps.shape, gt_exps.shape)

        results_dict = ground_truth_xai_eval(generated_exps, gt_exps)
        results_dict["generated_exps"] = generated_exps
        results_dict["gt_exps"] = gt_exps

        return f1, out, results_dict
    else:
        return f1, out

@torch.inference_mode()
def eval_mv4_cf(
    test_tuple,
    model,
    masked = True,
    reference_tuple = None,
    target_labels = None,
    change_eps = 1e-4,
    plausibility_z = 3.0,
    untargeted = False,
    batch_size = None,
):
    # Also evaluates models above v4
    def _to_tbd(x):
        # Convert to (T, B, d)
        if x is None:
            return None
        if x.ndim == 2:
            # (T, B) or (B, T) -> keep as (T, B, 1) heuristic on first dim
            return x.unsqueeze(-1)
        if x.ndim != 3:
            return x
        # Most internal tensors in this codebase are (T, B, d).
        # If first dim matches batch and second does not, treat as (B, T, d).
        if x.shape[0] == y.shape[0] and x.shape[1] != y.shape[0]:
            return x.transpose(0, 1)
        return x

    def _extract_cf(out_dict, src_tbd):
        if out_dict.get('X_cf') is not None:
            return _to_tbd(out_dict['X_cf'])
        if out_dict.get('reference_z') is not None:
            # In cf variant, reference_z[1] is transformed source (counterfactual candidate)
            return _to_tbd(out_dict['reference_z'][1])
        if out_dict.get('constraint_src') is not None:
            return _to_tbd(src_tbd + out_dict['constraint_src'])
        return None

    def _extract_cf_logits(out_dict, cf_x, time_x):
        if cf_x is None:
            return None
        return model.encoder_main(cf_x, time_x, captum_input=False)

    def _get_reference_stats(ref_x, src_x):
        x_ref = src_x if ref_x is None else ref_x
        x_ref = _to_tbd(x_ref)
        mu = x_ref.mean(dim=1, keepdim=True)
        std = x_ref.std(dim=1, keepdim=True, unbiased=False)
        std = torch.clamp(std, min=1e-6)
        return mu, std

    X, times, y = test_tuple
    if batch_size is not None and batch_size > 0 and X.shape[1] > batch_size:
        weighted_metrics = None
        total = 0
        y_true_chunks, y_pred_chunks = [], []
        out_chunks, chunk_sizes = [], []

        for start in range(0, X.shape[1], batch_size):
            end = min(start + batch_size, X.shape[1])
            sub_tuple = (X[:, start:end, :], times[:, start:end], y[start:end])
            sub_targets = None if target_labels is None else target_labels[start:end]
            _, sub_out, sub_metrics = eval_mv4_cf(
                sub_tuple,
                model,
                masked=masked,
                reference_tuple=reference_tuple,
                target_labels=sub_targets,
                change_eps=change_eps,
                plausibility_z=plausibility_z,
                untargeted=untargeted,
                batch_size=None,
            )
            n_sub = int(end - start)
            chunk_sizes.append(n_sub)
            out_chunks.append(sub_out)
            total += n_sub
            if weighted_metrics is None:
                weighted_metrics = {k: 0.0 for k in sub_metrics.keys()}
            for k, v in sub_metrics.items():
                weighted_metrics[k] += float(v) * n_sub

            cf_cls = sub_out['pred_cf'].argmax(dim=-1)
            if sub_targets is not None:
                y_true_chunks.append(sub_targets.detach().cpu())
            else:
                y_true_chunks.append(sub_out['pred'].argmax(dim=-1).detach().cpu())
            y_pred_chunks.append(cf_cls.detach().cpu())

        y_true_all = torch.cat(y_true_chunks).numpy()
        y_pred_all = torch.cat(y_pred_chunks).numpy()
        f1 = f1_score(y_true_all, y_pred_all, average='macro')
        cf_metrics = {k: v / max(total, 1) for k, v in weighted_metrics.items()}

        out = {}
        for key in ('pred', 'pred_cf', 'ste_mask'):
            vals = [chunk.get(key) for chunk in out_chunks if chunk.get(key) is not None]
            if len(vals) != len(out_chunks):
                continue
            if vals[0].ndim > 0 and all(v.shape[0] == n for v, n in zip(vals, chunk_sizes)):
                out[key] = torch.cat(vals, dim=0)
            elif vals[0].ndim > 1 and all(v.shape[1] == n for v, n in zip(vals, chunk_sizes)):
                out[key] = torch.cat(vals, dim=1)
            else:
                out[key] = vals[-1]

        return f1, out, cf_metrics

    model.eval()
    out = model(X, times, captum_input = False)
    pred_cf = out['pred_cf']

    src = _to_tbd(out.get('smooth_src', X))
    cf = _extract_cf(out, src)
    cf_logits = _extract_cf_logits(out, cf, times)

    use_untargeted_metrics = untargeted or (target_labels is None)

    if (cf is not None) and (cf_logits is not None):
        cf_cls = cf_logits.argmax(dim=-1)
        orig_cls_t = y.to(device=cf_cls.device).long().view(-1)
        if target_labels is not None:
            ynp = target_labels.cpu().numpy()
            f1 = f1_score(ynp, cf_cls.cpu().numpy(), average='macro')
        else:
            f1 = f1_score(orig_cls_t.cpu().numpy(), cf_cls.cpu().numpy(), average='macro')
    else:
        prednp = pred_cf.argmax(dim=1).detach().cpu().numpy()
        if target_labels is not None:
            ynp = target_labels.cpu().numpy()
            f1 = f1_score(ynp, prednp, average='macro')
        else:
            orig_cls = out['pred'].argmax(dim=-1).detach().cpu().numpy()
            f1 = f1_score(orig_cls, prednp, average='macro')

    cf_metrics = {
        'validity': 0.0,
        'confidence': 0.0,
        'sparsity': 0.0,
        'sparsity_count': 0.0,
        'proximity_l1': 0.0,
        'proximity_l2': 0.0,
        'plausibility': 0.0,
    }

    if (cf is not None) and (cf_logits is not None):
        cf_cls = cf_logits.argmax(dim=-1)
        orig_cls_t = y.to(device=cf_cls.device).long().view(-1)
        cf_prob = cf_logits.softmax(dim=-1)
        if target_labels is not None:
            target_labels_t = target_labels.to(device=cf_cls.device).long().view(-1)
            validity_vec = (cf_cls == target_labels_t).float()
            confidence_vec = cf_prob.gather(1, target_labels_t.view(-1, 1)).squeeze(1)
        elif use_untargeted_metrics:
            validity_vec = (cf_cls != orig_cls_t).float()
            alt_conf = cf_prob.clone()
            idx = torch.arange(cf_prob.shape[0], device=cf_prob.device, dtype=torch.long)
            alt_conf[idx, orig_cls_t] = -1e9
            confidence_vec = alt_conf.max(dim=1).values
        else:
            validity_vec = (cf_cls == cf_cls).float()
            confidence_vec = cf_prob.gather(1, cf_cls.view(-1, 1)).squeeze(1)

        delta = (cf - src).abs()
        changed = (delta > change_eps).float()
        sparsity_ratio = changed.mean(dim=(0, 2))
        sparsity_count = changed.sum(dim=(0, 2))
        proximity_l1 = delta.mean(dim=(0, 2))
        proximity_l2 = torch.sqrt((delta ** 2).mean(dim=(0, 2)) + 1e-12)

        if reference_tuple is not None:
            ref_x = reference_tuple[0]
        else:
            ref_x = X
        mu_ref, std_ref = _get_reference_stats(ref_x, src)
        mu_ref = mu_ref.to(device=cf.device, dtype=cf.dtype)
        std_ref = std_ref.to(device=cf.device, dtype=cf.dtype)
        plausibility_vec = ((cf >= (mu_ref - plausibility_z * std_ref)) & (cf <= (mu_ref + plausibility_z * std_ref))).float()
        plausibility_vec = plausibility_vec.mean(dim=(0, 2))

        cf_metrics = {
            'validity': validity_vec.mean().item(),
            'confidence': confidence_vec.mean().item(),
            'sparsity': sparsity_ratio.mean().item(),
            'sparsity_count': sparsity_count.mean().item(),
            'proximity_l1': proximity_l1.mean().item(),
            'proximity_l2': proximity_l2.mean().item(),
            'plausibility': plausibility_vec.mean().item(),
        }

    return f1, out, cf_metrics


@torch.inference_mode()
def eval_mv4_cf_his(
    test_tuple,
    model,
    masked = True,
    reference_tuple = None,
    target_labels = None,
    change_eps = 1e-4,
    plausibility_z = 3.0,
    untargeted = False,
    batch_size = None,
):
    # Also evaluates models above v4
    def _to_tbd(x):
        # Convert to (T, B, d)
        if x is None:
            return None
        if x.ndim == 2:
            # (T, B) or (B, T) -> keep as (T, B, 1) heuristic on first dim
            return x.unsqueeze(-1)
        if x.ndim != 3:
            return x
        # Most internal tensors in this codebase are (T, B, d).
        # If first dim matches batch and second does not, treat as (B, T, d).
        if x.shape[0] == y.shape[0] and x.shape[1] != y.shape[0]:
            return x.transpose(0, 1)
        return x

    def _extract_cf(out_dict, src_tbd):
        if out_dict.get('X_cf') is not None:
            return _to_tbd(out_dict['X_cf'])
        if out_dict.get('reference_z') is not None:
            # In cf variant, reference_z[1] is transformed source (counterfactual candidate)
            return _to_tbd(out_dict['reference_z'][1])
        if out_dict.get('constraint_src') is not None:
            return _to_tbd(src_tbd + out_dict['constraint_src'])
        return None

    def _extract_cf_logits(out_dict, cf_x, time_x):
        if out_dict.get('pred_cf') is not None:
            return out_dict['pred_cf']
        if out_dict.get('pred_mask') is not None:
            return out_dict['pred_mask']
        if cf_x is None:
            return None
        return model.encoder_main(cf_x, time_x, captum_input=False)

    def _get_reference_stats(ref_x, src_x):
        x_ref = src_x if ref_x is None else ref_x
        x_ref = _to_tbd(x_ref)
        mu = x_ref.mean(dim=1, keepdim=True)
        std = x_ref.std(dim=1, keepdim=True, unbiased=False)
        std = torch.clamp(std, min=1e-6)
        return mu, std

    X, times, y = test_tuple
    if batch_size is not None and batch_size > 0 and X.shape[1] > batch_size:
        weighted_metrics = None
        total = 0
        y_true_chunks, y_pred_chunks = [], []
        out_chunks, chunk_sizes = [], []

        for start in range(0, X.shape[1], batch_size):
            end = min(start + batch_size, X.shape[1])
            sub_tuple = (X[:, start:end, :], times[:, start:end], y[start:end])
            sub_targets = None if target_labels is None else target_labels[start:end]
            _, sub_out, sub_metrics = eval_mv4_cf(
                sub_tuple,
                model,
                masked=masked,
                reference_tuple=reference_tuple,
                target_labels=sub_targets,
                change_eps=change_eps,
                plausibility_z=plausibility_z,
                untargeted=untargeted,
                batch_size=None,
            )
            n_sub = int(end - start)
            chunk_sizes.append(n_sub)
            out_chunks.append(sub_out)
            total += n_sub
            if weighted_metrics is None:
                weighted_metrics = {k: 0.0 for k in sub_metrics.keys()}
            for k, v in sub_metrics.items():
                weighted_metrics[k] += float(v) * n_sub

            cf_cls = sub_out['pred_cf'].argmax(dim=-1)
            if untargeted or sub_targets is None:
                y_true_chunks.append(sub_out['pred'].argmax(dim=-1).detach().cpu())
            else:
                y_true_chunks.append(sub_targets.detach().cpu())
            y_pred_chunks.append(cf_cls.detach().cpu())

        y_true_all = torch.cat(y_true_chunks).numpy()
        y_pred_all = torch.cat(y_pred_chunks).numpy()
        f1 = f1_score(y_true_all, y_pred_all, average='macro')
        cf_metrics = {k: v / max(total, 1) for k, v in weighted_metrics.items()}

        out = {}
        for key in ('pred', 'pred_cf', 'ste_mask'):
            vals = [chunk.get(key) for chunk in out_chunks if chunk.get(key) is not None]
            if len(vals) != len(out_chunks):
                continue
            if vals[0].ndim > 0 and all(v.shape[0] == n for v, n in zip(vals, chunk_sizes)):
                out[key] = torch.cat(vals, dim=0)
            elif vals[0].ndim > 1 and all(v.shape[1] == n for v, n in zip(vals, chunk_sizes)):
                out[key] = torch.cat(vals, dim=1)
            else:
                out[key] = vals[-1]

        return f1, out, cf_metrics

    model.eval()
    out = model(X, times, captum_input = False)
    pred_cf = out['pred_cf']

    src = _to_tbd(out.get('smooth_src', X))
    cf = _extract_cf(out, src)
    cf_logits = _extract_cf_logits(out, cf, times)

    use_untargeted_metrics = untargeted or (target_labels is None)

    if (cf is not None) and (cf_logits is not None):
        orig_cls = out['pred'].argmax(dim=-1)
        cf_cls = cf_logits.argmax(dim=-1)
        if use_untargeted_metrics:
            f1 = f1_score(orig_cls.cpu().numpy(), cf_cls.cpu().numpy(), average='macro')
        else:
            ynp = target_labels.cpu().numpy()
            f1 = f1_score(ynp, cf_cls.cpu().numpy(), average='macro')
    else:
        prednp = pred_cf.argmax(dim=1).detach().cpu().numpy()
        if use_untargeted_metrics:
            orig_cls = out['pred'].argmax(dim=-1).detach().cpu().numpy()
            f1 = f1_score(orig_cls, prednp, average='macro')
        else:
            ynp = target_labels.cpu().numpy()
            f1 = f1_score(ynp, prednp, average='macro')

    cf_metrics = {
        'validity': 0.0,
        'confidence': 0.0,
        'sparsity': 0.0,
        'sparsity_count': 0.0,
        'proximity_l1': 0.0,
        'proximity_l2': 0.0,
        'plausibility': 0.0,
    }

    if (cf is not None) and (cf_logits is not None):
        orig_cls = target_labels #out['pred'].argmax(dim=-1)
        cf_cls = cf_logits.argmax(dim=-1)
        cf_prob = cf_logits.softmax(dim=-1)
        if use_untargeted_metrics:
            validity_vec = (cf_cls != orig_cls).float()
            alt_conf = cf_prob.clone()
            idx = torch.arange(cf_prob.shape[0], device=cf_prob.device, dtype=torch.long)
            alt_conf[idx, orig_cls.long()] = -1e9
            confidence_vec = alt_conf.max(dim=1).values
        else:
            validity_vec = (cf_cls == target_labels).float()
            if target_labels is not None:
                confidence_vec = cf_prob.gather(1, target_labels.long().view(-1, 1)).squeeze(1)
            else:
                confidence_vec = cf_prob.gather(1, cf_cls.view(-1, 1)).squeeze(1)

        delta = (cf - src).abs()
        changed = (delta > change_eps).float()
        sparsity_ratio = changed.mean(dim=(0, 2))
        sparsity_count = changed.sum(dim=(0, 2))
        proximity_l1 = delta.mean(dim=(0, 2))
        proximity_l2 = torch.sqrt((delta ** 2).mean(dim=(0, 2)) + 1e-12)

        if reference_tuple is not None:
            ref_x = reference_tuple[0]
        else:
            ref_x = X
        mu_ref, std_ref = _get_reference_stats(ref_x, src)
        mu_ref = mu_ref.to(device=cf.device, dtype=cf.dtype)
        std_ref = std_ref.to(device=cf.device, dtype=cf.dtype)
        plausibility_vec = ((cf >= (mu_ref - plausibility_z * std_ref)) & (cf <= (mu_ref + plausibility_z * std_ref))).float()
        plausibility_vec = plausibility_vec.mean(dim=(0, 2))

        cf_metrics = {
            'validity': validity_vec.mean().item(),
            'confidence': confidence_vec.mean().item(),
            'sparsity': sparsity_ratio.mean().item(),
            'sparsity_count': sparsity_count.mean().item(),
            'proximity_l1': proximity_l1.mean().item(),
            'proximity_l2': proximity_l2.mean().item(),
            'plausibility': plausibility_vec.mean().item(),
        }

    return f1, out, cf_metrics


@torch.no_grad()
def eval_mv4_cf_targeted(
    test_tuple,
    model,
    masked = True,
    reference_tuple = None,
    target_labels = None,
    change_eps = 1e-4,
    plausibility_z = 3.0,
):
    # Also evaluates models above v4
    def _to_tbd(x):
        # Convert to (T, B, d)
        if x is None:
            return None
        if x.ndim == 2:
            # (T, B) or (B, T) -> keep as (T, B, 1) heuristic on first dim
            return x.unsqueeze(-1)
        if x.ndim != 3:
            return x
        # Most internal tensors in this codebase are (T, B, d).
        # If first dim matches batch and second does not, treat as (B, T, d).
        if x.shape[0] == y.shape[0] and x.shape[1] != y.shape[0]:
            return x.transpose(0, 1)
        return x

    def _extract_cf(out_dict, src_tbd):
        if out_dict.get('X_cf') is not None:
            return _to_tbd(out_dict['X_cf'])
        if out_dict.get('reference_z') is not None:
            # In cf variant, reference_z[1] is transformed source (counterfactual candidate)
            return _to_tbd(out_dict['reference_z'][1])
        if out_dict.get('constraint_src') is not None:
            return _to_tbd(src_tbd + out_dict['constraint_src'])
        return None

    def _extract_cf_logits(out_dict, cf_x, time_x):
        if out_dict.get('pred_cf') is not None:
            return out_dict['pred_cf']
        if out_dict.get('pred_mask') is not None:
            return out_dict['pred_mask']
        if cf_x is None:
            return None
        return model.encoder_main(cf_x, time_x, captum_input=False)

    def _get_reference_stats(ref_x, src_x):
        x_ref = src_x if ref_x is None else ref_x
        x_ref = _to_tbd(x_ref)
        mu = x_ref.mean(dim=1, keepdim=True)
        std = x_ref.std(dim=1, keepdim=True, unbiased=False)
        std = torch.clamp(std, min=1e-6)
        return mu, std

    model.eval()
    X, times, y = test_tuple
    out = model(X, times, captum_input = False)
    pred = out['pred_cf']
    # if masked:
    #     pred = out['pred_cf']
    # else:
        # pred = out['pred']

    # ynp = y.cpu().numpy()
    # INSERT_YOUR_CODE
    ynp = target_labels.cpu().numpy() if target_labels is not None else y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')

    src = _to_tbd(out.get('smooth_src', X))
    cf = _extract_cf(out, src)
    cf_logits = _extract_cf_logits(out, cf, times)

    cf_metrics = {
        'validity': 0.0,
        'confidence': 0.0,
        'sparsity': 0.0,
        'sparsity_count': 0.0,
        'proximity_l1': 0.0,
        'proximity_l2': 0.0,
        'plausibility': 0.0,
    }

    if (cf is not None) and (cf_logits is not None):
        orig_cls = out['pred'].argmax(dim=-1)
        cf_cls = cf_logits.argmax(dim=-1)
        validity_vec = (cf_cls == target_labels).float()

        cf_prob = cf_logits.softmax(dim=-1)
        if target_labels is not None:
            confidence_vec = cf_prob.gather(1, target_labels.long().view(-1, 1)).squeeze(1)
        else:
            confidence_vec = cf_prob.gather(1, cf_cls.view(-1, 1)).squeeze(1)

        delta = (cf - src).abs()
        changed = (delta > change_eps).float()
        sparsity_ratio = changed.mean(dim=(0, 2))
        sparsity_count = changed.sum(dim=(0, 2))
        proximity_l1 = delta.mean(dim=(0, 2))
        proximity_l2 = torch.sqrt((delta ** 2).mean(dim=(0, 2)) + 1e-12)

        if reference_tuple is not None:
            ref_x = reference_tuple[0]
        else:
            ref_x = X
        mu_ref, std_ref = _get_reference_stats(ref_x, src)
        mu_ref = mu_ref.to(device=cf.device, dtype=cf.dtype)
        std_ref = std_ref.to(device=cf.device, dtype=cf.dtype)
        plausibility_vec = ((cf >= (mu_ref - plausibility_z * std_ref)) & (cf <= (mu_ref + plausibility_z * std_ref))).float()
        plausibility_vec = plausibility_vec.mean(dim=(0, 2))

        cf_metrics = {
            'validity': validity_vec.mean().item(),
            'confidence': confidence_vec.mean().item(),
            'sparsity': sparsity_ratio.mean().item(),
            'sparsity_count': sparsity_count.mean().item(),
            'proximity_l1': proximity_l1.mean().item(),
            'proximity_l2': proximity_l2.mean().item(),
            'plausibility': plausibility_vec.mean().item(),
        }

    return f1, out, cf_metrics

@torch.no_grad()
def eval_mv4_idexp(test_tuple, test_tuple_external, model, masked = False):
    # Also evaluates models above v4
    model.eval()
    X, times, y = test_tuple
    out = model(X, times, src_id = test_tuple_external[0], captum_input = False)

    if masked:
        pred = out['pred_mask']
    else:
        pred = out['pred']

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, out

@torch.no_grad()
def eval_mvts_transformer(test_tuple, model, batch_size = None, auprc = False, auroc = False):
    '''
    Returns f1 score
    '''
    model.eval()

    X, times, y = test_tuple

    if batch_size is not None:
        if isinstance(model, CNN) or isinstance(model, LSTM):
            pred = torch.cat(
                [model(xb, tb) for xb, tb in zip(torch.split(X, batch_size, dim=1), torch.split(times, batch_size, dim=1))],
                dim=0
            )
        else: 
            pred, _ = batch_forwards_TransformerMVTS(model, X, times, batch_size = batch_size)
    else:
        pred = model(X, times)

    f1 = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro')

    pred_prob = pred.softmax(dim=-1).detach().cpu().numpy()
    # if pred_prob.shape[-1] == 2:
    #     pred_prob = pred_prob[:,1]

    yc = y.cpu().numpy()
    one_hot_y = np.zeros((yc.shape[0], yc.max() + 1))
    for i, yi in enumerate(yc):
        one_hot_y[i,yi] = 1

    if auprc:
        auprc_val = average_precision_score(one_hot_y, pred_prob, average = 'macro')
    if auroc:
        auroc_val = roc_auc_score(one_hot_y, pred_prob, average = 'macro', multi_class = 'ovr')

    if auprc and auroc:
        return f1, auprc_val, auroc_val
    elif auprc:
        return f1, auprc_val
    elif auroc:
        return f1, auroc_val
    else:
        return f1

@torch.no_grad()
def eval_and_select(X, times, model):
    '''
    Duo version
    Designed for SAT time mask selector
    Assumed X is size (T,d), times is (T,1) or (T,)
    '''

    model.eval()

    if len(times.shape) < 2:
        # Reshape times
        times = times.unsqueeze(-1)

    if len(X.shape) < 3:
        # Unsqueeze to batch
        X = X.unsqueeze(1)

    pred, mask = model(X, times, captum_input = False)

    #mask = (sat_scores > 0.5).squeeze()
    #print('mask size', mask.shape, mask.sum())
    #selected_X = X[mask,:,:]
    selected_X = model.apply_mask(X, times, mask = mask)

    return selected_X, pred.argmax(dim=1).detach().cpu()

@torch.no_grad()
def eval_and_select_nonduo(X, times, model):
    '''
    Designed for SAT time mask selector
    Assumed X is size (T,d), times is (T,1) or (T,)
    '''

    model.eval()

    if len(times.shape) < 2:
        # Reshape times
        times = times.unsqueeze(-1)

    if len(X.shape) < 3:
        # Unsqueeze to batch
        X = X.unsqueeze(1)

    pred, sat_mask, all_attns = model(X, times)

    mask = (sat_mask > 0.5).squeeze()
    print('mask size', mask.shape, mask.sum())
    selected_X = X[mask,:,:]

    return selected_X, pred.argmax(dim=1).detach().cpu()