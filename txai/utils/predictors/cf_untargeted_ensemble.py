"""
Ensemble of targeted CF checkpoints: per instance, pick the target model whose
softmax(pred_cf)[target] is highest among models valid when orig_pred != target.
Used by experiments/*/bc_model_ptype_cf_untargeted.py.
"""
import torch


def safe_torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, tuple):
        return ckpt_obj[0]
    if isinstance(ckpt_obj, dict):
        if 'state_dict' in ckpt_obj and isinstance(ckpt_obj['state_dict'], dict):
            return ckpt_obj['state_dict']
        return ckpt_obj
    raise ValueError(f'Unsupported checkpoint format: {type(ckpt_obj)}')


def merge_by_batch_dim0(tensors, best_idx):
    # tensors[j] shape: (B, C) or (B, ...).
    # For each sample b, pick row b from the selected target model best_idx[b].
    bsz = int(best_idx.shape[0])
    merged = [tensors[int(best_idx[b])][b, ...] for b in range(bsz)]
    return torch.stack(merged, dim=0)


def merge_by_batch_dim1(tensors, best_idx):
    # tensors[j] shape: (T, B, d) style.
    # For each sample b, pick column b from the selected target model best_idx[b].
    bsz = int(best_idx.shape[0])
    merged = [tensors[int(best_idx[b])][:, b, ...] for b in range(bsz)]
    return torch.stack(merged, dim=1)


def build_best_cf_out(outs, target_labels, fallback_src):
    K = len(outs)
    if K == 0:
        raise ValueError('no target models in outs')

    orig_cls = outs[0]['pred'].argmax(dim=-1)
    device = orig_cls.device
    dtype = outs[0]['pred_cf'].dtype
    bsz = int(orig_cls.shape[0])

    # conf_mat[b, j]: confidence of sample b under target model j.
    # Start from -inf so invalid targets can stay impossible during argmax.
    conf_mat = torch.full((bsz, K), float('-inf'), device=device, dtype=dtype)

    for j, tgt in enumerate(target_labels):
        t = int(tgt)
        prob = outs[j]['pred_cf'].softmax(dim=-1)
        col = prob[:, t]
        # If original prediction is already target t, this target is invalid
        # for untargeted "change to another class" behavior.
        invalid = orig_cls == t
        conf_mat[:, j] = col.masked_fill(invalid, float('-inf'))

    best_idx = conf_mat.argmax(dim=1)
    row_max = conf_mat[torch.arange(bsz, device=device), best_idx]
    # Safety fallback: if all targets were masked invalid for a sample,
    # choose the first target different from the original class.
    bad = ~torch.isfinite(row_max)
    if bad.any():
        for b in torch.where(bad)[0].tolist():
            for j, tgt in enumerate(target_labels):
                if int(orig_cls[b].item()) != int(tgt):
                    best_idx[b] = j
                    break

    merged_out = {}
    # Keep original prediction from any model (they share the same frozen encoder_main).
    merged_out['pred'] = outs[0]['pred']
    merged_out['pred_cf'] = merge_by_batch_dim0([o['pred_cf'] for o in outs], best_idx)

    src = outs[0].get('smooth_src', fallback_src)
    merged_out['smooth_src'] = src

    xcf_list = [o.get('X_cf', None) for o in outs]
    if all(x is not None for x in xcf_list):
        # Merge selected counterfactual time-series per sample.
        merged_out['X_cf'] = merge_by_batch_dim1(xcf_list, best_idx)

    refz_list = [o.get('reference_z', None) for o in outs]
    if all((rz is not None and isinstance(rz, tuple) and len(rz) >= 2) for rz in refz_list):
        xprime_list = [rz[1] for rz in refz_list]
        # Preserve the (src, transformed_src) contract expected by eval_mv4_cf.
        merged_out['reference_z'] = (src, merge_by_batch_dim1(xprime_list, best_idx))

    return merged_out


class BestTargetCFModel(torch.nn.Module):
    def __init__(self, model_list, target_labels):
        super().__init__()
        self.model_list = torch.nn.ModuleList(model_list)
        self.target_labels = [int(t) for t in target_labels]

    @property
    def encoder_main(self):
        # Expose encoder_main so eval_mv4_cf can recompute logits on merged CF.
        # All target checkpoints share this frozen encoder, so index 0 is sufficient.
        return self.model_list[0].encoder_main

    def forward(self, src, times, captum_input=False):
        # Run all target-conditioned models, then ensemble by per-sample best target.
        outs = [m(src, times, captum_input=captum_input) for m in self.model_list]
        return build_best_cf_out(outs, self.target_labels, fallback_src=src)
