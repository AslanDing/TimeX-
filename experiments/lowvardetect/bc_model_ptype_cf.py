import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../timesynth-0.2.4')))

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import torch
torch.cuda.set_device(2)
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.utils.predictors.loss_cl import LabelConsistencyLoss, SimCLRLoss, EmbedConsistencyLoss
from txai.utils.predictors.select_models import simloss_on_val_cononly, simloss_on_val_laonly, simloss_on_val_wboth, cosine_sim_for_simclr
from txai.utils.data import process_Synth
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.eval import eval_mv4_cf
from txai.models.run_model_utils import batch_forwards
from txai.models.bc_model_cf import TimeXModel, AblationParameters, transformer_default_args
from txai.trainers.train_mv6_consistency_cf import train_mv6_consistency

import random


def naming_convention(args):
    name = "bc_full_split_cf={}.pt"
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
    name = "our_" + name
    if args.lam != 1.0:
        name = name[:-3] + f"_lam_cf={args.lam}.pt"
    return name


def _tensor_tbd_to_numpy_samples(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu()
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if x.ndim != 3:
        raise ValueError(f"Expected time series tensor with 2 or 3 dims, got shape={tuple(x.shape)}")
    return x.permute(1, 0, 2).reshape(x.shape[1], -1).numpy().astype(np.float32)


def _extract_cf_tensor(out_dict, src=None):
    if out_dict.get("X_cf") is not None:
        return out_dict["X_cf"]
    if out_dict.get("reference_z") is not None:
        return out_dict["reference_z"][1]
    if src is not None and out_dict.get("constraint_src") is not None:
        return src + out_dict["constraint_src"]
    return None


def compute_lof_plausibility(reference_x, cf_x, n_neighbors=20, max_ref=1000, seed=42):
    ref_np = _tensor_tbd_to_numpy_samples(reference_x)
    cf_np = _tensor_tbd_to_numpy_samples(cf_x)
    if ref_np is None or cf_np is None or ref_np.shape[0] < 3 or cf_np.shape[0] == 0:
        return float("nan")
    if max_ref is not None and max_ref > 0 and ref_np.shape[0] > max_ref:
        rng = np.random.default_rng(seed)
        ref_np = ref_np[rng.choice(ref_np.shape[0], size=max_ref, replace=False)]
    n_neighbors = min(int(n_neighbors), max(2, ref_np.shape[0] - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, metric="euclidean", n_jobs=-1)
    lof.fit(ref_np)
    cf_pred = lof.predict(cf_np)
    return float((cf_pred == 1).mean())


def _to_tbd_vis(x, batch_size):
    """Normalize to (T, B, d), matching eval_mv4_cf._to_tbd."""
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu()
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if x.ndim != 3:
        return x
    if x.shape[0] == batch_size and x.shape[1] != batch_size:
        return x.transpose(0, 1)
    return x


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
    if delta.ndim == 1:
        return delta > change_eps
    return delta.max(dim=-1).values > change_eps


def _sample_sparsity_stats(src, cf, batch_idx, change_eps=1e-4):
    """Per-sample sparsity stats using the same rule as eval_mv4_cf."""
    delta = (cf[:, batch_idx, :] - src[:, batch_idx, :]).abs()
    changed_elem = delta > change_eps
    sparsity = changed_elem.float().mean().item()
    sparsity_count = int(changed_elem.sum().item())
    if changed_elem.ndim == 1:
        changed_ts = changed_elem
    else:
        changed_ts = changed_elem.any(dim=-1)
    return sparsity, sparsity_count, changed_ts


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
    sparsity=None,
    sparsity_count=None,
    n_elem_total=None,
):
    t_np = t.numpy() if torch.is_tensor(t) else np.asarray(t)
    src_np = src.numpy() if torch.is_tensor(src) else np.asarray(src)
    cf_np = cf.numpy() if torch.is_tensor(cf) else np.asarray(cf)
    changed_np = changed.numpy() if torch.is_tensor(changed) else np.asarray(changed)
    changed_idx = np.flatnonzero(changed_np)
    ts_text = ", ".join(f"{t_np[i]:.2f}" for i in changed_idx[:12])
    if changed_idx.size > 12:
        ts_text += ", ..."

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(t_np, src_np, color="tab:blue", linewidth=1.5, label="original")
    axes[0].set_ylabel("value")
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        f"Original input | source={int(source_label)} | sample={sample_rank} (idx={batch_idx})"
    )

    axes[1].fill_between(t_np, 0.0, changed_np.astype(np.float32), color="tab:orange", alpha=0.35, step="mid")
    axes[1].plot(t_np, changed_np.astype(np.float32), color="tab:red", linewidth=1.2, drawstyle="steps-mid")
    for i in changed_idx:
        axes[1].axvline(t_np[i], color="tab:red", alpha=0.25, linewidth=0.8)
    axes[1].set_ylim(-0.05, 1.15)
    axes[1].set_ylabel("changed")
    sparsity_text = ""
    if sparsity is not None and sparsity_count is not None:
        denom = int(n_elem_total) if n_elem_total is not None else int(sparsity_count / max(sparsity, 1e-12))
        sparsity_text = (
            f" | sparsity={sparsity:.3f} (n_elem={int(sparsity_count)}/{denom})"
        )
    axes[1].set_title(
        f"Changed timestamps (n={int(changed_idx.size)})"
        + sparsity_text
        + (f" | t=[{ts_text}]" if changed_idx.size > 0 else "")
    )

    axes[2].plot(t_np, cf_np, color="tab:green", linewidth=1.5, label="counterfactual")
    axes[2].set_ylabel("value")
    axes[2].set_xlabel("time")
    axes[2].legend(loc="upper right")
    valid = int(cf_pred_label) == int(target_label)
    axes[2].set_title(
        f"CF result | pred={int(cf_pred_label)} (target={int(target_label)}, valid={valid})"
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


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
    os.makedirs(save_dir, exist_ok=True)

    src = out_dict.get("smooth_src", None)
    cf = _extract_cf_tensor(out_dict, src=src)
    if src is None or cf is None:
        print(f"[vis] skip label-pair vis for target={target_label}: missing smooth_src/CF tensor")
        return

    if torch.is_tensor(source_labels):
        batch_size = int(source_labels.shape[0])
    else:
        batch_size = int(len(source_labels))
    src = _to_tbd_vis(src, batch_size)
    cf = _to_tbd_vis(cf, batch_size)
    times_cpu = times.detach().cpu() if torch.is_tensor(times) else None
    mask = out_dict.get("ste_mask", None)
    if mask is not None:
        mask = mask.detach().cpu()
    pred_cf = out_dict.get("pred_cf", None)
    if pred_cf is not None:
        pred_cf = pred_cf.detach().cpu().argmax(dim=-1)

    n_saved = 0
    n_skipped_pairs = 0
    for source_label in range(n_classes):
        if int(source_label) == int(target_label):
            continue
        pair_dir = os.path.join(
            save_dir,
            f"source={int(source_label)}_target={int(target_label)}",
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
            print(f"[vis] no samples for source={source_label} -> target={target_label}, skip.")
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
            sparsity, sparsity_count, changed = _sample_sparsity_stats(
                src, cf, batch_idx, change_eps=change_eps
            )
            cf_pred_label = int(pred_cf[batch_idx]) if pred_cf is not None else -1
            save_path = os.path.join(pair_dir, f"example_{rank:02d}_idx={batch_idx}.png")
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
                sparsity=sparsity,
                sparsity_count=sparsity_count,
                n_elem_total=int(src.shape[0] * src.shape[2]),
            )
            n_saved += 1
    print(
        f"[vis] target={target_label}: saved {n_saved} figure(s) under {save_dir} "
        f"({n_skipped_pairs} empty source->target pair(s))"
    )


def visualize_counterfactual_examples(out_dict, times, y_true, target_label, save_dir, max_examples=5):
    os.makedirs(save_dir, exist_ok=True)

    src = out_dict.get("smooth_src", None)
    cf = _extract_cf_tensor(out_dict, src=src)
    if src is None or cf is None:
        print(f"[vis] skip: missing smooth_src/CF tensor for target={target_label}")
        return

    if torch.is_tensor(y_true):
        batch_size = int(y_true.shape[0])
    else:
        batch_size = int(len(y_true))
    src = _to_tbd_vis(src, batch_size)
    cf = _to_tbd_vis(cf, batch_size)
    times_cpu = times.detach().cpu() if torch.is_tensor(times) else None
    y_cpu = y_true.detach().cpu() if torch.is_tensor(y_true) else None
    mask = out_dict.get("ste_mask", None)
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
            m = _batch_mask_at_index(mask, b, src.shape[0])

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t.numpy(), x.numpy(), label="original", linewidth=1.4)
        axes[0].plot(t.numpy(), xcf.numpy(), label="counterfactual", linewidth=1.4)
        axes[0].set_ylabel("value")
        axes[0].legend(loc="upper right")
        axes[0].set_title(
            f"CF vs Original | y={int(y_cpu[b]) if y_cpu is not None else '?'} -> target={int(target_label)} | sample={b}"
        )

        axes[1].plot(t.numpy(), delta.numpy(), color="tab:red", linewidth=1.4)
        axes[1].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        axes[1].set_ylabel("cf - orig")

        if m is not None:
            axes[2].plot(t.numpy(), m.numpy(), color="tab:green", linewidth=1.2, label="mask")
            axes[2].set_ylabel("mask")
            axes[2].set_ylim(-0.05, 1.05)
            axes[2].legend(loc="upper right")
        else:
            axes[2].plot(t.numpy(), np.abs(delta.numpy()), color="tab:purple", linewidth=1.2)
            axes[2].set_ylabel("|cf-orig|")

        axes[2].set_xlabel("time")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"cf_example_{b}.png"), dpi=160)
        plt.close(fig)
    print(f"[vis] saved {num_examples} CF figure(s) to {save_dir}")


def main(args):
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
    tencoder_path = model_dir + "/models/transformer_new2_split={}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 4
    clf_criterion = Poly1CrossEntropyLoss(num_classes=n_classes, epsilon=1.0, weight=None, reduction="mean")

    sim_criterion_label = LabelConsistencyLoss()
    if args.simclr:
        sim_criterion_cons = SimCLRLoss()
        sc_expand_args = {"simclr_training": True, "num_negatives_simclr": 64}
    else:
        sim_criterion_cons = EmbedConsistencyLoss(normalize_distance=True)
        sc_expand_args = {"simclr_training": False, "num_negatives_simclr": 64}

    if args.no_la:
        sim_criterion = sim_criterion_cons
        selection_criterion = simloss_on_val_cononly(sim_criterion)
        label_matching, embedding_matching = False, True
    elif args.no_con:
        sim_criterion = sim_criterion_label
        selection_criterion = simloss_on_val_laonly(sim_criterion)
        label_matching, embedding_matching = True, False
    else:
        sim_criterion = [sim_criterion_cons, sim_criterion_label]
        selection_criterion = simloss_on_val_wboth([cosine_sim_for_simclr, sim_criterion_label], lam=1.0) if args.simclr else simloss_on_val_wboth(sim_criterion, lam=1.0)
        label_matching, embedding_matching = True, True

    metric_keys = ["f1_macro", "validity", "confidence", "sparsity", "sparsity_count", "proximity_l1", "proximity_l2", "plausibility", "cf_inference_time_per_sample"]
    all_results = {k: {"values": [], "weights": [], "weighted_sum": 0.0, "total_weight": 0.0} for k in metric_keys}
    targs = dict(transformer_default_args)

    base_path = os.path.abspath(os.path.dirname(__file__) + '../../../datasets/LowVarDetect')
    split_time_results = []
    for split_no in [1, 2,3,  4, 5]:

    # for split_no in [5]:
        d = process_Synth(split_no=split_no, device=device, base_path=base_path)
        val, test = d["val"], d["test"]
        train_X = d["train_loader"].X
        train_times = d["train_loader"].times
        mu = train_X.mean(dim=1)
        std = train_X.std(unbiased=True, dim=1)
        targs.update({"trans_dim_feedforward": 32, "trans_dropout": 0.1, "nlayers": 1, "norm_embedding": True})
        abl = AblationParameters(equal_g_gt=args.eq_ge, g_pret_equals_g=args.eq_pret, label_based_on_mask=True, ptype_assimilation=True, side_assimilation=True, use_ste=(not args.no_ste), archtype="transformer")
        loss_weight_dict = {"gsat": 1.0, "connect": 2.0}
        probe = TimeXModel(
            d_inp=val[0].shape[-1],
            max_len=val[0].shape[0],
            n_classes=n_classes,
            n_prototypes=50,
            gsat_r=0.5,
            transformer_args=targs,
            ablation_parameters=abl,
            loss_weight_dict=loss_weight_dict,
            masktoken_stats=(mu, std),
            tau=1.0,
        )
        probe.encoder_main.load_state_dict(torch.load(tencoder_path.format(split_no)))
        probe.to(device)
        probe.eval()
        with torch.inference_mode():
            train_out = batch_forwards(probe, train_X, train_times, batch_size=64)
            train_y_pred = train_out["pred"].argmax(dim=1).to(device)
            val_out = batch_forwards(probe, val[0], val[1], batch_size=64)
            val_y_pred = val_out["pred"].argmax(dim=1).to(device)
            test_out = batch_forwards(probe, test[0], test[1], batch_size=64)
            test_y_pred = test_out["pred"].argmax(dim=1).to(device)
        del probe
        if args.train:
            dset = DatasetwInds(train_X, train_times, train_y_pred)
            train_loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=True)
        train_tuple = (train_X, train_times, train_y_pred)
        val_pred_tuple = (val[0], val[1], val_y_pred)
        val_reference_tuple = (val[0], val[1], val[2])
        split_cf_inference_time_total = 0.0
        split_cf_n_eval = 0
        for target_label in range(n_classes):
        # for target_label in [3]:
                    
            my_seed=42
            random.seed(my_seed)
            np.random.seed(my_seed)
            torch.manual_seed(my_seed)
            torch.cuda.manual_seed_all(my_seed)
            model = TimeXModel(d_inp=val[0].shape[-1], max_len=val[0].shape[0], n_classes=n_classes, n_prototypes=50, gsat_r=0.1, transformer_args=targs, ablation_parameters=abl, loss_weight_dict=loss_weight_dict, masktoken_stats=(mu, std), tau=1.0)
            model.encoder_main.load_state_dict(torch.load(tencoder_path.format(split_no)))
            model.to(device)
            for p in model.encoder_main.parameters():
                p.requires_grad = False
            save_path = os.path.join(model_dir, "models", (naming_convention(args)[:-3] + f"_target={target_label}.pt")).format(split_no)
            if args.train:
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
                train_mv6_consistency(model, optimizer=optimizer, train_loader=train_loader,
                                    clf_criterion=clf_criterion, sim_criterion=sim_criterion,
                                    beta_exp=1.0, beta_sim=0.5, val_tuple=val_pred_tuple,
                                    val_reference_tuple=val_reference_tuple,
                                    num_epochs=1000, save_path=save_path, train_tuple=train_tuple,
                                    early_stopping=True, selection_criterion=selection_criterion, label_matching=label_matching,
                                    embedding_matching=embedding_matching, use_scheduler=False, use_warmup_cosine_scheduler=True,
                                    target_label=target_label, **sc_expand_args)
            sdict, _ = torch.load(save_path, weights_only=False)
            model.load_state_dict(sdict)
            model.eval()
            test_mask = (test_y_pred != target_label)
            if int(test_mask.sum().item()) == 0:
                continue
            test_filtered = (test[0][:, test_mask, :], test[1][:, test_mask], test_y_pred[test_mask])
            target_labels = torch.full((test_filtered[2].shape[0],), int(target_label), device=test_filtered[2].device, dtype=torch.long)
            reference_mask = (test[2] == target_label)
            reference_tuple = (test[0][:, reference_mask], test[1][:, reference_mask], test[2][reference_mask])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cf_start_time = time.perf_counter()
            f1, out_dict, cf_metrics = eval_mv4_cf(test_filtered, model, target_labels=target_labels, reference_tuple=reference_tuple)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cf_inference_time_total = time.perf_counter() - cf_start_time
            cf_tensor = _extract_cf_tensor(out_dict, src=out_dict.get("smooth_src", test_filtered[0]))
            lof_reference_mask = (d["train_loader"].y.to(train_X.device) == target_label)
            lof_reference_X = train_X #[:, lof_reference_mask, :]
            lof_plausibility = compute_lof_plausibility(
                lof_reference_X,
                cf_tensor,
                n_neighbors=args.lof_n_neighbors,
                max_ref=args.lof_max_ref,
                seed=42 + split_no * 10 + target_label,
            )
            cf_metrics["plausibility"] = lof_plausibility
            n_eval = int(test_filtered[2].shape[0])
            cf_inference_time_per_sample = cf_inference_time_total / max(n_eval, 1)
            cf_metrics["cf_inference_time_per_sample"] = cf_inference_time_per_sample
            split_cf_inference_time_total += cf_inference_time_total
            split_cf_n_eval += n_eval
            print(f'split={split_no}, target={target_label}, n_eval={n_eval}, test_f1={f1:.4f}, cf_time_total={cf_inference_time_total:.6f}s, cf_time_per_sample={cf_inference_time_per_sample * 1000.0:.4f}ms, lof_plausibility={lof_plausibility:.4f}, cf_metrics={cf_metrics}')
            if args.num_vis_cf > 0:
                vis_dir = os.path.join(model_dir, "cf_visualizations", f"split={split_no}", f"target={target_label}")
                visualize_counterfactual_examples(
                    out_dict=out_dict,
                    times=test_filtered[1],
                    y_true=test_filtered[2],
                    target_label=target_label,
                    save_dir=vis_dir,
                    max_examples=args.num_vis_cf,
                )

            if args.vis_split is not None and int(split_no) == int(args.vis_split):
                label_pair_vis_dir = os.path.join(
                    model_dir,
                    "cf_label_pair_visualizations",
                    f"split={split_no}",
                    f"target={target_label}",
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
                    seed=args.vis_seed + split_no * 100 + target_label,
                )

            metric_values = {"f1_macro": f1, **cf_metrics}
            for k in metric_keys:
                v = float(metric_values[k])
                all_results[k]["values"].append(v)
                all_results[k]["weights"].append(float(n_eval))
                all_results[k]["weighted_sum"] += v * float(n_eval)
                all_results[k]["total_weight"] += float(n_eval)

        split_cf_inference_time_per_sample = split_cf_inference_time_total / max(split_cf_n_eval, 1)
        split_time_results.append({
            "split": int(split_no),
            "n_eval": int(split_cf_n_eval),
            "cf_time_total": float(split_cf_inference_time_total),
            "cf_time_per_sample": float(split_cf_inference_time_per_sample),
        })
        print(f"[split_time] split={split_no}, cf_time_total={split_cf_inference_time_total:.6f}s, cf_time_per_sample={split_cf_inference_time_per_sample * 1000.0:.4f}ms, n_eval={split_cf_n_eval}")

    print("============================================================================================")
    print("Per-split CF inference time")
    for rec in split_time_results:
        print(f"split={rec['split']}, cf_time_total={rec['cf_time_total']:.6f}s, cf_time_per_sample={rec['cf_time_per_sample'] * 1000.0:.4f}ms, n_eval={rec['n_eval']}")
    if len(split_time_results) > 0:
        split_total_times = np.array([rec["cf_time_total"] for rec in split_time_results], dtype=np.float64)
        split_per_sample_ms = np.array([rec["cf_time_per_sample"] * 1000.0 for rec in split_time_results], dtype=np.float64)
        print(
            f"per_split_cf_time_total: mean = {split_total_times.mean():.6f}s, "
            f"std = {split_total_times.std(ddof=1 if len(split_total_times) > 1 else 0):.6f}s"
        )
        print(
            f"per_split_cf_time_per_sample: mean = {split_per_sample_ms.mean():.4f}ms, "
            f"std = {split_per_sample_ms.std(ddof=1 if len(split_per_sample_ms) > 1 else 0):.4f}ms"
        )

    print("============================================================================================")
    for k in metric_keys:
        rec = all_results[k]
        if rec["total_weight"] <= 0:
            print(f"{k}: no valid samples aggregated.")
            continue
        vals = np.array(rec["values"], dtype=np.float64)
        weights = np.array(rec["weights"], dtype=np.float64)
        wmean = rec["weighted_sum"] / rec["total_weight"]
        wvar = np.average((vals - wmean) ** 2, weights=weights)
        wstd = np.sqrt(max(wvar, 0.0))
        eff_n = (weights.sum() ** 2) / (np.square(weights).sum() + 1e-12)
        wsem = wstd / np.sqrt(max(eff_n, 1.0))
        print(f"{k}: weighted_mean = {wmean:.4f} +- {wsem:.4f} (total_n={int(rec['total_weight'])})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ablations = parser.add_mutually_exclusive_group()
    ablations.add_argument("--eq_ge", action="store_true")
    ablations.add_argument("--eq_pret", action="store_true")
    ablations.add_argument("--ge_rand_init", action="store_true")
    ablations.add_argument("--no_ste", action="store_true")
    ablations.add_argument("--simclr", action="store_true")
    ablations.add_argument("--no_la", action="store_true")
    ablations.add_argument("--no_con", action="store_true")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--num_vis_cf", type=int, default=0)
    parser.add_argument("--vis_split", type=int, default=1, help="if set, save label-pair CF visualizations for this split only")
    parser.add_argument("--vis_per_pair", type=int, default=20, help="max examples per (source_label, target_label) pair")
    parser.add_argument("--vis_change_eps", type=float, default=1e-4, help="threshold for marking changed timestamps")
    parser.add_argument("--vis_seed", type=int, default=42, help="RNG seed for sampling label-pair visualization examples")
    parser.add_argument("--train", action="store_true", help="train target-specific CF models before evaluation; default only loads saved checkpoints")
    parser.add_argument("--lof_n_neighbors", type=int, default=5, help="number of neighbors for LOF plausibility")
    parser.add_argument("--lof_max_ref", type=int, default=1000, help="maximum target-class training samples used to fit each LOF model")
    main(parser.parse_args())

"""
"""
