import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../timesynth-0.2.4')))

import torch
torch.cuda.set_device(7)
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import time
from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.utils.predictors.loss_cl import LabelConsistencyLoss, SimCLRLoss, EmbedConsistencyLoss
from txai.utils.predictors.select_models import simloss_on_val_cononly, simloss_on_val_laonly, simloss_on_val_wboth, cosine_sim_for_simclr
from txai.utils.data.preprocess import process_PAM
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.eval import eval_mv4_cf
from txai.models.run_model_utils import batch_forwards
from txai.models.bc_model_cf import TimeXModel, AblationParameters, transformer_default_args
from txai.trainers.train_mv6_consistency_cf import train_mv6_consistency


is_timex = False

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
    else:
        name = "bc_full_split_cf={}.pt"
    name = "our_" + name
    if args.lam != 1.0:
        name = name[:-3] + f"_lam_cf={args.lam}.pt"
    return name


def visualize_counterfactual_examples(out_dict, times, y_true, target_label, save_dir, max_examples=5):
    os.makedirs(save_dir, exist_ok=True)
    src = out_dict.get("smooth_src", None)
    cf = out_dict.get("X_prime_cf", None)
    if src is None or cf is None:
        return
    src = src.detach().cpu()
    cf = cf.detach().cpu()
    times_cpu = times.detach().cpu()
    y_cpu = y_true.detach().cpu()
    num_examples = min(int(src.shape[1]), int(max_examples))
    for b in range(num_examples):
        t = times_cpu[:, b]
        x = src[:, b, 0]
        xcf = cf[:, b, 0]
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(t.numpy(), x.numpy(), label="original")
        ax.plot(t.numpy(), xcf.numpy(), label="counterfactual")
        ax.set_title(f"y={int(y_cpu[b])} -> target={int(target_label)} | sample={b}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"cf_example_{b}.png"), dpi=160)
        plt.close(fig)


def main(args):
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./../../datasets/PAM/"))
    tencoder_path = model_dir + "/models/transformer_split={}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 8
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

    metric_keys = ["f1_macro", "validity", "confidence", "sparsity", "sparsity_count", "proximity_l1", "proximity_l2", "plausibility"]
    all_results = {k: {"values": [], "weights": [], "weighted_sum": 0.0, "total_weight": 0.0} for k in metric_keys}
    targs = dict(transformer_default_args)

    for split_no in [1, 2, 3, 4, 5]:
    # for split_no in [5]:
        train_pam, val, test = process_PAM(split_no=split_no, device=device, base_path=base_path, gethalf=True)
        val_tuple = (val.X, val.time, val.y)
        test_tuple = (test.X, test.time, test.y)
        train_X = train_pam.X
        train_times = train_pam.time
        mu = train_X.mean(dim=1)
        std = train_X.std(unbiased=True, dim=1)
        targs.update({"norm_embedding": False})
        abl_params = AblationParameters(equal_g_gt=args.eq_ge, g_pret_equals_g=args.eq_pret, label_based_on_mask=True, ptype_assimilation=True, side_assimilation=True, 
                        # use_ste=(not args.no_ste), archtype="transformer")
                        use_ste=True, archtype="transformer", cnn_dim=16)
        loss_weight_dict = {"gsat": 1.0, "connect": 0.0}
        probe = TimeXModel(
            d_inp=val_tuple[0].shape[-1],
            max_len=val_tuple[0].shape[0],
            n_classes=n_classes,
            n_prototypes=50,
            gsat_r=0.1,
            transformer_args=targs,
            ablation_parameters=abl_params,
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
            val_out = batch_forwards(probe, val_tuple[0], val_tuple[1], batch_size=64)
            val_y_pred = val_out["pred"].argmax(dim=1).to(device)
            test_out = batch_forwards(probe, test_tuple[0], test_tuple[1], batch_size=64)
            test_y_pred = test_out["pred"].argmax(dim=1).to(device)
        del probe
        dset = DatasetwInds(train_X.to(device), train_times.to(device), train_y_pred)
        train_loader = torch.utils.data.DataLoader(dset, batch_size=256, shuffle=True)
        train_tuple = (train_X, train_times, train_y_pred)
        val_pred_tuple = (val_tuple[0], val_tuple[1], val_y_pred)
        val_reference_tuple = (val_tuple[0], val_tuple[1], val_tuple[2])
        for target_label in range(n_classes):
        # for target_label in [7]:

            my_seed=45
            random.seed(my_seed)
            np.random.seed(my_seed)
            torch.manual_seed(my_seed)
            torch.cuda.manual_seed_all(my_seed)

            model = TimeXModel(d_inp=val_tuple[0].shape[-1], max_len=val_tuple[0].shape[0], n_classes=n_classes, n_prototypes=50, 
                                gsat_r=0.3, transformer_args=targs, ablation_parameters=abl_params, 
                                loss_weight_dict=loss_weight_dict, masktoken_stats=(mu, std), tau=5.0)
            model.encoder_main.load_state_dict(torch.load(tencoder_path.format(split_no)))
            model.to(device)
            time_start = time.time()
            for p in model.encoder_main.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5) # 1e-3
            save_path = os.path.join(model_dir, "models", (naming_convention(args)[:-3] + f"_target={target_label}.pt")).format(split_no)
            train_mv6_consistency(model, optimizer=optimizer, 
                                    train_loader=train_loader, clf_criterion=clf_criterion, 
                                    sim_criterion=sim_criterion, beta_exp=2.0, beta_sim=1.0, 
                                    val_tuple=val_pred_tuple, val_reference_tuple=val_reference_tuple, 
                                    num_epochs=5, save_path=save_path, 
                                    train_tuple=train_tuple, early_stopping=True, 
                                    selection_criterion=selection_criterion, 
                                    label_matching=label_matching, 
                                    embedding_matching=embedding_matching, use_scheduler=True, 
                                    target_label=target_label, **sc_expand_args)
            time_end_train = time.time()
            print(f"Time taken for split {split_no}, target {target_label}: {time_end_train - time_start:.2f} seconds")
            sdict, _ = torch.load(save_path, weights_only=False)
            model.load_state_dict(sdict)
            test_mask = (test_y_pred != target_label)
            if int(test_mask.sum().item()) == 0:
                continue
            test_filtered = (test_tuple[0][:, test_mask, :], test_tuple[1][:, test_mask], test_y_pred[test_mask])
            target_labels = torch.full((test_filtered[2].shape[0],), int(target_label), device=test_filtered[2].device, dtype=torch.long)
            reference_mask = (test_tuple[2] == target_label)
            reference_tuple = (test_tuple[0][:, reference_mask], test_tuple[1][:, reference_mask], test_tuple[2][reference_mask])
            f1, out_dict, cf_metrics = eval_mv4_cf(test_filtered, model, target_labels=target_labels, reference_tuple=reference_tuple)
            n_eval = int(test_filtered[2].shape[0])
            time_end_eval = time.time()
            print(f"Time taken for split {split_no}, target {target_label}: {time_end_eval - time_end_train:.5f} seconds")
            
            print(f'split={split_no}, target={target_label}, n_eval={n_eval}, test_f1={f1:.4f}, cf_metrics={cf_metrics}')

            print(f"Average time per evaluation: {(time_end_eval - time_end_train) / n_eval:.5f} seconds")
            if args.num_vis_cf > 0:
                visualize_counterfactual_examples(out_dict, test_filtered[1], test_filtered[2], target_label, os.path.join(model_dir, "cf_visualizations", f"split={split_no}", f"target={target_label}"), args.num_vis_cf)
            metric_values = {"f1_macro": f1, **cf_metrics}
            for k in metric_keys:
                v = float(metric_values[k])
                all_results[k]["values"].append(v)
                all_results[k]["weights"].append(float(n_eval))
                all_results[k]["weighted_sum"] += v * float(n_eval)
                all_results[k]["total_weight"] += float(n_eval)

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
    parser.add_argument("--num_vis_cf", type=int, default=5)
    main(parser.parse_args())


"""
f1_macro: weighted_mean = 0.0972 +- 0.0292 (total_n=18690)
validity: weighted_mean = 0.3088 +- 0.0495 (total_n=18690)
confidence: weighted_mean = 0.3012 +- 0.0470 (total_n=18690)
sparsity: weighted_mean = 0.1311 +- 0.0042 (total_n=18690)
sparsity_count: weighted_mean = 1337.2686 +- 42.9969 (total_n=18690)
proximity_l1: weighted_mean = 0.0847 +- 0.0051 (total_n=18690)
proximity_l2: weighted_mean = 0.2409 +- 0.0131 (total_n=18690)
plausibility: weighted_mean = 0.9259 +- 0.0048 (total_n=18690)







f1_macro: weighted_mean = 0.4327 +- 0.0446 (total_n=18690)
validity: weighted_mean = 0.9716 +- 0.0071 (total_n=18690)
confidence: weighted_mean = 0.9420 +- 0.0095 (total_n=18690)
sparsity: weighted_mean = 0.1089 +- 0.0015 (total_n=18690)
sparsity_count: weighted_mean = 1110.3583 +- 15.7166 (total_n=18690)
proximity_l1: weighted_mean = 0.0727 +- 0.0036 (total_n=18690)
proximity_l2: weighted_mean = 0.2238 +- 0.0105 (total_n=18690)
plausibility: weighted_mean = 0.9297 +- 0.0039 (total_n=18690)

Epoch 100, Val F1 = 0.2289, Val Sparsity = 0.0980
Best Epoch: 2    Val Metric = 1.7530     Val F1 = 0.3317
Time taken for split 1, target 0: 479.21 seconds
Time taken for split 1, target 0: 0.15 seconds
Average time per evaluation: 0.00 seconds




"""
