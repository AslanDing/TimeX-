import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../timesynth-0.2.4')))

import torch
torch.cuda.set_device(6)
import argparse
import numpy as np
from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.utils.predictors.loss_cl import LabelConsistencyLoss, SimCLRLoss, EmbedConsistencyLoss
from txai.utils.predictors.select_models import simloss_on_val_cononly, simloss_on_val_laonly, simloss_on_val_wboth, cosine_sim_for_simclr
from txai.utils.data.preprocess import process_Boiler_OLD
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.eval import eval_mv4_cf
from txai.models.run_model_utils import batch_forwards
from txai.models.bc_model_cf import TimeXModel, AblationParameters, transformer_default_args
from txai.trainers.train_mv6_consistency_cf import train_mv6_consistency
import random
import time


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


def main(args):
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./../../datasets/Boiler/"))
    tencoder_path = model_dir + "/models/transformer_split={}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 2
    clf_criterion = Poly1CrossEntropyLoss(num_classes=n_classes, epsilon=1.0, weight=None, reduction="mean")

    sim_criterion_label = LabelConsistencyLoss()
    if args.simclr:
        sim_criterion_cons = SimCLRLoss()
        sc_expand_args = {"simclr_training": True, "num_negatives_simclr": 64}
    else:
        sim_criterion_cons = EmbedConsistencyLoss(normalize_distance=False)
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
    targs.update({"trans_dim_feedforward": 32, "trans_dropout": 0.25, "norm_embedding": True})

    for split_no in [1, 2, 3, 4, 5]:
    # for split_no in [1]:
        
        train_b, val, test = process_Boiler_OLD(split_no=split_no, device=device, base_path=base_path)
        val_tuple = val
        test_tuple = test
        train_X, train_times, _ = train_b
        mu = train_X.mean(dim=1)
        std = train_X.std(unbiased=True, dim=1)
        abl = AblationParameters(equal_g_gt=args.eq_ge, g_pret_equals_g=args.eq_pret, label_based_on_mask=True, ptype_assimilation=True, side_assimilation=True, 
                                # use_ste=(not args.no_ste), archtype="transformer")
                                use_ste=True, archtype="transformer", cnn_dim=32)
                                # archtype="transformer", cnn_dim=32)
        loss_weight_dict = {"gsat": 0.0, "connect": 0.0}
        probe = TimeXModel(
            d_inp=val_tuple[0].shape[-1],
            max_len=val_tuple[0].shape[0],
            n_classes=n_classes,
            n_prototypes=50,
            gsat_r=0.1,
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
            val_out = batch_forwards(probe, val_tuple[0], val_tuple[1], batch_size=64)
            val_y_pred = val_out["pred"].argmax(dim=1).to(device)
            test_out = batch_forwards(probe, test_tuple[0], test_tuple[1], batch_size=64)
            test_y_pred = test_out["pred"].argmax(dim=1).to(device)
        del probe
        dset = DatasetwInds(train_X, train_times, train_y_pred)
        train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)
        train_tuple = (train_X, train_times, train_y_pred)
        val_pred_tuple = (val_tuple[0], val_tuple[1], val_y_pred)
        val_reference_tuple = (val_tuple[0], val_tuple[1], val_tuple[2])
        for target_label in range(n_classes):
        # for target_label in [1]:
                        
            my_seed=42
            random.seed(my_seed)
            np.random.seed(my_seed)
            torch.manual_seed(my_seed)
            torch.cuda.manual_seed_all(my_seed)
            time_start = time.time()
            model = TimeXModel(d_inp=val_tuple[0].shape[-1], max_len=val_tuple[0].shape[0], n_classes=n_classes, 
                        n_prototypes=50, gsat_r=0.2, transformer_args=targs, ablation_parameters=abl, 
                        loss_weight_dict=loss_weight_dict, masktoken_stats=(mu, std), tau=5.0)
            model.encoder_main.load_state_dict(torch.load(tencoder_path.format(split_no)))
            model.to(device)
            for p in model.encoder_main.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5)
            save_path = os.path.join(model_dir, "models", (naming_convention(args)[:-3] + f"_target={target_label}.pt")).format(split_no)
            train_mv6_consistency(model, optimizer=optimizer, train_loader=train_loader, clf_criterion=clf_criterion, sim_criterion=sim_criterion, 
                                    beta_exp=1.0, beta_sim=0.5, val_tuple=val_pred_tuple, val_reference_tuple=val_reference_tuple, 
                                    num_epochs=20, save_path=save_path, train_tuple=train_tuple, early_stopping=True, 
                                    selection_criterion=selection_criterion, label_matching=label_matching, 
                                    embedding_matching=embedding_matching, use_scheduler=True, target_label=target_label, **sc_expand_args)
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
            f1, _, cf_metrics = eval_mv4_cf(test_filtered, model, target_labels=target_labels, reference_tuple=reference_tuple)
            n_eval = int(test_filtered[2].shape[0])
            time_end_eval = time.time()
            print(f"Time taken for split {split_no}, target {target_label}: {time_end_eval - time_end_train:.4f} seconds")
            print(f'split={split_no}, target={target_label}, n_eval={n_eval}, test_f1={f1:.4f}, cf_metrics={cf_metrics}')
            print(f"Average time per evaluation: {(time_end_eval - time_end_train) / n_eval:.4f} seconds")
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
    parser.add_argument("--num_vis_cf", type=int, default=0)
    main(parser.parse_args())

"""

f1_macro: weighted_mean = 0.2138 +- 0.0505 (total_n=90115)
validity: weighted_mean = 0.3054 +- 0.0912 (total_n=90115)
confidence: weighted_mean = 0.3156 +- 0.0901 (total_n=90115)
sparsity: weighted_mean = 0.1094 +- 0.0021 (total_n=90115)
sparsity_count: weighted_mean = 78.7616 +- 1.4999 (total_n=90115)
proximity_l1: weighted_mean = 0.0775 +- 0.0191 (total_n=90115)
proximity_l2: weighted_mean = 0.2365 +- 0.0567 (total_n=90115)
plausibility: weighted_mean = 0.8572 +- 0.0090 (total_n=90115)

"""



"""
f1_macro: weighted_mean = 0.4405 +- 0.0553 (total_n=90115)
validity: weighted_mean = 0.7277 +- 0.0654 (total_n=90115)
confidence: weighted_mean = 0.6718 +- 0.0768 (total_n=90115)
sparsity: weighted_mean = 0.5018 +- 0.0011 (total_n=90115)
sparsity_count: weighted_mean = 361.2787 +- 0.7795 (total_n=90115)
proximity_l1: weighted_mean = 0.1318 +- 0.0468 (total_n=90115)
proximity_l2: weighted_mean = 0.1999 +- 0.0660 (total_n=90115)
plausibility: weighted_mean = 0.5792 +- 0.0261 (total_n=90115)

f1_macro: weighted_mean = 0.5043 +- 0.0553 (total_n=90115)
validity: weighted_mean = 0.9037 +- 0.0494 (total_n=90115)
confidence: weighted_mean = 0.8450 +- 0.0831 (total_n=90115)
sparsity: weighted_mean = 0.1012 +- 0.0026 (total_n=90115)
sparsity_count: weighted_mean = 72.8302 +- 1.8657 (total_n=90115)
proximity_l1: weighted_mean = 0.0662 +- 0.0135 (total_n=90115)
proximity_l2: weighted_mean = 0.2160 +- 0.0425 (total_n=90115)
plausibility: weighted_mean = 0.8658 +- 0.0091 (total_n=90115)


Epoch 50, Val F1 = 1.0000, Val Sparsity = 0.1042
Best Epoch: 7    Val Metric = 1.9394     Val F1 = 1.0000
Time taken for split 1, target 0: 1654.36 seconds
Time taken for split 1, target 0: 0.04 seconds

"""

