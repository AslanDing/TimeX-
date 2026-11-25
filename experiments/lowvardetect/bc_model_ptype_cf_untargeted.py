"""
Untargeted CF eval for LowVarDetect. Splits 1,2,3,5 match bc_model_ptype_cf.py.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../timesynth-0.2.4')))

import torch
import argparse
import numpy as np

torch.cuda.set_device(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from txai.models.run_model_utils import batch_forwards
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4_cf
from txai.utils.predictors.cf_untargeted_ensemble import BestTargetCFModel, extract_state_dict, safe_torch_load
from txai.models.bc_model_cf import TimeXModel, AblationParameters, transformer_default_args

import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def naming_convention_targeted(args):
    name = 'bc_full_split_cf={}.pt'
    if args.eq_ge:
        name = 'bc_eqge_split_cf={}.pt'
    elif args.eq_pret:
        name = 'bc_eqpret_split_cf={}.pt'
    elif args.ge_rand_init:
        name = 'bc_gerand_split_cf={}.pt'
    elif args.no_ste:
        name = 'bc_noste_split_cf={}.pt'
    elif args.simclr:
        name = 'bc_simclr_split_cf={}.pt'
    elif args.no_la:
        name = 'bc_nola_split_cf={}.pt'
    elif args.no_con:
        name = 'bc_nocon_split_cf={}.pt'
    name = 'our_' + name
    if args.lam != 1.0:
        name = name[:-3] + f'_lam_cf={args.lam}.pt'
    return name


def targeted_checkpoint_path(args, split_i: int, target_label: int) -> str:
    model_suffix = naming_convention_targeted(args)
    model_suffix = model_suffix[:-3] + f'_target={target_label}.pt'
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
    return os.path.join(model_path, 'models', model_suffix).format(split_i)


def main(args):
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
    tencoder_path = model_dir + '/models/transformer_new2_split={}.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = 4
    targs = dict(transformer_default_args)

    cf_metric_keys = [
        'f1_macro',
        'validity',
        'confidence',
        'sparsity',
        'sparsity_count',
        'proximity_l1',
        'proximity_l2',
        'plausibility',
    ]
    all_results = {
        k: {'values': [], 'weights': [], 'weighted_sum': 0.0, 'total_weight': 0.0} for k in cf_metric_keys
    }
    base_path = os.path.abspath(os.path.dirname(__file__) + '../../../datasets/LowVarDetect')

    for split_no in [1, 2, 3, 5]:
        d = process_Synth(split_no=split_no, device=device, base_path=base_path)
        train_tuple = (d['train_loader'].X, d['train_loader'].times, d['train_loader'].y)
        val, test = d['val'], d['test']
        mu = train_tuple[0].mean(dim=1)
        std = train_tuple[0].std(unbiased=True, dim=1)
        targs.update({'trans_dim_feedforward': 32, 'trans_dropout': 0.1, 'nlayers': 1, 'norm_embedding': True})
        abl = AblationParameters(
            equal_g_gt=args.eq_ge,
            g_pret_equals_g=args.eq_pret,
            label_based_on_mask=True,
            ptype_assimilation=True,
            side_assimilation=True,
            use_ste=(not args.no_ste),
            archtype='transformer',
        )
        probe = TimeXModel(
            d_inp=val[0].shape[-1],
            max_len=val[0].shape[0],
            n_classes=n_classes,
            n_prototypes=50,
            gsat_r=0.1,
            transformer_args=targs,
            ablation_parameters=abl,
            loss_weight_dict={'gsat': 1.0, 'connect': 2.0},
            masktoken_stats=(mu, std),
            tau=1.0,
        )
        probe.encoder_main.load_state_dict(torch.load(tencoder_path.format(split_no)))
        probe.to(device)
        probe.eval()
        with torch.inference_mode():
            val_out = batch_forwards(probe, val[0], val[1], batch_size=64)
            val_y_pred = val_out['pred'].argmax(dim=1).to(device)
            test_out = batch_forwards(probe, test[0], test[1], batch_size=64)
            test_y_pred = test_out['pred'].argmax(dim=1).to(device)
        del probe

        val_pred_tuple = (val[0], val[1], val_y_pred)
        val_reference_tuple = (val[0], val[1], val[2])

        for target_label in range(n_classes):
            test_target_mask = test_y_pred != target_label
            if test_target_mask.sum() > 0:
                acc = (
                    test_y_pred[test_target_mask]
                    == test[2].to(device)[test_target_mask]
                ).float().mean().item()
                print(
                    f'Test accuracy for target_label={target_label}: {acc:.4f} '
                    f'({int(test_target_mask.sum().item())} samples)'
                )
            else:
                print(f'No test samples predicted as target_label={target_label}')

        reference_tuple_train = (
            train_tuple[0].to(device),
            train_tuple[1].to(device),
            train_tuple[2].to(device),
        )

        def _new_model():
            m = TimeXModel(
                d_inp=val[0].shape[-1],
                max_len=val[0].shape[0],
                n_classes=n_classes,
                n_prototypes=50,
                gsat_r=0.1,
                transformer_args=targs,
                ablation_parameters=abl,
                loss_weight_dict={'gsat': 1.0, 'connect': 2.0},
                masktoken_stats=(mu, std),
                tau=1.0,
            )
            m.encoder_main.load_state_dict(torch.load(tencoder_path.format(split_no)))
            m.to(device)
            for p in m.encoder_main.parameters():
                p.requires_grad = False
            return m

        loaded_models = []
        loaded_targets = []
        for target_label in range(n_classes):
            spath = targeted_checkpoint_path(args, split_no, target_label)
            if not os.path.isfile(spath):
                print(f'split={split_no}, target={target_label}: missing {spath}, skip.')
                continue
            model = _new_model()
            model.load_state_dict(extract_state_dict(safe_torch_load(spath)))
            loaded_models.append(model)
            loaded_targets.append(int(target_label))

        if len(loaded_models) == 0:
            print(f'split={split_no}: no checkpoints, skip.')
            continue

        model = BestTargetCFModel(loaded_models, loaded_targets).to(device)
        _, _, cf_metrics_val = eval_mv4_cf(
            val_pred_tuple,
            model,
            reference_tuple=val_reference_tuple,
            target_labels=None,
            untargeted=True,
        )
        print(f'split={split_no}: val cf_metrics={cf_metrics_val}')

        test_tuple_pred = (test[0], test[1], test_y_pred)
        reference_tuple_test = (test[0], test[1], test[2])
        n_eval = int(test_y_pred.shape[0])
        f1, _, cf_metrics = eval_mv4_cf(
            test_tuple_pred,
            model,
            reference_tuple=reference_tuple_test,
            target_labels=None,
            untargeted=True,
        )
        print(f'split={split_no}, n_eval={n_eval}, test_f1={f1:.4f}, cf_metrics={cf_metrics}, targets={loaded_targets}')

        metric_values = {'f1_macro': f1, **cf_metrics}
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ablations = parser.add_mutually_exclusive_group()
    ablations.add_argument('--eq_ge', action='store_true')
    ablations.add_argument('--eq_pret', action='store_true')
    ablations.add_argument('--ge_rand_init', action='store_true')
    ablations.add_argument('--no_ste', action='store_true')
    ablations.add_argument('--simclr', action='store_true')
    ablations.add_argument('--no_la', action='store_true')
    ablations.add_argument('--no_con', action='store_true')
    parser.add_argument('--lam', type=float, default=1.0)
    main(parser.parse_args())
"""
f1_macro: weighted_mean = 0.0000 +- 0.0000 (total_n=4000)
validity: weighted_mean = 1.0000 +- 0.0000 (total_n=4000)
confidence: weighted_mean = 0.9993 +- 0.0002 (total_n=4000)
sparsity: weighted_mean = 0.2205 +- 0.0487 (total_n=4000)
sparsity_count: weighted_mean = 88.1880 +- 19.4776 (total_n=4000)
proximity_l1: weighted_mean = 0.2313 +- 0.0141 (total_n=4000)
proximity_l2: weighted_mean = 0.5516 +- 0.0231 (total_n=4000)
plausibility: weighted_mean = 0.9819 +- 0.0038 (total_n=4000)


f1_macro: weighted_mean = 0.0000 +- 0.0000 (total_n=4000)
validity: weighted_mean = 1.0000 +- 0.0000 (total_n=4000)
confidence: weighted_mean = 0.9965 +- 0.0018 (total_n=4000)
sparsity: weighted_mean = 0.1052 +- 0.0022 (total_n=4000)
sparsity_count: weighted_mean = 42.0908 +- 0.8646 (total_n=4000)
proximity_l1: weighted_mean = 0.0923 +- 0.0031 (total_n=4000)
proximity_l2: weighted_mean = 0.3131 +- 0.0053 (total_n=4000)
plausibility: weighted_mean = 0.9919 +- 0.0015 (total_n=4000)
"""
