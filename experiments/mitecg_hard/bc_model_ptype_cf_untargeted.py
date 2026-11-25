"""
Untargeted CF eval for MITECG hard split. See cf_untargeted_ensemble.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../timesynth-0.2.4')))

import torch
import argparse
import numpy as np

torch.cuda.set_device(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from txai.models.run_model_utils import batch_forwards
from txai.utils.data.preprocess import process_MITECG
from txai.utils.predictors.eval import eval_mv4_cf
from txai.utils.predictors.cf_untargeted_ensemble import BestTargetCFModel, extract_state_dict, safe_torch_load
from txai.models.bc_model_cf import TimeXModel, AblationParameters, transformer_default_args

import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def naming_convention_targeted(args):
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
    elif args.cnn:
        name = 'bc_cnn_split_cf={}.pt'
    elif args.lstm:
        name = 'bc_lstm_split_cf={}.pt'
    else:
        name = 'bc_full_split_cf={}.pt'
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
    eval_bs = args.eval_batch_size
    if eval_bs is not None and eval_bs <= 0:
        eval_bs = None
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
    if args.lstm:
        arch = 'lstm'
        tencoder_path = model_dir + '/models/MITECG-Hard_lstm_split={}.pt'
    elif args.cnn:
        arch = 'cnn'
        tencoder_path = model_dir + '/models/MITECG-Hard_cnn_split={}.pt'
    else:
        arch = 'transformer'
        tencoder_path = model_dir + '/models/transformer_exc_split={}.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = 2
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
    mitecg_base = '/TimeX/datasets/MITECG/'
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./../../datasets/MITECG/"))
        
    for split_no in [1, 2, 3, 4, 5]:
        train_ecg, val, test, _ = process_MITECG(
            split_no=split_no, device=device, hard_split=True, need_binarize=True, base_path=base_path
        )
        # Keep full training tensors on CPU: plausibility only needs per-timestep mean/std over
        # the train split; holding (T x N_train x d) on GPU alongside val/test often causes OOM.
        train_tuple = (train_ecg.X.cpu(), train_ecg.time.cpu(), train_ecg.y.cpu())
        del train_ecg
        val_tuple = (val.X, val.time, val.y)
        test_tuple = (test.X, test.time, test.y)
        mu = train_tuple[0].mean(dim=1).to(device)
        std = train_tuple[0].std(unbiased=True, dim=1).to(device)
        targs.update(
            {'trans_dim_feedforward': 64, 'trans_dropout': 0.1, 'nlayers': 1, 'stronger_clf_head': False, 'norm_embedding': True}
        )
        abl = AblationParameters(
            equal_g_gt=args.eq_ge,
            g_pret_equals_g=args.eq_pret,
            label_based_on_mask=True,
            ptype_assimilation=True,
            side_assimilation=True,
            use_ste=(not args.no_ste),
            archtype=arch,
        )
        probe = TimeXModel(
            d_inp=val_tuple[0].shape[-1],
            max_len=val_tuple[0].shape[0],
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
        pred_bs = eval_bs if eval_bs is not None else 256
        with torch.inference_mode():
            val_out = batch_forwards(probe, val_tuple[0], val_tuple[1], batch_size=pred_bs)
            val_y_pred = val_out['pred'].argmax(dim=1).to(device)
            test_out = batch_forwards(probe, test_tuple[0], test_tuple[1], batch_size=pred_bs)
            test_y_pred = test_out['pred'].argmax(dim=1).to(device)
        del probe

        val_pred_tuple = (val_tuple[0], val_tuple[1], val_y_pred)
        val_reference_tuple = (val_tuple[0], val_tuple[1], val_tuple[2])

        for target_label in range(n_classes):
            test_target_mask = test_y_pred != target_label
            if test_target_mask.sum() > 0:
                acc = (
                    test_y_pred[test_target_mask]
                    == test_tuple[2].to(device)[test_target_mask]
                ).float().mean().item()
                print(
                    f'Test accuracy for target_label={target_label}: {acc:.4f} '
                    f'({int(test_target_mask.sum().item())} samples)'
                )
            else:
                print(f'No test samples predicted as target_label={target_label}')

        reference_tuple_train = train_tuple

        def _new_model():
            m = TimeXModel(
                d_inp=val_tuple[0].shape[-1],
                max_len=val_tuple[0].shape[0],
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
            batch_size=eval_bs,
        )
        print(f'split={split_no}: val cf_metrics={cf_metrics_val}')

        test_tuple_pred = (test_tuple[0], test_tuple[1], test_y_pred)
        reference_tuple_test = (test_tuple[0], test_tuple[1], test_tuple[2])
        n_eval = int(test_y_pred.shape[0])
        f1, _, cf_metrics = eval_mv4_cf(
            test_tuple_pred,
            model,
            reference_tuple=reference_tuple_test,
            target_labels=None,
            untargeted=True,
            batch_size=eval_bs,
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
    ablations.add_argument('--lstm', action='store_true')
    ablations.add_argument('--cnn', action='store_true')
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=256,
        help='Max batch size along the sample dimension for eval_mv4_cf; use <=0 for full-tensor eval (high VRAM).',
    )
    main(parser.parse_args())

"""
f1_macro: weighted_mean = 0.0482 +- 0.0060 (total_n=100012)
validity: weighted_mean = 0.9505 +- 0.0063 (total_n=100012)
confidence: weighted_mean = 0.9415 +- 0.0067 (total_n=100012)
sparsity: weighted_mean = 0.1129 +- 0.0233 (total_n=100012)
sparsity_count: weighted_mean = 40.6399 +- 8.3899 (total_n=100012)
proximity_l1: weighted_mean = 0.0834 +- 0.0148 (total_n=100012)
proximity_l2: weighted_mean = 0.2634 +- 0.0196 (total_n=100012)
plausibility: weighted_mean = 0.9832 +- 0.0006 (total_n=100012)

f1_macro: weighted_mean = 0.2112 +- 0.0930 (total_n=100012)
validity: weighted_mean = 0.6882 +- 0.1493 (total_n=100012)
confidence: weighted_mean = 0.6793 +- 0.1465 (total_n=100012)
sparsity: weighted_mean = 0.0540 +- 0.0191 (total_n=100012)
sparsity_count: weighted_mean = 19.4226 +- 6.8932 (total_n=100012)
proximity_l1: weighted_mean = 0.0839 +- 0.0408 (total_n=100012)
proximity_l2: weighted_mean = 0.3057 +- 0.0975 (total_n=100012)
plausibility: weighted_mean = 0.9673 +- 0.0105 (total_n=100012)

"""