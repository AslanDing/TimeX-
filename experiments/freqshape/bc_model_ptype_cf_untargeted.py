"""
Untargeted counterfactual evaluation for FreqShape (TimeX++ BC model).

Mirrors `bc_model_ptype_cf.py`: same data prep, pseudo-labels from the frozen
encoder, model skeleton + checkpoint layout. Instead of training/evaluating per
`target_label`, loads the targeted checkpoints produced by that script and runs
`BestTargetCFModel` (per-instance best valid target by softmax confidence on
`pred_cf`), then `eval_mv4_cf(..., untargeted=True)`.

Shared ensemble logic: `txai.utils.predictors.cf_untargeted_ensemble`.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../timesynth-0.2.4')))

import torch
import argparse
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

torch.cuda.set_device(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from txai.models.run_model_utils import batch_forwards
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4_cf
from txai.utils.predictors.cf_untargeted_ensemble import (
    BestTargetCFModel,
    extract_state_dict,
    safe_torch_load,
)

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

my_seed = 42
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)

is_timex = False

if is_timex:
    from txai.models.bc_model4 import TimeXModel, AblationParameters, transformer_default_args
else:
    from txai.models.bc_model_cf import TimeXModel, AblationParameters, transformer_default_args


def naming_convention(args):
    """Same as bc_model_ptype_cf.py — targeted checkpoints: our_*_split_cf={split}_target={k}.pt"""
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

    if not is_timex:
        name = 'our_' + name

    if args.lam != 1.0:
        name = name[:-3] + '_lam_cf={}'.format(args.lam) + '.pt'

    return name


def targeted_checkpoint_path(args, split_i: int, target_label: int) -> str:
    model_suffix = naming_convention(args)
    model_suffix = model_suffix[:-3] + f'_target={target_label}.pt'
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
    spath = os.path.join(model_path, 'models', model_suffix)
    return spath.format(split_i)


def visualize_counterfactual_examples(out_dict, times, y_true, save_dir, max_examples=5):
    os.makedirs(save_dir, exist_ok=True)

    src = out_dict.get('smooth_src', None)
    cf = out_dict.get('X_cf', None)
    if src is None or cf is None:
        print('[vis] skip: missing smooth_src/X_cf (untargeted)')
        return

    src = src.detach().cpu()
    cf = cf.detach().cpu()
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
        axes[0].set_title(
            f'Untargeted CF | y_pred={int(y_cpu[b]) if y_cpu is not None else "?"} | sample={b}'
        )

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


def main(args):
    model_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
    if args.lstm:
        arch = 'lstm'
        tencoder_path = model_dir_path + '/models/Freqshape_lstm_split={}.pt'
    elif args.cnn:
        arch = 'cnn'
        tencoder_path = model_dir_path + '/models/Freqshape_cnn_split={}.pt'
    else:
        arch = 'transformer'
        tencoder_path = model_dir_path + '/models/Scomb_transformer_split={}.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    targs = transformer_default_args

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
        k: {
            'values': [],
            'weights': [],
            'weighted_sum': 0.0,
            'total_weight': 0.0,
        }
        for k in cf_metric_keys
    }
    n_classes = 4
    test_range = range(1, 6)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/FreqShape/'))
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))

    for i in test_range:
        D = process_Synth(split_no=i, device=device, base_path=base_path)

        val, test = D['val'], D['test']

        mu = D['train_loader'].X.mean(dim=1)
        std = D['train_loader'].X.std(unbiased=True, dim=1)

        targs['trans_dim_feedforward'] = 16
        targs['trans_dropout'] = 0.1
        targs['nlayers'] = 1
        targs['norm_embedding'] = False

        abl_params = AblationParameters(
            equal_g_gt=args.eq_ge,
            g_pret_equals_g=args.eq_pret,
            label_based_on_mask=True,
            ptype_assimilation=True,
            side_assimilation=True,
            use_ste=(not args.no_ste),
            archtype=arch,
        )
        loss_weight_dict = {
            'gsat': 1.0,
            'connect': 1.0,
        }

        probe = TimeXModel(
            d_inp=val[0].shape[-1],
            max_len=val[0].shape[0],
            n_classes=n_classes,
            n_prototypes=50,
            gsat_r=0.1,
            transformer_args=targs,
            ablation_parameters=abl_params,
            loss_weight_dict=loss_weight_dict,
            masktoken_stats=(mu, std),
        )
        probe.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
        probe.to(device)
        probe.eval()
        with torch.inference_mode():
            train_out = batch_forwards(probe, D['train_loader'].X, D['train_loader'].times, batch_size=64)
            train_y_pred = train_out['pred'].argmax(dim=1).to(device)
            val_out = batch_forwards(probe, val[0], val[1], batch_size=64)
            val_y_pred = val_out['pred'].argmax(dim=1).to(device)
            test_out = batch_forwards(probe, test[0], test[1], batch_size=64)
            test_y_pred = test_out['pred'].argmax(dim=1).to(device)
        del probe

        val_pred_tuple = (val[0], val[1], val_y_pred)
        val_reference_tuple = (val[0], val[1], val[2])

        for target_label in [0, 1, 2, 3]:
            test_target_mask = test_y_pred != target_label
            if test_target_mask.sum() > 0:
                true_target = D['test'][2].to(device) == target_label
                acc = (
                    test_y_pred[test_target_mask]
                    == D['test'][2].to(device)[test_target_mask]
                ).float().mean().item()
                print(
                    f'Test accuracy for target_label={target_label}: {acc:.4f} '
                    f'({int(test_target_mask.sum().item())} samples)'
                )
            else:
                print(f'No test samples predicted as target_label={target_label}')

        def _new_model():
            m = TimeXModel(
                d_inp=val[0].shape[-1],
                max_len=val[0].shape[0],
                n_classes=n_classes,
                n_prototypes=50,
                gsat_r=0.1,
                transformer_args=targs,
                ablation_parameters=abl_params,
                loss_weight_dict=loss_weight_dict,
                masktoken_stats=(mu, std),
            )
            m.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
            m.to(device)
            if is_timex:
                m.init_prototypes(
                    train=(
                        D['train_loader'].X.to(device),
                        D['train_loader'].times.to(device),
                        train_y_pred,
                    )
                )
                if not args.ge_rand_init:
                    m.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))
            for param in m.encoder_main.parameters():
                param.requires_grad = False
            return m

        loaded_models = []
        loaded_targets = []
        for target_label in range(n_classes):
            spath = targeted_checkpoint_path(args, i, target_label)
            if not os.path.isfile(spath):
                print(f'split={i}, target={target_label}: missing checkpoint {spath}, skip.')
                continue
            model = _new_model()
            sdict = extract_state_dict(safe_torch_load(spath))
            model.load_state_dict(sdict)
            loaded_models.append(model)
            loaded_targets.append(int(target_label))
            print(f'split={i}, loaded targeted checkpoint for target={target_label}')

        if len(loaded_models) == 0:
            print(f'split={i}: no targeted checkpoints found, skip.')
            continue

        ensemble = BestTargetCFModel(loaded_models, loaded_targets).to(device)

        _, _, cf_metrics_val = eval_mv4_cf(
            val_pred_tuple,
            ensemble,
            reference_tuple=val_reference_tuple,
            target_labels=None,
            untargeted=True,
        )
        print(f'split={i}: val (untargeted) cf_metrics={cf_metrics_val}')

        test_X, test_T = test[0], test[1]
        test_tuple_pred = (test_X, test_T, test_y_pred)
        reference_tuple = (test_X, test_T, test[2])

        n_eval = int(test_y_pred.shape[0])
        f1, out_dict, cf_metrics = eval_mv4_cf(
            test_tuple_pred,
            ensemble,
            reference_tuple=reference_tuple,
            target_labels=None,
            untargeted=True,
        )
        print(
            f'split={i}, n_eval={n_eval}, test_f1={f1:.4f}, cf_metrics={cf_metrics}, '
            f'ensemble_targets={loaded_targets}'
        )

        if args.num_vis_cf > 0:
            vis_dir = os.path.join(
                model_path, 'cf_visualizations', f'split={i}', 'untargeted'
            )
            visualize_counterfactual_examples(
                out_dict=out_dict,
                times=test_T,
                y_true=test_y_pred,
                save_dir=vis_dir,
                max_examples=args.num_vis_cf,
            )

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
    ablations.add_argument('--eq_ge', action='store_true', help='G = G_E')
    ablations.add_argument('--eq_pret', action='store_true', help='G_pret = G')
    ablations.add_argument('--ge_rand_init', action='store_true', help="Randomly initialized G_E, i.e. don't copy")
    ablations.add_argument('--no_ste', action='store_true', help='Does not use STE')
    ablations.add_argument('--simclr', action='store_true', help='Uses SimCLR loss instead of consistency loss')
    ablations.add_argument('--no_la', action='store_true', help='No label alignment - just consistency loss')
    ablations.add_argument('--no_con', action='store_true', help='No consistency loss - just label')
    ablations.add_argument('--lstm', default=False, action='store_true')
    ablations.add_argument('--cnn', default=False, action='store_true')

    parser.add_argument('--r', type=float, default=0.5, help='r for GSAT loss')
    parser.add_argument('--lam', type=float, default=1.0, help='lambda between label alignment and consistency loss')
    parser.add_argument('--num_vis_cf', type=int, default=5, help='number of counterfactual visualization examples per split')

    args = parser.parse_args()

    main(args)
"""
f1_macro: weighted_mean = 0.0000 +- 0.0000 (total_n=5000)
validity: weighted_mean = 1.0000 +- 0.0000 (total_n=5000)
confidence: weighted_mean = 0.9993 +- 0.0001 (total_n=5000)
sparsity: weighted_mean = 0.2334 +- 0.0092 (total_n=5000)
sparsity_count: weighted_mean = 11.6678 +- 0.4584 (total_n=5000)
proximity_l1: weighted_mean = 0.4200 +- 0.0160 (total_n=5000)
proximity_l2: weighted_mean = 0.8717 +- 0.0203 (total_n=5000)
plausibility: weighted_mean = 0.9744 +- 0.0015 (total_n=5000)

"""


