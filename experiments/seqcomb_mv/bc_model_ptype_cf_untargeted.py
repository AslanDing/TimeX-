"""
Untargeted CF eval for SeqCombMV2 (seqcomb_mv). See cf_untargeted_ensemble.
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

torch.cuda.set_device(3)

from txai.models.run_model_utils import batch_forwards
from txai.utils.data import process_Synth
from txai.synth_data.simple_spike import SpikeTrainDataset
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


def visualize_counterfactual_examples(out_dict, times, y_true, save_dir, max_examples=5):
    os.makedirs(save_dir, exist_ok=True)
    src = out_dict.get('smooth_src', None)
    cf = out_dict.get('X_cf', None)
    if src is None or cf is None:
        print('[vis] skip: missing smooth_src/X_cf')
        return
    src = src.detach().cpu()
    cf = cf.detach().cpu()
    times_cpu = times.detach().cpu() if torch.is_tensor(times) else None
    y_cpu = y_true.detach().cpu() if torch.is_tensor(y_true) else None
    num_examples = min(int(src.shape[1]), int(max_examples))
    for b in range(num_examples):
        t = times_cpu[:, b] if times_cpu is not None else torch.arange(src.shape[0], dtype=torch.float32)
        x = src[:, b, 0]
        xcf = cf[:, b, 0]
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(t.numpy(), x.numpy(), label='original')
        ax.plot(t.numpy(), xcf.numpy(), label='counterfactual')
        ax.set_title(f'y={int(y_cpu[b]) if y_cpu is not None else "?"} | sample={b}')
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'cf_example_{b}.png'), dpi=160)
        plt.close(fig)
    print(f'[vis] saved {num_examples} figure(s) to {save_dir}')


def main(args):
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
    if args.lstm:
        arch = 'lstm'
        tencoder_path = model_dir + '/models/ScombMV_lstm_split={}.pt'
    elif args.cnn:
        arch = 'cnn'
        tencoder_path = model_dir + '/models/ScombMV_cnn_split={}.pt'
    else:
        arch = 'transformer'
        tencoder_path = model_dir + '/models/transformer_split={}.pt'

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
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/SeqCombMV2'))

    for split_no in [1, 2, 3, 4, 5]:
        d = process_Synth(split_no=split_no, device=device, base_path=base_path)
        train_tuple = (d['train_loader'].X, d['train_loader'].times, d['train_loader'].y)
        val, test = d['val'], d['test']
        mu = train_tuple[0].mean(dim=1)
        std = train_tuple[0].std(unbiased=True, dim=1)
        targs.update({'trans_dim_feedforward': 128, 'trans_dropout': 0.25, 'nlayers': 2, 'norm_embedding': False})
        abl_params = AblationParameters(
            equal_g_gt=args.eq_ge,
            g_pret_equals_g=args.eq_pret,
            label_based_on_mask=True,
            ptype_assimilation=True,
            side_assimilation=True,
            use_ste=(not args.no_ste),
            archtype=arch,
        )
        probe = TimeXModel(
            d_inp=val[0].shape[-1],
            max_len=val[0].shape[0],
            n_classes=n_classes,
            n_prototypes=50,
            gsat_r=0.1,
            transformer_args=targs,
            ablation_parameters=abl_params,
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
                ablation_parameters=abl_params,
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

        test_X, test_T = test[0], test[1]
        test_tuple_pred = (test_X, test_T, test_y_pred)
        reference_tuple_test = (test_X, test_T, test[2])
        n_eval = int(test_y_pred.shape[0])
        f1, out_dict, cf_metrics = eval_mv4_cf(
            test_tuple_pred,
            model,
            reference_tuple=reference_tuple_test,
            target_labels=None,
            untargeted=True,
        )
        print(f'split={split_no}, n_eval={n_eval}, test_f1={f1:.4f}, cf_metrics={cf_metrics}, targets={loaded_targets}')

        if args.num_vis_cf > 0:
            vis_dir = os.path.join(model_dir, 'cf_visualizations', f'split={split_no}', 'from_targeted_best_per_instance')
            visualize_counterfactual_examples(out_dict, test_T, test_y_pred, vis_dir, args.num_vis_cf)

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
    parser.add_argument('--num_vis_cf', type=int, default=0)
    main(parser.parse_args())

"""
Transformer
f1_macro: weighted_mean = 0.0000 +- 0.0000 (total_n=5000)
validity: weighted_mean = 1.0000 +- 0.0000 (total_n=5000)
confidence: weighted_mean = 1.0000 +- 0.0000 (total_n=5000)
sparsity: weighted_mean = 0.3783 +- 0.0920 (total_n=5000)
sparsity_count: weighted_mean = 302.6310 +- 73.6256 (total_n=5000)
proximity_l1: weighted_mean = 0.1178 +- 0.0100 (total_n=5000)
proximity_l2: weighted_mean = 0.2912 +- 0.0078 (total_n=5000)
plausibility: weighted_mean = 0.9631 +- 0.0039 (total_n=5000)


"""


"""

f1_macro: weighted_mean = 0.0000 +- 0.0000 (total_n=5000)
validity: weighted_mean = 1.0000 +- 0.0000 (total_n=5000)
confidence: weighted_mean = 0.9999 +- 0.0001 (total_n=5000)
sparsity: weighted_mean = 0.0851 +- 0.0127 (total_n=5000)
sparsity_count: weighted_mean = 68.1098 +- 10.1621 (total_n=5000)
proximity_l1: weighted_mean = 0.0129 +- 0.0010 (total_n=5000)
proximity_l2: weighted_mean = 0.0462 +- 0.0044 (total_n=5000)
plausibility: weighted_mean = 0.9775 +- 0.0007 (total_n=5000)

f1_macro: weighted_mean = 0.0000 +- 0.0000 (total_n=5000)
validity: weighted_mean = 1.0000 +- 0.0000 (total_n=5000)
confidence: weighted_mean = 0.9989 +- 0.0007 (total_n=5000)
sparsity: weighted_mean = 0.1002 +- 0.0055 (total_n=5000)
sparsity_count: weighted_mean = 80.1830 +- 4.3677 (total_n=5000)
proximity_l1: weighted_mean = 0.0897 +- 0.0110 (total_n=5000)
proximity_l2: weighted_mean = 0.3360 +- 0.0225 (total_n=5000)
plausibility: weighted_mean = 0.9485 +- 0.0116 (total_n=5000)
"""

