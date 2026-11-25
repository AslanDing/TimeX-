import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../timesynth-0.2.4')))

import numpy as np
import torch

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.mvts_test_metrics import eval_accuracy_macro_f1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/FreqShape/'))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './models/'))

    test_accs, test_f1s = [], []

    for i in range(1, 6):
        ckpt = os.path.join(models_dir, 'Scomb_transformer_split={}.pt'.format(i))
        if not os.path.isfile(ckpt):
            raise FileNotFoundError('Missing checkpoint: {}'.format(ckpt))

        D = process_Synth(split_no=i, device=device, base_path=dataset_path)
        val, test = D['val'], D['test']

        model = TransformerMVTS(
            d_inp=val[0].shape[-1],
            max_len=val[0].shape[0],
            n_classes=4,
            trans_dim_feedforward=16,
            trans_dropout=0.1,
            d_pe=16,
        )
        model.to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        acc, f1 = eval_accuracy_macro_f1(test, model)
        test_accs.append(acc)
        test_f1s.append(f1)
        print('Split {} — Test accuracy: {:.4f}, macro-F1: {:.4f}'.format(i, acc, f1))

    print(
        'All splits — mean accuracy: {:.4f} (std {:.4f}), mean macro-F1: {:.4f} (std {:.4f})'.format(
            float(np.mean(test_accs)),
            float(np.std(test_accs)),
            float(np.mean(test_f1s)),
            float(np.std(test_f1s)),
        )
    )


if __name__ == '__main__':
    main()
