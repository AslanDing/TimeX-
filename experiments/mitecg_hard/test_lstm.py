import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../timesynth-0.2.4')))

import numpy as np
import torch

from txai.models.encoders.simple import LSTM
from txai.utils.data.preprocess import process_MITECG
from txai.mvts_test_metrics import eval_accuracy_macro_f1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MITECG_KWARGS = dict(
    hard_split=True,
    normalize=False,
    balance_classes=False,
    div_time=False,
    need_binarize=True,
    exclude_pac_pvc=True,
)


def main():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/MITECG'))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))

    test_accs, test_f1s = [], []
    for i in range(1, 6):
        ckpt = os.path.join(models_dir, 'MITECG-Hard_lstm_split={}.pt'.format(i))
        if not os.path.isfile(ckpt):
            raise FileNotFoundError('Missing checkpoint: {}'.format(ckpt))

        _, val, test, _ = process_MITECG(
            split_no=i, device=device, base_path=dataset_path, **MITECG_KWARGS
        )
        val = (val.X, val.time, val.y)
        test = (test.X, test.time, test.y)

        model = LSTM(d_inp=val[0].shape[-1], n_classes=2)
        model.to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        acc, f1 = eval_accuracy_macro_f1(test, model, batch_size=32)
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
