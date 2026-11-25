import os
import sys

# Add the parent directory of TimeXplusplus to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../timesynth-0.2.4')))

import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.trainers.train_transformer import train
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.utils.predictors import eval_mvts_transformer
from txai.synth_data.simple_spike import SpikeTrainDataset

torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 4,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

for i in range(1, 6):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/FreqShape/'))
    print(path)
    D = process_Synth(split_no = i, device = device, base_path = path)
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    model = TransformerMVTS(
        d_inp = val[0].shape[-1],
        max_len = val[0].shape[0],
        n_classes = 4,
        trans_dim_feedforward = 16,
        trans_dropout = 0.1,
        d_pe = 16,
        # aggreg = 'mean',
        # norm_embedding = True
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.1)
    
    spath = os.path.abspath(os.path.join(os.path.dirname(__file__),'./models/Scomb_transformer_split={}.pt'.format(i)))

    model, loss, auc = train(
        model,
        train_loader,
        val_tuple = val, 
        n_classes = 4,
        num_epochs = 100,
        save_path = spath,
        optimizer = optimizer,
        show_sizes = False,
        use_scheduler = False,
    )

    f1 = eval_mvts_transformer(test, model)
    print('Test F1: {:.4f}'.format(f1))