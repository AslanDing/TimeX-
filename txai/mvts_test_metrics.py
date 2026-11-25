"""Accuracy and macro-F1 on MVTS (T, N, d) tuples; matches eval_mvts_transformer batching."""
import torch
from sklearn.metrics import accuracy_score, f1_score

from txai.models.encoders.simple import CNN, LSTM
from txai.models.run_model_utils import batch_forwards_TransformerMVTS


@torch.no_grad()
def eval_accuracy_macro_f1(test_tuple, model, batch_size=None):
    """
    test_tuple: (X, times, y) with X (T, N, d), times (T, N), y (N,).
    batch_size: if set, same semantics as eval_mvts_transformer (CNN/LSTM split on dim 1;
                Transformer uses batch_forwards_TransformerMVTS).
    """
    model.eval()
    X, times, y = test_tuple

    if batch_size is not None:
        if isinstance(model, (CNN, LSTM)):
            pred = torch.cat(
                [
                    model(xb, tb)
                    for xb, tb in zip(
                        torch.split(X, batch_size, dim=1),
                        torch.split(times, batch_size, dim=1),
                    )
                ],
                dim=0,
            )
        else:
            pred, _ = batch_forwards_TransformerMVTS(model, X, times, batch_size=batch_size)
    else:
        pred = model(X, times)

    y_true = y.cpu().numpy()
    y_pred = pred.argmax(dim=1).detach().cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1
