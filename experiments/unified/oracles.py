"""Rule-based oracles that decode the class from a synthetic series using the known generating
process (txai/synth_data). Used for the semantic-validity test of counterfactuals: a CF is
semantically valid if the oracle assigns the target class to it, i.e. the CF actually contains
the target-class pattern rather than an adversarial perturbation.
"""
import numpy as np

FS_MAP = {(0, 10): 0, (1, 10): 1, (0, 17): 2, (1, 17): 3}


def _runs(mask):
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def freqshape_factors(x, thr=1.0, sign_agree=0.75):
    """Returns (shape, period): shape 0=down / 1=up spikes, period 10 / 17; -1 if not identifiable."""
    x = np.asarray(x).reshape(-1)
    runs = _runs(np.abs(x) > thr)
    shape, period = -1, -1
    if len(runs) >= 1:
        signs = np.array([np.sign(x[a:b].mean()) for a, b in runs])
        up = (signs > 0).mean()
        if up >= sign_agree:
            shape = 1
        elif up <= 1 - sign_agree:
            shape = 0
    if len(runs) >= 2:
        centers = np.array([(a + b - 1) / 2 for a, b in runs])
        spacing = np.median(np.diff(centers))
        if 7 <= spacing < 13.5:
            period = 10
        elif 13.5 <= spacing <= 21:
            period = 17
    return shape, period


FS_FACTORS = {v: k for k, v in FS_MAP.items()}


def freqshape_oracle(x, thr=1.0, sign_agree=0.75):
    """x: (T,) or (T,1). Classes: (spike sign, period) with sign 0=down, 1=up and period 10/17."""
    shape, period = freqshape_factors(x, thr, sign_agree)
    if shape == -1 or period == -1:
        return -1
    return FS_MAP[(shape, period)]


def seqcomb_oracle(x, win=15, slope_thr=None):
    """x: (T, D). Class 0: no inserted ramp; 1: two decreasing ramps; 2: two increasing ramps;
    3: an increasing ramp followed (later in time) by a decreasing ramp."""
    x = np.asarray(x)
    T, D = x.shape
    t = np.arange(win) - (win - 1) / 2
    denom = (t ** 2).sum()
    found = []  # (start_time, sign)
    for d in range(D):
        s = np.convolve(x[:, d], t[::-1] / denom, mode='valid')  # least-squares slope of each window
        thr = slope_thr if slope_thr is not None else 0.25
        m = np.abs(s) > thr
        for a, b in _runs(m):
            if b - a >= 3:
                seg = s[a:b]
                found.append((a, np.sign(seg[np.argmax(np.abs(seg))]), d))
    if len(found) == 0:
        return 0
    found.sort()
    signs = [f[1] for f in found]
    if all(sg < 0 for sg in signs):
        return 1
    if all(sg > 0 for sg in signs):
        return 2
    if signs[0] > 0 and signs[-1] < 0:
        return 3
    return -1
