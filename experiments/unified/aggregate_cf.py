"""Aggregation of the temporal, target-conditioned editor (results/C1_temporal) and, when present, of the joint-training
runs (results/C2_joint).

Table-4/10 style rows: weighted mean over (split, target) groups with weights = number of evaluated instances and the
weighted standard error with the effective sample size (identical to aggregate_cf_local.wstats), plus per-split
values (instance-weighted over targets) and paired (split, target) comparisons with the local-editor runs (results/main).
Table-11 style (PAM): mean over instances within a split, then mean +- SE over splits (score_cf.summarize_pi).
Writes results/C1_temporal/{table4.csv, per_split.csv, paired.csv, pam.csv, train_time.csv, table12.csv, cf_temporal*.tex}
and results/C2_joint/{table10.csv, per_split.csv, paired.csv, attr.csv, cf_joint.tex}.
"""
import os, glob, json, re
import numpy as np, pandas as pd

R2 = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.join(R2, 'results')
C1 = os.path.join(R2, 'results', 'C1_temporal')
C2 = os.path.join(R2, 'results', 'C2_joint')
DSS = ['freqshape', 'seqcomb_uv', 'seqcomb_mv', 'lowvar', 'mitecg', 'epilepsy']
NAMES = {'freqshape': 'FreqShapes', 'seqcomb_uv': 'SeqComb-UV', 'seqcomb_mv': 'SeqComb-MV', 'lowvar': 'LowVar',
         'mitecg': 'ECG', 'epilepsy': 'Epilepsy', 'pam': 'PAM'}
KEYS = ['validity', 'confidence', 'sparsity', 'prox_l1', 'prox_l2', 'prox_l2_valid']
SRC = {'main_learned': (os.path.join(R1, 'main'), 'learned'), 'main_nested': (os.path.join(R1, 'main'), 'nested'),
       'c1_pertarget': (os.path.join(C1, 'main'), 'pertarget'), 'c1_cond': (os.path.join(C1, 'main'), 'cond'),
       'c1m_pertarget': (os.path.join(C1, 'matched'), 'pertarget'), 'c1m_cond': (os.path.join(C1, 'matched'), 'cond'),
       'c2_joint': (C2, 'joint')}
LABEL = {'main_learned': 'TimeX$_{cf}$++ (paper, learned mask)', 'main_nested': 'TimeX$_{cf}$++ (paper, nested mask)',
         'c1_pertarget': 'Temporal $\\psi_{cf}$, one model per target', 'c1_cond': 'Temporal $\\psi_{cf}$, one model for all targets',
         'c1m_pertarget': 'Temporal $\psi_{cf}$, one model per target, sparsity-matched selection',
         'c1m_cond': 'Temporal $\psi_{cf}$, one model for all targets, sparsity-matched selection',
         'c2_joint': 'Joint training, nested mask'}


def load(method, ds):
    d, v = SRC[method]
    out = []
    for f in sorted(glob.glob(os.path.join(d, ds, f'split=*_target=*_{v}.json'))):
        s = json.load(open(f))['summary']
        out.append(s)
    return out


def wstats(S):
    w = np.array([s['n'] for s in S], float)
    r = {'n_groups': len(S), 'n': int(w.sum()), 'n_splits': len({s['split'] for s in S})}
    for k in KEYS:
        v = np.array([s.get(k, np.nan) for s in S], float); ok = ~np.isnan(v)
        m = float((v[ok] * w[ok]).sum() / w[ok].sum())
        sd = float(np.sqrt(np.average((v[ok] - m) ** 2, weights=w[ok])))
        eff = w[ok].sum() ** 2 / (np.square(w[ok]).sum() + 1e-12)
        r[k] = m; r[k + '_se'] = sd / np.sqrt(max(eff, 1.0))
    return r


def per_split(S):
    rows = {}
    for s in S:
        rows.setdefault(s['split'], []).append(s)
    out = []
    for sp, ss in sorted(rows.items()):
        w = np.array([s['n'] for s in ss], float)
        out.append({'split': sp, 'n_targets': len(ss), **{k: float(np.nansum(np.array([s.get(k, np.nan) for s in ss], float) * w) / w.sum()) for k in KEYS}})
    return out


def paired(A, B):
    a = {(s['split'], s['target']): s for s in A}; b = {(s['split'], s['target']): s for s in B}
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    dv = np.array([a[k]['validity'] - b[k]['validity'] for k in keys])
    ds_ = np.array([a[k]['sparsity'] - b[k]['sparsity'] for k in keys])
    dl = np.array([a[k]['prox_l2'] - b[k]['prox_l2'] for k in keys])
    from scipy.stats import wilcoxon
    p = float(wilcoxon(dv).pvalue) if np.any(dv != 0) and len(dv) > 1 else np.nan
    return {'n_cells': len(keys), 'd_validity': float(dv.mean()), 'cells_valid_higher': int((dv > 0).sum()),
            'cells_valid_lower': int((dv < 0).sum()), 'wilcoxon_p_validity': p, 'd_sparsity': float(ds_.mean()),
            'cells_sparser': int((ds_ < 0).sum()), 'd_prox_l2': float(dl.mean())}


def fmt(m, se):
    return f'{m:.3f}\\std{{{se:.3f}}}'


def tex_table4(T, methods, caption, label, path):
    L = [r'\begin{table*}[t]', r'\centering', r'\setlength{\tabcolsep}{3pt}', r'\caption{' + caption + '}', r'\label{' + label + '}',
         r'\maxsizebox{\linewidth}{!}{', r'\begin{tabular}{ll|ccccc}', r'\toprule',
         r'\textbf{Dataset} & \textbf{Model} & \textbf{Validity} & \textbf{Confidence} & \textbf{Sparsity} & \textbf{Prox.-L1} & \textbf{Prox.-L2} \\', r'\midrule']
    for ds in DSS:
        rows = [T[(T.dataset == ds) & (T.method == m)] for m in methods]
        rows = [r.iloc[0] for r in rows if len(r)]
        for i, r in enumerate(rows):
            first = (r'\multirow{' + str(len(rows)) + '}{*}{' + NAMES[ds] + '}') if i == 0 else ''
            L.append(f"{first} & {LABEL[r.method]} & " + ' & '.join(fmt(r[k], r[k + '_se']) for k in ['validity', 'confidence', 'sparsity', 'prox_l1', 'prox_l2']) + r' \\')
        if rows and ds != DSS[-1]:
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}']
    open(path, 'w').write('\n'.join(L) + '\n')


def pam_rows():
    rows = []
    e7 = json.load(open(os.path.join(R1, 'e7', 'pam_summary.json')))
    for k, lab in [('comte', 'CoMTE'), ('abcf', 'AB-CF'), ('mcels', 'M-CELS'), ('confetti', 'CONFETTI'),
                   ('ours_paper_ckpt', 'paper ckpt (5 epochs, r=0.3)'), ('ours_e7_long', 'retrained (30 epochs, r=0.3)'),
                   ('ours_e7_long_r0.1', 'retrained (30 epochs, r=0.1)')]:
        if k in e7:
            v = e7[k]
            rows.append({'method': k, 'label': lab, 'source': 'local editor and baselines', **{m: v[a] for m, a in
                         [('validity', 'valid'), ('confidence', 'conf'), ('sparsity', 'spars_0.0001'), ('prox_l1', 'prox_l1'), ('prox_l2', 'prox_l2')]},
                         **{m + '_se': v[a + '_se'] for m, a in [('validity', 'valid'), ('confidence', 'conf'), ('sparsity', 'spars_0.0001'), ('prox_l1', 'prox_l1'), ('prox_l2', 'prox_l2')]},
                         'n_splits': v.get('n_splits')})
    for key, d, v, lab in [('c1_cond', os.path.join(C1, 'pam', 'pam'), 'cond', 'temporal psi_cf, one model for all targets'),
                           ('c1_pertarget', os.path.join(C1, 'pam', 'pam'), 'pertarget', 'temporal psi_cf, one model per target'),
                           ('c1m_cond', os.path.join(C1, 'pam_matched', 'pam'), 'cond', 'temporal psi_cf, one model for all targets, sparsity-matched'),
                           ('c1m_pertarget', os.path.join(C1, 'pam_matched', 'pam'), 'pertarget', 'temporal psi_cf, one model per target, sparsity-matched'),
                           ('c2_nested', os.path.join(C2, 'pam_nested', 'pam'), 'nested', 'nested mask (30 epochs, r=0.3)')]:
        by = {}
        for f in glob.glob(os.path.join(d, f'split=*_target=*_{v}.json')):
            j = json.load(open(f)); s = j['summary']['split']; pi = j['per_instance']
            by.setdefault(s, []).append(pi)
        if not by:
            continue
        per = []
        for s, pis in sorted(by.items()):
            cat = lambda k, sub=None: np.concatenate([np.asarray(p[k] if sub is None else p[k][sub], float) for p in pis])
            per.append([cat('valid').mean(), cat('conf').mean(), cat('spars', '0.0001').mean(), cat('prox_l1').mean(), cat('prox_l2').mean(), len(pis)])
        per = np.array(per)
        r = {'method': key, 'label': lab, 'source': 'temporal editor', 'n_splits': len(per), 'n_targets_total': int(per[:, 5].sum())}
        for i, m in enumerate(['validity', 'confidence', 'sparsity', 'prox_l1', 'prox_l2']):
            r[m] = float(per[:, i].mean()); r[m + '_se'] = float(per[:, i].std() / np.sqrt(len(per)))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    T, P, PR, TT = [], [], [], []
    for ds in DSS:
        S = {m: load(m, ds) for m in SRC}
        for m, ss in S.items():
            if ss:
                T.append({'dataset': ds, 'method': m, **wstats(ss)})
                P += [{'dataset': ds, 'method': m, **r} for r in per_split(ss)]
        for m in ['c1_pertarget', 'c1_cond', 'c1m_pertarget', 'c1m_cond', 'c2_joint', 'main_nested']:
            for ref in ['main_learned', 'main_nested']:
                if m == ref or not S[m] or not S[ref]:
                    continue
                r = paired(S[m], S[ref])
                if r:
                    PR.append({'dataset': ds, 'a': m, 'b': ref, **r})
        for m in ['c1_pertarget', 'c1_cond', 'c1m_pertarget', 'c1m_cond', 'c2_joint']:
            by = {}
            for s in S[m]:
                by.setdefault(s['split'], {})[s['target']] = s.get('train_time_s', np.nan)
            for sp, d in by.items():  # cond / joint: one training per split (same value for all targets)
                TT.append({'dataset': ds, 'method': m, 'split': sp,
                           'train_time_s': float(np.nanmax(list(d.values())) if not m.endswith('pertarget') else np.nansum(list(d.values())))})
        for sp in sorted({s['split'] for s in S['main_learned']}):
            TT.append({'dataset': ds, 'method': 'main_learned (incl. eval)', 'split': sp,
                       'train_time_s': float(np.nansum([s.get('time_s', np.nan) for s in S['main_learned'] if s['split'] == sp]))})
    T = pd.DataFrame(T); P = pd.DataFrame(P); PR = pd.DataFrame(PR); TT = pd.DataFrame(TT)
    os.makedirs(C1, exist_ok=True); os.makedirs(C2, exist_ok=True)
    c1m = ['main_learned', 'c1_pertarget', 'c1_cond', 'c1m_pertarget', 'c1m_cond']; c2m = ['main_learned', 'main_nested', 'c2_joint']
    T[T.method.isin(c1m)].to_csv(os.path.join(C1, 'table4.csv'), index=False)
    T[T.method.isin(c2m)].to_csv(os.path.join(C2, 'table10.csv'), index=False)
    if len(P):
        P[P.method.isin(c1m)].to_csv(os.path.join(C1, 'per_split.csv'), index=False)
        P[P.method.isin(c2m)].to_csv(os.path.join(C2, 'per_split.csv'), index=False)
    if len(PR):
        PR[PR.a.str.startswith('c1')].to_csv(os.path.join(C1, 'paired.csv'), index=False)
        PR[~PR.a.str.startswith('c1')].to_csv(os.path.join(C2, 'paired.csv'), index=False)
    if len(TT):
        TT.groupby(['dataset', 'method']).train_time_s.agg(['mean', 'count']).reset_index().to_csv(os.path.join(C1, 'train_time.csv'), index=False)
    pd.set_option('display.width', 250)
    if len(T):
        print(T[['dataset', 'method', 'n_groups', 'validity', 'validity_se', 'confidence', 'sparsity', 'prox_l1', 'prox_l2', 'prox_l2_valid']].round(3).to_string())
    if len(PR):
        print(PR.round(4).to_string())
    tex_table4(T, c1m, 'Targeted counterfactuals with the temporal editing generator. Protocol of Table~\\ref{tab:cf_target}; mean $\\pm$ standard error over five splits and all target classes.',
               'tab:cf_temporal', os.path.join(C1, 'cf_temporal.tex'))
    tex_table4(T, c2m, 'Joint training of the attribution and counterfactual modes with the nested mask. Protocol of Table~\\ref{tab:cf_target}; mean $\\pm$ standard error over five splits and all target classes.',
               'tab:cf_joint', os.path.join(C2, 'cf_joint.tex'))
    # PAM (Table 11 protocol)
    try:
        PAM = pam_rows(); PAM.to_csv(os.path.join(C1, 'pam.csv'), index=False)
        print(PAM[['method', 'n_splits', 'validity', 'confidence', 'sparsity', 'prox_l1', 'prox_l2']].round(3).to_string())
    except Exception as e:
        print('PAM:', e)
    # attribution of the joint model
    A = []
    for ds in DSS:
        for f in sorted(glob.glob(os.path.join(C2, ds, 'split=*_attr.json'))):
            j = json.load(open(f))
            if 'joint' in j:
                A.append({'dataset': ds, 'split': j['split'], **{f'joint_{k}': j['joint'][k] for k in ['auprc', 'aup', 'aur']},
                          **{f'separate_{k}': j['separate'][k] for k in ['auprc', 'aup', 'aur']},
                          'pi_mean_joint': j['pi_test_mean'], 'pi_mean_separate': j.get('separate_pi_test_mean')})
    if A:
        A = pd.DataFrame(A); A.to_csv(os.path.join(C2, 'attr_per_split.csv'), index=False)
        g = A.groupby('dataset')
        S_ = pd.DataFrame({'n_splits': g.split.count(),
                           **{f'{w}_{k}_mean': g[f'{w}_{k}'].mean() for w in ['joint', 'separate'] for k in ['auprc', 'aup', 'aur']},
                           **{f'{w}_{k}_sd': g[f'{w}_{k}'].std(ddof=0) for w in ['joint', 'separate'] for k in ['auprc', 'aup', 'aur']},
                           'splits_joint_higher_auprc': g.apply(lambda d: int((d.joint_auprc > d.separate_auprc).sum()))}).reset_index()
        S_.to_csv(os.path.join(C2, 'attr.csv'), index=False)
        print(S_.round(3).to_string())
    # Table 12 (FreqShapes)
    f = os.path.join(C1, 'semantic', 'freqshape_table12.json')
    if os.path.exists(f):
        J = json.load(open(f))
        rows = []
        for k, v in J.items():
            a = v['agg']
            rows.append({'model': k, **{m: a.get(m) for m in ['valid', 'oracle_valid', 'oracle_shape_ok', 'oracle_period_ok', 'valid_cnn', 'valid_lstm', 'lof_inlier', 'mmd2',
                                                              'spars_0.0001', 'prox_l2', 'oracle_valid_given_valid']}})
        D = pd.DataFrame(rows); D.to_csv(os.path.join(C1, 'table12.csv'), index=False)
        print(D.round(3).to_string())


if __name__ == '__main__':
    main()
