"""Rows of TimeX_cf++ for the main counterfactual tables under the final protocol.
Aggregation identical to the original scripts: weighted mean over (split, target) groups with weights = number
of evaluated instances, and weighted standard error with the effective sample size.
Writes results/main_rows.json, results/tex_final/{cf_ablation_body,cf_models_rows,nested_rows}.tex and prints LaTeX rows:
python aggregate_cf_local.py"""
import os, re, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_D = os.environ.get('TIMEX_DIST', 'kl'); _W = float(os.environ.get('TIMEX_DIST_W', '0.1'))
RES = os.path.join(HERE, 'results' + ('' if _D == 'kl' else f'_{_D}' + ('' if _W == 0.1 else f'_w{_W:g}')))
KEYS = ['validity', 'confidence', 'sparsity', 'prox_l1', 'prox_l2']
DSS = ['freqshape', 'seqcomb_uv', 'seqcomb_mv', 'lowvar', 'mitecg', 'epilepsy']
ABL = ['no_ste', 'no_lc', 'no_kl', 'no_bound', 'no_con']
ABL_SPLITS = (1, 2, 3)


def _split(f):
    return int(re.search(r'split=(\d+)', os.path.basename(f)).group(1))


def select_by_val(tags, ds, variant='learned'):
    """Per (split, target), keep the run with the highest validation confidence among several training
    configurations (e.g. learning rates); returns the selected files and how often each tag was chosen."""
    best = {}
    for tag in tags:
        for f in glob.glob(os.path.join(RES, tag, ds, f'split=*_target=*_{variant}.json')):
            s = json.load(open(f))['summary']
            vc = s.get('best_val_conf', float('nan'))
            key = os.path.basename(f)
            if key not in best or (vc == vc and vc > best[key][1]):
                best[key] = (f, vc, tag)
    counts = {t: sum(1 for v in best.values() if v[2] == t) for t in tags}
    return sorted(v[0] for v in best.values()), counts


def wstats(tag, ds, variant, splits=None, files=None):
    fs = files if files is not None else sorted(glob.glob(os.path.join(RES, tag, ds, f'split=*_target=*_{variant}.json')))
    if splits is not None:
        fs = [f for f in fs if _split(f) in splits]
    if not fs:
        return None
    J = [json.load(open(f)) for f in fs]
    S = [j['summary'] for j in J]
    w = np.array([s['n'] for s in S], dtype=float)
    out = {'n_groups': len(S), 'n': int(w.sum()), 'splits': sorted(set(s['split'] for s in S)),
           'max_epochs': sorted(set(str(s.get('max_epochs', j.get('args', {}).get('max_epochs', '?'))) for s, j in zip(S, J))),
           'restarts': sorted(set(str(s.get('restarts', j.get('args', {}).get('restarts', '?'))) for s, j in zip(S, J)))}
    for k in KEYS + ['prox_l2_valid', 'sparsity_1e-2']:
        v = np.array([s.get(k, np.nan) for s in S], dtype=float)
        ok = ~np.isnan(v)
        if not ok.any():
            continue
        m = float((v[ok] * w[ok]).sum() / w[ok].sum())
        sd = float(np.sqrt(np.average((v[ok] - m) ** 2, weights=w[ok])))
        eff = w[ok].sum() ** 2 / (np.square(w[ok]).sum() + 1e-12)
        out[k] = (m, sd / np.sqrt(max(eff, 1.0)))
    return out


def untargeted(ds, rule):
    fs = glob.glob(os.path.join(RES, 'main', ds, 'split=*_target=*_learned.json'))
    if not fs:
        return None
    per = {}
    for f in fs:
        j = json.load(open(f)); s = j['summary']['split']; pi = j['per_instance']
        for n, i in enumerate(j['idx']):
            per.setdefault(s, {}).setdefault(i, []).append((pi['valid'][n], pi['conf'][n], pi['spars']['0.0001'][n], pi['prox_l1'][n], pi['prox_l2'][n]))
    vals = []
    for s, inst in sorted(per.items()):
        ch = []
        for c in inst.values():
            okc = [x for x in c if x[0] >= 0.5]
            ch.append(max(c, key=lambda x: x[1]) if (rule == 'maxconf' or not okc) else min(okc, key=lambda x: x[4]))
        vals.append(np.mean(ch, 0))
    vals = np.array(vals)
    return {k: (float(vals[:, i].mean()), float(vals[:, i].std() / np.sqrt(len(vals)))) for i, k in enumerate(KEYS)} | {'n_splits': len(vals)}


def cells(st, keys=KEYS):
    if st is None:
        return ' & '.join(['--'] * len(keys))
    return ' & '.join(f'{st[k][0]:.3f} \\std{{ {st[k][1]:.3f} }}' for k in keys)


def row(name, st, keys=KEYS):
    return f'{name} & ' + cells(st, keys)


ABL_NAMES = {'no_ste': 'w/o STE', 'no_lc': r'w/o $\mathcal{L}_{\mathrm{LC}}$',
             'no_kl': r'w/o $\mathcal{L}_{\mathrm{MMD}}$' if _D == 'mmd' else r'w/o $\mathcal{L}_{\mathrm{KL}}$',
             'no_bound': r'w/o $\mathcal{L}_{bound}$', 'no_con': r'w/o $\mathcal{L}_{\text{con}}$'}
DS_NAMES = {'freqshape': 'FreqShapes', 'seqcomb_uv': 'SeqComb-UV', 'seqcomb_mv': 'SeqComb-MV', 'lowvar': 'LowVar',
            'mitecg': 'ECG', 'epilepsy': 'Epilepsy'}


def tex_files(out):
    d = os.path.join(RES, 'tex_final'); os.makedirs(d, exist_ok=True)
    L = []
    for j, ds in enumerate(['freqshape', 'seqcomb_mv', 'mitecg']):
        if j:
            L += [r'         \midrule', r'         \midrule']
        L.append(r'          \multicolumn{6}{c}{\textbf{%s}} \\' % DS_NAMES[ds])
        L.append(r'         \rowcolor{mygreen} {\ourscf} & ' + cells(out.get(f'abl_main/{ds}')) + r' \\')
        for i, a in enumerate(ABL):
            pre = r'         \rowcolor{gray!15} ' if i % 2 else '         '
            L.append(pre + ABL_NAMES[a] + ' & ' + cells(out.get(f'abl_{a}/{ds}/learned')) + r' \\')
    open(os.path.join(d, 'cf_ablation_body.tex'), 'w').write('\n'.join(L) + '\n')
    M = []
    for ds in ['freqshape', 'seqcomb_mv', 'mitecg']:
        pick = {}
        for arch in ['lstm', 'cnn']:
            for t in [f'arch_{arch}_sel', f'arch_{arch}500', f'arch_{arch}']:
                if out.get(f'{t}/{ds}/learned'):
                    pick[arch] = t; break
        lstm, cnn = out.get(f'{pick.get("lstm")}/{ds}/learned'), out.get(f'{pick.get("cnn")}/{ds}/learned')
        M.append(f'% {ds}: lstm={pick.get("lstm")} cnn={pick.get("cnn")}')
        M.append(r'          \rowcolor{mygreen} {\ourscf} & ' + cells(lstm) + ' & ' + cells(cnn) + r' \\')
    open(os.path.join(d, 'cf_models_rows.tex'), 'w').write('\n'.join(M) + '\n')
    N = []
    for ds in DSS:
        N.append(f'% {ds} learned vs nested')
        N.append(row(r'{\ourscf}', out.get(f'main/{ds}/learned')) + r' \\')
        N.append(row(r'{\ourscf} (nested)', out.get(f'main/{ds}/nested')) + r' \\')
    open(os.path.join(d, 'nested_rows.tex'), 'w').write('\n'.join(N) + '\n')
    T = [r'\begin{table}[t]', r'\centering', r'\setlength{\tabcolsep}{3pt}',
         r'\caption{Targeted counterfactuals of {\ourscf} with the learned mask and with the nested mask, which restricts the learned mask to the $\lceil 2r\,T d\rceil$ coordinates with the highest attribution probability $\bm{\pi}$ of {\oursa}. Same protocol as Table~\ref{tab:cf_target}; mean $\pm$ standard error over five splits and all target classes.}',
         r'\label{tab:nested}', r'\maxsizebox{\linewidth}{!}{', r'\begin{tabular}{ll|ccccc}', r'\toprule',
         r'\textbf{Dataset} & \textbf{Mask} & \textbf{Validity} & \textbf{Confidence} & \textbf{Sparsity} & \textbf{Prox.-L1} & \textbf{Prox.-L2} \\', r'\midrule']
    for i, ds in enumerate(DSS):
        if i:
            T.append(r'\midrule')
        T.append(f'\\multirow{{2}}{{*}}{{{DS_NAMES[ds]}}} & learned & ' + cells(out.get(f'main/{ds}/learned')) + r' \\')
        T.append(' & nested & ' + cells(out.get(f'main/{ds}/nested')) + r' \\')
    T += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(os.path.join(d, 'rev_nested.tex'), 'w').write('\n'.join(T) + '\n')


if __name__ == '__main__':
    out = {}
    for ds in DSS:
        for tag, var in [('main', 'learned'), ('main', 'nested'), ('arch_cnn', 'learned'), ('arch_lstm', 'learned'),
                         ('arch_cnn500', 'learned'), ('arch_lstm500', 'learned')]:
            st = wstats(tag, ds, var)
            if st is not None:
                out[f'{tag}/{ds}/{var}'] = st
        for arch in ['cnn', 'lstm']:  # SeqComb-MV: learning rate selected per (split, target) on validation confidence
            fs, counts = select_by_val([f'arch_{arch}500', f'arch_{arch}_lr2e-3'], ds)
            if fs and counts.get(f'arch_{arch}_lr2e-3'):
                st = wstats(None, ds, 'learned', files=fs)
                st['selected_counts'] = counts
                out[f'arch_{arch}_sel/{ds}/learned'] = st
            for t in [f'arch_{arch}_lr2e-3', f'arch_{arch}_orig']:
                st = wstats(t, ds, 'learned')
                if st is not None:
                    out[f'{t}/{ds}/learned'] = st
        st = wstats('main', ds, 'learned', ABL_SPLITS)
        if st is not None:
            out[f'abl_main/{ds}'] = st
        for a in ABL:
            st = wstats(f'abl_{a}', ds, 'learned', ABL_SPLITS)
            if st is not None:
                out[f'abl_{a}/{ds}/learned'] = st
        for rule in ['maxconf', 'minl2']:
            u = untargeted(ds, rule)
            if u is not None:
                out[f'untargeted_{rule}/{ds}'] = u
    json.dump(out, open(os.path.join(RES, 'main_rows.json'), 'w'), indent=1)
    tex_files(out)
    for k, st in out.items():
        print(f'% {k}  groups={st.get("n_groups", "")} splits={st.get("splits", st.get("n_splits"))} '
              f'ep={st.get("max_epochs", "")} rs={st.get("restarts", "")} spars1e-2={st.get("sparsity_1e-2", ("", ""))[0]} l2v={st.get("prox_l2_valid", ("", ""))[0]}')
        print(row(k, st) + ' \\\\')
