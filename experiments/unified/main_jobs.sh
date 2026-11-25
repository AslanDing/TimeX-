#!/bin/bash
# Job lists for the main results of TimeX++, 5 splits per dataset, with the budgets of the paper runs:
#   TimeX++-a  : attribution models (train_attr.py)
#   TimeX++-cf : counterfactuals with the temporal, target-conditioned editor, one model per split for all
#                target classes (train_cf.py --variant cond)
# Prerequisites: datasets under datasets/ and the black boxes under experiments/<dataset>/models/
# (experiments/<dataset>/train_transformer.py).
#
# usage (from experiments/unified):
#   bash main_jobs.sh [python]
#   python run_queue.py jobs_main_train.txt --gpus 0,1 --per_gpu 2   # training + evaluation of both explainers
#   python run_queue.py jobs_main_eval.txt  --gpus 0   --per_gpu 1   # untargeted CFs, occlusion, aggregated rows
cd "$(dirname "$0")"
PY=${1:-${PY:-python}}
TR=jobs_main_train.txt; EV=jobs_main_eval.txt; : > $TR; : > $EV

# ---- TimeX++-a (writes ckpt/, cache/, results/e0)
for ds in freqshape seqcomb_uv seqcomb_mv lowvar mitecg epilepsy pam; do for s in 1 2 3 4 5; do
  printf "attr_%s_%s\t%s train_attr.py --ds %s --split %s\n" $ds $s "$PY" $ds $s >> $TR
done; done

# ---- TimeX++-cf, targeted (writes ckpt/C1/main, results/C1_temporal/main)
# budget = per-target budget x (number of classes - 1)
declare -A B=( [freqshape]="300 45" [seqcomb_uv]="450 45" [seqcomb_mv]="450 45" [lowvar]="450 45" [mitecg]="30 6" [epilepsy]="100 15" )
for ds in freqshape seqcomb_uv seqcomb_mv lowvar mitecg epilepsy; do set -- ${B[$ds]}; for s in 1 2 3 4 5; do
  printf "cf_%s_%s\t%s train_cf.py --ds %s --split %s --variant cond --max_epochs %s --patience %s --restarts 2 --nsamples 5\n" \
    $ds $s "$PY" $ds $s $1 $2 >> $TR
done; done
# PAM (8 classes): first 50 eligible test instances per target, one mask sample (writes results/C1_temporal/pam)
for s in 1 2 3 4 5; do
  printf "cf_pam_%s\t%s train_cf.py --ds pam --split %s --variant cond --pam --tag pam --max_epochs 210 --patience 35 --restarts 2\n" \
    $s "$PY" $s >> $TR
done

# ---- after training
printf "attr_rows\t%s aggregate_attr.py\n" "$PY" >> $EV                 # AUPRC / AUP / AUR
printf "attr_occlusion\t%s occlusion_real.py --ds epilepsy,pam\n" "$PY" >> $EV   # real data, no ground truth
printf "cf_untargeted\t%s untargeted_cf.py\n" "$PY" >> $EV                                   # untargeted CFs
printf "cf_rows\t%s aggregate_cf.py\n" "$PY" >> $EV                                         # targeted CF rows + PAM
wc -l $TR $EV
