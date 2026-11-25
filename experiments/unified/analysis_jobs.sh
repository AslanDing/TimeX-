#!/bin/bash
# Further analyses with the temporal editor (train_cf_variants.py / untargeted_cf.py / analyze_cf.py): nested mask, ablations,
# mask sources, prior r sweep, CNN/LSTM black boxes. Writes jobs_d2_1.txt, jobs_d2_2.txt, jobs_d2_3.txt into the
# current directory (experiments/unified); run each with run_queue.py.
# usage: bash analysis_jobs.sh [<python>] [<python of a second machine>]   (both default to `python`)
PO=${1:-python}; PA=${2:-$PO}
O=jobs_d2_1.txt; A1=jobs_d2_2.txt; A3=jobs_d2_3.txt; : > $O; : > $A1; : > $A3
# main protocol (Table 4) budgets x (C-1):  ds:E:P:C
declare -A MAIN=( [seqcomb_uv]=450:45 [seqcomb_mv]=450:45 [epilepsy]=100:15 [lowvar]=450:45 [mitecg]=30:6 [freqshape]=300:45 )
# Table 9 / Fig. operating points budgets (e1/e4 jobs) x (C-1)
declare -A SUB=( [freqshape]=180:30 [seqcomb_mv]=240:30 [mitecg]=30:6 [seqcomb_uv]=240:30 [lowvar]=240:30 )
# ---- evaluation-only parts
printf "untargeted_cf\t$PO untargeted_cf.py\n" >> $O
printf "d2_robust\t$PO analyze_cf.py --part robust\n" >> $O
printf "d2_pi\t$PO analyze_cf.py --part pi\n" >> $O
printf "d2_timing\t$PO analyze_cf.py --part timing\n" >> $O
# ---- Table 10: nested (prior rule), main protocol, 5 splits
for ds in seqcomb_uv seqcomb_mv epilepsy lowvar mitecg freqshape; do IFS=: read E P <<< "${MAIN[$ds]}"; for s in 1 2 3 4 5; do
  printf "d2_nested_%s_%s\t$PO train_cf_variants.py --ds %s --split %s --tag nested --mask nested --nested_rule prior --data main --max_epochs %s --patience %s --restarts 2 --nsamples 5\n" $ds $s $ds $s $E $P >> $O; done; done
# ---- Table 18: ablations, main protocol, splits 1-3
for ds in seqcomb_mv mitecg freqshape; do IFS=: read E P <<< "${MAIN[$ds]}"; for ab in no_ste no_bound no_kl no_con no_lc; do for s in 1 2 3; do
  printf "d2_abl_%s_%s_%s\t$PO train_cf_variants.py --ds %s --split %s --tag abl_%s --abl %s --data main --max_epochs %s --patience %s --restarts 2 --nsamples 5\n" $ab $ds $s $ds $s $ab $ab $E $P >> $O; done; done; done
# ---- Table 9: mask sources, analysis subsets, splits 1-3 (paper datasets first)
for ds in seqcomb_mv mitecg freqshape; do IFS=: read E P <<< "${SUB[$ds]}"; for s in 1 2 3; do
  for m in learned attr shift shuffle rand full; do
    printf "d2_ms_%s_%s_%s\t$PO train_cf_variants.py --ds %s --split %s --tag ms --mask %s --data subset --max_epochs %s --patience %s --restarts 2 --nsamples 1\n" $m $ds $s $ds $s $m $E $P >> $O; done
  printf "d2_ms_nested_%s_%s\t$PO train_cf_variants.py --ds %s --split %s --tag ms --mask nested --nested_rule k --data subset --max_epochs %s --patience %s --restarts 2 --nsamples 1\n" $ds $s $ds $s $E $P >> $O
done; done
# ---- Figures: r sweep (operating points), analysis subsets, splits 1-3 (SeqComb-MV: 1 restart)
for ds in mitecg freqshape; do IFS=: read E P <<< "${SUB[$ds]}"; for r in 0.05 0.1 0.2 0.3 0.5; do for s in 1 2 3; do
  printf "d2_sweep_%s_r%s_%s\t$PO train_cf_variants.py --ds %s --split %s --tag sweep_r%s --r %s --data subset --max_epochs %s --patience %s --restarts 2 --nsamples 1 --first100\n" $ds $r $s $ds $s $r $r $E $P >> $O; done; done; done
for r in 0.05 0.1 0.2 0.3 0.5; do for s in 1 2 3; do
  printf "d2_sweep_seqcomb_mv_r%s_%s\t$PA train_cf_variants.py --ds seqcomb_mv --split %s --tag sweep_r%s --r %s --data subset --max_epochs 240 --patience 30 --restarts 1 --nsamples 1 --first100\n" $r $s $s $r $r >> $A3; done; done
# ---- Table 17: CNN / LSTM black boxes, main protocol
for arch in cnn lstm; do for s in 1 2 3 4 5; do
  printf "d2_arch_%s_mitecg_%s\t$PO train_cf_variants.py --ds mitecg --split %s --tag arch_%s --arch %s --data main --max_epochs 10 --patience 6 --restarts 2 --nsamples 5\n" $arch $s $s $arch $arch >> $O
  printf "d2_arch_%s_seqcomb_mv_%s\t$PA train_cf_variants.py --ds seqcomb_mv --split %s --tag arch_%s --arch %s --data main --max_epochs 450 --patience 45 --restarts 2 --nsamples 5\n" $arch $s $s $arch $arch >> $A1
  printf "d2_arch_%s_lr2e-3_seqcomb_mv_%s\t$PA train_cf_variants.py --ds seqcomb_mv --split %s --tag arch_%s_lr2e-3 --arch %s --lr 2e-3 --data main --max_epochs 450 --patience 45 --restarts 2 --nsamples 5\n" $arch $s $s $arch $arch >> $A1
  printf "d2_arch_%s_freqshape_%s\t$PA train_cf_variants.py --ds freqshape --split %s --tag arch_%s --arch %s --data main --max_epochs 300 --patience 45 --restarts 2 --nsamples 5\n" $arch $s $s $arch $arch >> $A1
done; done
# ---- Table 9 extension (A4 datasets): SeqComb-UV, LowVar
for ds in seqcomb_uv lowvar; do IFS=: read E P <<< "${SUB[$ds]}"; for s in 1 2 3; do
  for m in learned attr shift shuffle rand full; do
    printf "d2_ms_%s_%s_%s\t$PO train_cf_variants.py --ds %s --split %s --tag ms --mask %s --data subset --max_epochs %s --patience %s --restarts 2 --nsamples 1\n" $m $ds $s $ds $s $m $E $P >> $O; done
  printf "d2_ms_nested_%s_%s\t$PO train_cf_variants.py --ds %s --split %s --tag ms --mask nested --nested_rule k --data subset --max_epochs %s --patience %s --restarts 2 --nsamples 1\n" $ds $s $ds $s $E $P >> $O
done; done
wc -l $O $A1 $A3
