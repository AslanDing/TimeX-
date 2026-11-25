#!/bin/bash
# Final-protocol jobs of the local (coordinate-wise) editor: main CF table, nested variant, CNN/LSTM black boxes,
# CF ablations, plus the TimeX++-a prerequisites. Writes three job lists (one per machine) for run_queue.py.
cd "$(dirname "$0")"
ep(){ case $1 in freqshape) echo "--max_epochs 100 --patience 15";; seqcomb_uv|seqcomb_mv|lowvar) echo "--max_epochs 150 --patience 15";;
       mitecg) echo "--max_epochs 30 --patience 6";; epilepsy) echo "--max_epochs 100 --patience 15";; esac; }
nt(){ case $1 in mitecg|epilepsy) echo 2;; *) echo 4;; esac; }
COMMON="--main --no_agree --restarts 2 --nsamples 5"
rm -f jobs_fin_1.txt jobs_fin_2.txt jobs_fin_3.txt
# A: prerequisites (list 1)
for ds in seqcomb_uv lowvar epilepsy; do for s in 1 2 3 4 5; do
  printf "fA_e0_${ds}_${s}\tpython train_attr.py --ds $ds --split $s\n" >> jobs_fin_1.txt; done; done
for s in 1 2 3 4 5; do printf "fA_ecgcache_${s}\tpython train_attr.py --ds mitecg --split $s --cache_only\n" >> jobs_fin_1.txt; done
printf "fA_reeval_noleak\tpython reeval_attr.py --noleak\n" >> jobs_fin_1.txt
# B: main table, learned mask (list 1, slow ECG jobs first)
for ds in mitecg seqcomb_mv seqcomb_uv lowvar epilepsy freqshape; do for s in 1 2 3 4 5; do for t in $(seq 0 $(( $(nt $ds) - 1 ))); do
  printf "fB_${ds}_${s}_${t}\tpython train_cf_local.py --ds $ds --split $s --targets $t --variants learned --tag main $COMMON $(ep $ds)\n" >> jobs_fin_1.txt
done; done; done
# B': nested (needs the attribution pi cache of the split)
for ds in mitecg seqcomb_mv seqcomb_uv lowvar epilepsy freqshape; do for s in 1 2 3 4 5; do for t in $(seq 0 $(( $(nt $ds) - 1 ))); do
  printf "fN_${ds}_${s}_${t}\twhile [ ! -f cache/${ds}/attr_pi_split=${s}.pt ]; do sleep 30; done; python train_cf_local.py --ds $ds --split $s --targets $t --variants nested --nested_rule prior --tag main $COMMON $(ep $ds)\n" >> jobs_fin_1.txt
done; done; done
# C: CNN / LSTM black boxes (list 2)
for arch in cnn lstm; do for ds in mitecg seqcomb_mv freqshape; do for s in 1 2 3 4 5; do for t in $(seq 0 $(( $(nt $ds) - 1 ))); do
  E=$(ep $ds); [ $ds = mitecg ] && E="--max_epochs 10 --patience 6"
  printf "fC_${arch}_${ds}_${s}_${t}\tpython train_cf_local.py --ds $ds --split $s --targets $t --variants learned --arch $arch --tag arch_$arch $COMMON $E\n" >> jobs_fin_2.txt
done; done; done; done
# D: ablations (list 3)
for ab in no_lc no_bound no_kl no_con no_ste; do for ds in mitecg seqcomb_mv freqshape; do for s in 1 2 3 4 5; do for t in $(seq 0 $(( $(nt $ds) - 1 ))); do
  printf "fD_${ab}_${ds}_${s}_${t}\tpython train_cf_local.py --ds $ds --split $s --targets $t --variants learned --abl $ab --tag abl_$ab $COMMON $(ep $ds)\n" >> jobs_fin_3.txt
done; done; done; done
wc -l jobs_fin_*.txt
