#!/bin/bash
#SBATCH -J hpls-perf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH -o logs/hpls_perf_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"

echo "########## AGG sweep @ N=32768, NB=2048, native_tail=0 (pure Ozaki, no tail fallback) ##########"
for AGG in 1 2 3 4 6 8; do
  echo "== AGG=$AGG =="
  HPLS_NB=2048 HPLS_AGG=$AGG HPLS_BACKEND=ozaki HPLS_NATIVE_TAIL=0 HPLS_OZ_NT=1024 \
    ./hpl_single 32768 2>&1 | grep -E "HPLS_ozaki|PASSED|FAILED|MEM PLAN: total"
done

echo "########## Large-N runs, NB=2048, AGG=6, native_tail=8192 (realistic tail cutoff) ##########"
for N in 98304 122880; do
  echo "== N=$N =="
  HPLS_NB=2048 HPLS_AGG=6 HPLS_BACKEND=ozaki HPLS_NATIVE_TAIL=8192 HPLS_OZ_NT=1024 \
    ./hpl_single $N 2>&1 | tail -30
done
echo "ALL_DONE"
