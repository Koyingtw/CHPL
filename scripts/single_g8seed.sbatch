#!/bin/bash
#SBATCH -J hpls-g8seed
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH -o logs/hpls_g8seed_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"

for M in 12 11; do
  for S in 1 2 3; do
    echo "== moduli=$M seed=$S =="
    HPLS_NB=1024 HPLS_AGG=8 HPLS_BACKEND=gemmul8 HPLS_NATIVE_TAIL=8192 \
      HPLS_G8_MODULI=$M HPLS_G8_FAST=1 HPLS_G8_CHUNK=3072 HPLS_SEED=$S \
      ./hpl_single 122880 2>&1 | grep -E "HPLS_gemmul8|PASSED|FAILED"
  done
done
echo "ALL_DONE"
