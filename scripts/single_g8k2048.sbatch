#!/bin/bash
#SBATCH -J hpls-g8k2048
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -o logs/hpls_g8k2048_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
# k=2048 regime (AGG=1): what the distributed per-panel update faces.
# Compare gemmul8 vs ozaki vs the AGG=6 reference at the same N.
for CFG in "gemmul8 12 1" "gemmul8 13 1" "ozaki 0 0"; do
  set -- $CFG
  echo "== AGG=1 (K=2048) backend=$1 moduli=$2 fast=$3 =="
  HPLS_NB=2048 HPLS_AGG=1 HPLS_BACKEND=$1 HPLS_NATIVE_TAIL=8192 \
    HPLS_G8_MODULI=$2 HPLS_G8_FAST=$3 HPLS_G8_CHUNK=8192 \
    ./hpl_single 98304 2>&1 | grep -E "HPLS_|PASSED|FAILED|internal|wide_gemm  "
done
echo "== reference: AGG=6 gemmul8 m12 fast (from prior runs: 67.09) =="
echo "ALL_DONE"
