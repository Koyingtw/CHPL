#!/bin/bash
#SBATCH -J hpls-g8perf2
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH -o logs/hpls_g8perf2_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"

echo "########## 1) N=98304 NB=2048 AGG=6 moduli=13 (accuracy check at K=12288, full output) ##########"
HPLS_NB=2048 HPLS_AGG=6 HPLS_BACKEND=gemmul8 HPLS_NATIVE_TAIL=8192 HPLS_G8_MODULI=13 \
  ./hpl_single 98304 2>&1 | tail -24

echo "########## 2) N=122880 NB=1024 AGG=8 chunk=2048 moduli=14 (max-N head-to-head vs ozaki 47.16) ##########"
HPLS_NB=1024 HPLS_AGG=8 HPLS_BACKEND=gemmul8 HPLS_NATIVE_TAIL=8192 HPLS_G8_CHUNK=2048 \
  ./hpl_single 122880 2>&1 | tail -24

echo "########## 3) N=122880 NB=1024 AGG=8 chunk=2048 moduli=13 ##########"
HPLS_NB=1024 HPLS_AGG=8 HPLS_BACKEND=gemmul8 HPLS_NATIVE_TAIL=8192 HPLS_G8_CHUNK=2048 HPLS_G8_MODULI=13 \
  ./hpl_single 122880 2>&1 | tail -24
echo "ALL_DONE"
