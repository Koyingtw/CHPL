#!/bin/bash
#SBATCH -J hpls-g8perf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH -o logs/hpls_g8perf_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"

echo "########## 1) N=98304 gemmul8, teammate-matched: NB=1024 AGG=8 K=8192 chunk=16384 moduli=14 ##########"
HPLS_NB=1024 HPLS_AGG=8 HPLS_BACKEND=gemmul8 HPLS_NATIVE_TAIL=8192 \
  ./hpl_single 98304 2>&1 | tail -22

echo "########## 2) N=98304 gemmul8, our block structure: NB=2048 AGG=6 K=12288 ##########"
HPLS_NB=2048 HPLS_AGG=6 HPLS_BACKEND=gemmul8 HPLS_NATIVE_TAIL=8192 \
  ./hpl_single 98304 2>&1 | tail -22

echo "########## 3) N=98304 gemmul8, moduli=13 probe (accuracy watch) ##########"
HPLS_NB=1024 HPLS_AGG=8 HPLS_BACKEND=gemmul8 HPLS_NATIVE_TAIL=8192 HPLS_G8_MODULI=13 \
  ./hpl_single 98304 2>&1 | tail -8

echo "########## 4) N=110592 gemmul8, NB=1024 AGG=8 chunk=8192 (mem-diet for bigger N) ##########"
HPLS_NB=1024 HPLS_AGG=8 HPLS_BACKEND=gemmul8 HPLS_NATIVE_TAIL=8192 HPLS_G8_CHUNK=8192 \
  ./hpl_single 110592 2>&1 | tail -22
echo "ALL_DONE"
