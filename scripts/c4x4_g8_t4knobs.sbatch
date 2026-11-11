#!/bin/bash
#SBATCH -J c4x4-g8-4nkb
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 01:30:00
#SBATCH -o logs/c4x4_g8_4nkb_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
scontrol show hostnames $SLURM_JOB_NODELIST
cat > HPL_4kb.dat <<DAT
x
x
HPL.out
6
1
622592
1
4096
0
1
4
8
16.0
DAT
BASE="CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=11 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=3072 CHPL2D_WS=64 CHPL2D_XT=6144 NCCL_BUFFSIZE=2097152 NCCL_MIN_NCHANNELS=1 CUDA_DEVICE_MAX_CONNECTIONS=32"
XFLAGS="-x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x NCCL_MIN_NCHANNELS -x CUDA_DEVICE_MAX_CONNECTIONS -x CHPL2D_OZ -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_WCH -x CHPL2D_PRESEL -x CHPL2D_PRESEL2 -x CHPL2D_WS -x CHPL2D_XT -x NCCL_BUFFSIZE"
run() { env $BASE CHPL2D_WCH=$1 NCCL_MAX_NCHANNELS=$2 CHPL2D_PRESEL=$3 CHPL2D_PRESEL2=$4 timeout 600 mpirun -np 32 --bind-to none $XFLAGS ./chpl4x4 HPL_4kb.dat 2>&1 | grep -E "WC2|PASSED|FAILED|out of memory|CUDA error" | head -3; }
echo "==== baseline: WCH4 NCH8 P1 P2 ===="
run 4 8 1 1
echo "==== WCH8 ===="
run 8 8 1 1
echo "==== WCH2 ===="
run 2 8 1 1
echo "==== NCH12 ===="
run 4 12 1 1
echo "==== PRESEL2=0 ===="
run 4 8 1 0
echo "ALL_DONE"
