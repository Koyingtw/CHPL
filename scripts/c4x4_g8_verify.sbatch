#!/bin/bash
#SBATCH -J c4x4-g8-vfy
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 01:00:00
#SBATCH -o logs/c4x4_g8_vfy_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
scontrol show hostnames $SLURM_JOB_NODELIST
cat > HPL_vfy.dat <<DAT
x
x
HPL.out
6
1
460800
1
4096
0
1
4
4
16.0
DAT
COMMON="CHPL2D_WCH=4 CHPL2D_PRESEL=1 CHPL2D_PRESEL2=1 CHPL2D_WS=64 CHPL2D_XT=6144 NCCL_BUFFSIZE=2097152 NCCL_MIN_NCHANNELS=1 CUDA_DEVICE_MAX_CONNECTIONS=32"
XFLAGS="-x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x NCCL_MIN_NCHANNELS -x CUDA_DEVICE_MAX_CONNECTIONS -x CHPL2D_OZ -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_G8_MSPLIT -x CHPL2D_WCH -x CHPL2D_PRESEL -x CHPL2D_PRESEL2 -x CHPL2D_WS -x CHPL2D_XT -x NCCL_BUFFSIZE"
g8() { env CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=11 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=3072 CHPL2D_G8_MSPLIT=$1 NCCL_MAX_NCHANNELS=8 $COMMON timeout 700 mpirun -np 16 --bind-to none $XFLAGS ./chpl4x4 HPL_vfy.dat 2>&1 | grep -E "WC2|PASSED|FAILED|out of memory" | head -4; }
echo "==== barrier default (MSPLIT=0), expect ~989 ===="
g8 0
echo "==== repeat ===="
g8 0
echo "==== msplit=1 sanity (expect ~980) ===="
g8 1
echo "ALL_DONE"
