#!/bin/bash
#SBATCH -J g8-profile2n
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 01:30:00
#SBATCH -o logs/g8_profile2n_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
scontrol show hostnames $SLURM_JOB_NODELIST
cat > HPL_p2n.dat <<DAT
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
BASE="CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=11 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=3072 CHPL2D_WCH=4 CHPL2D_PRESEL=1 CHPL2D_PRESEL2=1 CHPL2D_WS=64 CHPL2D_XT=6144 NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=1 NCCL_BUFFSIZE=2097152 CUDA_DEVICE_MAX_CONNECTIONS=32"
XF="-x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x NCCL_MIN_NCHANNELS -x NCCL_BUFFSIZE -x CUDA_DEVICE_MAX_CONNECTIONS -x CHPL2D_OZ -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_WCH -x CHPL2D_PRESEL -x CHPL2D_PRESEL2 -x CHPL2D_WS -x CHPL2D_XT -x CHPL2D_TIMING"
echo "######## 2-NODE TIMING=1 ########"
env $BASE CHPL2D_TIMING=1 timeout 600 mpirun -np 16 --bind-to none $XF ./chpl4x4 HPL_p2n.dat 2>&1 | grep -E "WC2|PASSED|2dtiming" | head -18
echo "######## 2-NODE TIMING=2 (fine) ########"
env $BASE CHPL2D_TIMING=2 timeout 600 mpirun -np 16 --bind-to none $XF ./chpl4x4 HPL_p2n.dat 2>&1 | grep -E "WC2|PASSED|2dtiming. rank 0 |2dfine. r0 " | head -12
echo ALL_DONE
