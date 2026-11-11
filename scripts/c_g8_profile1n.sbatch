#!/bin/bash
#SBATCH -J g8-profile1n
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 01:30:00
#SBATCH -o logs/g8_profile1n_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
scontrol show hostnames $SLURM_JOB_NODELIST
cat > HPL_p1n.dat <<DAT
x
x
HPL.out
6
1
319488
1
4096
0
1
2
4
16.0
DAT
BASE="CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=11 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=2816 CHPL2D_PRESEL=1 NCCL_MAX_NCHANNELS=6"
XF="-x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x CHPL2D_OZ -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_PRESEL -x CHPL2D_TIMING"
echo "######## 1-NODE TIMING=1 ########"
env $BASE CHPL2D_TIMING=1 timeout 600 mpirun -np 8 --bind-to none $XF ./chpl2d_g8 HPL_p1n.dat 2>&1 | grep -E "WC2|PASSED|2dtiming" | head -10
echo "######## 1-NODE TIMING=2 (fine) ########"
env $BASE CHPL2D_TIMING=2 timeout 600 mpirun -np 8 --bind-to none $XF ./chpl2d_g8 HPL_p1n.dat 2>&1 | grep -E "WC2|PASSED|2dtiming. rank 0 |2dfine. r0 " | head -10
echo ALL_DONE
