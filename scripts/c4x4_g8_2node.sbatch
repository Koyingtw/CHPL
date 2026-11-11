#!/bin/bash
#SBATCH -J c4x4-g8-2n
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 02:00:00
#SBATCH -o logs/c4x4_g8_2node_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
scontrol show hostnames $SLURM_JOB_NODELIST
cat > HPL_g8gate.dat <<DAT
x
x
HPL.out
6
1
131072
1
2048
0
1
4
4
16.0
DAT
cat > HPL_g8prod.dat <<DAT
x
x
HPL.out
6
1
497664
1
2048
0
1
4
4
16.0
DAT
COMMON="CHPL2D_WCH=4 CHPL2D_PRESEL=1 CHPL2D_PRESEL2=1 CHPL2D_WS=64 CHPL2D_XT=6144 NCCL_BUFFSIZE=2097152 NCCL_MAX_NCHANNELS=4 NCCL_MIN_NCHANNELS=1 CUDA_DEVICE_MAX_CONNECTIONS=32"
XFLAGS="-x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x NCCL_MIN_NCHANNELS -x CUDA_DEVICE_MAX_CONNECTIONS -x CHPL2D_OZ -x CHPL2D_OZ_NT -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_WCH -x CHPL2D_PRESEL -x CHPL2D_PRESEL2 -x CHPL2D_WS -x CHPL2D_XT -x NCCL_BUFFSIZE -x CHPL2D_TIMING"
oz() { env CHPL2D_OZ=1 CHPL2D_OZ_NT=1024 CHPL2D_G8=0 $COMMON CHPL2D_TIMING=${2:-0} timeout 700 mpirun -np 16 --bind-to none $XFLAGS ./chpl4x4 $1 2>&1 | grep -E "WC2|PASSED|FAILED|G8 backend|s0_gemm" | head -6; }
g8() { env CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=$2 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=$3 $COMMON CHPL2D_TIMING=${4:-0} timeout 700 mpirun -np 16 --bind-to none $XFLAGS ./chpl4x4 $1 2>&1 | grep -E "WC2|PASSED|FAILED|G8 backend|s0_gemm|error|Error|out of memory" | head -8; }

echo "==== GATE: G8 m12 @131072 ===="
g8 HPL_g8gate.dat 12 2048
echo "==== GATE: G8 m13 @131072 ===="
g8 HPL_g8gate.dat 13 2048

echo "==== A/B round 1: OZ champion @497664 ===="
oz HPL_g8prod.dat
echo "==== A/B round 1: G8 m12 NT2048 @497664 ===="
g8 HPL_g8prod.dat 12 2048
echo "==== A/B round 2: OZ champion @497664 ===="
oz HPL_g8prod.dat
echo "==== A/B round 2: G8 m12 NT2048 @497664 ===="
g8 HPL_g8prod.dat 12 2048
echo "==== G8 m13 NT2048 @497664 ===="
g8 HPL_g8prod.dat 13 2048
echo "==== G8 m12 NT4096 @497664 (mem probe) ===="
g8 HPL_g8prod.dat 12 4096
echo "==== timing: G8 m12 NT2048 @497664 ===="
g8 HPL_g8prod.dat 12 2048 1
echo "==== timing: OZ @497664 ===="
oz HPL_g8prod.dat 1
echo "ALL_DONE"
