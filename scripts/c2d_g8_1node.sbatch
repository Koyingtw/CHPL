#!/bin/bash
#SBATCH -J c2d-g8-1n
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 02:00:00
#SBATCH -o logs/c2d_g8_1node_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
scontrol show hostnames $SLURM_JOB_NODELIST
cat > HPL_g81g.dat <<DAT
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
2
4
16.0
DAT
cat > HPL_g81n.dat <<DAT
x
x
HPL.out
6
1
348160
1
2048
0
1
2
4
16.0
DAT
XFLAGS="-x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x CHPL2D_OZ -x CHPL2D_OZ_NT -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_PRESEL -x CHPL2D_TIMING"
oz() { env CHPL2D_OZ=1 CHPL2D_G8=0 CHPL2D_PRESEL=1 NCCL_MAX_NCHANNELS=6 CHPL2D_TIMING=${2:-0} timeout 700 mpirun -np 8 --bind-to none $XFLAGS ./chpl2d_g8 $1 2>&1 | grep -E "WC2|PASSED|FAILED|s0_gemm" | head -5; }
g8() { env CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=$2 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=$3 CHPL2D_PRESEL=1 NCCL_MAX_NCHANNELS=6 CHPL2D_TIMING=${4:-0} timeout 700 mpirun -np 8 --bind-to none $XFLAGS ./chpl2d_g8 $1 2>&1 | grep -E "WC2|PASSED|FAILED|G8 backend|s0_gemm|out of memory" | head -8; }

echo "==== GATE: new binary OZ @131072 (vs champion behavior) ===="
oz HPL_g81g.dat
echo "==== GATE: G8 m12 @131072 ===="
g8 HPL_g81g.dat 12 2048
echo "==== round 1: OZ champion @348160 ===="
oz HPL_g81n.dat
echo "==== round 1: G8 m12 NT2048 @348160 ===="
g8 HPL_g81n.dat 12 2048
echo "==== round 2: OZ champion @348160 ===="
oz HPL_g81n.dat
echo "==== round 2: G8 m12 NT2048 @348160 ===="
g8 HPL_g81n.dat 12 2048
echo "==== G8 m12 NT1024 @348160 (mem-lean variant) ===="
g8 HPL_g81n.dat 12 1024
echo "==== timing: G8 m12 NT2048 ===="
g8 HPL_g81n.dat 12 2048 1
echo "==== timing: OZ ===="
oz HPL_g81n.dat 1
echo "ALL_DONE"
