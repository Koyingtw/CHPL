#!/bin/bash
#SBATCH -J c2d-g8-t1n
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 02:00:00
#SBATCH -o logs/c2d_g8_t1n_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
scontrol show hostnames $SLURM_JOB_NODELIST
mkdat() { cat > $1 <<DAT
x
x
HPL.out
6
1
$2
1
$3
0
1
2
4
16.0
DAT
}
mkdat HPL_s1.dat 339968 4096
mkdat HPL_s2.dat 331776 4096
XFLAGS="-x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x CHPL2D_OZ -x CHPL2D_OZ_NT -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_PRESEL -x CHPL2D_TIMING"
g8() { env CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=$2 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=$3 CHPL2D_PRESEL=1 NCCL_MAX_NCHANNELS=6 CHPL2D_TIMING=${4:-0} timeout 700 mpirun -np 8 --bind-to none $XFLAGS ./chpl2d_g8 $1 2>&1 | grep -E "WC2|PASSED|FAILED|G8 backend|s0_gemm|out of memory" | head -8; }

echo "==== 1n: N=339968 NB4096 m12 NT1024 ===="
g8 HPL_s1.dat 12 1024
echo "==== 1n: N=331776 NB4096 m12 NT2048 ===="
g8 HPL_s2.dat 12 2048
echo "==== 1n: N=331776 NB4096 m11 NT2048 ===="
g8 HPL_s2.dat 11 2048
echo "==== 1n: N=331776 NB4096 m12 NT2048 (repeat) ===="
g8 HPL_s2.dat 12 2048
echo "==== timing: N=331776 NB4096 m12 NT2048 ===="
g8 HPL_s2.dat 12 2048 1
echo "ALL_DONE"
