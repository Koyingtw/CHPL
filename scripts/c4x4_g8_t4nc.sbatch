#!/bin/bash
#SBATCH -J c4x4-g8-t4nc
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 02:00:00
#SBATCH -o logs/c4x4_g8_t4nc_%j.log
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
4096
0
1
4
8
16.0
DAT
}
mkdat HPL_y1.dat 622592
mkdat HPL_y2.dat 614400
COMMON="CHPL2D_WCH=4 CHPL2D_PRESEL=1 CHPL2D_PRESEL2=1 CHPL2D_WS=64 CHPL2D_XT=6144 NCCL_BUFFSIZE=2097152 NCCL_MIN_NCHANNELS=1 CUDA_DEVICE_MAX_CONNECTIONS=32"
XFLAGS="-x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x NCCL_MIN_NCHANNELS -x CUDA_DEVICE_MAX_CONNECTIONS -x CHPL2D_OZ -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_WCH -x CHPL2D_PRESEL -x CHPL2D_PRESEL2 -x CHPL2D_WS -x CHPL2D_XT -x NCCL_BUFFSIZE"
g8() { env CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=$2 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=$3 NCCL_MAX_NCHANNELS=$4 $COMMON timeout 700 mpirun -np 32 --bind-to none $XFLAGS ./chpl4x4 $1 2>&1 | grep -E "WC2|PASSED|FAILED|out of memory" | head -4; }
echo "==== 4n m11 NT3072 NCH4 @622592 ===="
g8 HPL_y1.dat 11 3072 4
echo "==== 4n m11 NT3072 NCH8 @622592 ===="
g8 HPL_y1.dat 11 3072 8
echo "==== 4n m11 NT3072 NCH4 @614400 (fallback if above OOM) ===="
g8 HPL_y2.dat 11 3072 4
echo "==== 4n m11 NT2048 NCH8 @622592 ===="
g8 HPL_y1.dat 11 2048 8
echo "ALL_DONE"
