#!/bin/bash
#SBATCH -J c2d-g8-1nsw3
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 02:00:00
#SBATCH -o logs/c2d_g8_1nsw3_%j.log
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
2
4
16.0
DAT
}
mkdat HPL_u40.dat 315392
mkdat HPL_u41.dat 319488
mkdat HPL_u42.dat 323584
XFLAGS="-x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x CHPL2D_OZ -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_PRESEL"
g8() { env CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=11 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=$2 CHPL2D_PRESEL=1 NCCL_MAX_NCHANNELS=6 timeout 600 mpirun -np 8 --bind-to none $XFLAGS ./chpl2d_g8 $1 2>&1 | grep -E "WC2|PASSED|FAILED|out of memory" | head -4; }
echo "==== NT2560 @319488 ===="
g8 HPL_u41.dat 2560
echo "==== NT2560 @323584 ===="
g8 HPL_u42.dat 2560
echo "==== NT3072 @315392 (mem probe) ===="
g8 HPL_u40.dat 3072
echo "==== NT2816 @315392 ===="
g8 HPL_u40.dat 2816
echo "==== 3-round stability: NT2560 @315392 ===="
for r in 1 2 3; do g8 HPL_u40.dat 2560; done
echo "ALL_DONE"
