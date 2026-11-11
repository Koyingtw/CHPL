#!/bin/bash
#SBATCH -J g8-v320-1n
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 01:30:00
#SBATCH -o logs/g8_v320_1n_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
mkdir -p "$CHPL_ROOT/logs"
cd $CHPL_ROOT/build
hostname
cat > HPL_v320_1n.dat <<DAT
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
XF="-x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x CHPL2D_OZ -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_PRESEL"
g8() { env CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=11 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=2816 CHPL2D_PRESEL=1 NCCL_MAX_NCHANNELS=6 timeout 600 mpirun -np 8 --bind-to none $XF ./chpl2d HPL_v320_1n.dat 2>&1 | grep -E "WC2|PASSED|FAILED|out of memory"; }
echo "==== GEMMul8 version check ===="
strings $CHPL_ROOT/gemmul8/lib/libgemmul8.a 2>/dev/null | grep -m1 "^3\.[0-9]*\.[0-9]*$"
echo "==== 1-node G8 champion (N=319488 NB=4096 m11 NT=2816), 3 rounds, GEMMul8 v3.2.0 ===="
for r in 1 2 3; do
  echo "-- round $r --"
  g8
done
echo "ALL_DONE"
