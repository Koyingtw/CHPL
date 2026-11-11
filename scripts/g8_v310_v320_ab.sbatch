#!/bin/bash
#SBATCH -J g8-v31-v32-ab
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 02:00:00
#SBATCH -o logs/g8_v31_v32_ab_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
mkdir -p "$CHPL_ROOT/logs"
cd $CHPL_ROOT/build
hostname
cat > HPL_ab.dat <<DAT
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
run() { env CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=11 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=2816 CHPL2D_PRESEL=1 NCCL_MAX_NCHANNELS=6 timeout 600 mpirun -np 8 --bind-to none $XF ./$1 HPL_ab.dat 2>&1 | grep -E "WC2|PASSED|FAILED|out of memory"; }
echo "==== SAME-NODE interleaved A/B: chpl2d_v310 (GEMMul8 v3.1.0) vs chpl2d (GEMMul8 v3.2.0) ===="
for r in 1 2 3 4; do
  echo "-- round $r: v3.1.0 --"
  run chpl2d_v310
  echo "-- round $r: v3.2.0 --"
  run chpl2d
done
echo "ALL_DONE"
