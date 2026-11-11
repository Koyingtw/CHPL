#!/bin/bash
#SBATCH -J g8-showdown1n
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 03:00:00
#SBATCH -o logs/g8_showdown1n_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
module load singularity/4.3.7 2>/dev/null || true
export TMPDIR=/dev/shm
export SINGULARITYENV_TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
hostname
cat > HPL_sd1n.dat <<DAT
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
S=${NV_HPL_SIF:-$HOME/hpc-benchmarks-26.02.sif}
XF="-x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x CHPL2D_OZ -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_PRESEL"
g8() { env CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=11 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=2816 CHPL2D_PRESEL=1 NCCL_MAX_NCHANNELS=6 timeout 600 mpirun -np 8 --bind-to none $XF ./chpl2d HPL_sd1n.dat 2>&1 | grep -E "WC2|PASSED|FAILED|out of memory" | head -3; }
nv() { timeout 1100 singularity run --nv -B /dev/shm:/tmp $S mpirun -np 8 /workspace/hpl.sh --dat $CHPL_ROOT/build/HPL_nv8.dat 2>&1 | grep -E "WC0 |WR0|PASSED|FAILED" | head -3; }
echo "==== sanity: rebuilt ./chpl2d (README path) runs the G8 champion ===="
g8
echo "==== G8-era 1-node showdown: 3 interleaved rounds ===="
for r in 1 2 3; do
  echo "-- round $r: CHPL G8 (N=319488 NB=4096) --"
  g8
  echo "-- round $r: NVIDIA container (N=366735 NB=1024, its tuned best) --"
  nv
done
echo "ALL_DONE"
