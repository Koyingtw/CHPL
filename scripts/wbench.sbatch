#!/bin/bash
#SBATCH -J wbench
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH -t 00:20:00
#SBATCH -o logs/wbench_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
echo "== 3840 MB (production W size), NCH=8 =="
env NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=1 NCCL_BUFFSIZE=2097152 mpirun -np 16 --bind-to none -x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x NCCL_MIN_NCHANNELS -x NCCL_BUFFSIZE ./wbench 3840
echo "== 3840 MB, NCH=16 =="
env NCCL_MAX_NCHANNELS=16 NCCL_MIN_NCHANNELS=1 NCCL_BUFFSIZE=2097152 mpirun -np 16 --bind-to none -x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x NCCL_MIN_NCHANNELS -x NCCL_BUFFSIZE ./wbench 3840
echo "== 960 MB (one WCH chunk) =="
env NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=1 NCCL_BUFFSIZE=2097152 mpirun -np 16 --bind-to none -x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x NCCL_MIN_NCHANNELS -x NCCL_BUFFSIZE ./wbench 960
echo "ALL_DONE"
