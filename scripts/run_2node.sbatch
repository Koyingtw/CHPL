#!/bin/bash
# 兩節點冠軍重現（16x H200, P4Q4, 792.2 TF 中位數）
# 請依叢集調整 -A / -p / module；於 repo 根目錄: sbatch scripts/run_2node.sbatch
#SBATCH -J chpl-2node
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 01:30:00
#SBATCH -o logs/run2node_%j.log

source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
run() { env CHPL2D_OZ=1 CHPL2D_OZ_NT=1024 CHPL2D_WCH=4 CHPL2D_PRESEL=1 \
  CHPL2D_WS=64 CHPL2D_XT=6144 NCCL_BUFFSIZE=2097152 NCCL_MAX_NCHANNELS=4 NCCL_MIN_NCHANNELS=1 \
  CUDA_DEVICE_MAX_CONNECTIONS=32 timeout 700 mpirun -np 16 --bind-to none \
  -x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x NCCL_MIN_NCHANNELS -x CUDA_DEVICE_MAX_CONNECTIONS \
  -x CHPL2D_OZ -x CHPL2D_OZ_NT -x CHPL2D_WCH -x CHPL2D_PRESEL \
  -x CHPL2D_WS -x CHPL2D_XT -x NCCL_BUFFSIZE \
  ./build/chpl4x4 "$1" 2>&1 | grep -E "WC2|PASSED|FAILED" | head -3; }
sed '6s/.*/131072/' dat/HPL_2node_p4q4.dat > /tmp/g44.dat && echo "== gate 131072 ==" && run /tmp/g44.dat
for i in 1 2 3; do echo "== round $i: N=497664 =="; run dat/HPL_2node_p4q4.dat; done
