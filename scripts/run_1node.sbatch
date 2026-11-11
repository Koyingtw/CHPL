#!/bin/bash
# 單節點冠軍重現（8x H200, P2Q4 + PRESEL, 403.3 TF 中位數）
# 請依叢集調整 -A / -p / module；於 repo 根目錄: sbatch scripts/run_1node.sbatch
#SBATCH -J chpl-1node
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 01:00:00
#SBATCH -o logs/run1node_%j.log

source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export NCCL_MAX_NCHANNELS=6
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
run() { CHPL2D_OZ=1 CHPL2D_PRESEL=1 timeout 1100 mpirun -np 8 --bind-to none \
  -x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x CHPL2D_OZ -x CHPL2D_PRESEL \
  ./build/chpl2d "$1" 2>&1 | grep -E "WC2|PASSED|FAILED" | head -3; }
# 正確性 gate（小 N）後跑正式 N=348160
sed '6s/.*/16384/'  dat/HPL_1node.dat > /tmp/g1.dat && echo "== gate 16384 ==" && run /tmp/g1.dat
sed '6s/.*/131072/' dat/HPL_1node.dat > /tmp/g2.dat && echo "== gate 131072 ==" && run /tmp/g2.dat
for i in 1 2 3; do echo "== round $i: N=348160 =="; run dat/HPL_1node.dat; done
