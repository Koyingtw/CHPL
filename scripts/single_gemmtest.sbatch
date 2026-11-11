#!/bin/bash
#SBATCH -J hpls-gemmtest
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 00:20:00
#SBATCH -o logs/hpls_gemmtest_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
echo "== small: m=64 k=256 n=32 (single tile, no tail) =="
./hpl_single --gemmtest 64 256 32
echo "== with tail: m=100 k=256 n=100, oz_nt will be 100 -> single tile =="
./hpl_single --gemmtest 100 256 100
echo "== multi-tile with tail: m=100 k=256 n=300 (oz_nt=min(256,300)=256, tail=44) =="
./hpl_single --gemmtest 100 256 300
