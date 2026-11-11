#!/bin/bash
# 兩節點正式對決：CHPL(P4Q4) vs NVIDIA HPC-Benchmarks 容器，同節點交錯 5 輪
# 需要 singularity 與 hpc-benchmarks sif（調整 SIF 路徑）
#SBATCH -J chpl-showdown
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 03:50:00
#SBATCH -o logs/showdown_%j.log

source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
module load singularity/4.3.7
export TMPDIR=/dev/shm
export SINGULARITYENV_TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SIF=${NV_HPL_SIF:-$HOME/hpc-benchmarks-26.02.sif}      # ← 調整
# 容器方最佳參數（完整標準 HPL.dat 格式，N=524288/NB1024/P4Q4）
cat > /tmp/HPL_nv.dat <<'NVDAT'
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
524288       Ns
1            # of NBs
1024         NBs
1            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
4            Ps
4            Qs
16.0         threshold
1            # of panel fact
0 1 2        PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
2 8          NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
0 1 2        RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
3 2          BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1 0          DEPTHs (>=0)
1            SWAP (0=bin-exch,1=long,2=mix)
192          swapping threshold
1            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
0            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
NVDAT
ours() { env CHPL2D_OZ=1 CHPL2D_OZ_NT=1024 CHPL2D_WCH=4 CHPL2D_PRESEL=1 \
  CHPL2D_WS=64 CHPL2D_XT=6144 NCCL_BUFFSIZE=2097152 NCCL_MAX_NCHANNELS=8 \
  CUDA_DEVICE_MAX_CONNECTIONS=32 timeout 700 mpirun -np 16 --bind-to none \
  -x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x CUDA_DEVICE_MAX_CONNECTIONS \
  -x CHPL2D_OZ -x CHPL2D_OZ_NT -x CHPL2D_WCH -x CHPL2D_PRESEL \
  -x CHPL2D_WS -x CHPL2D_XT -x NCCL_BUFFSIZE \
  ./build/chpl4x4 dat/HPL_2node_p4q4.dat 2>&1 | grep -E "WC2|PASSED|FAILED" | head -3; }
nv() { timeout 1000 srun --mpi=pmix -N 2 --ntasks-per-node=8 \
  singularity run --nv -B /dev/shm:/tmp "$SIF" /workspace/hpl.sh \
  --dat /tmp/HPL_nv.dat 2>&1 | grep -E "W[RC][0-9]|PASSED|FAILED" | head -3; }
sed '6s/.*/131072/' dat/HPL_2node_p4q4.dat > /tmp/g.dat
echo "== gate ==" && env CHPL2D_OZ=1 mpirun -np 16 --bind-to none -x LD_LIBRARY_PATH -x CHPL2D_OZ ./build/chpl4x4 /tmp/g.dat 2>&1 | grep -E "WC2|PASSED|FAILED" | head -2
for r in 1 2 3 4 5; do
  echo "== round $r: CHPL =="; ours
  echo "== round $r: container =="; nv
done
