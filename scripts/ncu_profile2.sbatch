#!/bin/bash
#SBATCH -J ncu-prof2
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 01:30:00
#SBATCH -o logs/ncu_prof2_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
NCU=/usr/local/cuda-13.0/bin/ncu
which $NCU; $NCU --version | head -2
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
G8ENV="HPLS_NB=4096 HPLS_AGG=1 HPLS_BACKEND=gemmul8 HPLS_G8_MODULI=11 HPLS_G8_FAST=1 HPLS_G8_CHUNK=2816 HPLS_NATIVE_TAIL=0"

echo "######## A2: tensor-pipe instruction counters on the int8 GEMM (GMMA=WGMMA proof) ########"
TM=$($NCU --query-metrics 2>/dev/null | grep -oE "sm__inst_executed_pipe_tensor_op_[a-z0-9]+" | sort -u | paste -sd, -)
echo "tensor metrics found: [$TM]"
[ -z "$TM" ] && TM="sm__inst_executed_pipe_tensor.sum"
env $G8ENV $NCU -k "regex:xmma_gemm_i8i32" --launch-skip 20 --launch-count 2 \
  --metrics "$TM,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed" \
  ./hpl_single 24576 2>&1 | grep -vE "^==PROF== Profiling" | tail -40

echo "######## A3: SOL on GEMMul8 side kernels (mod / undo_scaling / scaling) ########"
env $G8ENV $NCU -k "regex:(mod_hi2mid_ge_kernel|undo_scaling_kernel_special|scaling_rowwise_full)" \
  --launch-skip 24 --launch-count 6 --section SpeedOfLight --section MemoryWorkloadAnalysis \
  ./hpl_single 24576 2>&1 | grep -vE "^==PROF== Profiling" | head -150
echo ALL_DONE
