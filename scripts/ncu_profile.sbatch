#!/bin/bash
#SBATCH -J ncu-profile
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 02:00:00
#SBATCH -o logs/ncu_profile_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
export TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
NCU=/opt/nvidia/nsight-compute/2025.3.1/ncu
NSYS=/usr/local/bin/nsys
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
G8ENV="HPLS_NB=4096 HPLS_AGG=1 HPLS_BACKEND=gemmul8 HPLS_G8_MODULI=11 HPLS_G8_FAST=1 HPLS_G8_CHUNK=2816 HPLS_NATIVE_TAIL=0"

echo "######## A1: nsys kernel breakdown, hpl_single N=65536 (production k=4096 mix) ########"
env $G8ENV $NSYS profile -o /dev/shm/hpls_prof --force-overwrite true ./hpl_single 65536 > /dev/null 2>&1
$NSYS stats --report cuda_gpu_kern_sum /dev/shm/hpls_prof.nsys-rep 2>/dev/null | head -30

echo "######## A2: ncu GMMA/tensor-pipe counters on the int8 GEMM kernel ########"
TENSOR_METRICS=$($NCU --query-metrics 2>/dev/null | grep -oE "sm__inst_executed_pipe_tensor_op_[a-z0-9]+" | sort -u | paste -sd, -)
echo "available tensor-pipe metrics: $TENSOR_METRICS"
env $G8ENV $NCU -k "regex:(nvjet|xmma|cutlass|gemm|sm90)" --launch-skip 30 --launch-count 2 \
  --metrics "$TENSOR_METRICS,sm__cycles_elapsed.avg,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed" \
  ./hpl_single 24576 2>&1 | grep -vE "^==PROF== Profiling|^\s*$" | head -60

echo "######## A3: ncu SOL on GEMMul8 side kernels (scaling/mod/CRT) ########"
env $G8ENV $NCU -k "regex:gemmul8" --launch-skip 60 --launch-count 9 --set speedoflight \
  ./hpl_single 24576 2>&1 | grep -vE "^==PROF== Profiling" | head -120

echo "######## B: nsys rank-0 kernel breakdown, 1-node chpl2d champion mix N=131072 ########"
cat > HPL_prof1n.dat <<DAT
x
x
HPL.out
6
1
131072
1
4096
0
1
2
4
16.0
DAT
cat > /dev/shm/wrap0.sh <<'W'
#!/bin/bash
if [ "$OMPI_COMM_WORLD_RANK" = "0" ]; then
  exec /usr/local/bin/nsys profile -o /dev/shm/chpl2d_r0 --force-overwrite true ./chpl2d HPL_prof1n.dat
else
  exec ./chpl2d HPL_prof1n.dat
fi
W
chmod +x /dev/shm/wrap0.sh
env CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=11 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=2816 CHPL2D_PRESEL=1 NCCL_MAX_NCHANNELS=6 \
  timeout 600 mpirun -np 8 --bind-to none \
  -x LD_LIBRARY_PATH -x NCCL_MAX_NCHANNELS -x CHPL2D_OZ -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_PRESEL \
  /dev/shm/wrap0.sh 2>&1 | grep -E "WC2|PASSED|FAILED" | head -3
$NSYS stats --report cuda_gpu_kern_sum /dev/shm/chpl2d_r0.nsys-rep 2>/dev/null | head -35
cp /dev/shm/hpls_prof.nsys-rep /dev/shm/chpl2d_r0.nsys-rep $CHPL_ROOT/logs/ 2>/dev/null
echo ALL_DONE
