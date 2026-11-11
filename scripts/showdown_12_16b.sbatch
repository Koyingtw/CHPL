#!/bin/bash
#SBATCH -J sd-12-16b
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 02:30:00
#SBATCH -o logs/sd_12_16b_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
module load singularity/4.3.7 2>/dev/null || true
export TMPDIR=/dev/shm
export SINGULARITYENV_TMPDIR=/dev/shm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
scontrol show hostnames $SLURM_JOB_NODELIST
# pick only InfiniBand-link-layer HCAs (guards against mixed IB/RoCE pairing at
# asymmetric rank counts; harmless when the fabric is homogeneous)
IBLIST=$(for d in /sys/class/infiniband/*; do n=$(basename $d); ll=$(cat $d/ports/1/link_layer 2>/dev/null); [ "$ll" = "InfiniBand" ] && echo -n "$n:1,"; done | sed 's/,$//')
echo "IB-only HCA list: $IBLIST"
export NCCL_IB_HCA="=$IBLIST"
export SINGULARITYENV_NCCL_IB_HCA="=$IBLIST"
export SINGULARITYENV_NVSHMEM_HCA_LIST="$IBLIST"
mkdat() { cat > $1 <<DAT
x
x
HPL.out
6
1
$2
1
4096
0
1
$3
$4
16.0
DAT
}
mkdat HPL_o12g.dat 131072 4 3
mkdat HPL_o12a.dat 397312 4 3
mkdat HPL_o12b.dat 405504 4 3
mkdat HPL_o16.dat  460800 4 4
S=${NV_HPL_SIF:-$HOME/hpc-benchmarks-26.02.sif}
G8="CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=11 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=3072 CHPL2D_WCH=4 CHPL2D_PRESEL=1 CHPL2D_PRESEL2=1 CHPL2D_WS=64 CHPL2D_XT=6144 NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=1 NCCL_BUFFSIZE=2097152 CUDA_DEVICE_MAX_CONNECTIONS=32"
XF="-x LD_LIBRARY_PATH -x NCCL_IB_HCA -x NCCL_MAX_NCHANNELS -x NCCL_MIN_NCHANNELS -x NCCL_BUFFSIZE -x CUDA_DEVICE_MAX_CONNECTIONS -x CHPL2D_OZ -x CHPL2D_G8 -x CHPL2D_G8_MODULI -x CHPL2D_G8_FAST -x CHPL2D_G8_NT -x CHPL2D_WCH -x CHPL2D_PRESEL -x CHPL2D_PRESEL2 -x CHPL2D_WS -x CHPL2D_XT"
ours() { env $G8 timeout 600 mpirun -np $2 --bind-to none $XF ./chpl4x4 $1 2>&1 | grep -E "WC2|PASSED|FAILED|out of memory|CUDA error|NCCL error" | head -5; }
nv16() { timeout 1100 srun --mpi=pmix -N 2 --ntasks-per-node=8 singularity run --nv -B /dev/shm:/tmp $S /workspace/hpl.sh --dat $CHPL_ROOT/build/HPL_nv16.dat 2>&1 | grep -E "WR|WC0|PASSED|FAILED|not accessible|same number" | head -4; }
nv12() { timeout 1100 srun --mpi=pmix -N 2 -n 12 --ntasks-per-node=6 singularity run --nv -B /dev/shm:/tmp $S /workspace/hpl.sh --dat $CHPL_ROOT/build/$1 2>&1 | grep -E "WR|WC0|PASSED|FAILED|not accessible|same number" | head -4; }
echo "==== our gate: 12 GPU @131072 ===="
ours HPL_o12g.dat 12
echo "########## 12-GPU showdown (ours 8+4, container 6+6) ##########"
for r in 1 2; do
  echo "-- round $r: OURS 12 (N=397312 NB4096 P4Q3) --"
  ours HPL_o12a.dat 12
  echo "-- round $r: CONTAINER 12 (N=448512 NB1024 P4Q3) --"
  nv12 HPL_nv12a.dat
done
echo "==== mem probes: ours @405504 | container @454656 ===="
ours HPL_o12b.dat 12
nv12 HPL_nv12b.dat
echo "########## 16-GPU showdown ##########"
for r in 1 2; do
  echo "-- round $r: OURS 16 (N=460800) --"
  ours HPL_o16.dat 16
  echo "-- round $r: CONTAINER 16 (N=524288) --"
  nv16
done
echo ALL_DONE
