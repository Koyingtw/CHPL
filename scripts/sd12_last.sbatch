#!/bin/bash
#SBATCH -J sd12-last
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -t 00:50:00
#SBATCH -o logs/sd12_last_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
module load x86-nvhpc/26.3
module load singularity/4.3.7 2>/dev/null || true
export TMPDIR=/dev/shm
export SINGULARITYENV_TMPDIR=/dev/shm
IBLIST=$(for d in /sys/class/infiniband/*; do n=$(basename $d); ll=$(cat $d/ports/1/link_layer 2>/dev/null); [ "$ll" = "InfiniBand" ] && echo -n "$n:1,"; done | sed 's/,$//')
export SINGULARITYENV_NCCL_IB_HCA="=$IBLIST"
export SINGULARITYENV_NVSHMEM_HCA_LIST="$IBLIST"
export SINGULARITYENV_NVSHMEM_DISABLE_NVLS=1
export SINGULARITYENV_HPL_FCT_COMM_POLICY=1
export SINGULARITYENV_HPL_NVSHMEM_SWAP=0
export SINGULARITYENV_HPL_P2P_AS_BCAST=1
CHPL_ROOT=${CHPL_ROOT:-$SLURM_SUBMIT_DIR}
cd "$CHPL_ROOT/build"
S=${NV_HPL_SIF:-$HOME/hpc-benchmarks-26.02.sif}
nv12() { timeout 600 srun --mpi=pmix -n 12 --distribution=cyclic singularity run --nv -B /dev/shm:/tmp $S /workspace/hpl.sh --dat $CHPL_ROOT/build/$1 2>&1 | grep -E "WR|WC0|PASSED|FAILED|failed|rror" | head -5; }
echo "==== container 12: all-NCCL steering, P4Q3 ===="
nv12 HPL_nv12a.dat
echo "==== container 12: all-NCCL steering, P3Q4 ===="
nv12 HPL_nv12c.dat
echo ALL_DONE
