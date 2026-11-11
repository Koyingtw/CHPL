#!/bin/bash
#SBATCH -J emu-h200
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o logs/emu_h200_replay_%j.log
source /etc/profile.d/modules.sh 2>/dev/null || true
PROBES=$WORK/HPL_EmuGEMM_build/probes
nvidia-smi --query-gpu=name,compute_cap --format=csv
echo "=== gemmul8 3.1.0 baseline (production + square shapes) ==="
$PROBES/g8_baseline_probe 11 1
$PROBES/g8_square
$PROBES/g8_accuracy
echo "=== EmuGEMM (Emu-II CRT) ==="
PY=$WORK/conda_envs/emu_gemm/bin/python
export EG_JIT_CACHE_DIR=$WORK/HPL_EmuGEMM_build/jit_cache
export LD_LIBRARY_PATH=$WORK/conda_envs/emu_gemm/lib:$LD_LIBRARY_PATH
export PATH=$WORK/conda_envs/emu_gemm/bin:$PATH
cd $PROBES
$PY emu_vs_g8_bench.py
$PY sanity_square.py
echo "ALL_DONE"
