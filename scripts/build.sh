#!/bin/bash
# 建置 CHPL 全部二進位（需要 nvhpc 模組提供的 MPI 與 NCCL、CUDA 13.x）
# 用法: bash scripts/build.sh   （在 repo 根目錄執行）
#
# GEMMul8 相依（G8 後端，冠軍配置必需）：
#   git clone https://github.com/RIKEN-RCCS/GEMMul8 gemmul8   # 置於 repo 根目錄
#   （或 export G8_HOME=/path/to/gemmul8）
#   本腳本會自動以「最小化建置」只編 gemm<double,INT8> 需要的 9 個目標檔
#   （完整 make 會編數百個模板組合，可能耗盡編譯資源）。
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/build"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load x86-nvhpc/26.3            # ← 依叢集調整
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
MPI_HOME=$(dirname "$(dirname "$(which mpicc)")")
NCCL_HOME=$NVHPC_ROOT/comm_libs/nccl
[ -d "$NCCL_HOME/include" ] || NCCL_HOME=$NVHPC_ROOT/comm_libs/13.1/nccl
G8_HOME=${G8_HOME:-$ROOT/gemmul8}

# ---- GEMMul8 最小化靜態庫（只在缺 lib 時建） ----
if [ ! -f "$G8_HOME/lib/libgemmul8.a" ]; then
  if [ ! -d "$G8_HOME" ]; then
    echo "缺少 GEMMul8：git clone https://github.com/RIKEN-RCCS/GEMMul8 '$G8_HOME'" >&2
    exit 1
  fi
  echo "== building minimal GEMMul8 (9 objects) =="
  make -C "$G8_HOME" lib/libgemmul8.a -j4 \
    BACKEND=cuda CUDA_PATH="$CUDA_HOME" GPU_ARCH=90 \
    INST_MK="src/gemm/cu_recipe/instantiations.mk \
             src/oz2/scaling/accu/cu_recipe/instantiations.mk \
             src/oz2/scaling/fast/cu_recipe/instantiations.mk \
             src/oz2/scaling/general/cu_recipe/instantiations.mk \
             src/oz2/mod/cu_recipe/instantiations.mk \
             src/oz2/undo_scaling/cu_recipe/instantiations.mk \
             src/worksize/cu_recipe/instantiations.mk \
             src/trsm/cu_recipe/instantiations.mk" \
    GEMM_TYPE_TRIPLES_REAL=d_d_d GEMM_TYPE_TRIPLES_COMPLEX= GEMMLT_BACKEND_CODES= \
    ACCU_TYPE_CODES_REAL=d ACCU_TYPE_CODES_COMPLEX= ACCU_BACKEND_CODES=i8 \
    ACCU_TRI_PATTERNS=full_nonunit ACCU_UPLO_CODES= ACCU_OUT_UPLO_CODES=out_full \
    ACCU_OUT_UPLO_CODES_TRI= \
    FAST_TYPE_CODES_REAL=d FAST_TYPE_CODES_COMPLEX= FAST_BACKEND_CODES=i8 \
    FAST_TRI_PATTERNS=full_nonunit FAST_UPLO_CODES= \
    GENERAL_TYPE_CODES_REAL=d GENERAL_TYPE_CODES_COMPLEX= GENERAL_BACKEND_CODES=i8 \
    GENERAL_TRI_PATTERNS=full_nonunit GENERAL_UPLO_CODES= \
    TRSM_TYPE_PAIRS_REAL= TRSM_TYPE_PAIRS_COMPLEX= TRSM_LT_BACKEND_CODES= \
    MOD_COMPLEX_CODES=real MOD_BACKEND_CODES=i8 MOD_UPLO_CODES= MOD_UPLO_CODES_WITH_FULL=full \
    UNDO_TRIPLES=d_d_d UNDO_BACKEND_CODES=i8 UNDO_UPLO_CODES= UNDO_UPLO_CODES_WITH_FULL=full \
    UNDO_SYR2K_TRIPLES= UNDO_HER2K_TRIPLES= \
    WORKSIZE_COMPLEX_CODES=real WORKSIZE_BACKEND_CODES=i8
fi

build_one() {   # $1 = 原始檔  $2 = 輸出名
  "$CUDA_HOME/bin/nvcc" -O3 -arch=sm_90a -std=c++20 \
    -I"$MPI_HOME/include" -I"$NCCL_HOME/include" -I"$G8_HOME/include" \
    -o "$ROOT/build/$2" "$ROOT/src/$1" \
    -L"$MPI_HOME/lib" -L"$NCCL_HOME/lib" -L"$CUDA_HOME/lib64" -L"$G8_HOME/lib" \
    -lmpi -lnccl -lgemmul8 -lcublas -lcublasLt -lcusolver -lcuda -lnvidia-ml -ldl \
    -Xlinker -rpath="$MPI_HOME/lib" -Xlinker -rpath="$NCCL_HOME/lib" \
    -Xlinker -rpath="$CUDA_HOME/lib64"
  echo "BUILT $2"
}
build_one chpl4x4.cu chpl4x4     # P=4 版（兩節點/四節點冠軍）
build_one chpl2d.cu  chpl2d      # P=2 版（單節點冠軍）

# 單 GPU 基準（hpl_single，額外需要 curand）
"$CUDA_HOME/bin/nvcc" -O3 -arch=sm_90a -std=c++20 \
  -I"$G8_HOME/include" \
  -o "$ROOT/build/hpl_single" "$ROOT/src/hpl_single.cu" \
  -L"$G8_HOME/lib" -L"$CUDA_HOME/lib64" \
  -lgemmul8 -lcublas -lcublasLt -lcusolver -lcurand -lcuda -lnvidia-ml -ldl \
  -Xlinker -rpath="$CUDA_HOME/lib64"
echo "BUILT hpl_single"
