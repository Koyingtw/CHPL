#!/usr/bin/env bash
# Configure, build and install CHPL.
set -euo pipefail

usage() {
  cat <<USAGE
Usage: ./install.sh [options]

  -h|--help              print this help message
  -g|--debug             build type Debug (default: Release)
  --prefix=<dir>         install location (default: build/chpl-install)
  --with-cuda=<dir>      CUDA toolkit (default: \$CUDA_HOME or /usr/local/cuda)
  --with-mpi=<dir>       MPI install (default: the mpicc found in PATH)
  --with-nccl=<dir>      NCCL install (default: \$NCCL_HOME, then the CUDA toolkit)
  --arch=<sm>            CUDA architecture (default: 90a, i.e. H100/H200)
  -j <n>                 parallel build jobs (default: nproc)
USAGE
}

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_TYPE=Release
PREFIX="$ROOT/build/chpl-install"
CUDA_DIR="${CUDA_HOME:-/usr/local/cuda}"
MPI_DIR=""
NCCL_DIR="${NCCL_HOME:-}"
ARCH=90a
JOBS="$(nproc 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -g|--debug) BUILD_TYPE=Debug ;;
    --prefix=*) PREFIX="${1#*=}" ;;
    --with-cuda=*) CUDA_DIR="${1#*=}" ;;
    --with-mpi=*) MPI_DIR="${1#*=}" ;;
    --with-nccl=*) NCCL_DIR="${1#*=}" ;;
    --arch=*) ARCH="${1#*=}" ;;
    -j) JOBS="$2"; shift ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

if [[ ! -f "$ROOT/gemmul8/Makefile" ]]; then
  echo "== fetching GEMMul8 submodule =="
  git -C "$ROOT" submodule update --init --recursive
fi

CMAKE_ARGS=(
  -S "$ROOT" -B "$ROOT/build"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
  -DCMAKE_CUDA_COMPILER="$CUDA_DIR/bin/nvcc"
  -DCUDAToolkit_ROOT="$CUDA_DIR"
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
  -DGEMMUL8_JOBS="$JOBS"
)
[[ -n "$MPI_DIR" ]] && CMAKE_ARGS+=(-DMPI_HOME="$MPI_DIR" -DMPI_CXX_COMPILER="$MPI_DIR/bin/mpicxx")
[[ -n "$NCCL_DIR" ]] && CMAKE_ARGS+=(-DNCCL_ROOT="$NCCL_DIR")

cmake "${CMAKE_ARGS[@]}"
cmake --build "$ROOT/build" -j "$JOBS"
cmake --install "$ROOT/build"
echo "CHPL installed to $PREFIX/bin"
