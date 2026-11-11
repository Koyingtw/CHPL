# CHPL
CHPL is a CUDA implementation of the [HPL][] (High-Performance Linpack) benchmark for NVIDIA Hopper GPUs (H200 / H100, `sm_90a`). It is a rewrite of the netlib HPL-2.3 solver in which the trailing-matrix update runs as int8-emulated FP64 DGEMM via [GEMMul8][] (Ozaki scheme II), while panel factorization and pivoting stay in exact FP64. Every run passes the standard HPL residual check.

On H200 it outperforms the NVIDIA HPC-Benchmarks 26.02 container (same nodes, interleaved runs):

| GPUs | CHPL | NVIDIA HPC-Benchmarks 26.02 |
|---|---|---|
| 8 × H200 (1 node) | 470.8 TFLOPS | 385.9 TFLOPS |
| 16 × H200 (2 nodes) | 984.4 TFLOPS | 782.5 TFLOPS |

Both were measured back to back on the same nodes. The best results per scale are listed under [Performance evaluation](#performance-evaluation).

## Requirements
* Git
* CMake (3.18 or later)
* CUDA Toolkit 13.x (cuBLAS, cuBLASLt, cuSOLVER, cuRAND)
* MPI (Open MPI / HPC-X)
* [NCCL][]
* NVIDIA Hopper GPU (H200 or H100)

The NVIDIA HPC SDK provides MPI and NCCL and works out of the box.

## Quickstart CHPL build and install

#### Install script
You can build CHPL using the `install.sh` script
```
# Clone CHPL using git (GEMMul8 is a submodule)
git clone --recurse-submodules https://github.com/Koyingtw/CHPL.git

# Go to CHPL directory
cd CHPL

# Run install.sh script
# Command line options:
#    -h|--help            - prints this help message
#    -g|--debug           - Set build type to Debug (otherwise build Release)
#    --prefix=<dir>       - Path to CHPL install location (Default: build/chpl-install)
#    --with-cuda=<dir>    - Path to CUDA toolkit (Default: $CUDA_HOME or /usr/local/cuda)
#    --with-mpi=<dir>     - Path to MPI install (Default: mpicc found in PATH)
#    --with-nccl=<dir>    - Path to NCCL install (Default: $NCCL_HOME, then the CUDA toolkit)
#    --arch=<sm>          - CUDA architecture (Default: 90a)
#    -j <n>               - Parallel build jobs (Default: nproc)
./install.sh
```
`install.sh` builds a minimal static GEMMul8 library (only the `gemm<double, INT8>` path, 9 object files) from the `gemmul8/` submodule, then the CHPL executables. After building, the executables are placed in `build/chpl-install/bin`:

| Executable | Process grid | Use |
|---|---|---|
| `chpl_p4` | P = 4, Q = ranks / 4 | 2 or more nodes (P4Q4 on 16 GPUs, P4Q8 on 32 GPUs) |
| `chpl_p2` | P = 2, Q = 4 or 8 (8 or 16 ranks) | a single 8-GPU node |
| `chpl_single` | 1 GPU, no MPI | single-GPU runs |

With the NVIDIA HPC SDK (MPI in `PATH`), for example:
```
./install.sh --with-cuda=/usr/local/cuda-13.0 --with-nccl=$NVHPC_ROOT/comm_libs/nccl
```

## Running CHPL benchmark application
CHPL provides a wrapper script for launching via `mpirun`, `mpirun_chpl`. It picks the executable from the process grid and applies the tuned settings listed under [Tuning](#tuning). The script has two run modes:
```
mpirun_chpl -P <P> -Q <Q> -N <N> --NB <NB>
# where
# P       - is the number of rows in the MPI grid (1, 2 or 4)
# Q       - is the number of columns in the MPI grid
# N       - is the total number of rows/columns of the global matrix
# NB      - is the panel size in the blocking algorithm (default 4096; 1024 for -P 1 -Q 1)
```
This run script will launch a total of np=PxQ MPI processes, one per GPU. `-P 1 -Q 1` runs `chpl_single` without MPI.

The second run mode takes an input file:
```
mpirun_chpl -P <P> -Q <Q> -i <input>
# where
# input   - is the input filename (default: the HPL.dat installed next to the
#           script, sized for 16 GPUs)
```

The executables read the problem size, block size and residual threshold from a standard HPL input file (lines 6, 8 and 13). The process grid comes from the number of MPI ranks, so the remaining lines are ignored. If the file is missing or shorter than 13 lines, the executables fall back to N=16384, NB=1024:
```
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
460800       Ns
1            # of NBs
4096         NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
4            Ps
4            Qs
16.0         threshold
```

Extra launcher options can be passed through `MPIRUN_OPTS` (for example `MPIRUN_OPTS="--hostfile hosts"`), and `MPIRUN` selects another launcher.

Ranks are mapped as `rank = P * qc + pr`, and each group of P consecutive ranks (one process column) must be placed on the same node. Block placement with 8 ranks per node (the default for `mpirun` and `srun`) satisfies this. Users launching through a workload manager such as Slurm can call the executables directly, as long as the tuned environment variables are exported:
```
srun -N 2 --ntasks-per-node=8 --gpus-per-node=8 ./build/chpl-install/bin/chpl_p4 HPL.dat
```

## Performance evaluation
CHPL is weak scaled so that the global matrix fills the memory of all GPUs. The best results on 141 GB H200 GPUs (median of 5 runs) were obtained with:
* 1 H200: `mpirun_chpl -P 1 -Q 1 -N 122880` (75.2 TFLOPS)
* 8 H200: `mpirun_chpl -P 2 -Q 4 -N 319488 --NB 4096` (472.0 TFLOPS)
* 16 H200: `mpirun_chpl -P 4 -Q 4 -N 460800 --NB 4096` (989.1 TFLOPS)
* 32 H200: `mpirun_chpl -P 4 -Q 8 -N 622592 --NB 4096` (1853 TFLOPS)

Performance is reported at the end of the run in FP64 floating point operations per second.

## Tuning
`mpirun_chpl` sets the following defaults for the MPI solvers, and only for variables that are not already set in the environment. The executables' built-in defaults are more conservative, so export these when launching the executables directly.

| Variable | Default | Description |
|---|---|---|
| `CHPL2D_G8` | 1 | GEMMul8 (Ozaki-II) trailing update |
| `CHPL2D_OZ` | 0 | Ozaki-I trailing update (used only when `CHPL2D_G8=0`) |
| `CHPL2D_G8_MODULI` | 11 | number of moduli; 12 is more conservative (lower residual, about 5% slower) |
| `CHPL2D_G8_FAST` | 1 | GEMMul8 fast mode |
| `CHPL2D_G8_NT` | 3072 (P=2: 2816) | column tile width of the emulated GEMM |
| `CHPL2D_PRESEL` / `CHPL2D_PRESEL2` | 1 / 1 | pivot preselection one panel ahead (stage 1 / stage 2, P=4 only) |
| `CHPL2D_WCH` | 4 | number of chunks for the inter-node panel broadcast |
| `CHPL2D_WS` | 64 | cuBLAS workspace in MB |
| `CHPL2D_XT` | 6144 | column tile width for the row exchange and staging |
| `NCCL_MAX_NCHANNELS` / `NCCL_MIN_NCHANNELS` | 8 (P=2: 6) / 1 | fewer NCCL channels leave more SMs to the GEMM |
| `NCCL_BUFFSIZE` | 2097152 | NCCL buffer size |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 32 | hardware work queues |

`CHPL2D_OZ=1` (with `CHPL2D_G8=0`) selects the Ozaki-I digit-slicing backend instead, which works best with NB=2048.

`chpl_single` is configured with `HPLS_*` variables (run it without arguments to list them). For `-P 1 -Q 1`, `mpirun_chpl` sets `HPLS_BACKEND=gemmul8 HPLS_NB=1024 HPLS_AGG=8 HPLS_G8_MODULI=12 HPLS_G8_FAST=1 HPLS_G8_CHUNK=3072`.

Notes:
* NB=4096 is the best block size for the GEMMul8 backend. NB of 6144 or more exceeds the shared-memory limit of one of the kernels.
* Memory per GPU is dominated by the local matrix, `(N/P) * (N/Q) * 8` bytes, plus panel buffers, the GEMMul8 workspace and fixed buffers that grow with NB (about 36 GB per GPU in total for P4Q4, N=460800, NB=4096); see [docs/TUNING.md](docs/TUNING.md). If N is too large, the run may fail late with an NCCL or CUDA error instead of a clear out-of-memory message.
* If `/tmp` is read-only on compute nodes, set `TMPDIR=/dev/shm`.

## Documentation
* [docs/DESIGN.md](docs/DESIGN.md) - how CHPL works: data distribution, panel factorization, emulated DGEMM backends, communication
* [docs/TUNING.md](docs/TUNING.md) - environment variable reference, choosing N and NB, accuracy margin, benchmarking methodology
* [CHANGELOG.md](CHANGELOG.md) - release notes

## Testing CHPL
At the end of each benchmark run, residual error checking is computed, and PASSED or FAILED is printed to output.

A quick check of each code path:
```
chpl_single --selftest
mpirun_chpl -P 2 -Q 4 -N 65536
mpirun_chpl -P 4 -Q 4 -N 98304
```
`--selftest` prints `SELFTEST_PASS`, and the MPI runs should report PASSED.

## License
CHPL is released under the [BSD 3-Clause License](LICENSE). It is derived from netlib HPL-2.3; the original HPL copyright notice is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). GEMMul8 is distributed under its own MIT License.

[HPL]: https://www.netlib.org/benchmark/hpl/
[GEMMul8]: https://github.com/RIKEN-RCCS/GEMMul8
[NCCL]: https://github.com/NVIDIA/nccl
