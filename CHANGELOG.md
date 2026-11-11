# Changelog

## 1.0.0

First public release.

* `chpl_p4`: multi-GPU solver with a P = 4 process grid (Q = ranks / 4), for two or more 8-GPU nodes.
* `chpl_p2`: multi-GPU solver with a P = 2 process grid (8 or 16 ranks), for a single 8-GPU node.
* `chpl_single`: single-GPU solver with selectable native, Ozaki-I and GEMMul8 trailing-update backends.
* Trailing update through int8-emulated FP64 DGEMM: GEMMul8 (Ozaki scheme II, default) or Ozaki-I digit slicing.
* Tournament pivoting (CALU) with fp32 candidate selection and exact fp64 refactorization.
* Intra-node transfers through CUDA IPC and copy engines; inter-node broadcasts through NCCL.
* CMake build and `install.sh`, with a minimal GEMMul8 v3.2.0 build from the `gemmul8/` submodule.
* `mpirun_chpl` launcher with tuned defaults for H200.
