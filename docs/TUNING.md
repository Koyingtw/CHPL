# CHPL Tuning Guide

This guide explains how to get the best performance out of CHPL while keeping the HPL residual check comfortably passing. It covers the recommended configurations, every runtime environment variable, how to choose N, NB and P, and how to benchmark reliably.

All measurements in this document were taken on NVIDIA H200 SXM GPUs (141 GB HBM3e, 8 GPUs per node, NVLink within a node, InfiniBand between nodes) with GEMMul8 v3.2.0. On other hardware (for example 80 GB H100), use the memory budget in [Choosing N and NB](#choosing-n-and-nb) to re-derive N, and re-sweep `CHPL2D_G8_NT` and `NCCL_MAX_NCHANNELS`.

Contents:
1. [Recommended configurations](#recommended-configurations)
2. [Environment variable reference](#environment-variable-reference)
3. [Choosing N and NB](#choosing-n-and-nb)
4. [Accuracy margin](#accuracy-margin)
5. [Choosing P](#choosing-p)
6. [Benchmarking methodology](#benchmarking-methodology)
7. [Platform notes](#platform-notes)

---

## Recommended configurations

`mpirun_chpl` applies the tuned settings below automatically (only for variables that are not already set in the environment), so in most cases the command line in the last column is all you need.

| Scale | Grid | Binary | N | NB | TFLOPS (median of 5) | Residual | Command |
|---|---|---|---|---|---|---|---|
| 1 GPU | 1 × 1 | `chpl_single` | 122880 | 1024 | 75.2 | 0.31–0.37 | `mpirun_chpl -P 1 -Q 1 -N 122880` |
| 8 GPUs (1 node) | P2Q4 | `chpl_p2` | 319488 | 4096 | 472.0 | 2.00 | `mpirun_chpl -P 2 -Q 4 -N 319488 --NB 4096` |
| 16 GPUs (2 nodes) | P4Q4 | `chpl_p4` | 460800 | 4096 | 989.1 | 1.94 | `mpirun_chpl -P 4 -Q 4 -N 460800 --NB 4096` |
| 32 GPUs (4 nodes) | P4Q8 | `chpl_p4` | 622592 | 4096 | 1853 | 1.40 | `mpirun_chpl -P 4 -Q 8 -N 622592 --NB 4096` |

Run-to-run spread on a fixed set of nodes was below 0.1% for every multi-GPU configuration. Residuals are deterministic: the same N, NB, grid and settings produce a bit-identical residual on every run.

### Settings behind each configuration

Multi-GPU (`chpl_p2`, `chpl_p4`), as exported by `mpirun_chpl`:

```bash
CHPL2D_OZ=0 CHPL2D_G8=1 CHPL2D_G8_MODULI=11 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=3072 \
CHPL2D_WCH=4 CHPL2D_PRESEL=1 CHPL2D_PRESEL2=1 CHPL2D_WS=64 CHPL2D_XT=6144 \
NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=1 NCCL_BUFFSIZE=2097152 \
CUDA_DEVICE_MAX_CONNECTIONS=32
```

For P = 2 (single node) the wrapper changes two values: `CHPL2D_G8_NT=2816` and `NCCL_MAX_NCHANNELS=6`. `CHPL2D_PRESEL2` is exported for P = 2 as well but is ignored by `chpl_p2`.

Single GPU (`chpl_single`), as exported by `mpirun_chpl`:

```bash
HPLS_NB=1024 HPLS_AGG=8 HPLS_BACKEND=gemmul8 HPLS_G8_MODULI=12 HPLS_G8_FAST=1 \
HPLS_G8_CHUNK=3072
chpl_single 122880
```

With `HPLS_G8_MODULI=11` the same single-GPU run reaches 79.2–79.9 TFLOPS, but the residual rises to 4.7–6.0 (threshold 16.0). The wrapper uses 12 moduli for the single GPU for that reason; see [Accuracy margin](#accuracy-margin).

### Launching without the wrapper

The tuned values are **not** the compiled-in defaults. Run directly (for example through `srun`) without exporting the variables above, `chpl_p2` and `chpl_p4` fall back to a plain FP64 cuBLAS trailing update, and `chpl_single` to the native backend. Both still pass, but at a fraction of the performance. Check the start of the output to confirm which backend is active:

```
[chpl_p4] G8 backend on: moduli=11 fast=1 nt=3072 work=... GiB (m-split x4)
```

If that line is missing, GEMMul8 is not in use.

### Other GPU counts

`chpl_p4` runs on any number of ranks divisible by 4, as long as each process column stays on one node (see [Choosing P](#choosing-p)). For example, 12 GPUs as P4Q3 with 8 + 4 block placement reached 724.2 TFLOPS (best of 3) at N = 405504, NB = 4096 with the same settings; N = 409600 ran out of memory. Start from the memory budget and step N down by NB until the run fits.

### Ozaki-I fallback

The Ozaki-I backend (`CHPL2D_G8=0 CHPL2D_OZ=1`) is kept as an alternative engine. It is 15–25% slower than GEMMul8 at every scale, and its best block size is NB = 2048. The best Ozaki-I results were:

| Scale | Grid | N | NB | TFLOPS | Settings that differ from the GEMMul8 set |
|---|---|---|---|---|---|
| 8 GPUs | P2Q4 | 348160 | 2048 | 403 | `CHPL2D_G8=0 CHPL2D_OZ=1 NCCL_MAX_NCHANNELS=6` |
| 16 GPUs | P4Q4 | 497664 | 2048 | 790 | `CHPL2D_G8=0 CHPL2D_OZ=1 NCCL_MAX_NCHANNELS=4` |
| 32 GPUs | P4Q8 | 688128 | 2048 | 1593 | `CHPL2D_G8=0 CHPL2D_OZ=1` |

Ozaki-I residuals at these sizes are about 0.5–0.7.

---

## Environment variable reference

Each table lists the compiled-in default ("Code") and the value `mpirun_chpl` exports ("Wrapper"). "Binary" says which executables read the variable: a variable not read by a binary is silently ignored by it.

### Backend selection

| Variable | Binary | Code | Wrapper | Description |
|---|---|---|---|---|
| `CHPL2D_G8` | p2, p4 | 0 | 1 | 1 = GEMMul8 (Ozaki-II, CRT) int8-emulated trailing update. Takes precedence over `CHPL2D_OZ`. |
| `CHPL2D_OZ` | p2, p4 | 0 | 0 | 1 = Ozaki-I int8 digit-slicing trailing update. Only takes effect with `CHPL2D_G8=0`. |

With both set to 0 the trailing update is native FP64 cuBLAS DGEMM. This is useful as a reference, not for performance. Only the large trailing update is emulated: panel factorization, pivoting, the triangular solves and the panel broadcast stay in exact FP64.

### GEMMul8 backend

| Variable | Binary | Code | Wrapper | Description |
|---|---|---|---|---|
| `CHPL2D_G8_MODULI` | p2, p4 | 12 | 11 | Number of CRT moduli. Each extra modulus costs one more full-k int8 GEMM. Going from 12 to 11 is 5–8% faster and raises the residual by about 13×. 11 is the lowest value that passes. See [Accuracy margin](#accuracy-margin). |
| `CHPL2D_G8_FAST` | p2, p4 | 1 | 1 | GEMMul8 fast scaling mode. On HPL matrices it is both faster (it skips a scaling GEMM) and more accurate than the accurate mode. Keep at 1. |
| `CHPL2D_G8_NT` | p2, p4 | 2048 | 3072 (P=2: 2816) | Column tile width of each emulated GEMM call. After N, this is the most important performance knob. See below. |

`CHPL2D_G8_NT` is a trade between GEMM efficiency and memory. The GEMMul8 workspace grows with `(moduli + 4) × (N/P) × (NT + 1024)` bytes, so a wider tile leaves less room for the matrix. Measured at NB = 4096:

| Grid, N | NT = 1024 | 2048 | 2560 | 2816 | 3072 | 3584 |
|---|---|---|---|---|---|---|
| P2Q4, N = 315392–319488 | 386 | 427 | 466 | **472** | 468 | — |
| P4Q4, N = 460800 | — | 883 | — | — | **977–989** | out of memory |

Use the widest NT that still fits at the N you want. Giving up a little N for a wider NT is usually worth it: at 8 GPUs, going from NT = 1024 to 2816 at almost the same N gained 22%.

### Ozaki-I backend

These apply only with `CHPL2D_G8=0 CHPL2D_OZ=1`.

| Variable | Binary | Code | Description |
|---|---|---|---|
| `CHPL2D_OZ_NT` | p2, p4 | 1024 | Column tile width of the int8 GEMMs. 1024 is clearly best. 768 is about 5% slower. 1280 is much slower (cuBLASLt has no good algorithm for that width). 1536 and 2048 are 2–3% slower. |
| `CHPL2D_OZ_ALGO` | p2, p4 | 3 | Index into the cuBLASLt heuristic list used for the int8 GEMMs. 0 and 3 are within 1% of each other. 1 is about 25% slower. |
| `CHPL2D_OZ_S`, `CHPL2D_OZ_D` | p2, p4 | 6, 5 | Number of int8 slices and fused diagonals. D = 5 is the accuracy floor (D = 4 fails the residual check). Do not change. |

The wrapper does not set any of these and leaves them at their code defaults.

### Pivot preselection

| Variable | Binary | Code | Wrapper | Description |
|---|---|---|---|---|
| `CHPL2D_PRESEL` | p2, p4 | 0 | 1 | Stage-1 preselection: the fp32 candidate selection for panel k+1 runs on a separate stream while panel k's update is still in progress, which takes it off the critical path. Worth about 1–2%. |
| `CHPL2D_PRESEL2` | **p4 only** | 0 | 1 | Stage-2 preselection: also moves the final winner selection and the exact refactorization off the critical path. Requires `CHPL2D_PRESEL=1`. Worth 1.9% at 32 GPUs with NB = 4096. Residuals are bit-identical with and without it. `chpl_p2` does not read this variable. |

Preselection has shown a rare race at small N (one residual failure in early testing at N = 131072, P = 4). It has not been seen at production sizes. A bad run always fails the residual check and never produces a silently wrong PASSED. If a small validation run fails with preselection on, rerun it, and rerun with `CHPL2D_PRESEL=0` to confirm.

### Blocking and resources

| Variable | Binary | Code | Wrapper | Description |
|---|---|---|---|---|
| `CHPL2D_XT` | p2, p4 | 8192 | 6144 | Row tile width for staging and row-swap buffers. 6144 performs the same as 8192 and frees about 0.25 GB per GPU for a larger N. |
| `CHPL2D_WCH` | p2, p4 | 4 | 4 | Number of chunks in the inter-node panel (W) broadcast. Clamped to 1–8. Only matters across nodes. At 32 GPUs: 2 is about 1.3% slower, 8 is neutral to slightly slower. |
| `CHPL2D_WS` | p2, p4 | 256 | 64 | cuBLAS workspace per handle, in MB. 64 costs no performance and saves about 0.4 GB per GPU. |

### NCCL and CUDA

| Variable | Code (NCCL/CUDA default) | Wrapper | Description |
|---|---|---|---|
| `NCCL_MAX_NCHANNELS` | NCCL default | 8 (P=2: 6) | Each NCCL channel occupies SMs that the GEMM could otherwise use. Use the fewest channels that still hide the panel broadcast. At 16 GPUs with NB = 4096: 6 is 0.3% slower than 8, 12 is 0.8% slower, 16 is 0.7% slower, and 4 is 1.3% slower. With Ozaki-I and NB = 2048 (smaller broadcasts), 4 was best across nodes and 6 within one node. |
| `NCCL_MIN_NCHANNELS` | NCCL default | 1 | Lets NCCL use fewer channels than its default minimum. |
| `NCCL_BUFFSIZE` | 4194304 | 2097152 | 2 MiB is faster than the 4 MiB default and uses less memory. 1 MiB is slower. |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 8 | 32 | Number of hardware work queues. Helps with the many concurrent streams and NCCL kernels (about +0.3%). 64 is slightly slower. |

Other NCCL settings that were tested and made no difference: `NCCL_PROTO=Simple`, `NCCL_ALGO`, `NCCL_CROSS_NIC`, `NCCL_IB_QPS_PER_CONNECTION`, `NCCL_CHUNKSIZE`. Leave them at their defaults. `mpirun_chpl` forwards any `CHPL2D_*` or `NCCL_*` variable you export to all ranks.

### Single-GPU (`chpl_single`)

`chpl_single` takes N on the command line and reads `HPLS_*` variables. Run it without arguments to print the list. It does not read any `CHPL2D_*` variable.

| Variable | Code | Wrapper | Description |
|---|---|---|---|
| `HPLS_BACKEND` | `native` | `gemmul8` | Trailing-update backend: `native` (cuBLAS DGEMM), `ozaki` (Ozaki-I) or `gemmul8`. |
| `HPLS_NB` | 2048 | 1024 | Panel width. `mpirun_chpl --NB` sets it. |
| `HPLS_AGG` | 6 | 8 | Number of panels aggregated into one trailing update, so K = NB × AGG. NB = 1024 / AGG = 8 and NB = 2048 / AGG = 4 (both K = 8192) are within 0.3% of each other. K = 12288 is slightly faster at N ≤ 114688 but does not fit at N = 122880. |
| `HPLS_G8_MODULI` | 14 | 12 | Number of CRT moduli (see [Accuracy margin](#accuracy-margin)). |
| `HPLS_G8_FAST` | 0 | 1 | GEMMul8 fast mode. Faster and more accurate on HPL matrices. |
| `HPLS_G8_CHUNK` | 16384 | 3072 | Column chunk width of each GEMMul8 call. 3072 is best at N = 122880. 2048 and 4096 are slower, and 8192 or more does not fit. |
| `HPLS_G8_SKIPA` | 1 | — | Quantize A once per K block and reuse it across chunks. Keep at 1. |
| `HPLS_OZ_NT`, `HPLS_OZ_ALGO` | 1024, 3 | — | Same meaning as the `CHPL2D_OZ_*` variables, for `HPLS_BACKEND=ozaki`. |
| `HPLS_NATIVE_TAIL` | 8192 | — | The trailing update falls back to native DGEMM once fewer than this many rows remain. |
| `HPLS_SEED` | 42 | — | Matrix generator seed. Use it to check the residual on several matrices. |

`chpl_single` checks its memory plan before allocating. If N does not fit, it prints how much is missing, suggests reducing N or K, and exits with code 4. `chpl_single --selftest` runs a small case with each backend.

### Experimental options

These variables exist in the code but did not pay off. Leave them unset.

| Variable | Binary | What it does |
|---|---|---|
| `CHPL2D_G8_MSPLIT` | p4 | Splits the W broadcast by rows and consumes it chunk by chunk. It hides the broadcast wait but moves the stall into the GEMM, so wall time does not change (and is slightly worse). |
| `CHPL2D_PKGPRE` | p4 | Prefetches the pivot package broadcast on a separate communicator. 0.7% slower because of SM contention. |
| `CHPL2D_WI8`, `CHPL2D_WI8V2` | p4 (Ozaki-I, multi-node) | Broadcasts W as int8 slices instead of FP64. 1.7–6% slower at 32 GPUs. |

### Debug variables (not for production)

These variables add synchronization, events or deliberate delays. Use them only to investigate a problem and never in a benchmark run.

| Variable | Binary | Purpose |
|---|---|---|
| `CHPL2D_TIMING` | p2, p4 | 1 = coarse per-stream time breakdown at the end of the run. 2 = fine per-panel breakdown of the update and the panel chain. |
| `CHPL2D_PROG` | p2, p4 | Prints per-panel progress to stderr. Use it to find where a hung run stopped. |
| `CHPL2D_PSYNC`, `CHPL2D_PDUMP` | p2, p4 | Preselection bisection modes and pivot-candidate checksums. |
| `CHPL2D_DLY`, `CHPL2D_DLY2`, `CHPL2D_DLYW` | p2, p4 (`DLY2`, `DLYW`: p4 only) | Inject delays into the critical path to measure how sensitive the run is to each stage. |
| `CHPL2D_OZ_BENCH`, `CHPL2D_OZ_ALGD` | p4 | Ozaki-I only: benchmark cuBLASLt algorithms at startup, or pin them per diagonal. |

---

## Choosing N and NB

### N: as large as memory allows

Performance rises with N until the run no longer fits, because the per-panel latency of the panel chain is spread over more trailing-update work. The recommended values above are close to the memory limit of a 141 GB H200.

Per-GPU memory for the GEMMul8 backend is approximately (bytes, with m = N/P local rows and n = N/Q local columns):

```
matrix          8 × m × n
panel buffers   20 × m × NB  +  16 × NB × n        (W double buffer + fp32 panel + U double buffer)
GEMMul8         (moduli + 1) × m × NB  +  (moduli + 4) × m × (NT + 1024)
fixed           ~208 × NB²  +  (2P + 5) × NB × XT × 8   (pivoting, preselection, staging buffers)
```

On top of this come the CUDA context, NCCL, cuSOLVER and cuBLAS workspaces, and these grow with the number of peers and communicators. At NB = 4096 the non-matrix part is far from negligible:

| Config | Matrix | Panel buffers | GEMMul8 | Fixed | Estimate |
|---|---|---|---|---|---|
| P2Q4, N = 319488, NT = 2816 | 102 GB | 18 GB | 17 GB | 5 GB | 143 GB |
| P4Q4, N = 460800, NT = 3072 | 106 GB | 17 GB | 13 GB | 6 GB | 142 GB |
| P4Q8, N = 622592, NT = 3072 | 97 GB | 18 GB | 17 GB | 6 GB | 138 GB |

On 141 GB H200 GPUs, the largest configurations that ran came to roughly 138–144 GB by this estimate. The next step up in N failed in each case: 323584 at P2Q4, 630784 at P4Q8, and 409600 at P4Q3. Use the formula to get a starting point, then search downward in steps of NB. Settings that change the budget:

* `CHPL2D_G8_MODULI` 11 → 12 adds about m × (NB + NT + 1024) bytes (about 1 GB at P4Q4).
* `CHPL2D_XT=6144` and `CHPL2D_WS=64` together save about 0.6 GB compared with the code defaults.
* `NCCL_BUFFSIZE=2097152` saves NCCL buffer memory compared with the 4 MiB default.

N does not have to be a multiple of NB (460800 = 112.5 × 4096). All recommended values are multiples of 2048.

### Symptoms of an N that is too large

Running out of memory does not always produce a clear out-of-memory message:

* `cudaMalloc` failure at startup: the static buffers do not fit. This is the easy case.
* `NCCL ... unhandled cuda error`, or a CUDA error late in the run: an asynchronous allocation inside NCCL or a library failed after the static buffers were placed. A run that dies near the end at a new, larger N is almost always this.

If a run fails this way, reduce N by NB (or narrow `CHPL2D_G8_NT`) before you look for other causes.

### Panel balance

The number of panels, ceil(N / NB), and how it falls across the Q process columns affects load balance at the end of the factorization. A larger N is therefore not always faster: in one measurement a larger N ran slower than a slightly smaller one. When fine-tuning N near the memory limit, try the neighboring values a step of NB apart instead of assuming that the largest N that fits is the fastest.

### NB: 4096 for GEMMul8, 2048 for Ozaki-I

The block size is tied to the backend:

* **GEMMul8: NB = 4096.** A larger NB halves the number of panels, and so the total latency of the serial panel chain, and it doubles k for each emulated GEMM, which halves GEMMul8's per-element modular reduction and reconstruction overhead. Both effects outweigh the smaller N that the larger buffers force. At NB = 2048, GEMMul8 is slower than Ozaki-I. NB = 5120 runs but is about 4% slower (947 vs 989 TFLOPS at 16 GPUs), so the curve peaks at 4096.
* **Ozaki-I: NB = 2048.** Its int8 work is proportional to k, so it gains nothing from a larger NB and loses N (716 vs 792 TFLOPS at 16 GPUs with NB = 4096).
* **NB ≥ 6144 cannot run.** The row-swap list kernel needs 40 × NB + 1024 bytes of shared memory per block, which exceeds the 227 KB per-block limit of Hopper at NB = 6144. The run fails at startup with `invalid argument` from `cudaFuncSetAttribute`.
* **Other block sizes** (1536, 2560, 3072) have failed silently or hung inside cuSOLVER in the multi-GPU solvers. Stay with 2048 or 4096.
* **Single GPU:** `chpl_single` uses a separate aggregation parameter, so NB = 1024 with AGG = 8 is the recommended setting there.

---

## Accuracy margin

The emulated trailing update is not bit-exact FP64, so the residual depends on the emulation settings. HPL's check is `||Ax-b||∞ / (ε · (||A||∞ · ||x||∞ + ||b||∞) · N) < 16.0`. Keep in mind:

**Moduli versus residual.** Each modulus removed multiplies the residual by about 13 and saves 5–8% of run time. Measured residuals:

| Setting | Multi-GPU (NB = 4096, sizes above) | Single GPU (N = 122880, K = 8192) |
|---|---|---|
| 13 moduli | ~0.01 | 0.03 |
| 12 moduli | 0.06–0.14 | 0.31–0.37 |
| 11 moduli (wrapper default, multi-GPU) | 1.4–2.2 | 4.7–6.0 |
| 10 moduli | expected to fail | expected to fail |

The multi-GPU solvers update the urgent (next-panel) columns in exact FP64 and run at moderate N per GPU, which keeps 11 moduli about 7× below the threshold. The single-GPU solver aggregates the update over K = 8192 and has less margin, so the wrapper uses 12 moduli there.

**The residual grows with N.** It grows roughly linearly with N. The margins above hold for the recommended sizes. On a larger system (more nodes, larger N), estimate the residual by scaling linearly from the nearest measured point. If the estimate gets within a factor of 2–3 of 16, switch to `CHPL2D_G8_MODULI=12` (about 5% slower).

**Fast mode is the accurate choice.** `CHPL2D_G8_FAST=1` / `HPLS_G8_FAST=1` gave lower residuals than the accurate mode on HPL's random matrices (for example 0.029 vs 0.765 with 13 moduli on a single GPU), and it is faster.

**Always check PASSED.** Every run ends with:

```
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=   1.943372891117 ...... PASSED
```

A performance number from a run that did not print PASSED is not valid. When you change moduli, N or the matrix seed (`HPLS_SEED` for the single GPU), check the residual again. Residuals are deterministic for a given configuration, so an unexpected change in the residual at an unchanged configuration indicates a problem worth investigating.

---

## Choosing P

CHPL has two multi-GPU solvers with a fixed number of process rows:

| | `chpl_p2` (P = 2) | `chpl_p4` (P = 4) |
|---|---|---|
| Ranks | 8 or 16 (P2Q4, P2Q8) | any multiple of 4 (Q = ranks / 4) |
| Best for | one 8-GPU node | two or more nodes |
| Preselection | stage 1 only | stage 1 and stage 2 |

**One node: P = 2.** With two process rows the pivoting and row-exchange protocol is simplest, and within one node the panel moves over NVLink. `chpl_p4` as P4Q2 on one node is not recommended: it was slower in tests (the 4-way exchange and doubled staging outweigh the shorter pivot factorization at Q = 2) and has failed at some sizes.

**Two or more nodes: P = 4.** Compared with P = 2 at the same GPU count, P = 4 halves the height of the pivot-candidate factorization (a shorter critical path), halves the per-rank panel broadcast volume and reduces the per-GPU panel buffers. The freed memory allows a larger N. In earlier measurements at 16 GPUs, P4Q4 ran a 4% larger N than P2Q8 and was faster. The 4-way row exchange is routed through the diagonal owner, so it costs little more than a 2-way exchange.

**Each process column must be on one node.** Ranks map as `rank = P × qc + pr`, and the P ranks of a process column (`qc` fixed) exchange U blocks and pivot rows through CUDA IPC and copy engines, which only works within a node. Block placement with consecutive ranks on the same node meets this requirement:

* 8 ranks per node (the default for `mpirun` and `srun` with 8 GPUs per node) gives each node exactly 4 process columns for P = 2 and 2 for P = 4.
* For rank counts that are not a multiple of 8 (for example 12 ranks as 8 + 4), keep block placement. `chpl_p4` prints a reminder at startup. Cyclic placement breaks the requirement.

---

## Benchmarking methodology

Differences between nodes are larger than most tuning effects, so compare carefully:

* **Node-to-node variance is 1–2%.** The same binary and configuration can differ by 1–2% between two sets of nodes, mostly because the GPUs run at different clocks under the power limit. In one case a 0.9% gap that looked like a library regression was this variance: a same-node A/B showed the two versions were identical.
* **Compare on the same nodes, interleaved.** Run A and B alternately (A, B, A, B, ...) within one allocation on the same nodes, at least 3–5 rounds each. Interleaving cancels slow drift (temperature, other jobs on the fabric). Do not compare numbers from different allocations or different nodes when the expected difference is a few percent or less.
* **Use medians.** Report the median of the rounds and the min–max range. On a fixed node set, CHPL's spread is typically below 0.1–0.3%, so a difference that does not separate the ranges is not established.
* **Tune each implementation separately.** When you compare against another HPL implementation, sweep its own N, NB and grid. Its best settings and memory limit differ from CHPL's (for example, implementations with FP64 DGEMM tend to use NB = 1024). Implementations that run FP64 tensor cores near the power limit can also be more sensitive to the node's clock behavior than CHPL's int8 path, which widens their spread.
* **Count only PASSED runs.**
* **Re-sweep at each scale.** A knob that was decisive at one scale must be swept again at others. For example, `CHPL2D_G8_NT` worth +10% at 16 GPUs was worth +22% at 8 GPUs, where it had initially been tested only at a value that was too small.

---

## Platform notes

* **CUDA library path.** Put the lib64 directory of the CUDA toolkit you built against first in `LD_LIBRARY_PATH`. Loading cuBLAS and cuBLASLt from different CUDA versions (for example one from the HPC SDK and one from a system toolkit) can segfault at the first GEMM. `mpirun_chpl` forwards `LD_LIBRARY_PATH` to all ranks.
* **Read-only `/tmp`.** `mpirun_chpl -N` writes a temporary input file to `$TMPDIR` (default `/tmp`). On compute nodes where `/tmp` is read-only, set `TMPDIR=/dev/shm`.
* **Mixed InfiniBand / RoCE adapters.** If some nodes have both IB and RoCE ports, NCCL may pair incompatible ports when the layout is asymmetric ("Remote IB device is incompatible with the local RoCE"). Restrict NCCL to the IB ports with `NCCL_IB_HCA==<list of IB HCAs>`. Each port's link layer is in `/sys/class/infiniband/<hca>/ports/<n>/link_layer`.
* **Do not free and reallocate large buffers between runs.** CHPL allocates all large GPU buffers once and keeps them for the whole run. Repeatedly freeing and reallocating buffers of this size within one process has been observed to corrupt data at the driver level. If you embed the solver or loop over several problem sizes in one process, keep this pattern or start a new process for each run.
* **Rebuild all binaries together.** An executable built before a feature existed silently ignores that feature's environment variables and runs the fallback path. After you update the source, rebuild every `chpl_*` binary and check the backend line at startup.
* **Bad GPUs.** A node whose GPUs fail `cudaGetDeviceCount` or `cudaSetDevice` fails every binary. Exclude it rather than retrying.
