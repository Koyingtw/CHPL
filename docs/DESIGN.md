# CHPL design

This document describes how CHPL solves the HPL system on Hopper GPUs: how the
matrix is distributed, how panels are factored and pivoted, how the trailing
update is emulated with int8 tensor cores, and how data moves between GPUs. It
is written for readers who want to understand or modify the code. Build and run
instructions are in the [README](../README.md); environment variables and
sizing are in [TUNING.md](TUNING.md).

## 1. Overview

CHPL solves a dense system `Ax = b` of order N with LU factorization and
row pivoting, then checks the result with the standard HPL scaled residual.
There are three executables, all in `src/`:

| Source | Grid | Notes |
|---|---|---|
| `chpl_p4.cu` | P = 4, Q = ranks / 4 | multi-node; 4-way tournament, star-shaped row exchange, two-stage preselection |
| `chpl_p2.cu` | P = 2, Q = 4 or 8 | single 8-GPU node (or 16 ranks); 2-way tournament, pairwise row exchange |
| `chpl_single.cu` | 1 GPU, no MPI | blocked right-looking LU with exact partial pivoting; native, Ozaki-I or GEMMul8 update |

`chpl_p4` and `chpl_p2` share the same structure; `chpl_p4` generalizes the
pivot tournament and the row exchange from 2 to 4 process rows. The rest of
this document describes the distributed solvers unless noted otherwise.

The division of labour is:

* **Exact FP64:** panel factorization, the diagonal block, its triangular
  inverses, the U12 row panel, the update of the next panel's columns, and the
  back substitution.
* **Emulated FP64 (int8 tensor cores):** the bulk of the trailing-matrix
  update `A22 -= W * M`, which accounts for almost all of the `2/3 N^3` flops.
* **FP32:** only the *choice* of pivot rows. The numbers that go into the
  factors are always recomputed in FP64.

The matrix is generated on the GPU from a counter-based hash of the global
element index (entries uniform in [-0.5, 0.5)), so any rank can regenerate
any element. The right-hand side `b` is stored as column N of an `N x (N+1)`
matrix and is carried through the elimination, as in netlib HPL; the lower
factor L is therefore never needed after its panel is applied. The timer
covers factorization and back substitution, not matrix generation, and the
reported rate uses the HPL operation count `2/3 N^3 + 3/2 N^2`.

## 2. Data distribution and process grid

* **2D block-cyclic layout** with block size NB. Rank `r` has grid
  coordinates `pr = r % P`, `qc = r / P`, i.e. `rank = P * qc + pr`. Global
  row block `i` lives on process row `i % P`; global column block `j` on
  process column `j % Q`.
* Each rank stores its local part `mloc x nloc` (about `N/P x N/Q`) in one
  column-major FP64 array. The leading dimension is rounded up to a multiple
  of 128 and then padded by 64 elements to avoid pathological strides.
* **Same-node process columns.** The P ranks of a process column (P
  consecutive ranks) must share a node. All traffic inside a process column
  (pivot candidates, row exchange, U12 staging, back-substitution partial
  sums) then stays on NVLink, and only traffic along process rows can cross
  nodes. Block placement with 8 ranks per node satisfies this for both P = 2
  and P = 4.

Communicators:

| Communicator | Members | Used for |
|---|---|---|
| `nworld` | all ranks | pivot winner list, inverse diagonal pieces, final gather of x |
| `nrow` | the Q ranks with the same `pr` | panel (W) broadcast across nodes, inter-node back-substitution reduction, residual sums |
| `ncol` | the P ranks with the same `qc` | tournament candidate gather, diagonal block broadcast |

In addition, ranks on the same node open each other's buffers with CUDA IPC
(section 6); the bulk of intra-node traffic goes through that path, not NCCL.

## 3. Panel factorization

Each panel of NB columns is factored with **tournament pivoting** (CALU):

1. Every rank in the owning process column converts its local rows of the
   panel to FP32 and runs a tall `getrf` on them. The first NB pivot rows it
   selects are its candidates (only the row indices are kept).
2. The candidates' original FP64 rows and their global row numbers are sent
   over `ncol` to process row 0 of that column.
3. That rank stacks the `P * NB` candidates, converts them to FP32 and runs one
   more `getrf` to select the NB winners.
4. The winners' FP64 rows are gathered into a square `NB x NB` block and
   factored with an FP64 `getrf`. Its own partial pivoting is composed into
   the winner order. Two triangular solves produce `inv(L11)` and `inv(U11)`.
5. The winner list goes to all ranks over `nworld`; the inverse pieces follow
   (`nworld`), and the factored diagonal block is broadcast down `ncol` to the
   rank that owns the diagonal block.

FP32 only decides which rows become pivots. Any choice made by a tournament
is a valid pivot order for CALU, and the factorization itself is computed
entirely in FP64 from the original rows, so FP32 selection costs no accuracy.

The lower panel is not formed explicitly ("L21 fold"). With
`L21 = A21 * inv(U11)` and `U12 = inv(L11) * A12`, the trailing update is

```
A22 -= L21 * U12 = W * M,   W = A21 (pivoted panel rows),   M = inv(U11) * U12
```

so the panel rows W are broadcast unchanged and each rank forms M for its
own columns.

`chpl_single` uses a different scheme: exact partial-pivoting `getrf` on each
NB-wide panel inside a super-block of `K = AGG * NB` columns, then a single
trailing update of depth K with the selected backend (small trailing blocks
fall back to native DGEMM).

## 4. Trailing-update backends

The update `C -= W * M` (W is `m x NB`, M is `NB x n`) is the hot spot. Three
engines are available; `mpirun_chpl` selects GEMMul8.

### 4.1 GEMMul8, Ozaki scheme II (`CHPL2D_G8=1`, default with NB = 4096)

[GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) emulates DGEMM with the
Chinese Remainder Theorem. Rows of W and columns of M are scaled by powers of
two and rounded to integers; the integers are reduced modulo a set of
pairwise coprime moduli whose residues fit in int8; one exact int8 GEMM
(int32 accumulation) is run per modulus; and the products are recombined with
the CRT and scaled back to FP64. The number of moduli sets the precision
(`CHPL2D_G8_MODULI`, 11 by default; 12 is more conservative).

* **Fast mode** (`CHPL2D_G8_FAST=1`) chooses the scaling from cheap bounds on
  the inputs. Accurate mode spends an extra low-precision product estimating
  the magnitude of the result. On HPL's random matrices fast mode is
  accurate enough and avoids the extra product.
* **Column tiling.** Each chunk of trailing columns is processed in tiles
  of `CHPL2D_G8_NT` columns; a remainder narrower than 1024 is merged into the
  previous tile. For each tile, `M = inv(U11) * U12` is computed with FP64
  DGEMM and then passed to `gemmul8::gemm`. A wider tile uses the GEMMs more
  efficiently but needs a larger workspace, which reduces the N that fits.
* **Quantize W once.** The first call of a panel converts W to residues; all
  later tiles of the panel reuse them (`skip_scalA`).
* **Why NB = 4096.** The modular reduction and CRT steps cost
  `O(moduli * m * n)` and do not depend on the inner dimension, so doubling
  NB halves their weight relative to the int8 GEMMs. It also halves the
  number of panels, and with them the total length of the serial panel chain
  (section 7). NB of 6144 or more exceeds the shared-memory limit of the
  kernel that builds the exchange lists.

### 4.2 Ozaki scheme I (`CHPL2D_OZ=1`, best with NB = 2048)

The alternative backend splits each FP64 value into int8 digits:

```
x = s * sum_{p=0..S-1} x_p * 128^-(p+1),   x_p in [-128, 127],   s = power of two
```

with a per-row scale for W and a per-column scale for M. The default is S = 6
slices. Every slicing step multiplies by a power of two and rounds, so the
digits are exact. Base 128 instead of 256 keeps each rounded digit inside the
asymmetric int8 range (base 256 would need the digit +128).

The product needs the slice pairs `(p, q)` with `p + q <= D` (D = 5 by
default, 21 pairs). All pairs on one anti-diagonal `d = p + q` share the
weight `128^-(d+2)`, so they can be summed by one GEMM. To do this, A's slices
are concatenated along K in forward order and B's slices in reverse order:
the operands for diagonal d are then contiguous K ranges of length
`(d + 1) * NB`. **The 21 slice-pair products become 6 cuBLASLt int8 GEMMs**,
and the int32 output falls from 21 to 6 planes per element.

A reconstruction kernel combines the 6 int32 planes with the row and column
scales and subtracts the result from C. C is processed in tiles of
`CHPL2D_OZ_NT` columns (1024) with double-buffered int32 planes, so
reconstruction of one tile overlaps the GEMMs of the next. The M slices are
produced tile by tile on the panel stream while U12 is staged; W is sliced
once per panel. int8 to int32 GEMMs are exact, so any cuBLASLt algorithm gives
bit-identical results and the fastest one can be used.

The work of scheme I grows with the inner dimension for every slice pair, so
it does not benefit from NB = 4096 the way scheme II does. Using fewer
diagonals than the default fails the residual check at large N.

### 4.3 Native FP64

With both backends off, the update is plain cuBLAS DGEMM. The native path is
also used for chunks narrower than 4096 columns and for the urgent update of
the next panel's columns (section 7), which is on the critical path.

## 5. Why int8 emulation wins on H200

* **Throughput.** H200's dense int8 tensor-core peak is roughly 30 times its
  FP64 tensor-core peak. Emulation performs more operations than the FP64
  flops it replaces (about 11 full-depth int8 GEMMs with 11 moduli, or 21
  slice-pair products for scheme I), but still delivers more FP64-equivalent
  throughput. For example, `chpl_single` reaches 75.2 TFLOPS on one H200
  (README), above the 67 TFLOPS FP64 tensor-core peak.
* **Power and clock.** A large DGEMM on H200 is limited by the 700 W board
  power, so the SM clock settles well below its maximum. The int8 path uses
  less energy per operation and runs closer to the maximum clock, so the
  emulated update gains on both counts.
* **Precision budget.** HPL does not require a DGEMM accurate to the last bit.
  It requires the scaled residual
  `||Ax - b||_oo / (eps * (||A||_oo * ||x||_oo + ||b||_oo) * N)` to be below
  16. The backends are therefore tuned to the smallest precision that keeps
  the residual comfortably below the threshold, not to full FP64 accuracy.
  Under this criterion scheme II needs much less int8 work than scheme I: its
  number of moduli can be lowered in small steps, while scheme I has a hard
  minimum (S = 6, D = 5). Schemes must be compared at the target accuracy,
  not at maximum accuracy. TUNING.md discusses the accuracy margin.

Every numerical shortcut in CHPL (emulated update, FP32 selection,
preselection) is checked by the residual test at the end of every run; a
result outside the bound is reported as FAILED.

## 6. Communication

### 6.1 Intra-node: CUDA IPC and copy engines

At startup, ranks on the same node (found with `MPI_COMM_TYPE_SHARED`) open
each other's W, U, exchange, back-substitution and flag buffers with
`cudaIpcOpenMemHandle`. Transfers are `cudaMemcpyAsync` pulls from the peer's
buffer, executed by the copy engines and leaving the SMs free for the GEMMs.

Synchronization uses an array of **monotonic 64-bit flag counters** per rank.
A producer writes the sequence number of the item it has finished
(`flag_set` after `__threadfence_system`). A consumer runs a one-thread
kernel on its own stream that waits until the peer's counter reaches the
expected value. Counters only increase, so no locks, resets or ABA problems
arise, and the host never blocks.

| Flags | Meaning |
|---|---|
| 0 / 1 | W packed (owner) / W pulled (readers) |
| 2 / 3 | U12 chunk staged (`4k + chunk`) / U12 chunk consumed |
| 4 / 5 | back-substitution step reached / x piece ready |
| 6 / 7 | exchange tile packed / exchange tile consumed |

A producer reuses a buffer only after every reader has advanced the matching
"consumed" counter. W, U and exchange buffers are double-buffered by panel or
tile parity.

### 6.2 Row exchange

After the winners are known, a kernel builds the net permutation on the
device. Each winner row moves to its slot in the diagonal block; each
displaced diagonal row moves to the slot the winner vacated. The pairing order
is the same on all ranks, so the per-peer send and receive lists match
without negotiation. Row counts stay on the device and kernel grids are sized
for the worst case (at most NB rows per peer), so no device-to-host
synchronization is needed.

* **P = 2:** each rank swaps rows with its single partner in the process
  column, through IPC.
* **P = 4, star exchange:** a winner's destination is always the process row
  that owns the diagonal block, and a displaced row's source is always that
  row. Non-owner process rows therefore never exchange rows with each other,
  and the exchange is a star centred on the owner. Each rank packs outgoing
  rows into slots indexed by `[tile parity][destination row]`. Receivers
  scatter directly from the sender's IPC slot.

Exchanges are applied only to trailing columns. Columns to the left of the
current panel are not read again (back substitution uses U only), so they
are never swapped. Columns are exchanged in tiles of `CHPL2D_XT`, in the
same chunk order as the update (section 7).

### 6.3 Panel broadcast (W) and U12

* **U12** is computed by the diagonal owner row as `inv(L11) * A12` in
  `CHPL2D_XT` tiles and published chunk by chunk (flag 2). The other ranks in
  the process column pull each chunk over IPC as soon as it is ready.
* **W, single node:** W is published by the panel owner and pulled by the
  other ranks of the process row over IPC (flags 0/1).
* **W, multi-node:** W is broadcast along `nrow` with NCCL on a dedicated
  stream, split along the panel width into `CHPL2D_WCH` chunks (4 by
  default). Because a GEMM can be split along K into a sum, the urgent update
  consumes W chunk by chunk as K slabs, and the critical path pays only the
  first chunk's latency. This is the only large inter-node transfer per panel.
* **Host throttling.** In multi-node runs the host waits at the end of each
  iteration until W of that panel has arrived. Deep queues of NCCL operations
  on several communicators can otherwise deadlock. The wait is free because
  the next iteration's device work depends on that W anyway.

## 7. Pipeline, lookahead and preselection

Each rank uses several CUDA streams:

| Stream | Priority | Work |
|---|---|---|
| `sP` | high | panel chain: tournament, exchange, W, U12 staging, urgent update |
| `sW` | high | multi-node W broadcast |
| `s0` | low | trailing update (emulated GEMM) |
| `sSel` | low | preselection (FP32 selection, stage-2 tournament) |

The local trailing columns of panel k are split into four chunks: the
columns of panel k+1, the columns of panel k+2, then the first and second
halves of the remainder. Work per panel:

* **chainA(k)** on `sP`: tournament factorization of panel k (section 3).
* **chainB(k)** on `sP`: build exchange lists; exchange panel k's own
  columns; publish W; receive the inverse pieces; then, chunk by chunk,
  exchange rows, stage U12 and emit the update. Chunk 0 (panel k+1's
  columns) is updated immediately on `sP` with native DGEMM.
* **Update(k)** on `s0`: chunks 1 to 3 with the emulated GEMM, each starting
  when its U12 chunk is ready.

This is a depth-1 lookahead. Panel k+2's columns are updated first among the
`s0` chunks, so the next two panels can start early. The serial chain per
panel is

```
urgent(k-1) -> exchange(k) -> W(k) -> inverse pieces(k) -> urgent(k) -> chainA(k+1) -> ...
```

Most of the design moves work off this chain or overlaps the transfers on it
with computation.

**Preselection (`CHPL2D_PRESEL=1`).** Once chainB(k) has exchanged and
updated the columns of panel k+1, those columns hold their final values, but
chainB(k) still has the remaining exchanges and U12 staging to do. At that
point the FP32 copy of panel k+1 is taken on `sP` (for ordering), and the tall
FP32 selection `getrf` runs on `sSel`, overlapping the rest of chainB(k).
chainA(k+1) waits for the result with an event. The input data is the same,
so the selected pivots are bit-identical to inline selection. Preselection
writes into private buffers; chainA copies them into the shared candidate
buffers after the event, so no buffer is written by two streams. Delay probes
(`CHPL2D_DLY`, `CHPL2D_DLY2`) inject a spin kernel at these points and
measure how much of each step is exposed on the critical path.

**Stage-2 preselection (`CHPL2D_PRESEL2=1`, `chpl_p4` only).** The whole
tournament for panel k+1 also runs on `sSel`: candidate gather over `ncol`,
stacked FP32 selection, FP64 factorization of the diagonal block and its
inverses. Results go to private winner and inverse buffers, and chainA(k+1)
only copies them.

## 8. Back substitution and residual check

After the last panel, U and the transformed right-hand side are distributed
over the grid. Back substitution runs from the last block to the first, on
the device and without host synchronization:

1. Each rank keeps a partial sum vector of the right-hand side minus its
   local `U * x` contributions. For block j, the diagonal owner adds the
   partial sums of its process row: IPC pulls on one node, `ncclReduce` over
   `nrow` across nodes.
2. The owner solves the `NB x NB` triangular system (`trsv`) and publishes
   the x piece (flag 5).
3. The ranks of that process column pull the x piece and subtract its
   contribution from their partial sums with `gemv`. Flag 4 tracks progress
   so buffers are not overwritten early.

The pieces of x are combined with an all-reduce over `nworld`. For
verification, A and b are regenerated from the seed in NB-column chunks into
a now-idle panel buffer. `Ax`, the row sums of `|A|` and `b` are reduced over
`nrow`, and the scaled residual uses `eps = 2^-53`. The output line follows
the HPL format (`WC2 N NB P Q Time Gflops`) and ends with PASSED or FAILED.

## 9. Memory layout

Per GPU, the main allocations are:

| Buffer | Size |
|---|---|
| local matrix | `lda * nloc * 8` bytes, about `(N/P) * (N/Q) * 8` |
| W (2 buffers) | `2 * mloc * NB * 8` |
| U12 (2 buffers) | `2 * NB * nloc * 8` |
| exchange packs | `(2P + 2) * NB * XT * 8` |
| U12/M staging (3) | `3 * NB * XT * 8` |
| candidates, diagonal block and inverses | a few `NB^2 * 8` (more with PRESEL2) |
| GEMMul8 workspace | depends on `mloc`, `CHPL2D_G8_NT + 1024`, NB and the number of moduli |
| cuBLAS workspaces | `CHPL2D_WS` MB per handle |

All buffers are allocated once at startup and reused for the whole run. The
W and U12 buffers grow with NB and the GEMMul8 workspace with
`CHPL2D_G8_NT`, so a larger NB or tile width lowers the largest N that fits.
TUNING.md covers how to choose N, NB and the tile widths.
