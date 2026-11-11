/*
 * CHPL — CUDA HPL: GPU-resident High Performance Linpack for NVIDIA Hopper (H100/H200)
 *
 * A rewrite of the netlib hpl-2.3 benchmark solver core for CUDA:
 *   - Entire matrix resident in GPU HBM (no host staging)
 *   - 1 x Q column-block-cyclic layout (one MPI rank per GPU)
 *   - Device-side panel factorization (custom fused kernels or cuSOLVER)
 *   - Trailing update via cuBLAS DGEMM / DTRSM (FP64 tensor cores)
 *   - NCCL panel broadcast over NVLink / IB
 *   - Depth-1 lookahead: next panel factored on a high-priority stream
 *     while the bulk trailing update proceeds on the update stream.
 *   - Fully asynchronous host enqueue: no host<->device sync inside the
 *     factorization loop.
 *
 * Keeps hpl-2.3 semantics: HPL.dat input format, N x (N+1) augmented system,
 * LU with partial pivoting (pivots applied only to the trailing submatrix,
 * exactly like HPL), residual check
 *   ||Ax-b||_oo / (eps * (||A||_oo * ||x||_oo + ||b||_oo) * N)  <  threshold
 * and HPL-style result output.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <nccl.h>
#define NVSHMEM_HOSTLIB_ONLY   /* host-issued collectives only, no RDC */
#include <nvshmem.h>
#include <nvshmemx.h>

#define CHECK_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[rank %d] CUDA error %s:%d: %s\n", g_rank, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define CHECK_CUBLAS(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "[rank %d] cuBLAS error %s:%d: %d\n", g_rank, __FILE__, __LINE__, (int)s_); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define CHECK_CUSOLVER(x) do { cusolverStatus_t s_ = (x); if (s_ != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "[rank %d] cuSOLVER error %s:%d: %d\n", g_rank, __FILE__, __LINE__, (int)s_); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define CHECK_NCCL(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { \
    fprintf(stderr, "[rank %d] NCCL error %s:%d: %s\n", g_rank, __FILE__, __LINE__, ncclGetErrorString(r_)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define CHECK_MPI(x) do { int e_ = (x); if (e_ != MPI_SUCCESS) { \
    fprintf(stderr, "[rank %d] MPI error %s:%d\n", g_rank, __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

static int g_rank = 0, g_size = 1;
static int g_nvshmem = 0;   /* panel broadcasts via nvshmemx on-stream colls */

typedef unsigned long long u64;

/* ------------------------------------------------------------------ */
/* Matrix generation: counter-based RNG, uniform in [-0.5, 0.5).      */
/* Element (i,j) of the global N x (N+1) system depends only on       */
/* (seed, i, j) so any rank can regenerate any entry (verification).  */
/* ------------------------------------------------------------------ */
__device__ __forceinline__ double chpl_elem(u64 seed, u64 idx)
{
    u64 z = (idx + 1ULL) * 0x9E3779B97F4A7C15ULL + seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

/* Generate ncols columns; column c of dst is global column gcol0st[c].
 * Rows are the full 0..m-1 (global row == local row in 1xQ layout). */
__global__ void matgen_kernel(double *dst, size_t ldd, int m,
                              const long long *gcols, int ncols,
                              long long ncols_glob, u64 seed)
{
    int c = blockIdx.y;
    if (c >= ncols) return;
    long long gc = gcols[c];
    double *col = dst + (size_t)c * ldd;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < m;
         r += gridDim.x * blockDim.x) {
        col[r] = chpl_elem(seed, (u64)r * (u64)ncols_glob + (u64)gc);
    }
}

/* ------------------------------------------------------------------ */
/* Row interchanges (HPL_dlaswp equivalent) applied to a contiguous   */
/* range of local columns.  ipiv[i] is 1-based relative to row r0.    */
/* One thread per column, swaps applied sequentially (order matters). */
/* ------------------------------------------------------------------ */
__global__ void laswp_kernel(double *A, size_t lda, int r0,
                             const int *ipiv, int jb,
                             size_t c0, int ncols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncols) return;
    double *col = A + (c0 + (size_t)c) * lda + r0;
    for (int i = 0; i < jb; i++) {
        int p = ipiv[i] - 1;
        if (p != i) {
            double t = col[i];
            col[i] = col[p];
            col[p] = t;
        }
    }
}

/* ------------------------------------------------------------------ */
/* fp64 -> fp32 panel cast (pivot-selection-only getrf)                */
__global__ void d2s_kernel(const double *A, size_t lda, float *F, size_t ldf,
                           long long m, int n)
{
    int c = blockIdx.y;
    if (c >= n) return;
    const double *Ac = A + (size_t)c * lda;
    float *Fc = F + (size_t)c * ldf;
    for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x; r < m;
         r += (long long)gridDim.x * blockDim.x)
        Fc[r] = (float)Ac[r];
}

/* no-pivot LU of a w x w diagonal block (w <= 64) in shared memory,
 * plus inv(L) (unit lower) and inv(U); single block of 64 threads.   */
__global__ void nplu_diag_kernel(double *A11, size_t lda, int w,
                                 double *invL, double *invU)
{
    __shared__ double a[64][65], il[64][65], iu[64][65];
    int t = threadIdx.x;
    if (t < w)
        for (int j = 0; j < w; j++) a[t][j] = A11[(size_t)j * lda + t];
    __syncthreads();
    for (int c = 0; c < w; c++) {
        if (t > c && t < w) {
            double l = a[t][c] / a[c][c];
            a[t][c] = l;
            for (int j = c + 1; j < w; j++) a[t][j] -= l * a[c][j];
        }
        __syncthreads();
    }
    if (t < w) {
        /* forward substitution, column t of inv(L) (unit diagonal) */
        for (int i = 0; i < w; i++) {
            double x = (i == t) ? 1.0 : 0.0;
            for (int k = 0; k < i; k++) x -= a[i][k] * il[k][t];
            il[i][t] = x;
        }
        /* back substitution, column t of inv(U) */
        for (int i = w - 1; i >= 0; i--) {
            double x = (i == t) ? 1.0 : 0.0;
            for (int k = i + 1; k < w; k++) x -= a[i][k] * iu[k][t];
            iu[i][t] = x / a[i][i];
        }
        for (int j = 0; j < w; j++) {
            A11[(size_t)j * lda + t] = a[t][j];
            invL[(size_t)j * 64 + t] = il[t][j];
            invU[(size_t)j * 64 + t] = iu[t][j];
        }
    }
}

/* ------------------------------------------------------------------ */
/* monotonic sequence flags for the cross-process copy-engine panel
 * broadcast; one slot per (kind, panel parity) so each slot is written
 * from a single stream and stays monotone */
__global__ void flag_set_kernel(volatile long long *flag, long long v)
{
    __threadfence_system();
    *flag = v;
}
__global__ void flag_wait_kernel(const volatile long long *f0,
                                 const volatile long long *f1,
                                 const volatile long long *f2, long long v)
{
    while ((f0 && *f0 < v) || (f1 && *f1 < v) || (f2 && *f2 < v))
        __nanosleep(200);
}

/* Resolved-permutation row exchange (replaces sequential laswp):     */
/* simulate the pivot swap sequence once, producing                   */
/*   srcTop[i]   : original panel-relative row whose content is the   */
/*                 final content of top row i (gathered into U12      */
/*                 staging, so top rows of A are never written back)  */
/*   dst/srcBelow: displaced rows that must be written below the      */
/*                 panel (at most jb of them)                         */
/* ------------------------------------------------------------------ */
#define PERM_HASH 4096
#define PERM_MAXJB 2048
__global__ void resolve_perm_kernel(const int *ipiv, int jb,
                                    int *srcTop, int *dstBelow, int *srcBelow,
                                    int *d_nbelow)
{
    __shared__ int top[PERM_MAXJB];
    __shared__ int hkey[PERM_HASH];
    __shared__ int hval[PERM_HASH];
    __shared__ int cnt;
    int tid = threadIdx.x;
    for (int i = tid; i < jb; i += blockDim.x) top[i] = i;
    for (int i = tid; i < PERM_HASH; i += blockDim.x) hkey[i] = -1;
    if (tid == 0) cnt = 0;
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < jb; i++) {
            int p = ipiv[i] - 1;
            if (p == i) continue;
            if (p < jb) {
                int t = top[i]; top[i] = top[p]; top[p] = t;
            } else {
                unsigned h = ((unsigned)p * 2654435761u) & (PERM_HASH - 1);
                int occ = p, slot;
                while (true) {
                    if (hkey[h] == p) { occ = hval[h]; slot = h; break; }
                    if (hkey[h] == -1) { slot = h; break; }
                    h = (h + 1) & (PERM_HASH - 1);
                }
                hkey[slot] = p;
                hval[slot] = top[i];
                top[i] = occ;
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < jb; i += blockDim.x) srcTop[i] = top[i];
    __syncthreads();
    for (int i = tid; i < PERM_HASH; i += blockDim.x) {
        if (hkey[i] >= 0 && hval[i] != hkey[i]) {
            int j = atomicAdd(&cnt, 1);
            dstBelow[j] = hkey[i];
            srcBelow[j] = hval[i];
        }
    }
    __syncthreads();
    if (tid == 0) *d_nbelow = cnt;
}

/* Fast row exchange, second generation.  Mapping: warp spans 32 row slots
 * (i) so the dense-side access (ld = jb) is fully coalesced; the scattered
 * A-side access is irreducibly 8B-granular.  threadIdx.y covers several
 * columns and each thread walks columns with an unrolled stride loop for
 * memory-level parallelism (the old kernels were latency-bound). */
__global__ void gather_rows_kernel(const double * __restrict__ A, size_t lda,
                                   int r0, const int * __restrict__ rowlist,
                                   int nrows, const int * __restrict__ d_n,
                                   size_t cs, long long nc,
                                   double * __restrict__ dst, int ldd)
{
    int n = (nrows >= 0) ? nrows : *d_n;
    int i = blockIdx.x * 32 + threadIdx.x;
    if (i >= n) return;
    const double *Ac = A + cs * lda + (size_t)(r0 + rowlist[i]);
    long long c = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    const long long cstep = (long long)gridDim.y * blockDim.y;
    for (; c + 3 * cstep < nc; c += 4 * cstep) {
        double v0 = Ac[c * lda];
        double v1 = Ac[(c + cstep) * lda];
        double v2 = Ac[(c + 2 * cstep) * lda];
        double v3 = Ac[(c + 3 * cstep) * lda];
        dst[i + (size_t)c * ldd] = v0;
        dst[i + (size_t)(c + cstep) * ldd] = v1;
        dst[i + (size_t)(c + 2 * cstep) * ldd] = v2;
        dst[i + (size_t)(c + 3 * cstep) * ldd] = v3;
    }
    for (; c < nc; c += cstep)
        dst[i + (size_t)c * ldd] = Ac[c * lda];
}

/* A[dstlist[i] + r0, column cs+c] = src[i + c*lds] */
__global__ void scatter_rows_kernel(double * __restrict__ A, size_t lda,
                                    int r0, const int * __restrict__ dstlist,
                                    const int * __restrict__ d_n, size_t cs,
                                    long long nc,
                                    const double * __restrict__ src, int lds)
{
    int n = *d_n;
    int i = blockIdx.x * 32 + threadIdx.x;
    if (i >= n) return;
    double *Ac = A + cs * lda + (size_t)(r0 + dstlist[i]);
    long long c = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    const long long cstep = (long long)gridDim.y * blockDim.y;
    for (; c + 3 * cstep < nc; c += 4 * cstep) {
        double v0 = src[i + (size_t)c * lds];
        double v1 = src[i + (size_t)(c + cstep) * lds];
        double v2 = src[i + (size_t)(c + 2 * cstep) * lds];
        double v3 = src[i + (size_t)(c + 3 * cstep) * lds];
        Ac[c * lda] = v0;
        Ac[(c + cstep) * lda] = v1;
        Ac[(c + 2 * cstep) * lda] = v2;
        Ac[(c + 3 * cstep) * lda] = v3;
    }
    for (; c < nc; c += cstep)
        Ac[c * lda] = src[i + (size_t)c * lds];
}

/* ------------------------------------------------------------------ */
/* Device-side panel factorization (dgetf2-style, blocked)            */
/* Panel: m x jb at (r0, column c0panel) inside local matrix, lda.    */
/* All kernels read pivot info from device memory: no host syncs.     */
/* ------------------------------------------------------------------ */

/* pass 1: per-block abs-max of column segment */
__global__ void iamax_part_kernel(const double *col, int len,
                                  double *pval, int *pidx)
{
    __shared__ double sval[256];
    __shared__ int    sidx[256];
    int tid = threadIdx.x;
    double best = -1.0; int bi = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < len;
         i += gridDim.x * blockDim.x) {
        double v = fabs(col[i]);
        if (v > best) { best = v; bi = i; }
    }
    sval[tid] = best; sidx[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sval[tid + s] > sval[tid]) {
            sval[tid] = sval[tid + s]; sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) { pval[blockIdx.x] = sval[0]; pidx[blockIdx.x] = sidx[0]; }
}

/* pass 2: reduce partials, then swap row (r0+c) <-> pivot row across the
 * whole panel width jb, and record ipiv[c] (1-based, relative to r0).
 * Single block. */
__global__ void iamax_swap_kernel(double *panel /* = A + c0*lda + r0 */,
                                  size_t lda, int m, int jb, int c,
                                  const double *pval, const int *pidx,
                                  int npart, int *ipiv, int *dinfo)
{
    __shared__ double sval[256];
    __shared__ int    sidx[256];
    __shared__ int    s_p;
    int tid = threadIdx.x;
    double best = -1.0; int bi = 0;
    for (int i = tid; i < npart; i += blockDim.x) {
        if (pval[i] > best) { best = pval[i]; bi = pidx[i]; }
    }
    sval[tid] = best; sidx[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sval[tid + s] > sval[tid]) {
            sval[tid] = sval[tid + s]; sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        int p = c + sidx[0];          /* absolute row within panel */
        s_p = p;
        ipiv[c] = p + 1;
        if (sval[0] == 0.0 && *dinfo == 0) *dinfo = c + 1;
    }
    __syncthreads();
    int p = s_p;
    if (p != c) {
        for (int j = tid; j < jb; j += blockDim.x) {
            double t = panel[(size_t)j * lda + c];
            panel[(size_t)j * lda + c] = panel[(size_t)j * lda + p];
            panel[(size_t)j * lda + p] = t;
        }
    }
}

/* Fused scale + rank-1 update inside the current sub-panel.
 * Column c: L(c+1:m, c) /= A(c,c); A(c+1:m, c+1:send) -= L * A(c, c+1:send)
 * Pivot value read from device memory (post-swap A[c,c]).             */
#define GETF2_MAXW 64
__global__ void scale_ger_kernel(double *panel, size_t lda, int m,
                                 int c, int send /* sub-panel col end */)
{
    __shared__ double urow[GETF2_MAXW];
    __shared__ double pivinv;
    int w = send - (c + 1);
    if (threadIdx.x == 0) pivinv = 1.0 / panel[(size_t)c * lda + c];
    for (int j = threadIdx.x; j < w; j += blockDim.x)
        urow[j] = panel[(size_t)(c + 1 + j) * lda + c];
    __syncthreads();
    for (int r = c + 1 + blockIdx.x * blockDim.x + threadIdx.x; r < m;
         r += gridDim.x * blockDim.x) {
        double l = panel[(size_t)c * lda + r] * pivinv;
        panel[(size_t)c * lda + r] = l;
        for (int j = 0; j < w; j++)
            panel[(size_t)(c + 1 + j) * lda + r] -= l * urow[j];
    }
}

/* ------------------------------------------------------------------ */
/* Verification helpers                                               */
/* ------------------------------------------------------------------ */

/* accumulate row sums of |A| for a chunk (m rows x ncols), ld ldc */
__global__ void rowsum_kernel(const double *chunk, size_t ldc, int m,
                              int ncols, double *rowsum)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;
    double s = 0.0;
    for (int c = 0; c < ncols; c++)
        s += fabs(chunk[(size_t)c * ldc + r]);
    rowsum[r] += s;
}

__global__ void axpby_res_kernel(const double *b, const double *ax,
                                 double *r, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i] - ax[i];
}

__global__ void setzero_kernel(double *p, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0;
}

/* write a jb x jb identity (column-major, ld jb) */
__global__ void seteye_kernel(double *p, int jb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < jb * jb) p[i] = (i / jb == i % jb) ? 1.0 : 0.0;
}

/* ------------------------------------------------------------------ */
/* Host-side helpers                                                   */
/* ------------------------------------------------------------------ */

struct Grid {
    int Q;                       /* number of ranks (1 x Q grid)       */
    int q;                       /* my rank                            */
};

struct RunCfg {
    int N, NB;
    double thresh;
    u64 seed;
    int use_cusolver_panel;
    int progress;
    int lookahead;
};

/* absmax of device vector using cublas (untimed phase helper) */
static double dev_absmax(cublasHandle_t h, const double *d, int n)
{
    if (n <= 0) return 0.0;
    int idx = 0;
    CHECK_CUBLAS(cublasIdamax(h, n, d, 1, &idx));
    double v = 0.0;
    CHECK_CUDA(cudaMemcpy(&v, d + (idx - 1), sizeof(double), cudaMemcpyDeviceToHost));
    return fabs(v);
}

/* exact: count local columns of rank q with global col < t */
static long long local_cols_below_exact(long long t, int NB, int Q, int q,
                                        long long ntot)
{
    if (t > ntot) t = ntot;
    long long cnt = 0;
    long long nblocks = (ntot + NB - 1) / NB;
    for (long long b = q; b < nblocks; b += Q) {
        long long lo = b * NB, hi = std::min(ntot, (b + 1) * (long long)NB);
        if (lo >= t) break;
        cnt += std::min(hi, t) - lo;
    }
    return cnt;
}

/* ------------------------------------------------------------------ */
/* Persistent device memory pool.  Allocated once, reused across runs. */
/* NOTE: per-run cudaFree/cudaMalloc cycling was observed to corrupt   */
/* freshly reallocated buffers on this driver (580.65.06) when a       */
/* different-size sweep point followed; a single long-lived allocation */
/* avoids the free/realloc reuse path entirely.                        */
/* ------------------------------------------------------------------ */
struct DevPool {
    double *A = nullptr, *panel[2] = {nullptr, nullptr};
    double *x = nullptr, *acc = nullptr, *tmp = nullptr;
    double *rowsum = nullptr, *b = nullptr, *r = nullptr;
    double *iaval = nullptr, *work = nullptr;
    double *tmpU = nullptr, *tmpP = nullptr;   /* U12 staging for trsm->gemm */
    double *tmpU2 = nullptr, *tmpP2 = nullptr; /* displaced-row staging      */
    int *srcTop[2] = {nullptr, nullptr}, *dstBelow[2] = {nullptr, nullptr},
        *srcBelow[2] = {nullptr, nullptr}, *nbelow[2] = {nullptr, nullptr};
    int *ipiv[2] = {nullptr, nullptr}, *iaidx = nullptr, *infoArr = nullptr;
    long long *gcol = nullptr, *gc2 = nullptr;
    size_t maxA = 0, maxP = 0;
    int maxN = 0, maxNB = 0, maxPanels = 0;
    long long maxNloc = 0;
    int lwork = 0;
};

static size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

static void pool_init(DevPool &P, const std::vector<int> &Ns,
                      const std::vector<int> &NBs, int Q, int q,
                      int use_cusolver, cusolverDnHandle_t csP)
{
    for (int N : Ns) {
        for (int NB : NBs) {
            long long ntot = (long long)N + 1;
            size_t lda = (((size_t)N + 127) & ~127ULL) + 64;
            long long nloc = local_cols_below_exact(ntot, NB, Q, q, ntot);
            P.maxA = std::max(P.maxA, lda * (size_t)nloc * sizeof(double));
            /* panel payload: m x jb factors followed by jb x jb inv(L11) */
            P.maxP = std::max(P.maxP,
                              ((size_t)N * NB + (size_t)NB * NB) * sizeof(double));
            P.maxN = std::max(P.maxN, N);
            P.maxNB = std::max(P.maxNB, NB);
            P.maxPanels = std::max(P.maxPanels, (N + NB - 1) / NB);
            P.maxNloc = std::max(P.maxNloc, nloc);
        }
    }
    const int IAMAX_NPART = 1024;
    size_t off = 0;
    auto carve = [&](size_t bytes) {
        size_t o = off;
        off += align_up(bytes, 256);
        return o;
    };
    size_t oA = carve(P.maxA);
    /* with NVSHMEM the panel buffers must live in the symmetric heap */
    size_t oP0 = g_nvshmem ? 0 : carve(P.maxP);
    size_t oP1 = g_nvshmem ? 0 : carve(P.maxP);
    size_t ox = carve((size_t)P.maxN * 8), oacc = carve((size_t)P.maxN * 8);
    size_t otmp = carve((size_t)P.maxNB * 8);
    size_t ors = carve((size_t)P.maxN * 8), ob = carve((size_t)P.maxN * 8),
           orr = carve((size_t)P.maxN * 8);
    size_t oiv = carve(IAMAX_NPART * 8), oii = carve(IAMAX_NPART * 4);
    size_t otU = carve((size_t)P.maxNB * P.maxNloc * 8);
    size_t otU2 = carve((size_t)P.maxNB * P.maxNloc * 8);
    size_t otP2 = carve(2 * (size_t)P.maxNB * P.maxNB * 8);
    size_t operm[2][4];
    for (int b = 0; b < 2; b++) {
        operm[b][0] = carve((size_t)P.maxNB * 4);
        operm[b][1] = carve((size_t)P.maxNB * 4);
        operm[b][2] = carve((size_t)P.maxNB * 4);
        operm[b][3] = carve(4);
    }
    size_t otP = carve(2 * (size_t)P.maxNB * P.maxNB * 8);  /* stage + stage2 */
    size_t oip0 = carve((size_t)P.maxNB * 4), oip1 = carve((size_t)P.maxNB * 4);
    size_t oinfo = carve((size_t)(P.maxPanels + 1) * 4);
    size_t ogcol = carve((size_t)P.maxNloc * 8);
    size_t ogc2 = carve((size_t)P.maxNB * 8);
    unsigned char *base = nullptr;
    CHECK_CUDA(cudaMalloc(&base, off));
    CHECK_CUDA(cudaMemset(base, 0, off));
    if (g_rank == 0)
        printf("[chpl] device pool: %.2f GiB (single allocation)\n",
               off / 1073741824.0);
    P.A = (double *)(base + oA);
    if (g_nvshmem) {
        P.panel[0] = (double *)nvshmem_malloc(P.maxP);
        P.panel[1] = (double *)nvshmem_malloc(P.maxP);
        if (!P.panel[0] || !P.panel[1]) {
            fprintf(stderr, "[rank %d] nvshmem_malloc(%zu) failed\n", g_rank,
                    P.maxP);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CHECK_CUDA(cudaMemset(P.panel[0], 0, P.maxP));
        CHECK_CUDA(cudaMemset(P.panel[1], 0, P.maxP));
    } else {
        P.panel[0] = (double *)(base + oP0);
        P.panel[1] = (double *)(base + oP1);
    }
    P.x = (double *)(base + ox);
    P.acc = (double *)(base + oacc);
    P.tmp = (double *)(base + otmp);
    P.rowsum = (double *)(base + ors);
    P.b = (double *)(base + ob);
    P.r = (double *)(base + orr);
    P.iaval = (double *)(base + oiv);
    P.iaidx = (int *)(base + oii);
    P.tmpU = (double *)(base + otU);
    P.tmpU2 = (double *)(base + otU2);
    P.tmpP2 = (double *)(base + otP2);
    for (int b = 0; b < 2; b++) {
        P.srcTop[b] = (int *)(base + operm[b][0]);
        P.dstBelow[b] = (int *)(base + operm[b][1]);
        P.srcBelow[b] = (int *)(base + operm[b][2]);
        P.nbelow[b] = (int *)(base + operm[b][3]);
    }
    P.tmpP = (double *)(base + otP);
    P.ipiv[0] = (int *)(base + oip0);
    P.ipiv[1] = (int *)(base + oip1);
    P.infoArr = (int *)(base + oinfo);
    P.gcol = (long long *)(base + ogcol);
    P.gc2 = (long long *)(base + ogc2);
    if (use_cusolver) {
        CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(csP, P.maxN, P.maxNB, P.A,
                                                   (int)((((size_t)P.maxN + 127) & ~127ULL) + 64),
                                                   &P.lwork));
        CHECK_CUDA(cudaMalloc(&P.work, (size_t)P.lwork * sizeof(double)));
    }
}

/* ------------------------------------------------------------------ */
/* One benchmark run: factor + solve + verify for a given (N, NB)      */
/* ------------------------------------------------------------------ */
struct RunResult {
    double time_s;
    double gflops;
    double resid;
    int passed;
    int info;
    double norm_r, norm_A, norm_x, norm_b;
    time_t t_start, t_end;
};

static RunResult run_one(const RunCfg &cfg, const Grid &g, DevPool &pool,
                         cublasHandle_t &hU, cublasHandle_t &hP,
                         cublasHandle_t &hV, cusolverDnHandle_t csP,
                         cudaStream_t sU, cudaStream_t sP, cudaStream_t sC,
                         cudaStream_t sV, cudaStream_t sC2,
                         ncclComm_t nccl, ncclComm_t nccl2)
{
    RunResult res = {};
    const int N = cfg.N, NB = cfg.NB, Q = g.Q, q = g.q;
    const long long ntot = (long long)N + 1;          /* incl. b column   */
    const long long nblocks = (ntot + NB - 1) / NB;
    const int npanels = (N + NB - 1) / NB;

    /* ---- local columns ---- */
    std::vector<long long> h_gcol;
    for (long long b = q; b < nblocks; b += Q) {
        long long lo = b * NB, hi = std::min(ntot, (b + 1) * (long long)NB);
        for (long long c = lo; c < hi; c++) h_gcol.push_back(c);
    }
    const long long nloc = (long long)h_gcol.size();
    size_t lda = ((size_t)N + 127) & ~127ULL;
    lda += 64;                                         /* break 4K aliasing */

    size_t bytesA = (size_t)lda * nloc * sizeof(double);
    size_t bytesP = (size_t)N * NB * sizeof(double);
    if (g_rank == 0 && cfg.progress)
        printf("[chpl] N=%d NB=%d Q=%d npanels=%d nloc=%lld lda=%zu "
               "A: %.2f GiB/GPU  panels: %.2f GiB\n",
               N, NB, Q, npanels, nloc, lda,
               bytesA / 1073741824.0, 2.0 * bytesP / 1073741824.0);

    /* ---- persistent pool views (no per-run cudaMalloc/cudaFree) ---- */
    double *d_A = pool.A, *d_panel[2] = {pool.panel[0], pool.panel[1]};
    int *d_ipiv[2] = {pool.ipiv[0], pool.ipiv[1]};
    long long *d_gcol = pool.gcol;
    int *d_infoArr = pool.infoArr;
    const int IAMAX_NPART = 1024;
    double *d_iaval = pool.iaval;
    int *d_iaidx = pool.iaidx;
    double *d_work = pool.work;
    CHECK_CUDA(cudaMemset(d_infoArr, 0, (npanels + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_gcol, h_gcol.data(), nloc * sizeof(long long),
                          cudaMemcpyHostToDevice));

    /* ---- generate matrix ---- */
    {
        int nybatch = 32768;
        for (long long c0 = 0; c0 < nloc; c0 += nybatch) {
            int nc = (int)std::min((long long)nybatch, nloc - c0);
            dim3 grid(64, nc);
            matgen_kernel<<<grid, 256, 0, sU>>>(d_A + (size_t)c0 * lda, lda, N,
                                                d_gcol + c0, nc, ntot, cfg.seed);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(sU));
    }
    /* warmup */
    {
        double one = 1.0, zero = 0.0;
        CHECK_CUBLAS(cublasDgemm(hU, CUBLAS_OP_N, CUBLAS_OP_N, 256, 256, 256,
                                 &one, d_panel[0], 256, d_panel[1], 256, &zero,
                                 d_panel[0] + 256 * 256, 256));
        CHECK_CUDA(cudaStreamSynchronize(sU));
    }

    /* ---- events ---- */
    std::vector<cudaEvent_t> evPack(npanels + 2), evBcH(npanels + 2),
                             evBcP1(npanels + 2), evBcP2(npanels + 2),
                             evUpd(npanels + 2), evUseP(npanels + 2),
                             evUpdA(npanels + 2), evUpdB(npanels + 2),
                             evBcRaw(npanels + 2);
    for (int i = 0; i < npanels + 2; i++) {
        CHECK_CUDA(cudaEventCreateWithFlags(&evPack[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evBcH[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evBcP1[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evBcP2[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evUpd[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evUseP[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evUpdA[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evUpdB[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evBcRaw[i], cudaEventDisableTiming));
    }

    const double done_ = 1.0, dmone = -1.0;

    /* ---- CUDA IPC copy-engine panel broadcast (skip under NVSHMEM) ----
     * flags: [0]=pack seq (even panels), [1]=pack (odd), [2]=pull (even),
     * [3]=pull (odd) */
    static long long *d_flags = nullptr;
    static double *peer_panel[8][2] = {};
    static int *peer_ipiv[8][2] = {};
    static long long *peer_flags[8] = {};
    static int ce_init = 0;
    /* CE bcast measured ~0.3s slower than NCCL for 1x8 -> default off */
    int ce_bcast = 0;
    { const char *e = getenv("CHPL_CEBCAST"); if (e) ce_bcast = atoi(e) && Q > 1 && !g_nvshmem; }
    if (ce_bcast && !ce_init) {
        ce_init = 1;
        CHECK_CUDA(cudaMalloc(&d_flags, 8 * 8));
        CHECK_CUDA(cudaMemset(d_flags, 0, 8 * 8));
        struct HOff {
            cudaIpcMemHandle_t hbase, hflags;
            long long op[2], oip[2];
        } mine, all[8];
        CHECK_CUDA(cudaIpcGetMemHandle(&mine.hbase, pool.A));
        CHECK_CUDA(cudaIpcGetMemHandle(&mine.hflags, d_flags));
        for (int b = 0; b < 2; b++) {
            mine.op[b] = (long long)((char *)pool.panel[b] - (char *)pool.A);
            mine.oip[b] = (long long)((char *)pool.ipiv[b] - (char *)pool.A);
        }
        CHECK_MPI(MPI_Allgather(&mine, sizeof(HOff), MPI_BYTE, all,
                                sizeof(HOff), MPI_BYTE, MPI_COMM_WORLD));
        for (int r = 0; r < Q; r++) {
            char *pb;
            if (r == g_rank) pb = (char *)pool.A;
            else CHECK_CUDA(cudaIpcOpenMemHandle((void **)&pb, all[r].hbase,
                                cudaIpcMemLazyEnablePeerAccess));
            for (int b = 0; b < 2; b++) {
                peer_panel[r][b] = (double *)(pb + all[r].op[b]);
                peer_ipiv[r][b] = (int *)(pb + all[r].oip[b]);
            }
            if (r == g_rank) peer_flags[r] = d_flags;
            else CHECK_CUDA(cudaIpcOpenMemHandle((void **)&peer_flags[r],
                                all[r].hflags, cudaIpcMemLazyEnablePeerAccess));
        }
    }
    /* flags persist across sweep runs; seq values keep growing, so reset
     * per run by remembering the base */
    if (ce_bcast) CHECK_CUDA(cudaMemset(d_flags, 0, 8 * 8));

    /* fp32-selection + all-GEMM no-pivot LU panel factorization */
    /* measured slower than cuSOLVER (fact 2.83->4.20s) + an intermittent
     * residual failure -> default off, kept for reference */
    int nplu = 0;
    { const char *e = getenv("CHPL_NPLU"); if (e) nplu = atoi(e); }
    static float *d_selF = nullptr;      /* m x jb fp32 panel */
    static double *d_iLU = nullptr;      /* 2 x 64x64 inverses */
    static double *d_tmpL = nullptr;     /* (m x 64) L21 staging */
    static double *d_tmpA12 = nullptr;   /* (64 x NB) U12 staging */
    if (nplu && !d_selF) {
        CHECK_CUDA(cudaMalloc(&d_selF, (size_t)pool.maxN * pool.maxNB * 4));
        CHECK_CUDA(cudaMalloc(&d_iLU, 2 * 64 * 64 * 8));
        CHECK_CUDA(cudaMalloc(&d_tmpL, (size_t)pool.maxN * 64 * 8));
        CHECK_CUDA(cudaMalloc(&d_tmpA12, (size_t)64 * pool.maxNB * 8));
    }

    /* device panel factorization: panel at (r0, local col c0l), m x jb,
     * in place inside d_A; ipiv (1-based rel r0) written to d_ipiv_out. */
    auto factor_panel = [&](int r0, long long c0l, int m, int jb,
                            int *d_ipiv_out, int *d_info, cudaStream_t st) {
        double *panel = d_A + c0l * lda + r0;
        if (nplu) {
            /* fp32 getrf selects the pivots; the swaps are applied to the
             * fp64 panel and the factorization is redone without pivoting
             * as pure GEMM-shaped work (64-wide diag blocks in smem). */
            {
                dim3 gg(128, jb);
                d2s_kernel<<<gg, 256, 0, st>>>(panel, lda, d_selF, (size_t)m,
                                               m, jb);
            }
            CHECK_CUSOLVER(cusolverDnSgetrf(csP, m, jb, d_selF, m,
                                            (float *)d_work, d_ipiv_out,
                                            d_info));
            laswp_kernel<<<(jb + 127) / 128, 128, 0, st>>>(
                d_A, lda, r0, d_ipiv_out, std::min(m, jb), (size_t)c0l, jb);
            double *invL = d_iLU, *invU = d_iLU + 64 * 64;
            const double one_ = 1.0, mone_ = -1.0, zero_ = 0.0;
            for (int sub = 0; sub < jb; sub += 64) {
                int w = std::min(64, jb - sub);
                double *As = panel + (size_t)sub * lda + sub;
                nplu_diag_kernel<<<1, 64, 0, st>>>(As, lda, w, invL, invU);
                int mb = m - sub - w, tc = jb - sub - w;
                if (mb > 0) {
                    /* L21 = A21 * inv(U11) (staged: gemm can't run in place) */
                    CHECK_CUBLAS(cublasDgemm(hP, CUBLAS_OP_N, CUBLAS_OP_N, mb,
                                             w, w, &one_, As + w, (int)lda,
                                             invU, 64, &zero_, d_tmpL, mb));
                    CHECK_CUDA(cudaMemcpy2DAsync(As + w, lda * 8, d_tmpL,
                                                 (size_t)mb * 8,
                                                 (size_t)mb * 8, w,
                                                 cudaMemcpyDeviceToDevice,
                                                 st));
                }
                if (tc > 0) {
                    /* U12 = inv(L11) * A12 (staged) */
                    double *A12 = panel + (size_t)(sub + w) * lda + sub;
                    CHECK_CUDA(cudaMemcpy2DAsync(d_tmpA12, (size_t)w * 8, A12,
                                                 lda * 8, (size_t)w * 8, tc,
                                                 cudaMemcpyDeviceToDevice,
                                                 st));
                    CHECK_CUBLAS(cublasDgemm(hP, CUBLAS_OP_N, CUBLAS_OP_N, w,
                                             tc, w, &one_, invL, 64, d_tmpA12,
                                             w, &zero_, A12, (int)lda));
                    if (mb > 0)
                        CHECK_CUBLAS(cublasDgemm(
                            hP, CUBLAS_OP_N, CUBLAS_OP_N, mb, tc, w, &mone_,
                            As + w, (int)lda, A12, (int)lda, &one_,
                            panel + (size_t)(sub + w) * lda + sub + w,
                            (int)lda));
                }
            }
            return;
        }
        if (cfg.use_cusolver_panel) {
            /* note: cuSOLVER writes ipiv rel. to the submatrix (1-based) */
            CHECK_CUSOLVER(cusolverDnDgetrf(csP, m, jb, panel, (int)lda,
                                            d_work, d_ipiv_out, d_info));
            return;
        }
        const int ib = GETF2_MAXW;      /* sub-panel width */
        for (int s = 0; s < jb; s += ib) {
            int w = std::min(ib, jb - s);
            for (int c = s; c < s + w; c++) {
                int len = m - c;
                const double *colp = panel + (size_t)c * lda + c;
                int npart = std::min(IAMAX_NPART, (len + 255) / 256);
                iamax_part_kernel<<<npart, 256, 0, st>>>(colp, len, d_iaval, d_iaidx);
                iamax_swap_kernel<<<1, 256, 0, st>>>(panel, lda, m, jb, c,
                                                     d_iaval, d_iaidx, npart,
                                                     d_ipiv_out, d_info);
                if (c + 1 < m) {
                    int nrow = m - (c + 1);
                    int nblk = std::min(2048, (nrow + 255) / 256);
                    scale_ger_kernel<<<nblk, 256, 0, st>>>(panel, lda, m, c, s + w);
                }
            }
            /* pivot swaps were applied panel-wide inside iamax_swap_kernel */
            /* update trailing panel columns [s+w, jb) */
            int tc = jb - (s + w);
            if (tc > 0) {
                CHECK_CUBLAS(cublasDtrsm(hP, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                         CUBLAS_OP_N, CUBLAS_DIAG_UNIT, w, tc, &done_,
                                         panel + (size_t)s * lda + s, (int)lda,
                                         panel + (size_t)(s + w) * lda + s, (int)lda));
                if (m - (s + w) > 0)
                    CHECK_CUBLAS(cublasDgemm(hP, CUBLAS_OP_N, CUBLAS_OP_N,
                                             m - (s + w), tc, w, &dmone,
                                             panel + (size_t)s * lda + s + w, (int)lda,
                                             panel + (size_t)(s + w) * lda + s, (int)lda,
                                             &done_,
                                             panel + (size_t)(s + w) * lda + s + w, (int)lda));
            }
        }
    };

    /* plain column-major panel buffer, ld = m; inv(L11) at offset m*jb */
    auto pack_panel = [&](int r0, long long c0l, int m, int jb, int buf,
                          cudaStream_t st) {
        CHECK_CUDA(cudaMemcpy2DAsync(d_panel[buf], (size_t)m * sizeof(double),
                                     d_A + c0l * lda + r0, lda * sizeof(double),
                                     (size_t)m * sizeof(double), jb,
                                     cudaMemcpyDeviceToDevice, st));
    };

    /* replace the left unit-lower triangular solve with a GEMM against a
     * precomputed inv(L11) (broadcast with the panel):  DTRSM runs at
     * ~10 TF/s on H200 while DGEMM runs at 60+ TF/s. */
    const char *tgenv = getenv("CHPL_TRSMGEMM");
    int tg = tgenv ? atoi(tgenv) : 1;
    /* inv(L11) GEMM works for any NB; only fastswap needs jb <= 2048 */
    const char *fsenv = getenv("CHPL_FASTSWAP");
    int fastswap = (tg && fsenv) ? atoi(fsenv) : tg;
    if (NB > PERM_MAXJB) fastswap = 0;
    const char *dcenv = getenv("CHPL_DUALCOMM");
    const int dualcomm = nccl2 ? (dcenv ? atoi(dcenv) : 1) : 0;
    const double dzero = 0.0;

    /* trailing update of local columns [cs, cs+nc), pipelined against the
     * split broadcast — both trailing GEMMs keep the FULL k=jb:
     *   evs[0] head  (ipiv + inv(L11)) : laswp, U12 staging, U12 GEMM
     *   evs[1] part1 (rows [0,R1))     : trailing GEMM on rows [jb, R1)
     *   evs[2] part2 (rows [R1,m))     : trailing GEMM on rows [R1, m)
     * evs == nullptr: panel already resident (same-stream producer).
     * tev (optional, timing): {after row exchange, after U12 update} */
    auto update_cols = [&](int r0, int m, int jb, int buf, long long cs,
                           long long nc, cublasHandle_t hh, cudaStream_t st,
                           double *stage, cudaEvent_t *evs, cudaEvent_t *tev) {
        if (nc <= 0) {
            /* still record timing markers: ranks can run out of trailing
             * columns near the end of the factorization */
            if (tev) {
                CHECK_CUDA(cudaEventRecord(tev[0], st));
                CHECK_CUDA(cudaEventRecord(tev[1], st));
            }
            return;
        }
        if (evs) CHECK_CUDA(cudaStreamWaitEvent(st, evs[0], 0));
        if (fastswap) {
            /* single-pass exchange with the resolved net permutation:
             * gather P*A12 straight into the U12 staging buffer (top rows
             * of A are overwritten by the inv(L11) GEMM below, so they are
             * never written back), park displaced-row values in stage2 and
             * scatter them below the panel. */
            double *stage2 = stage + (size_t)jb * nc;
            dim3 blk(32, 8);
            dim3 grd((jb + 31) / 32,
                     (unsigned)std::min((nc + 7) / 8, (long long)512));
            gather_rows_kernel<<<grd, blk, 0, st>>>(d_A, lda, r0,
                                                    pool.srcTop[buf], jb,
                                                    nullptr, (size_t)cs, nc,
                                                    stage, jb);
            gather_rows_kernel<<<grd, blk, 0, st>>>(d_A, lda, r0,
                                                    pool.srcBelow[buf], -1,
                                                    pool.nbelow[buf],
                                                    (size_t)cs, nc, stage2, jb);
            scatter_rows_kernel<<<grd, blk, 0, st>>>(d_A, lda, r0,
                                                     pool.dstBelow[buf],
                                                     pool.nbelow[buf],
                                                     (size_t)cs, nc, stage2, jb);
            if (tev) CHECK_CUDA(cudaEventRecord(tev[0], st));
            CHECK_CUBLAS(cublasDgemm(hh, CUBLAS_OP_N, CUBLAS_OP_N,
                                     jb, (int)nc, jb, &done_,
                                     d_panel[buf] + (size_t)m * jb, jb,
                                     stage, jb, &dzero,
                                     d_A + cs * lda + r0, (int)lda));
        } else if (tg) {
            laswp_kernel<<<(int)((nc + 127) / 128), 128, 0, st>>>(
                d_A, lda, r0, d_ipiv[buf], jb, (size_t)cs, (int)nc);
            if (tev) CHECK_CUDA(cudaEventRecord(tev[0], st));
            /* U12 = inv(L11) * A12 via staging copy + GEMM */
            CHECK_CUDA(cudaMemcpy2DAsync(stage, (size_t)jb * sizeof(double),
                                         d_A + cs * lda + r0,
                                         lda * sizeof(double),
                                         (size_t)jb * sizeof(double), nc,
                                         cudaMemcpyDeviceToDevice, st));
            CHECK_CUBLAS(cublasDgemm(hh, CUBLAS_OP_N, CUBLAS_OP_N,
                                     jb, (int)nc, jb, &done_,
                                     d_panel[buf] + (size_t)m * jb, jb,
                                     stage, jb, &dzero,
                                     d_A + cs * lda + r0, (int)lda));
        } else {
            laswp_kernel<<<(int)((nc + 127) / 128), 128, 0, st>>>(
                d_A, lda, r0, d_ipiv[buf], jb, (size_t)cs, (int)nc);
            if (tev) CHECK_CUDA(cudaEventRecord(tev[0], st));
            if (evs) CHECK_CUDA(cudaStreamWaitEvent(st, evs[2], 0));
            CHECK_CUBLAS(cublasDtrsm(hh, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_N, CUBLAS_DIAG_UNIT, jb, (int)nc, &done_,
                                     d_panel[buf], m,
                                     d_A + cs * lda + r0, (int)lda));
        }
        if (tev) CHECK_CUDA(cudaEventRecord(tev[1], st));
        if (m - jb > 0) {
            if (evs) CHECK_CUDA(cudaStreamWaitEvent(st, evs[2], 0));
            CHECK_CUBLAS(cublasDgemm(hh, CUBLAS_OP_N, CUBLAS_OP_N,
                                     m - jb, (int)nc, jb, &dmone,
                                     d_panel[buf] + jb, m,
                                     d_A + cs * lda + r0, (int)lda, &done_,
                                     d_A + cs * lda + r0 + jb, (int)lda));
        }
    };

    /* resolve the net row permutation of a factored panel (fastswap path) */
    auto resolve_perm = [&](int buf, int jb, cudaStream_t st) {
        if (!fastswap) return;
        resolve_perm_kernel<<<1, 256, 0, st>>>(d_ipiv[buf], jb,
                                               pool.srcTop[buf],
                                               pool.dstBelow[buf],
                                               pool.srcBelow[buf],
                                               pool.nbelow[buf]);
    };

    /* compute inv(L11) of a factored panel into the extra payload region of
     * the panel buffer.  L11 is read from `src` with leading dim `srcld`
     * (owner path: in d_A; redundant path: inside the panel buffer). */
    auto make_linv = [&](const double *src, int srcld, int m, int jb, int buf,
                         cublasHandle_t hh, cudaStream_t st) {
        if (!tg) return;
        double *linv = d_panel[buf] + (size_t)m * jb;
        seteye_kernel<<<(jb * jb + 255) / 256, 256, 0, st>>>(linv, jb);
        CHECK_CUBLAS(cublasDtrsm(hh, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_UNIT, jb, jb, &done_,
                                 src, srcld, linv, jb));
    };

    /* Redundant tail factorization: for panels with m <= redT the RAW panel
     * is broadcast and every rank factors it locally (bit-identical result),
     * removing the factor stage from the inter-rank critical chain.  The
     * extra factor work overlaps each rank's own trailing GEMMs. */
    const char *rtenv = getenv("CHPL_REDFACT");
    const long long redT = rtenv ? atoll(rtenv) : 0;   /* measured: no gain */
    auto is_red = [&](int j) {
        return Q > 1 && cfg.use_cusolver_panel && cfg.lookahead && j > 0 &&
               (long long)(N - j * NB) <= redT;
    };

    /* optional phase timing (CHPL_TIMING=1): timing-capable event chains */
    const int timing = getenv("CHPL_TIMING") ? 1 : 0;
    std::vector<cudaEvent_t> tR0, tRL, tRT, tRG, tP0, tPF, tPP;
    if (timing) {
        tR0.resize(npanels); tRL.resize(npanels); tRT.resize(npanels);
        tRG.resize(npanels); tP0.resize(npanels + 1); tPF.resize(npanels + 1);
        tPP.resize(npanels + 1);
        for (int i = 0; i < npanels; i++) {
            CHECK_CUDA(cudaEventCreate(&tR0[i]));
            CHECK_CUDA(cudaEventCreate(&tRL[i]));
            CHECK_CUDA(cudaEventCreate(&tRT[i]));
            CHECK_CUDA(cudaEventCreate(&tRG[i]));
        }
        for (int i = 0; i < npanels + 1; i++) {
            CHECK_CUDA(cudaEventCreate(&tP0[i]));
            CHECK_CUDA(cudaEventCreate(&tPF[i]));
            CHECK_CUDA(cudaEventCreate(&tPP[i]));
        }
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    double t0 = MPI_Wtime();
    res.t_start = time(NULL);

    /* ---- factor panel 0 (owner) ---- */
    if (npanels > 0 && q == 0 % Q) {
        int jb = std::min(NB, N);
        long long c0l = local_cols_below_exact(0, NB, Q, q, ntot); /* = 0 */
        if (timing) CHECK_CUDA(cudaEventRecord(tP0[0], sP));
        factor_panel(0, c0l, N, jb, d_ipiv[0], d_infoArr + 0, sP);
        if (timing) CHECK_CUDA(cudaEventRecord(tPF[0], sP));
        resolve_perm(0, jb, sP);
        make_linv(d_A + c0l * lda, (int)lda, N, jb, 0, hP, sP);
        pack_panel(0, c0l, N, jb, 0, sP);
        if (timing) CHECK_CUDA(cudaEventRecord(tPP[0], sP));
    }
    CHECK_CUDA(cudaEventRecord(evPack[0], sP));
    if (Q == 1) {
        CHECK_CUDA(cudaEventRecord(evBcH[0], sP));
        CHECK_CUDA(cudaEventRecord(evBcP1[0], sP));
        CHECK_CUDA(cudaEventRecord(evBcP2[0], sP));
    }

    /* ---- main loop: fully asynchronous enqueue ---- */
    for (int k = 0; k < npanels; k++) {
        const int r0 = k * NB;
        const int m = N - r0;
        const int jb = std::min(NB, N - r0);
        const int root = k % Q;
        const int buf = k & 1;

        const int red = is_red(k);

        /* --- broadcast of panel k ---
         * non-red: head (ipiv + inv(L11)) then the factored panel.
         * red: single broadcast of the RAW panel; every rank then factors
         * it locally on sP (factor leaves the inter-rank critical chain). */
        if (Q > 1) {
            /* with dual communicators, even/odd panels broadcast on
             * independent comm/stream pairs so consecutive panel broadcasts
             * overlap (helps the latency-bound tail) */
            ncclComm_t bco = (dualcomm && (k & 1)) ? nccl2 : nccl;
            cudaStream_t bst = (dualcomm && (k & 1)) ? sC2 : sC;
            if (q == root) {
                CHECK_CUDA(cudaStreamWaitEvent(bst, evPack[k], 0));
            } else if (k >= 2) {
                CHECK_CUDA(cudaStreamWaitEvent(bst, evUpd[k - 2], 0));
                CHECK_CUDA(cudaStreamWaitEvent(bst, evUseP[k - 2], 0));
            }
            /* the big panel payload goes over NVSHMEM when enabled (lower
             * per-op overhead than NCCL ring broadcast); NVSHMEM team
             * collectives must not run concurrently, so they always use sC */
            cudaStream_t pst = g_nvshmem ? sC : bst;
            if (red) {
                if (g_nvshmem) {
                    if (pst != bst) {
                        CHECK_CUDA(cudaEventRecord(evBcH[k], bst));
                        CHECK_CUDA(cudaStreamWaitEvent(pst, evBcH[k], 0));
                    }
                    if (nvshmemx_broadcastmem_on_stream(NVSHMEM_TEAM_WORLD,
                            d_panel[buf], d_panel[buf],
                            (size_t)m * jb * sizeof(double), root, pst)) {
                        fprintf(stderr, "[rank %d] nvshmem bcast failed\n", g_rank);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                } else {
                    CHECK_NCCL(ncclBroadcast(d_panel[buf], d_panel[buf],
                                             (size_t)m * jb, ncclDouble, root,
                                             bco, bst));
                }
                CHECK_CUDA(cudaEventRecord(evBcRaw[k], pst));
            } else if (ce_bcast) {
                /* copy-engine pulls: zero SM usage, frees power for GEMM */
                const int par = k & 1;
                if (k >= 2 && (k - 2) % Q == q) {
                    /* peers must have consumed my k-2 panel before this
                     * buffer parity is overwritten (by my pull below) */
                    int t = 0; const volatile long long *w[9] = {};
                    for (int r = 0; r < Q; r++)
                        if (r != g_rank) w[t++] = peer_flags[r] + 2 + par;
                    for (int g = 0; g < t; g += 3)
                        flag_wait_kernel<<<1, 1, 0, bst>>>(
                            w[g], g + 1 < t ? w[g + 1] : nullptr,
                            g + 2 < t ? w[g + 2] : nullptr, (long long)k - 1);
                }
                if (q == root) {
                    flag_set_kernel<<<1, 1, 0, bst>>>(d_flags + 0 + par,
                                                      (long long)k + 1);
                } else {
                    flag_wait_kernel<<<1, 1, 0, bst>>>(
                        peer_flags[root] + 0 + par, nullptr, nullptr,
                        (long long)k + 1);
                    size_t bytes = ((size_t)m * jb +
                                    (tg ? (size_t)jb * jb : 0)) * 8;
                    CHECK_CUDA(cudaMemcpyAsync(d_panel[buf],
                                               peer_panel[root][buf], bytes,
                                               cudaMemcpyDeviceToDevice, bst));
                    CHECK_CUDA(cudaMemcpyAsync(d_ipiv[buf],
                                               peer_ipiv[root][buf],
                                               (size_t)jb * 4,
                                               cudaMemcpyDeviceToDevice, bst));
                    resolve_perm(buf, jb, bst);
                }
                flag_set_kernel<<<1, 1, 0, bst>>>(d_flags + 2 + par,
                                                  (long long)k + 1);
                CHECK_CUDA(cudaEventRecord(evBcH[k], bst));
                CHECK_CUDA(cudaEventRecord(evBcP1[k], bst));
                CHECK_CUDA(cudaEventRecord(evBcP2[k], bst));
            } else {
                CHECK_NCCL(ncclGroupStart());
                CHECK_NCCL(ncclBroadcast(d_ipiv[buf], d_ipiv[buf], jb, ncclInt32,
                                         root, bco, bst));
                if (tg)
                    CHECK_NCCL(ncclBroadcast(d_panel[buf] + (size_t)m * jb,
                                             d_panel[buf] + (size_t)m * jb,
                                             (size_t)jb * jb, ncclDouble, root,
                                             bco, bst));
                CHECK_NCCL(ncclGroupEnd());
                if (q != root) resolve_perm(buf, jb, bst);
                CHECK_CUDA(cudaEventRecord(evBcH[k], bst));
                if (g_nvshmem) {
                    if (pst != bst)
                        CHECK_CUDA(cudaStreamWaitEvent(pst, evBcH[k], 0));
                    if (nvshmemx_broadcastmem_on_stream(NVSHMEM_TEAM_WORLD,
                            d_panel[buf], d_panel[buf],
                            (size_t)m * jb * sizeof(double), root, pst)) {
                        fprintf(stderr, "[rank %d] nvshmem bcast failed\n", g_rank);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                } else {
                    CHECK_NCCL(ncclBroadcast(d_panel[buf], d_panel[buf],
                                             (size_t)m * jb, ncclDouble, root,
                                             bco, bst));
                }
                CHECK_CUDA(cudaEventRecord(evBcP1[k], pst));
                CHECK_CUDA(cudaEventRecord(evBcP2[k], pst));
            }
        }
        /* Q==1: evBcH/P1/P2[k] were recorded on sP right after pack */

        /* --- red: local (redundant, bit-identical) factorization of the
         * received raw panel, inside the panel buffer --- */
        if (red) {
            CHECK_CUDA(cudaStreamWaitEvent(sP, evBcRaw[k], 0));
            if (timing && k % Q == q) CHECK_CUDA(cudaEventRecord(tP0[k], sP));
            CHECK_CUSOLVER(cusolverDnDgetrf(csP, m, jb, d_panel[buf], m,
                                            d_work, d_ipiv[buf],
                                            d_infoArr + k));
            if (timing && k % Q == q) CHECK_CUDA(cudaEventRecord(tPF[k], sP));
            resolve_perm(buf, jb, sP);
            make_linv(d_panel[buf], m, m, jb, buf, hP, sP);
            if (q == root) {
                /* write U11 back for the back-substitution */
                long long c0l = local_cols_below_exact((long long)r0, NB, Q, q,
                                                       ntot);
                CHECK_CUDA(cudaMemcpy2DAsync(d_A + c0l * lda + r0,
                                             lda * sizeof(double),
                                             d_panel[buf],
                                             (size_t)m * sizeof(double),
                                             (size_t)jb * sizeof(double), jb,
                                             cudaMemcpyDeviceToDevice, sP));
            }
            if (timing && k % Q == q) CHECK_CUDA(cudaEventRecord(tPP[k], sP));
            CHECK_CUDA(cudaEventRecord(evBcH[k], sP));
            CHECK_CUDA(cudaEventRecord(evBcP1[k], sP));
            CHECK_CUDA(cudaEventRecord(evBcP2[k], sP));
        }

        /* --- column ranges --- */
        long long trail0 = local_cols_below_exact((long long)r0 + jb, NB, Q, q, ntot);
        long long urgent_nc = 0;
        int next_exists = (k + 1 < npanels);
        int i_own_next = next_exists && ((k + 1) % Q == q);
        int jb2 = next_exists ? std::min(NB, N - (k + 1) * NB) : 0;
        if (i_own_next) urgent_nc = jb2;
        long long rest0 = trail0 + urgent_nc;
        long long rest_nc = nloc - rest0;

        if (!cfg.lookahead) { urgent_nc = 0; rest0 = trail0; rest_nc = nloc - rest0; }

        /* --- lookahead: urgent update + factor panel k+1 on sP --- */
        cudaEvent_t bev[3] = {evBcH[k], evBcP1[k], evBcP2[k]};
        if (i_own_next && cfg.lookahead) {
            if (k >= 1) CHECK_CUDA(cudaStreamWaitEvent(sP, evUpd[k - 1], 0));
            update_cols(r0, m, jb, buf, trail0, urgent_nc, hP, sP, pool.tmpP,
                        bev, nullptr);
            CHECK_CUDA(cudaEventRecord(evUseP[k], sP));
            int r0n = (k + 1) * NB;
            int mn = N - r0n;
            if (is_red(k + 1)) {
                /* redundant path: ship the RAW panel; everyone factors it */
                if (k >= 1) CHECK_CUDA(cudaStreamWaitEvent(sP, evBcP2[k - 1], 0));
                if (Q > 1 && k >= 1)
                    CHECK_CUDA(cudaStreamWaitEvent(sP, evBcRaw[k - 1], 0));
                pack_panel(r0n, trail0, mn, jb2, (k + 1) & 1, sP);
                CHECK_CUDA(cudaEventRecord(evPack[k + 1], sP));
            } else {
                /* owner factors panel k+1 in place in d_A */
                if (timing) CHECK_CUDA(cudaEventRecord(tP0[k + 1], sP));
                factor_panel(r0n, trail0, mn, jb2, d_ipiv[(k + 1) & 1],
                             d_infoArr + (k + 1), sP);
                if (timing) CHECK_CUDA(cudaEventRecord(tPF[k + 1], sP));
                resolve_perm((k + 1) & 1, jb2, sP);
                if (k >= 1) CHECK_CUDA(cudaStreamWaitEvent(sP, evBcP2[k - 1], 0));
                if (Q > 1 && k >= 1)
                    CHECK_CUDA(cudaStreamWaitEvent(sP, evBcRaw[k - 1], 0));
                make_linv(d_A + trail0 * lda + r0n, (int)lda, mn, jb2,
                          (k + 1) & 1, hP, sP);
                pack_panel(r0n, trail0, mn, jb2, (k + 1) & 1, sP);
                CHECK_CUDA(cudaEventRecord(evPack[k + 1], sP));
                if (timing) CHECK_CUDA(cudaEventRecord(tPP[k + 1], sP));
            }
            if (Q == 1) {
                CHECK_CUDA(cudaEventRecord(evBcH[k + 1], sP));
                CHECK_CUDA(cudaEventRecord(evBcP1[k + 1], sP));
                CHECK_CUDA(cudaEventRecord(evBcP2[k + 1], sP));
            }
        } else {
            /* non-owners record marker events (never waited on with content) */
            CHECK_CUDA(cudaEventRecord(evUseP[k], sP));
            if (!(i_own_next && !cfg.lookahead))
                CHECK_CUDA(cudaEventRecord(evPack[k + 1], sP));
        }

        /* --- bulk trailing update on sU --- */
        if (timing) {
            CHECK_CUDA(cudaStreamWaitEvent(sU, evBcH[k], 0));
            CHECK_CUDA(cudaEventRecord(tR0[k], sU));
        }
        cudaEvent_t tpair[2];
        if (timing) { tpair[0] = tRL[k]; tpair[1] = tRT[k]; }
        /* the rest columns are split into two chunks on streams sU/sV so one
         * chunk's laswp/staging/U12 chain overlaps the other chunk's GEMMs
         * (dataset-partitioned stream pipelining) */
        long long ncA = (rest_nc >= 4096) ? rest_nc / 2 : rest_nc;
        long long ncB = rest_nc - ncA;
        if (k >= 1) {
            CHECK_CUDA(cudaStreamWaitEvent(sU, evUpdB[k - 1], 0));
            CHECK_CUDA(cudaStreamWaitEvent(sV, evUpdA[k - 1], 0));
        }
        update_cols(r0, m, jb, buf, rest0, ncA, hU, sU, pool.tmpU,
                    bev, timing ? tpair : nullptr);
        CHECK_CUDA(cudaEventRecord(evUpdA[k], sU));
        update_cols(r0, m, jb, buf, rest0 + ncA, ncB, hV, sV, pool.tmpU2,
                    bev, nullptr);
        CHECK_CUDA(cudaEventRecord(evUpdB[k], sV));
        if (timing) CHECK_CUDA(cudaEventRecord(tRG[k], sU));
        CHECK_CUDA(cudaStreamWaitEvent(sU, evUpdB[k], 0));
        CHECK_CUDA(cudaEventRecord(evUpd[k], sU));

        /* --- no-lookahead debug mode: factor panel k+1 strictly after the
         * full trailing update of iteration k --- */
        if (i_own_next && !cfg.lookahead) {
            CHECK_CUDA(cudaStreamWaitEvent(sP, evUpd[k], 0));
            int r0n = (k + 1) * NB;
            int mn = N - r0n;
            factor_panel(r0n, trail0, mn, jb2, d_ipiv[(k + 1) & 1],
                         d_infoArr + (k + 1), sP);
            resolve_perm((k + 1) & 1, jb2, sP);
            if (k >= 1) CHECK_CUDA(cudaStreamWaitEvent(sP, evBcP2[k - 1], 0));
            make_linv(d_A + trail0 * lda + r0n, (int)lda, mn, jb2,
                      (k + 1) & 1, hP, sP);
            pack_panel(r0n, trail0, mn, jb2, (k + 1) & 1, sP);
            CHECK_CUDA(cudaEventRecord(evPack[k + 1], sP));
            if (Q == 1) {
                CHECK_CUDA(cudaEventRecord(evBcH[k + 1], sP));
                CHECK_CUDA(cudaEventRecord(evBcP1[k + 1], sP));
                CHECK_CUDA(cudaEventRecord(evBcP2[k + 1], sP));
            }
        }

    }

    /* per-panel progress in HPL-NVIDIA style.  All work is already enqueued,
     * so waiting on events here does not stall the device pipeline. */
    if (cfg.progress && g_rank == 0) {
        const double totf = 2.0 / 3.0 * (double)N * (double)N * (double)N +
                            1.5 * (double)N * (double)N;
        double tprev = t0, fprev = 0.0;
        for (int k = 0; k < npanels; k++) {
            CHECK_CUDA(cudaEventSynchronize(evUpd[k]));
            double t = MPI_Wtime();
            long long nleft = (long long)N - (long long)(k + 1) * NB;
            if (nleft < 0) nleft = 0;
            double rl = (double)nleft / (double)N;
            double frac = 1.0 - rl * rl * rl;
            double done = totf * frac;
            double el = t - t0;
            double igf = (done - fprev) / (t - tprev + 1e-12) / 1e9;
            double gf = done / (el + 1e-12) / 1e9;
            printf("\033[0;31m Prog= %.2f%%\tN_left= %8lld\tTime= %6.2f\t"
                   "Time_left= %6.2f\tiGF= %8.2f\tGF= %8.2f\tiGF_per= %8.2f "
                   "\tGF_per= %8.2f\033[0m\n",
                   frac * 100.0, nleft, el,
                   el / (frac + 1e-12) - el, igf, gf, igf / Q, gf / Q);
            fflush(stdout);
            tprev = t; fprev = done;
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    if (timing) {
        double s_laswp = 0, s_trsm = 0, s_gemm = 0, s_fact = 0, s_pack = 0,
               s_stall = 0;
        float ms;
        for (int k = 0; k < npanels; k++) {
            CHECK_CUDA(cudaEventElapsedTime(&ms, tR0[k], tRL[k])); s_laswp += ms;
            CHECK_CUDA(cudaEventElapsedTime(&ms, tRL[k], tRT[k])); s_trsm += ms;
            CHECK_CUDA(cudaEventElapsedTime(&ms, tRT[k], tRG[k])); s_gemm += ms;
            if (k >= 1) {
                CHECK_CUDA(cudaEventElapsedTime(&ms, tRG[k - 1], tR0[k]));
                s_stall += ms;
            }
        }
        if (cfg.lookahead)
            for (int k = 0; k < npanels; k++) {
                if (k % Q != q) continue;  /* panels this rank factored */
                CHECK_CUDA(cudaEventElapsedTime(&ms, tP0[k], tPF[k])); s_fact += ms;
                CHECK_CUDA(cudaEventElapsedTime(&ms, tPF[k], tPP[k])); s_pack += ms;
            }
        printf("[timing] rank %d  laswp=%.2fs trsm=%.2fs gemm=%.2fs "
               "stall(sU)=%.2fs | panel: fact=%.2fs pack=%.2fs\n",
               g_rank, s_laswp / 1e3, s_trsm / 1e3, s_gemm / 1e3,
               s_stall / 1e3, s_fact / 1e3, s_pack / 1e3);
        fflush(stdout);
        for (int i = 0; i < npanels; i++) {
            cudaEventDestroy(tR0[i]); cudaEventDestroy(tRL[i]);
            cudaEventDestroy(tRT[i]); cudaEventDestroy(tRG[i]);
        }
        for (int i = 0; i < npanels + 1; i++) {
            cudaEventDestroy(tP0[i]); cudaEventDestroy(tPF[i]);
            cudaEventDestroy(tPP[i]);
        }
    }

    /* ---------------- back substitution: U x = y ---------------- */
    /* y lives in the b column (global col N) on its owner.          */
    double *d_x = pool.x, *d_acc = pool.acc, *d_tmp = pool.tmp;
    setzero_kernel<<<(N + 255) / 256, 256, 0, sP>>>(d_x, N);
    setzero_kernel<<<(N + 255) / 256, 256, 0, sP>>>(d_acc, N);
    {
        long long bblk = (long long)N / NB;
        int bowner = (int)(bblk % Q);
        if (q == bowner) {
            long long bcl = local_cols_below_exact((long long)N, NB, Q, q, ntot);
            /* local column index of global column N */
            CHECK_CUDA(cudaMemcpyAsync(d_acc, d_A + bcl * lda,
                                       (size_t)N * sizeof(double),
                                       cudaMemcpyDeviceToDevice, sP));
        }
    }

    /* fully asynchronous enqueue: per-rank ordering via stream sP, cross-rank
     * ordering via matched NCCL collectives — no host syncs in the loop */
    for (int j = npanels - 1; j >= 0; j--) {
        int r0 = j * NB;
        int jb = std::min(NB, N - r0);
        int owner = j % Q;
        if (Q > 1) {
            CHECK_NCCL(ncclReduce(d_acc + r0, d_tmp, jb, ncclDouble, ncclSum,
                                  owner, nccl, sP));
        } else {
            CHECK_CUDA(cudaMemcpyAsync(d_tmp, d_acc + r0, jb * sizeof(double),
                                       cudaMemcpyDeviceToDevice, sP));
        }
        if (q == owner) {
            long long cl = local_cols_below_exact((long long)r0, NB, Q, q, ntot);
            CHECK_CUBLAS(cublasDtrsv(hP, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                     CUBLAS_DIAG_NON_UNIT, jb,
                                     d_A + cl * lda + r0, (int)lda, d_tmp, 1));
            CHECK_CUDA(cudaMemcpyAsync(d_x + r0, d_tmp, jb * sizeof(double),
                                       cudaMemcpyDeviceToDevice, sP));
            if (r0 > 0) {
                CHECK_CUBLAS(cublasDgemv(hP, CUBLAS_OP_N, r0, jb, &dmone,
                                         d_A + cl * lda, (int)lda, d_tmp, 1,
                                         &done_, d_acc, 1));
            }
        }
    }
    if (Q > 1)
        CHECK_NCCL(ncclAllReduce(d_x, d_x, N, ncclDouble, ncclSum, nccl, sP));
    CHECK_CUDA(cudaStreamSynchronize(sP));

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    double t1 = MPI_Wtime();
    res.t_end = time(NULL);
    res.time_s = t1 - t0;
    res.gflops = ((2.0 / 3.0) * (double)N * (double)N * (double)N +
                  1.5 * (double)N * (double)N) / res.time_s / 1e9;

    /* check pivot infos */
    {
        std::vector<int> infos(npanels + 1);
        CHECK_CUDA(cudaMemcpy(infos.data(), d_infoArr, (npanels + 1) * sizeof(int),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i <= npanels; i++)
            if (infos[i] != 0) { res.info = infos[i]; break; }
        int ginfo = 0;
        CHECK_MPI(MPI_Allreduce(&res.info, &ginfo, 1, MPI_INT, MPI_MAX,
                                MPI_COMM_WORLD));
        res.info = ginfo;
    }

    /* ---------------- verification (untimed, like HPL) ---------------- */
    {
        double *d_ax = d_acc;                 /* reuse */
        double *d_rowsum = pool.rowsum, *d_b = pool.b, *d_r = pool.r;
        setzero_kernel<<<(N + 255) / 256, 256, 0, sU>>>(d_ax, N);
        setzero_kernel<<<(N + 255) / 256, 256, 0, sU>>>(d_rowsum, N);

        /* regenerate my blocks into d_panel[0] (chunk buffer, ld = N) */
        double *chunk = d_panel[0];
        std::vector<long long> h_gc2(NB);
        long long *d_gc2 = pool.gc2;
        for (long long b = q; b < nblocks; b += Q) {
            long long lo = b * NB;
            long long hi = std::min((long long)N, (b + 1) * (long long)NB);
            int mw = (int)(hi - lo);          /* matrix columns only */
            if (mw <= 0) continue;
            for (int c = 0; c < mw; c++) h_gc2[c] = lo + c;
            CHECK_CUDA(cudaMemcpyAsync(d_gc2, h_gc2.data(), mw * sizeof(long long),
                                       cudaMemcpyHostToDevice, sU));
            dim3 grid(64, mw);
            matgen_kernel<<<grid, 256, 0, sU>>>(chunk, (size_t)N, N, d_gc2, mw,
                                                ntot, cfg.seed);
            /* Ax += chunk * x[lo:hi] */
            CHECK_CUBLAS(cublasDgemv(hU, CUBLAS_OP_N, N, mw, &done_, chunk, N,
                                     d_x + lo, 1, &done_, d_ax, 1));
            rowsum_kernel<<<(N + 255) / 256, 256, 0, sU>>>(chunk, (size_t)N, N,
                                                           mw, d_rowsum);
        }
        /* b column (every rank regenerates it deterministically) */
        {
            h_gc2[0] = N;
            CHECK_CUDA(cudaMemcpyAsync(d_gc2, h_gc2.data(), sizeof(long long),
                                       cudaMemcpyHostToDevice, sU));
            dim3 grid(64, 1);
            matgen_kernel<<<grid, 256, 0, sU>>>(d_b, (size_t)N, N, d_gc2, 1,
                                                ntot, cfg.seed);
        }
        CHECK_CUDA(cudaStreamSynchronize(sU));
        if (Q > 1) {
            CHECK_NCCL(ncclAllReduce(d_ax, d_ax, N, ncclDouble, ncclSum, nccl, sU));
            CHECK_NCCL(ncclAllReduce(d_rowsum, d_rowsum, N, ncclDouble, ncclSum,
                                     nccl, sU));
            CHECK_CUDA(cudaStreamSynchronize(sU));
        }
        axpby_res_kernel<<<(N + 255) / 256, 256, 0, sU>>>(d_b, d_ax, d_r, N);
        CHECK_CUDA(cudaStreamSynchronize(sU));

        /* d_ax and d_rowsum were AllReduce-summed above, d_b regenerated by
         * every rank, d_x AllReduced after back-substitution: all norms below
         * are identical on every rank. */
        double norm_r = dev_absmax(hU, d_r, N);
        double norm_x = dev_absmax(hU, d_x, N);
        double norm_b = dev_absmax(hU, d_b, N);
        double normA_inf = dev_absmax(hU, d_rowsum, N);   /* rowsums >= 0 */
        /* relative machine precision, as computed by HPL_pdlamch (= DBL_EPSILON/2) */
        double eps = DBL_EPSILON / 2.0;
        res.resid = norm_r / (eps * (normA_inf * norm_x + norm_b) * (double)N);
        res.passed = (res.resid < cfg.thresh) && (res.info == 0);
        res.norm_r = norm_r;
        res.norm_A = normA_inf;
        res.norm_x = norm_x;
        res.norm_b = norm_b;
    }

    for (int i = 0; i < npanels + 2; i++) {
        cudaEventDestroy(evPack[i]); cudaEventDestroy(evBcH[i]);
        cudaEventDestroy(evBcP1[i]); cudaEventDestroy(evBcP2[i]);
        cudaEventDestroy(evUpdA[i]); cudaEventDestroy(evUpdB[i]);
        cudaEventDestroy(evBcRaw[i]);
        cudaEventDestroy(evUpd[i]); cudaEventDestroy(evUseP[i]);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return res;
}

/* ------------------------------------------------------------------ */
/* HPL.dat parsing (hpl-2.3 line layout)                               */
/* ------------------------------------------------------------------ */
struct DatCfg {
    std::vector<int> Ns, NBs;
    double thresh = 16.0;
};

static bool parse_hpl_dat(const char *path, DatCfg &out)
{
    FILE *f = fopen(path, "r");
    if (!f) return false;
    char line[4096];
    std::vector<std::string> lines;
    while (fgets(line, sizeof(line), f)) lines.push_back(line);
    fclose(f);
    if (lines.size() < 13) return false;
    auto ints = [&](const std::string &s, int maxn) {
        std::vector<int> v; const char *p = s.c_str(); char *e;
        while (*p && (int)v.size() < maxn) {
            long val = strtol(p, &e, 10);
            if (e == p) break;
            v.push_back((int)val); p = e;
        }
        return v;
    };
    std::vector<int> nN = ints(lines[4], 1);
    if (nN.empty()) return false;
    std::vector<int> Ns = ints(lines[5], nN[0]);
    std::vector<int> nNB = ints(lines[6], 1);
    std::vector<int> NBs = ints(lines[7], nNB[0]);
    if (Ns.empty() || NBs.empty()) return false;
    out.Ns = Ns; out.NBs = NBs;
    { const char *p = lines[12].c_str(); char *e; double th = strtod(p, &e);
      if (e != p) out.thresh = th; }
    return true;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    CHECK_MPI(MPI_Init(&argc, &argv));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &g_size));

    /* GPU selection: one rank per GPU on each node */
    MPI_Comm shm;
    CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank,
                                  MPI_INFO_NULL, &shm));
    int lrank = 0;
    CHECK_MPI(MPI_Comm_rank(shm, &lrank));
    int ndev = 0;
    CHECK_CUDA(cudaGetDeviceCount(&ndev));
    CHECK_CUDA(cudaSetDevice(lrank % ndev));

    Grid g; g.Q = g_size; g.q = g_rank;

    /* NCCL: two communicators so even/odd panel broadcasts can overlap */
    ncclComm_t nccl = nullptr, nccl2 = nullptr;
    if (g_size > 1) {
        ncclUniqueId id;
        if (g_rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
        CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
        CHECK_NCCL(ncclCommInitRank(&nccl, g_size, id, g_rank));
        ncclUniqueId id2;
        if (g_rank == 0) CHECK_NCCL(ncclGetUniqueId(&id2));
        CHECK_MPI(MPI_Bcast(&id2, sizeof(id2), MPI_BYTE, 0, MPI_COMM_WORLD));
        CHECK_NCCL(ncclCommInitRank(&nccl2, g_size, id2, g_rank));
    }

    /* NVSHMEM (optional): panel payload broadcasts via symmetric heap */
    {
        const char *nv = getenv("CHPL_NVSHMEM");
        g_nvshmem = (nv && g_size > 1) ? atoi(nv) : 0;
        if (g_nvshmem) {
            static MPI_Comm world = MPI_COMM_WORLD;
            nvshmemx_init_attr_t attr;
            memset(&attr, 0, sizeof(attr));
            nvshmemx_set_attr_mpi_comm_args(&world, &attr);
            if (nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr)) {
                fprintf(stderr, "[rank %d] nvshmem init failed, falling back "
                        "to NCCL\n", g_rank);
                g_nvshmem = 0;
            } else if (g_rank == 0) {
                /* MPI-comm bootstrap guarantees PE numbering == MPI rank */
                printf("[chpl] NVSHMEM enabled for panel broadcasts\n");
            }
        }
    }

    /* streams: sP (panel, high prio), sU/sV (chunked update), sC (comm) */
    int prLow, prHigh;
    CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&prLow, &prHigh));
    cudaStream_t sU, sP, sC, sV, sC2;
    CHECK_CUDA(cudaStreamCreateWithPriority(&sU, cudaStreamNonBlocking, prLow));
    CHECK_CUDA(cudaStreamCreateWithPriority(&sV, cudaStreamNonBlocking, prLow));
    CHECK_CUDA(cudaStreamCreateWithPriority(&sP, cudaStreamNonBlocking, prHigh));
    CHECK_CUDA(cudaStreamCreateWithPriority(&sC, cudaStreamNonBlocking, prHigh));
    CHECK_CUDA(cudaStreamCreateWithPriority(&sC2, cudaStreamNonBlocking, prHigh));

    cublasHandle_t hU, hP, hV;
    CHECK_CUBLAS(cublasCreate(&hU));
    CHECK_CUBLAS(cublasCreate(&hP));
    CHECK_CUBLAS(cublasCreate(&hV));
    CHECK_CUBLAS(cublasSetStream(hU, sU));
    CHECK_CUBLAS(cublasSetStream(hP, sP));
    CHECK_CUBLAS(cublasSetStream(hV, sV));
    {
        /* generous cuBLAS workspaces improve algorithm choice for the
         * skinny-k update GEMMs */
        size_t wsz = 256ULL << 20;
        void *w1 = nullptr, *w2 = nullptr, *w3 = nullptr;
        if (cudaMalloc(&w1, wsz) == cudaSuccess)
            CHECK_CUBLAS(cublasSetWorkspace(hU, w1, wsz));
        if (cudaMalloc(&w2, 64ULL << 20) == cudaSuccess)
            CHECK_CUBLAS(cublasSetWorkspace(hP, w2, 64ULL << 20));
        if (cudaMalloc(&w3, wsz) == cudaSuccess)
            CHECK_CUBLAS(cublasSetWorkspace(hV, w3, wsz));
    }
    cusolverDnHandle_t csP;
    CHECK_CUSOLVER(cusolverDnCreate(&csP));
    CHECK_CUSOLVER(cusolverDnSetStream(csP, sP));

    /* config */
    DatCfg dat;
    const char *datpath = (argc > 1) ? argv[1] : "HPL.dat";
    if (!parse_hpl_dat(datpath, dat)) {
        if (g_rank == 0)
            fprintf(stderr, "[chpl] could not parse %s, using defaults\n", datpath);
        dat.Ns = {8192}; dat.NBs = {512}; dat.thresh = 16.0;
    }

    RunCfg cfg;
    cfg.thresh = dat.thresh;
    cfg.seed = 42;
    const char *pv = getenv("CHPL_PANEL");
    cfg.use_cusolver_panel = (pv && !strcmp(pv, "custom")) ? 0 : 1;
    const char *pg = getenv("CHPL_PROGRESS");
    cfg.progress = pg ? atoi(pg) : 1;
    const char *pl = getenv("CHPL_LOOKAHEAD");
    cfg.lookahead = pl ? atoi(pl) : 1;

    DevPool pool;
    pool_init(pool, dat.Ns, dat.NBs, g.Q, g.q, cfg.use_cusolver_panel, csP);

    cudaDeviceProp devprop;
    CHECK_CUDA(cudaGetDeviceProperties(&devprop, 0));
    int curdev = 0;
    CHECK_CUDA(cudaGetDevice(&curdev));
    CHECK_CUDA(cudaGetDeviceProperties(&devprop, curdev));

    if (g_rank == 0) {
        printf("================================================================================\n");
        printf("CHPL 1.0  -- CUDA accelerated HPL benchmark  -- rewrite of netlib HPL-2.3\n");
        printf("================================================================================\n");
        printf("HPLinpack 2.3  --  High-Performance Linpack benchmark  --   December 2, 2018\n");
        printf("Written by A. Petitet and R. Clint Whaley,  Innovative Computing Laboratory, UTK\n");
        printf("Modified by Piotr Luszczek, Innovative Computing Laboratory, UTK\n");
        printf("Modified by Julien Langou, University of Colorado Denver\n");
        printf("================================================================================\n");
        printf("\n");
        printf("An explanation of the input/output parameters follows:\n");
        printf("T/V    : Wall time / encoded variant.\n");
        printf("N      : The order of the coefficient matrix A.\n");
        printf("NB     : The partitioning blocking factor.\n");
        printf("P      : The number of process rows.\n");
        printf("Q      : The number of process columns.\n");
        printf("Time   : Time in seconds to solve the linear system.\n");
        printf("Gflops : Rate of execution for solving the linear system.\n");
        printf("\n");
        printf("The following parameter values will be used:\n");
        printf("\n");
        printf("N      : ");
        for (int n : dat.Ns) printf("%7d ", n);
        printf("\n");
        printf("NB     : ");
        for (int nb : dat.NBs) printf("%7d ", nb);
        printf("\n");
        printf("PMAP   : Column-major process mapping\n");
        printf("P      : %7d \n", 1);
        printf("Q      : %7d \n", g_size);
        printf("PFACT  : device-side fused getf2 (%s)\n",
               cfg.use_cusolver_panel ? "cuSOLVER" : "custom");
        printf("BCAST  : ncclBcast \n");
        printf("DEPTH  : %7d \n", cfg.lookahead ? 1 : 0);
        printf("SWAP   : GPU laswp (trailing submatrix only)\n");
        printf("L1     : no-transposed form\n");
        printf("U      : no-transposed form\n");
        printf("EQUIL  : no\n");
        printf("ALIGN  : 8 double precision words\n");
        printf("\n");
        printf("--------------------------------------------------------------------------------\n");
        printf("\n");
        printf("- The matrix A is randomly generated for each test.\n");
        printf("- The following scaled residual check will be computed:\n");
        printf("      ||Ax-b||_oo / ( eps * ( || x ||_oo * || A ||_oo + || b ||_oo ) * N )\n");
        printf("- The relative machine precision (eps) is taken to be          %14.6e\n",
               DBL_EPSILON / 2.0);
        printf("- Computational tests pass if scaled residuals are less than   %13.1f\n",
               cfg.thresh);
        printf("\n");
        printf("--- DEVICE INFO ---\n");
        printf("  SM version          : %d%d\n", devprop.major, devprop.minor);
        printf("  Number of SMs       : %d\n", devprop.multiProcessorCount);
        printf("  Device memory       : %.1f GiB\n",
               devprop.totalGlobalMem / 1073741824.0);
        printf("-------------------\n");
        fflush(stdout);
    }

    int n_passed = 0, n_failed = 0, n_skipped = 0;
    for (int in = 0; in < (int)dat.Ns.size(); in++) {
        for (int ib = 0; ib < (int)dat.NBs.size(); ib++) {
            cfg.N = dat.Ns[in];
            cfg.NB = dat.NBs[ib];
            RunResult r = run_one(cfg, g, pool, hU, hP, hV, csP, sU, sP, sC,
                                  sV, sC2, nccl, nccl2);
            if (g_rank == 0) {
                char tstart[64], tend[64];
                struct tm tmv;
                localtime_r(&r.t_start, &tmv);
                strftime(tstart, sizeof(tstart), "%a %b %e %H:%M:%S %Y", &tmv);
                localtime_r(&r.t_end, &tmv);
                strftime(tend, sizeof(tend), "%a %b %e %H:%M:%S %Y", &tmv);
                printf("================================================================================\n");
                printf("T/V                N    NB     P     Q         Time          Gflops (   per GPU)\n");
                printf("--------------------------------------------------------------------------------\n");
                printf("%-4s%16d %5d %5d %5d %12.2f %15.3e (%10.3e)\n",
                       "WC0", cfg.N, cfg.NB, 1, g_size, r.time_s, r.gflops,
                       r.gflops / g_size);
                printf("\n");
                printf("HPL_pdgesv() start time %s\n", tstart);
                printf("HPL_pdgesv() end time   %s\n", tend);
                printf("\n");
                printf("--------------------------------------------------------------------------------\n");
                printf("||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)= %16.12f ...... %s\n",
                       r.resid, r.passed ? "PASSED" : "FAILED");
                printf("||Ax-b||_oo  . . . . . . . . . . . . . . . . . = %.16f\n", r.norm_r);
                printf("||A||_oo . . . . . . . . . . . . . . . . . . . = %.16f\n", r.norm_A);
                printf("||x||_oo . . . . . . . . . . . . . . . . . . . = %.16f\n", r.norm_x);
                printf("||b||_oo . . . . . . . . . . . . . . . . . . . = %.16f\n", r.norm_b);
                if (r.info != 0)
                    printf("WARNING: zero pivot at panel column %d\n", r.info);
                fflush(stdout);
            }
            if (r.passed) n_passed++; else n_failed++;
        }
    }

    if (g_rank == 0) {
        printf("================================================================================\n");
        printf("\n");
        printf("Finished %6d tests with the following results:\n",
               n_passed + n_failed + n_skipped);
        printf("         %6d tests completed and passed residual checks,\n", n_passed);
        printf("         %6d tests completed and failed residual checks,\n", n_failed);
        printf("         %6d tests skipped because of illegal input values.\n", n_skipped);
        printf("--------------------------------------------------------------------------------\n");
        printf("\n");
        printf("End of Tests.\n");
        printf("================================================================================\n");
        fflush(stdout);
    }

    cublasDestroy(hU);
    cublasDestroy(hP);
    cublasDestroy(hV);
    cusolverDnDestroy(csP);
    cudaStreamDestroy(sU);
    cudaStreamDestroy(sV);
    cudaStreamDestroy(sP);
    cudaStreamDestroy(sC);
    cudaStreamDestroy(sC2);
    if (nccl) ncclCommDestroy(nccl);
    if (nccl2) ncclCommDestroy(nccl2);
    if (g_nvshmem) nvshmemx_hostlib_finalize();
    MPI_Finalize();
    return 0;
}
