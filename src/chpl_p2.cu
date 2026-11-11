/*
 * chpl_p2 -- CUDA HPL for NVIDIA Hopper (H200/H100), P x Q grid with P = 2.
 *
 * Runs on 8 or 16 MPI ranks (P2Q4 / P2Q8), one rank per GPU. Best choice
 * for a single 8-GPU node.
 *
 *  - 2D block-cyclic distribution, rank r -> pr = r % P, qc = r / P
 *  - panel factorization with tournament pivoting (CALU), fp32 candidate
 *    selection and exact fp64 refactorization of the winning rows
 *  - trailing update with int8-emulated DGEMM: GEMMul8 (Ozaki-II, CRT,
 *    CHPL2D_G8=1) or Ozaki-I digit slicing (CHPL2D_OZ=1)
 *  - intra-node transfers through CUDA IPC + copy engines, NCCL otherwise
 *  - standard HPL residual check
 *
 * Usage: mpirun -np <8|16> chpl_p2 HPL.dat   (reads N, NB and threshold)
 *
 * Part of CHPL. SPDX-License-Identifier: BSD-3-Clause
 * Derived from the netlib HPL-2.3 benchmark; see THIRD_PARTY_NOTICES.md.
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
#include <array>

#include <mpi.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cublasLt.h>
#include <nccl.h>

#include <gemmul8.hpp> /* RIKEN GEMMul8 (Ozaki-II CRT DGEMM emulation) */

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
typedef unsigned long long u64;

static const int P = 2;
static int Q = 4;

/* ------------------------- kernels ------------------------------- */
__device__ __forceinline__ double chpl_elem(u64 seed, u64 idx)
{
    u64 z = (idx + 1ULL) * 0x9E3779B97F4A7C15ULL + seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

/* dst(lr, lc) = elem(grow[lr], gcol[lc]) */
__global__ void matgen2d_kernel(double *dst, size_t ldd, int mrows,
                                const long long *grow, const long long *gcol,
                                int ncols, long long ncg, u64 seed)
{
    int c = blockIdx.y;
    if (c >= ncols) return;
    long long gc = gcol[c];
    double *col = dst + (size_t)c * ldd;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < mrows;
         r += gridDim.x * blockDim.x)
        col[r] = chpl_elem(seed, (u64)grow[r] * (u64)ncg + (u64)gc);
}

/* gather rows (by local row index list) into dense pack, ld = nrows */
__global__ void grow_gather_kernel(const double *A, size_t lda,
                                   const int *rows, int nrows,
                                   size_t c0, long long nc, double *dst)
{
    int i = blockIdx.x * 32 + threadIdx.x;
    if (i >= nrows) return;
    const double *Ac = A + c0 * lda + rows[i];
    for (long long c = (long long)blockIdx.y * blockDim.y + threadIdx.y; c < nc;
         c += (long long)gridDim.y * blockDim.y)
        dst[i + (size_t)c * nrows] = Ac[c * lda];
}

/* scatter pack rows into local rows */
__global__ void grow_scatter_kernel(double *A, size_t lda, const int *rows,
                                    int nrows, size_t c0, long long nc,
                                    const double *src)
{
    int i = blockIdx.x * 32 + threadIdx.x;
    if (i >= nrows) return;
    double *Ac = A + c0 * lda + rows[i];
    for (long long c = (long long)blockIdx.y * blockDim.y + threadIdx.y; c < nc;
         c += (long long)gridDim.y * blockDim.y)
        Ac[c * lda] = src[i + (size_t)c * nrows];
}

/* gather with device-side row count (cnt[ci]) and explicit dst ld */
__global__ void gather_dyn_kernel(const double *A, size_t lda, const int *rows,
                                  const int *cnt, int ci, size_t c0,
                                  long long nc, double *dst, size_t ldd)
{
    int n = cnt[ci];
    int i = blockIdx.x * 32 + threadIdx.x;
    if (blockIdx.x * 32 >= n) return;
    if (i >= n) return;
    const double *Ac = A + c0 * lda + rows[i];
    for (long long c = (long long)blockIdx.y * blockDim.y + threadIdx.y; c < nc;
         c += (long long)gridDim.y * blockDim.y)
        dst[i + (size_t)c * ldd] = Ac[c * lda];
}

__global__ void scatter_dyn_kernel(double *A, size_t lda, const int *rows,
                                   const int *cnt, int ci, size_t c0,
                                   long long nc, const double *src, size_t lds)
{
    int n = cnt[ci];
    int i = blockIdx.x * 32 + threadIdx.x;
    if (blockIdx.x * 32 >= n) return;
    if (i >= n) return;
    double *Ac = A + c0 * lda + rows[i];
    for (long long c = (long long)blockIdx.y * blockDim.y + threadIdx.y; c < nc;
         c += (long long)gridDim.y * blockDim.y)
        Ac[c * lda] = src[i + (size_t)c * lds];
}

/* resolve a LAPACK swap sequence into the net top-n permutation using a full
 * occupancy array: occ[p] = original index now at position p.  Single block. */
__global__ void resolve_kernel(const int *ipiv, int n, int m, int *occ,
                               int *top)
{
    for (int p = threadIdx.x; p < m; p += blockDim.x) occ[p] = p;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            int p = ipiv[i] - 1;
            if (p != i) { int t = occ[i]; occ[i] = occ[p]; occ[p] = t; }
        }
        for (int i = 0; i < n; i++) top[i] = occ[i];
    }
}

/* candidate local rows + provenance (global row ids) from resolved top */
__global__ void prep_cand_kernel(const int *top, int n, long long plr0,
                                 const long long *grow, int *myrows, int *prov)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long lr = plr0 + top[i];
    myrows[i] = (int)lr;
    prov[i] = (int)grow[lr];
}

/* map stacked-getrf winners back to global row ids */
__global__ void winmap_kernel(const int *top2, int jb, int nc0,
                              const int *prov0, const int *prov1, int *win)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= jb) return;
    int t = top2[i];
    win[i] = (t < nc0) ? prov0[t] : prov1[t - nc0];
}

/* gather the jb winner rows (by stacked index) from the two candidate
 * blocks into a packed jb x jb fp64 block */
__global__ void win_gather_kernel(const int *top, int jb, int nc0, int nc1,
                                  const double *cand0, const double *cand1,
                                  double *D)
{
    int i = blockIdx.x * 32 + threadIdx.x;
    if (i >= jb) return;
    int t = top[i];
    const double *src; int lds;
    if (t < nc0) { src = cand0 + t; lds = nc0; }
    else         { src = cand1 + (t - nc0); lds = nc1; }
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < jb;
         j += gridDim.y * blockDim.y)
        D[i + (size_t)j * jb] = src[(size_t)j * lds];
}

__global__ void compose_win_kernel(const int *top3, const int *winpre, int jb,
                                   int *win)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < jb) win[i] = winpre[top3[i]];
}

/* monotonic sequence flags for cross-process copy-engine data plane */
__global__ void sumf_kernel(const float *x, size_t n, double *out)
{
    double s = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        s += x[i];
    __shared__ double sh[256];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (!threadIdx.x) atomicAdd(out, sh[0]);
}

__global__ void spin_kernel(long long cycles)
{
    long long t0 = clock64();
    while (clock64() - t0 < cycles) ;
}

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

/* device closed-form lrows_below for P=2 row ranks */
__device__ __forceinline__ long long d_lrb(long long g, int NB_, int pr_)
{
    long long nb = g / NB_;
    long long full = (nb > pr_) ? (nb - pr_ - 1) / 2 + 1 : 0;
    long long part = (nb % 2 == pr_) ? g - nb * NB_ : 0;
    return full * NB_ + part;
}

/* build the net-permutation exchange lists from the winner row ids.
 * Sequential single-block kernel (pairing of vacated
 * slots with displaced diagonal rows follows winner order; any deterministic
 * bijection is a valid pivoting choice).  Outputs the five d_rows lists and
 * cnt = {ng, nsend, n1, n2}. */
__global__ void buildlists_kernel(const int *win, long long gr0, int jb,
                                  int NB_, int pr_, int *rows, int *cnt)
{
    extern __shared__ int sh[];
    int *chg_d = sh;              /* dst global rows, 2*jb */
    int *chg_s = chg_d + 2 * jb;  /* src global rows, 2*jb */
    int *gpos  = chg_s + 2 * jb;  /* 2*jb */
    int *frees = gpos + 2 * jb;   /* jb */
    int *displ = frees + jb;      /* jb */
    char *iswin = (char *)(displ + jb); /* jb */
    for (int i = threadIdx.x; i < jb; i += blockDim.x) iswin[i] = 0;
    __syncthreads();
    if (threadIdx.x != 0) return;
    int nchg = 0, nf = 0, nd = 0;
    for (int i = 0; i < jb; i++) {
        long long w = win[i];
        if (w != gr0 + i) { chg_d[nchg] = (int)(gr0 + i); chg_s[nchg] = (int)w; nchg++; }
        if (w >= gr0 && w < gr0 + jb) iswin[w - gr0] = 1;
        else frees[nf++] = (int)w;
    }
    for (int i = 0; i < jb; i++)
        if (!iswin[i]) displ[nd++] = (int)(gr0 + i);
    for (int i = 0; i < nf; i++) { chg_d[nchg] = frees[i]; chg_s[nchg] = displ[i]; nchg++; }
    int ng = 0;
    for (int t = 0; t < nchg; t++) {
        long long src = chg_s[t];
        if ((int)((src / NB_) % 2) == pr_) {
            gpos[t] = ng;
            rows[ng++] = (int)d_lrb(src, NB_, pr_);
        } else gpos[t] = -1;
    }
    int nsend = 0, n1 = 0, n2 = 0;
    for (int t = 0; t < nchg; t++) {
        long long dst = chg_d[t], src = chg_s[t];
        int dpr = (int)((dst / NB_) % 2), spr = (int)((src / NB_) % 2);
        if (spr == pr_ && dpr != pr_)
            rows[2 * NB_ + nsend++] = (int)d_lrb(src, NB_, pr_);
        if (dpr != pr_) continue;
        int lr = (int)d_lrb(dst, NB_, pr_);
        if (spr == pr_) {
            rows[3 * NB_ + n1] = (int)d_lrb(src, NB_, pr_);
            rows[5 * NB_ + n1] = lr; n1++;
        } else rows[7 * NB_ + n2++] = lr;
    }
    cnt[0] = ng; cnt[1] = nsend; cnt[2] = n1; cnt[3] = n2;
}

/* cast a double panel to float (for pivot-selection-only getrf) */
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

__global__ void setzero_kernel(double *p, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0;
}

__global__ void seteye_kernel(double *p, int jb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < jb * jb) p[i] = (i / jb == i % jb) ? 1.0 : 0.0;
}

__global__ void rowsum_kernel(const double *chunk, size_t ldc, int m,
                              int ncols, double *rowsum)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;
    double s = 0.0;
    for (int c = 0; c < ncols; c++) s += fabs(chunk[(size_t)c * ldc + r]);
    rowsum[r] += s;
}

__global__ void axpby_res_kernel(const double *b, const double *ax,
                                 double *r, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i] - ax[i];
}

/* ---------------- Ozaki int8 fp64-emulation kernels ---------------- */
__global__ void ozrowscale_kernel(const double *A, size_t lda, int m, int k,
                                  double *rinv, double *rsc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double mx = 0.0;
    for (int j = 0; j < k; j++) mx = fmax(mx, fabs(A[(size_t)j * lda + i]));
    int e = (mx > 0.0) ? ilogb(mx) + 1 : 0;
    rinv[i] = scalbn(1.0, -e);
    rsc[i] = scalbn(1.0, e);
}

__global__ void ozcolscale_kernel(const double *B, size_t ldb, int k, int n,
                                  double *cinv, double *csc)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const double *col = B + (size_t)j * ldb;
    double mx = 0.0;
    for (int i = 0; i < k; i++) mx = fmax(mx, fabs(col[i]));
    int e = (mx > 0.0) ? ilogb(mx) + 1 : 0;
    cinv[j] = scalbn(1.0, -e);
    csc[j] = scalbn(1.0, e);
}

/* A (m x k, ld lda) -> s slices stored transposed (k x m, ld k) */
__global__ void ozsliceA_kernel(const double *A, size_t lda, int m, int k,
                                const double *rinv, signed char *S, int s)
{
    __shared__ double t[32][33];
    int bi = blockIdx.x * 32, bj = blockIdx.y * 32;
    for (int jj = threadIdx.y; jj < 32; jj += blockDim.y) {
        int i = bi + threadIdx.x, j = bj + jj;
        t[jj][threadIdx.x] =
            (i < m && j < k) ? A[(size_t)j * lda + i] * rinv[i] : 0.0;
    }
    __syncthreads();
    size_t ldS = (size_t)k * s;   /* slices concatenated along k */
    for (int ii = threadIdx.y; ii < 32; ii += blockDim.y) {
        int i = bi + ii, j = bj + threadIdx.x;
        if (i >= m || j >= k) continue;
        double e = t[threadIdx.x][ii];
        signed char *dst = S + (size_t)i * ldS + j;
        for (int p = 0; p < s; p++) {
            double a = rint(e * 128.0);
            if (a > 127.0) a = 127.0;
            e = e * 128.0 - a;
            dst[(size_t)p * k] = (signed char)a;
        }
    }
}

/* B (k x n, ld k packed) -> slices concatenated along k in REVERSE order
 * (slice p at row offset (s-1-p)*k), so the diagonal-d operand
 * [b_d; b_{d-1}; ...; b_0] is simply the suffix starting at (s-1-d)*k. */
__global__ void ozsliceB_kernel(const double *B, int k, int n,
                                const double *cinv, signed char *S, int s,
                                size_t ldS)
{
    int j = blockIdx.y;
    if (j >= n) return;
    double ci = cinv[j];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k;
         i += gridDim.x * blockDim.x) {
        double e = B[(size_t)j * k + i] * ci;
        signed char *dst = S + (size_t)j * ldS + i;
        for (int p = 0; p < s; p++) {
            double a = rint(e * 128.0);
            if (a > 127.0) a = 127.0;
            e = e * 128.0 - a;
            dst[(size_t)(s - 1 - p) * k] = (signed char)a;
        }
    }
}

/* C(tile) -= rsc[i]*csc[j] * sum_d Dd*128^-(d+2)  (up to 6 diagonals) */
__global__ void ozrecon_kernel(double *C, size_t ldc, int m, int n,
                               const int *D0, const int *D1, const int *D2,
                               const int *D3, const int *D4, const int *D5,
                               int ndiag, const double *rsc, const double *csc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;
    size_t off = (size_t)j * m + i;
    double sc = 1.0 / 16384.0;
    double acc = (double)D0[off] * sc;
    if (ndiag > 1) { sc /= 128.0; acc += (double)D1[off] * sc; }
    if (ndiag > 2) { sc /= 128.0; acc += (double)D2[off] * sc; }
    if (ndiag > 3) { sc /= 128.0; acc += (double)D3[off] * sc; }
    if (ndiag > 4) { sc /= 128.0; acc += (double)D4[off] * sc; }
    if (ndiag > 5) { sc /= 128.0; acc += (double)D5[off] * sc; }
    C[(size_t)j * ldc + i] -= rsc[i] * csc[j] * acc;
}

__global__ void addacc_kernel(double *dst, const double *own,
                              const double *t3, int NB_, int jb, int np)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= jb) return;
    double v = own[i];
    for (int o = 0; o < np; o++) v += t3[(size_t)o * NB_ + i];
    dst[i] = v;
}

/* ------------------------- host helpers --------------------------- */
static long long lrows_below(long long g, int NB, int pr, long long Nrows)
{
    if (g > Nrows) g = Nrows;
    long long cnt = 0;
    long long nblk = (Nrows + NB - 1) / NB;
    for (long long i = pr; i < nblk; i += P) {
        long long lo = i * NB, hi = std::min(Nrows, (i + 1) * (long long)NB);
        if (lo >= g) break;
        cnt += std::min(hi, g) - lo;
    }
    return cnt;
}
static long long lcols_below(long long c, int NB, int qc, long long ntot)
{
    if (c > ntot) c = ntot;
    long long cnt = 0;
    long long nblk = (ntot + NB - 1) / NB;
    for (long long j = qc; j < nblk; j += Q) {
        long long lo = j * NB, hi = std::min(ntot, (j + 1) * (long long)NB);
        if (lo >= c) break;
        cnt += std::min(hi, c) - lo;
    }
    return cnt;
}

static double dev_absmax(cublasHandle_t h, const double *d, long long n)
{
    if (n <= 0) return 0.0;
    int idx = 0;
    CHECK_CUBLAS(cublasIdamax(h, (int)n, d, 1, &idx));
    double v = 0.0;
    CHECK_CUDA(cudaMemcpy(&v, d + (idx - 1), 8, cudaMemcpyDeviceToHost));
    return fabs(v);
}

/* resolve a swap sequence (base+i <-> piv[i], ascending i) into the net
 * permutation: top[i] = original index whose content ends at base+i,
 * below = (position >= base+n, original index) pairs that changed. */
static void resolve_seq(const std::vector<long long> &piv, long long base,
                        int n, std::vector<long long> &top,
                        std::vector<std::pair<long long, long long>> &below)
{
    top.resize(n);
    for (int i = 0; i < n; i++) top[i] = base + i;
    below.clear();
    for (int i = 0; i < n; i++) {
        long long p = piv[i];
        if (p == base + i) continue;
        if (p < base + n) {
            std::swap(top[i], top[p - base]);
        } else {
            long long occ = p;
            bool found = false;
            for (auto &e : below)
                if (e.first == p) { occ = e.second; found = true; break; }
            if (found) {
                for (auto &e : below)
                    if (e.first == p) { e.second = top[i]; break; }
            } else {
                below.push_back({p, top[i]});
            }
            top[i] = occ;
        }
    }
}

/* ------------------------- main ----------------------------------- */
int main(int argc, char **argv)
{
    CHECK_MPI(MPI_Init(&argc, &argv));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &g_size));
    if (g_size != 8 && g_size != 16) {
        if (!g_rank) fprintf(stderr, "chpl_p2 requires 8 or 16 ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    Q = g_size / P;
    const int pr = g_rank % P, qc = g_rank / P;

    MPI_Comm shm;
    CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank,
                                  MPI_INFO_NULL, &shm));
    int lrank = 0; CHECK_MPI(MPI_Comm_rank(shm, &lrank));
    /* same-node map: ranks sharing this shm communicator */
    int samenode[16] = {0};
    {
        int wr[16]; wr[lrank] = g_rank;
        int shsz = 0; CHECK_MPI(MPI_Comm_size(shm, &shsz));
        std::vector<int> loc(shsz);
        CHECK_MPI(MPI_Allgather(&g_rank, 1, MPI_INT, loc.data(), 1, MPI_INT,
                                shm));
        for (int i = 0; i < shsz; i++)
            if (loc[i] < 16) samenode[loc[i]] = 1;
        (void)wr;
    }
    int ndev = 0; CHECK_CUDA(cudaGetDeviceCount(&ndev));
    CHECK_CUDA(cudaSetDevice(lrank % ndev));

    MPI_Comm mrow, mcol;
    CHECK_MPI(MPI_Comm_split(MPI_COMM_WORLD, pr, qc, &mrow));
    CHECK_MPI(MPI_Comm_split(MPI_COMM_WORLD, qc, pr, &mcol));
    ncclComm_t nworld, nrow, ncol;
    {
        ncclUniqueId id;
        if (!g_rank) CHECK_NCCL(ncclGetUniqueId(&id));
        CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
        CHECK_NCCL(ncclCommInitRank(&nworld, P * Q, id, g_rank));
        int rr; CHECK_MPI(MPI_Comm_rank(mrow, &rr));
        ncclUniqueId idr;
        if (!rr) CHECK_NCCL(ncclGetUniqueId(&idr));
        CHECK_MPI(MPI_Bcast(&idr, sizeof(idr), MPI_BYTE, 0, mrow));
        CHECK_NCCL(ncclCommInitRank(&nrow, Q, idr, rr));
        int cr; CHECK_MPI(MPI_Comm_rank(mcol, &cr));
        ncclUniqueId idc;
        if (!cr) CHECK_NCCL(ncclGetUniqueId(&idc));
        CHECK_MPI(MPI_Bcast(&idc, sizeof(idc), MPI_BYTE, 0, mcol));
        CHECK_NCCL(ncclCommInitRank(&ncol, P, idc, cr));
    }

    int N = 16384, NB = 1024;
    double thresh = 16.0;
    if (argc > 1) {
        FILE *f = fopen(argv[1], "r");
        if (f) {
            char ln[4096]; std::vector<std::string> L;
            while (fgets(ln, sizeof(ln), f)) L.push_back(ln);
            fclose(f);
            if (L.size() > 12) {
                N = atoi(L[5].c_str());
                NB = atoi(L[7].c_str());
                thresh = atof(L[12].c_str());
            }
        }
    }
    const u64 seed = 42;
    const long long ntot = (long long)N + 1;
    const int npan = (N + NB - 1) / NB;

    int prLow, prHigh;
    CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&prLow, &prHigh));
    cudaStream_t s0, sP;   /* s0 = trailing update, sP = panel chain + comms */
    CHECK_CUDA(cudaStreamCreateWithPriority(&s0, cudaStreamNonBlocking, prLow));
    CHECK_CUDA(cudaStreamCreateWithPriority(&sP, cudaStreamNonBlocking, prHigh));
    cudaStream_t sW;       /* multi-node: W broadcast off the sP critical path */
    CHECK_CUDA(cudaStreamCreateWithPriority(&sW, cudaStreamNonBlocking, prHigh));
    cublasHandle_t hb; CHECK_CUBLAS(cublasCreate(&hb));
    CHECK_CUBLAS(cublasSetStream(hb, s0));
    const size_t WSMB = (getenv("CHPL2D_WS") ? atoi(getenv("CHPL2D_WS")) : 256);
    { void *w; if (cudaMalloc(&w, WSMB << 20) == cudaSuccess)
          CHECK_CUBLAS(cublasSetWorkspace(hb, w, WSMB << 20)); }
    cublasHandle_t hbP; CHECK_CUBLAS(cublasCreate(&hbP));
    CHECK_CUBLAS(cublasSetStream(hbP, sP));
    { void *w; if (cudaMalloc(&w, WSMB << 20) == cudaSuccess)
          CHECK_CUBLAS(cublasSetWorkspace(hbP, w, WSMB << 20)); }
    cusolverDnHandle_t cs; CHECK_CUSOLVER(cusolverDnCreate(&cs));
    CHECK_CUSOLVER(cusolverDnSetStream(cs, sP));
    /* stale-selection: tall fp32 selection getrf runs one panel early on its
     * own stream/handle so it overlaps chainB + trailing update */
    cudaStream_t sSel;
    CHECK_CUDA(cudaStreamCreateWithPriority(&sSel, cudaStreamNonBlocking,
                                            prLow));
    cusolverDnHandle_t cs2; CHECK_CUSOLVER(cusolverDnCreate(&cs2));
    CHECK_CUSOLVER(cusolverDnSetStream(cs2, sSel));

    const long long mloc = lrows_below(N, NB, pr, N);
    const long long nloc = lcols_below(ntot, NB, qc, ntot);
    size_t lda = ((size_t)mloc + 127) & ~127ULL; lda += 64;
    std::vector<long long> h_grow, h_gcol;
    for (long long i = pr; i * NB < N; i += P)
        for (long long g = i * NB;
             g < std::min((long long)N, (i + 1) * (long long)NB); g++)
            h_grow.push_back(g);
    for (long long j = qc; j * NB < ntot; j += Q)
        for (long long c = j * NB;
             c < std::min(ntot, (j + 1) * (long long)NB); c++)
            h_gcol.push_back(c);

    double *d_A;      CHECK_CUDA(cudaMalloc(&d_A, lda * nloc * 8));
    long long *d_grow, *d_gcol;
    CHECK_CUDA(cudaMalloc(&d_grow, mloc * 8));
    CHECK_CUDA(cudaMalloc(&d_gcol, nloc * 8));
    CHECK_CUDA(cudaMemcpy(d_grow, h_grow.data(), mloc * 8, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gcol, h_gcol.data(), nloc * 8, cudaMemcpyHostToDevice));

    double *d_W[2];
    CHECK_CUDA(cudaMalloc(&d_W[0], (size_t)mloc * NB * 8));
    CHECK_CUDA(cudaMalloc(&d_W[1], (size_t)mloc * NB * 8));
    double *d_Wf;   /* fp32 panel (mloc x NB floats) or fp64 2NB x NB */
    CHECK_CUDA(cudaMalloc(&d_Wf,
        std::max((size_t)mloc * NB * 4, (size_t)2 * NB * NB * 8)));
    double *d_cand;   CHECK_CUDA(cudaMalloc(&d_cand, (size_t)2 * NB * NB * 8));
    double *d_pkg[2];
    CHECK_CUDA(cudaMalloc(&d_pkg[0], (size_t)3 * NB * NB * 8));
    CHECK_CUDA(cudaMalloc(&d_pkg[1], (size_t)3 * NB * NB * 8));
    double *d_U[2];
    CHECK_CUDA(cudaMalloc(&d_U[0], (size_t)NB * nloc * 8));
    CHECK_CUDA(cudaMalloc(&d_U[1], (size_t)NB * nloc * 8));
    const long long XT = getenv("CHPL2D_XT") ? atoll(getenv("CHPL2D_XT")) : 8192;
    int WCH = getenv("CHPL2D_WCH") ? atoi(getenv("CHPL2D_WCH")) : 4;
    if (WCH < 1) WCH = 1;
    if (WCH > 8) WCH = 8;
    long long exch_seq = 0; /* monotone tile counter for the CE exchange */
    double *d_stage;  CHECK_CUDA(cudaMalloc(&d_stage, (size_t)NB * XT * 8));
    double *d_Mt;     CHECK_CUDA(cudaMalloc(&d_Mt, (size_t)NB * XT * 8));
    double *d_Mfb;    CHECK_CUDA(cudaMalloc(&d_Mfb, (size_t)2 * NB * NB * 8));
    double *d_Ms;     CHECK_CUDA(cudaMalloc(&d_Ms, (size_t)NB * XT * 8));

    double *d_xpack;
    CHECK_CUDA(cudaMalloc(&d_xpack, (size_t)4 * NB * XT * 8));
    int *d_ipiv;      CHECK_CUDA(cudaMalloc(&d_ipiv, 2 * NB * 4));
    int *d_info;      CHECK_CUDA(cudaMalloc(&d_info, 4));
    int *d_rows;      CHECK_CUDA(cudaMalloc(&d_rows, 10 * NB * 4));
    float *d_sel;
    CHECK_CUDA(cudaMalloc(&d_sel, (size_t)2 * NB * NB * 4));
    int *d_top, *d_prov0, *d_prov1, *d_win, *d_cnt, *d_occ;
    CHECK_CUDA(cudaMalloc(&d_top, 2 * NB * 4));
    CHECK_CUDA(cudaMalloc(&d_prov0, NB * 4));
    CHECK_CUDA(cudaMalloc(&d_prov1, NB * 4));
    CHECK_CUDA(cudaMalloc(&d_win, NB * 4));
    CHECK_CUDA(cudaMalloc(&d_cnt, 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_occ,
        (size_t)(std::max(mloc, (long long)2 * NB) + NB) * 4));
    CHECK_CUDA(cudaFuncSetAttribute(buildlists_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 40 * NB + 1024));

    /* ---- Ozaki int8 fp64-emulation context (CHPL2D_OZ=1) ---- */
    int oz_on = getenv("CHPL2D_OZ") ? atoi(getenv("CHPL2D_OZ")) : 0;
    /* ---- GEMMul8 (Ozaki-II CRT) trailing-update backend (CHPL2D_G8=1).
     * Replaces the scheme-I fused-diagonal engine for the big s0 update
     * ONLY; W transport (fp64), U staging, urgent sP updates, pivoting are
     * all untouched.  W is quantized locally once per panel and reused
     * across every chunk/tile via skip_scalA; M = inv(U11)*U12 is computed
     * per tile on s0 from the persistent d_U/d_pkg, exactly like the fp64
     * fallback path (no sP-side B slicing needed at all). ---- */
    const int g8_on = getenv("CHPL2D_G8") ? atoi(getenv("CHPL2D_G8")) : 0;
    const int G8_MODULI =
        getenv("CHPL2D_G8_MODULI") ? atoi(getenv("CHPL2D_G8_MODULI")) : 12;
    const int G8_FAST =
        getenv("CHPL2D_G8_FAST") ? atoi(getenv("CHPL2D_G8_FAST")) : 1;
    const long long G8_NT =
        getenv("CHPL2D_G8_NT") ? atoll(getenv("CHPL2D_G8_NT")) : 2048;
    if (g8_on) oz_on = 0; /* the two update engines are mutually exclusive */
    const int OZ_S = getenv("CHPL2D_OZ_S") ? atoi(getenv("CHPL2D_OZ_S")) : 6;
    const int OZ_D = getenv("CHPL2D_OZ_D") ? atoi(getenv("CHPL2D_OZ_D")) : 5;
    const long long OZ_MINN = 4096;
    const long long OZ_NT = getenv("CHPL2D_OZ_NT") ? atoll(getenv("CHPL2D_OZ_NT")) : 1024;
    const int OZ_ALGO = getenv("CHPL2D_OZ_ALGO") ? atoi(getenv("CHPL2D_OZ_ALGO")) : 3;
    /* fraction (percent) of update columns done in fp64 CONCURRENTLY with the
     * int8 path: the DMMA pipe is otherwise idle while IMMA runs. */
    const int OZ_FP = getenv("CHPL2D_OZ_FP") ? atoi(getenv("CHPL2D_OZ_FP")) : 25;
    signed char *ozSA = nullptr, *ozSB = nullptr;
    int *ozDg[2][6] = {};
    cudaStream_t sOz = nullptr, sFp = nullptr;
    cublasHandle_t hbF = nullptr;
    cudaEvent_t evOzGemm[2], evOzRecon[2], evFpDone;
    double *ozRinv = nullptr, *ozRsc = nullptr, *ozCinv = nullptr,
           *ozCsc = nullptr;
    cublasLtHandle_t ozLt = nullptr;
    cublasLtMatmulPreference_t ozPref = nullptr;
    void *ozWs = nullptr; const size_t ozWsz = 256u << 20;
    if (oz_on) {
        CHECK_CUDA(cudaMalloc(&ozSA, (size_t)OZ_S * mloc * NB));
        CHECK_CUDA(cudaMalloc(&ozSB, (size_t)OZ_S * NB * nloc));
        for (int ss = 0; ss < 2; ss++)
            for (int d = 0; d < 6; d++)
                CHECK_CUDA(cudaMalloc(&ozDg[ss][d],
                                      (size_t)mloc * OZ_NT * 4));
        CHECK_CUDA(cudaStreamCreateWithPriority(&sOz, cudaStreamNonBlocking,
                                                prLow));
        CHECK_CUDA(cudaStreamCreateWithPriority(&sFp, cudaStreamNonBlocking,
                                                prLow));
        CHECK_CUBLAS(cublasCreate(&hbF));
        CHECK_CUBLAS(cublasSetStream(hbF, sFp));
        { void *w; size_t f = WSMB < 128 ? WSMB : 128;
          if (cudaMalloc(&w, f << 20) == cudaSuccess)
              CHECK_CUBLAS(cublasSetWorkspace(hbF, w, f << 20)); }
        CHECK_CUDA(cudaEventCreateWithFlags(&evFpDone,
                                            cudaEventDisableTiming));
        for (int ss = 0; ss < 2; ss++) {
            CHECK_CUDA(cudaEventCreateWithFlags(&evOzGemm[ss],
                                                cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&evOzRecon[ss],
                                                cudaEventDisableTiming));
            CHECK_CUDA(cudaEventRecord(evOzRecon[ss], s0));
        }
        CHECK_CUDA(cudaMalloc(&ozRinv, mloc * 8));
        CHECK_CUDA(cudaMalloc(&ozRsc, mloc * 8));
        CHECK_CUDA(cudaMalloc(&ozCinv, nloc * 8));
        CHECK_CUDA(cudaMalloc(&ozCsc, nloc * 8));
        CHECK_CUBLAS(cublasLtCreate(&ozLt));
        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&ozPref));
        CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(ozPref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ozWsz, sizeof(ozWsz)));
        CHECK_CUDA(cudaMalloc(&ozWs, ozWsz));
        if (!g_rank) printf("[chpl_p2] OZ int8 emulation on: S=%d D=%d\n",
                            OZ_S, OZ_D);
    }
    void *g8_work = nullptr;
    double *d_Mg8 = nullptr;
    cublasHandle_t hbG8 = nullptr;
    if (g8_on) {
        /* workspace sized for the largest call: m = mloc, n = G8_NT + 1024
         * (column tails < 1024 are absorbed into the previous tile), k = NB,
         * with the skip_scalA slice reserved for W reuse across tiles */
        const size_t g8_wsz = gemmul8::workSize<false, gemmul8::Backend::INT8>(
            (size_t)mloc, (size_t)(G8_NT + 1024), (size_t)NB, G8_MODULI,
            true, false);
        CHECK_CUDA(cudaMalloc(&g8_work, g8_wsz));
        CHECK_CUDA(cudaMalloc(&d_Mg8, (size_t)NB * (G8_NT + 1024) * 8));
        CHECK_CUBLAS(cublasCreate(&hbG8));
        CHECK_CUBLAS(cublasSetStream(hbG8, s0));
        { void *w; if (cudaMalloc(&w, 64u << 20) == cudaSuccess)
              CHECK_CUBLAS(cublasSetWorkspace(hbG8, w, 64u << 20)); }
        if (!g_rank)
            printf("[chpl_p2] G8 backend on: moduli=%d fast=%d nt=%lld "
                   "work=%.2f GiB\n", G8_MODULI, G8_FAST, G8_NT,
                   g8_wsz / 1073741824.0);
    }
    /* one emulated update: C(m x n, ld lda) -= W(m x k) * M(k x n),
     * using pre-sliced SA (this iteration) and SB (this chunk, col offset
     * cB into ozCinv/ozCsc).  n tiled by OZ_NT. */
    auto oz_gemm = [&](long long m, long long n, int kk, double *Cp,
                       long long cB, cudaStream_t st) {
        cublasLtMatmulDesc_t opI;
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&opI, CUBLAS_COMPUTE_32I,
                                              CUDA_R_32I));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opI,
            CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opI,
            CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        const size_t ldS = (size_t)OZ_S * kk;
        /* per-diagonal layouts + heuristics, hoisted out of the tile loop */
        long long ntail = (n % OZ_NT) && (n > OZ_NT) ? n % OZ_NT : 0;
        cublasLtMatrixLayout_t laD[6], lbD[6], lcD[6], lbDt[6], lcDt[6];
        cublasLtMatmulHeuristicResult_t hD[6][4], hDt[6][4];
        for (int d = 0; d <= OZ_D; d++) {
            long long kd = (long long)(std::min(d, OZ_S - 1) + 1) * kk;
            int nh = 0;
            CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&laD[d], CUDA_R_8I, kd, m,
                                                    ldS));
            CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lbD[d], CUDA_R_8I, kd,
                std::min(n, OZ_NT), ldS));
            CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lcD[d], CUDA_R_32I, m,
                std::min(n, OZ_NT), m));
            CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(ozLt, opI, laD[d],
                lbD[d], lcD[d], lcD[d], ozPref, 4, hD[d], &nh));
            if (nh > OZ_ALGO) hD[d][0] = hD[d][OZ_ALGO];
            if (!nh) {
                fprintf(stderr, "[rank %d] oz: no algo d=%d m=%lld\n",
                        g_rank, d, m);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (ntail) {
                CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lbDt[d], CUDA_R_8I,
                    kd, ntail, ldS));
                CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lcDt[d], CUDA_R_32I,
                    m, ntail, m));
                CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(ozLt, opI, laD[d],
                    lbDt[d], lcDt[d], lcDt[d], ozPref, 4, hDt[d], &nh));
                if (nh > OZ_ALGO) hDt[d][0] = hDt[d][OZ_ALGO];
                if (!nh) {
                    fprintf(stderr, "[rank %d] oz: no tail algo d=%d\n",
                            g_rank, d);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
        for (long long t0 = 0; t0 < n; t0 += OZ_NT) {
            long long nt = std::min(OZ_NT, n - t0);
            int istail = (nt != OZ_NT && n > OZ_NT);
            const signed char *Bt = ozSB + (size_t)(cB + t0) * ldS;
            const int set = (int)((t0 / OZ_NT) & 1);
            CHECK_CUDA(cudaStreamWaitEvent(st, evOzRecon[set], 0));
            for (int d = 0; d <= OZ_D; d++) {
                /* fused diagonal: [a_0..a_min(d,S-1)] x [b_d..b_0] suffix */
                int lo = (d >= OZ_S) ? d - OZ_S + 1 : 0; /* first valid p */
                (void)lo; /* OZ_D <= OZ_S-1 keeps lo == 0 */
                int ia = 1, ib = 0;
                CHECK_CUBLAS(cublasLtMatmul(ozLt, opI, &ia, ozSA,
                    laD[d], Bt + (size_t)(OZ_S - 1 - d) * kk,
                    istail ? lbDt[d] : lbD[d], &ib, ozDg[set][d],
                    istail ? lcDt[d] : lcD[d], ozDg[set][d],
                    istail ? lcDt[d] : lcD[d],
                    istail ? &hDt[d][0].algo : &hD[d][0].algo, ozWs, ozWsz,
                    st));
            }
            CHECK_CUDA(cudaEventRecord(evOzGemm[set], st));
            CHECK_CUDA(cudaStreamWaitEvent(sOz, evOzGemm[set], 0));
            {
                dim3 blk(64, 4), grd((unsigned)((m + 63) / 64),
                                     (unsigned)((nt + 3) / 4));
                ozrecon_kernel<<<grd, blk, 0, sOz>>>(
                    Cp + (size_t)t0 * lda, lda, (int)m, (int)nt,
                    ozDg[set][0], ozDg[set][1], ozDg[set][2], ozDg[set][3],
                    ozDg[set][4], ozDg[set][5], OZ_D + 1, ozRsc,
                    ozCsc + cB + t0);
            }
            CHECK_CUDA(cudaEventRecord(evOzRecon[set], sOz));
        }
        for (int d = 0; d <= OZ_D; d++) {
            cublasLtMatrixLayoutDestroy(laD[d]);
            cublasLtMatrixLayoutDestroy(lbD[d]);
            cublasLtMatrixLayoutDestroy(lcD[d]);
            if (ntail) { cublasLtMatrixLayoutDestroy(lbDt[d]);
                         cublasLtMatrixLayoutDestroy(lcDt[d]); }
        }
        cublasLtMatmulDescDestroy(opI);
        /* downstream (evUpdC) must see all reconstructions done */
        CHECK_CUDA(cudaStreamWaitEvent(st, evOzRecon[0], 0));
        CHECK_CUDA(cudaStreamWaitEvent(st, evOzRecon[1], 0));
    };

    double *d_x, *d_acc, *d_tmp, *d_rs, *d_b, *d_r;
    CHECK_CUDA(cudaMalloc(&d_x, (size_t)N * 8));
    CHECK_CUDA(cudaMalloc(&d_acc, (size_t)mloc * 8));
    CHECK_CUDA(cudaMalloc(&d_tmp, (size_t)NB * 8));
    double *d_tmp3; CHECK_CUDA(cudaMalloc(&d_tmp3, (size_t)7 * NB * 8));
    CHECK_CUDA(cudaMalloc(&d_rs, (size_t)mloc * 8));
    CHECK_CUDA(cudaMalloc(&d_b, (size_t)mloc * 8));
    CHECK_CUDA(cudaMalloc(&d_r, (size_t)mloc * 8));

    /* ---- CUDA IPC data plane: peers pull W/U over copy engines ----
     * flags layout (per rank): [0]=W pack seq, [1]=W pull seq,
     * [2]=U pack seq, [3]=U pull seq.  Seq values are monotone counters. */
    long long *d_flags; CHECK_CUDA(cudaMalloc(&d_flags, 8 * 8));
    CHECK_CUDA(cudaMemset(d_flags, 0, 8 * 8));
    struct PeerBufs { double *W[2], *U[2], *acc, *tmp, *xp; long long *flags; };
    PeerBufs peer[16] = {};
    {
        struct H { cudaIpcMemHandle_t hW[2], hU[2], hF, hAcc, hTmp, hXp; };
        H mine, all[16];
        CHECK_CUDA(cudaIpcGetMemHandle(&mine.hW[0], d_W[0]));
        CHECK_CUDA(cudaIpcGetMemHandle(&mine.hW[1], d_W[1]));
        CHECK_CUDA(cudaIpcGetMemHandle(&mine.hU[0], d_U[0]));
        CHECK_CUDA(cudaIpcGetMemHandle(&mine.hU[1], d_U[1]));
        CHECK_CUDA(cudaIpcGetMemHandle(&mine.hF, d_flags));
        CHECK_CUDA(cudaIpcGetMemHandle(&mine.hAcc, d_acc));
        CHECK_CUDA(cudaIpcGetMemHandle(&mine.hTmp, d_tmp));
        CHECK_CUDA(cudaIpcGetMemHandle(&mine.hXp, d_xpack));
        CHECK_MPI(MPI_Allgather(&mine, sizeof(H), MPI_BYTE, all, sizeof(H),
                                MPI_BYTE, MPI_COMM_WORLD));
        for (int r = 0; r < P * Q; r++) {
            if (!samenode[r] && r != g_rank) continue; /* IPC is intra-node */
            if (r == g_rank) {
                peer[r].W[0] = d_W[0]; peer[r].W[1] = d_W[1];
                peer[r].U[0] = d_U[0]; peer[r].U[1] = d_U[1];
                peer[r].flags = d_flags; peer[r].acc = d_acc;
                peer[r].tmp = d_tmp; peer[r].xp = d_xpack;
                continue;
            }
            for (int b = 0; b < 2; b++) {
                CHECK_CUDA(cudaIpcOpenMemHandle((void **)&peer[r].W[b],
                    all[r].hW[b], cudaIpcMemLazyEnablePeerAccess));
                CHECK_CUDA(cudaIpcOpenMemHandle((void **)&peer[r].U[b],
                    all[r].hU[b], cudaIpcMemLazyEnablePeerAccess));
            }
            CHECK_CUDA(cudaIpcOpenMemHandle((void **)&peer[r].flags,
                all[r].hF, cudaIpcMemLazyEnablePeerAccess));
            CHECK_CUDA(cudaIpcOpenMemHandle((void **)&peer[r].acc,
                all[r].hAcc, cudaIpcMemLazyEnablePeerAccess));
            CHECK_CUDA(cudaIpcOpenMemHandle((void **)&peer[r].tmp,
                all[r].hTmp, cudaIpcMemLazyEnablePeerAccess));
            CHECK_CUDA(cudaIpcOpenMemHandle((void **)&peer[r].xp,
                all[r].hXp, cudaIpcMemLazyEnablePeerAccess));
        }
    }
    /* same-pr row peers (for W) and the ncol partner (for U) */
    int rowpeer[7]; { int t = 0;
        for (int q2 = 0; q2 < Q; q2++)
            if (q2 != qc) rowpeer[t++] = pr + 2 * q2; }
    const int upartner = (1 - pr) + 2 * qc;
    /* the W data plane uses CE pulls only when all row peers share the
     * node (single-node runs); across nodes it broadcasts via NCCL */
    int wce = 1;
    for (int t = 0; t < Q - 1; t++)
        if (!samenode[rowpeer[t]]) wce = 0;
    int lwork = 0, lwork2 = 0, lworks = 0;
    CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(cs, (int)mloc, NB, d_Wf,
                                               (int)std::max(mloc, (long long)1),
                                               &lwork));
    CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(cs, 2 * NB, NB, d_cand, 2 * NB,
                                               &lwork2));
    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(cs, (int)mloc, NB,
                                               (float *)d_Wf,
                                               (int)std::max(mloc, (long long)1),
                                               &lworks));
    lwork = std::max(std::max(lwork, lwork2), (lworks + 1) / 2);
    double *d_work;  CHECK_CUDA(cudaMalloc(&d_work, (size_t)lwork * 8));
    /* private buffers for the stale-selection stream */
    float *d_work2;  CHECK_CUDA(cudaMalloc(&d_work2, (size_t)lworks * 4));
    int *d_ipiv2;    CHECK_CUDA(cudaMalloc(&d_ipiv2, 2 * NB * 4));
    int *d_info2;    CHECK_CUDA(cudaMalloc(&d_info2, 4));
    int *d_top2;     CHECK_CUDA(cudaMalloc(&d_top2, 2 * NB * 4));
    int *d_rows2;    CHECK_CUDA(cudaMalloc(&d_rows2, NB * 4));
    int *d_provP;    CHECK_CUDA(cudaMalloc(&d_provP, NB * 4));
    int *d_occ2;
    CHECK_CUDA(cudaMalloc(&d_occ2,
        (size_t)(std::max(mloc, (long long)2 * NB) + NB) * 4));

    for (long long c0 = 0; c0 < nloc; c0 += 32768) {
        int nc = (int)std::min((long long)32768, nloc - c0);
        dim3 gg(64, nc);
        matgen2d_kernel<<<gg, 256, 0, s0>>>(d_A + c0 * lda, lda, (int)mloc,
                                            d_grow, d_gcol + c0, nc, ntot, seed);
    }
    CHECK_CUDA(cudaStreamSynchronize(s0));

    const double done_ = 1.0, dmone = -1.0, dzero = 0.0;
    std::vector<int> h_ipiv(2 * NB);
    std::vector<long long> h_win(NB);
    std::vector<int> myrows_i(2 * NB);
    std::vector<long long> provg(2 * NB, -1);

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    double t0 = MPI_Wtime();
    time_t ts0 = time(NULL);

    /* events: W-ready, per-chunk U-ready, per-chunk update-done */
    std::vector<cudaEvent_t> evReadyW(npan + 1), evUpd(npan + 1), evXg0(npan + 1);
    std::vector<std::array<cudaEvent_t, 8>> evWc(npan + 1);
    std::vector<cudaEvent_t> evD2s(npan + 1), evSel(npan + 1), evCand(npan + 1);
    const int presel = getenv("CHPL2D_PRESEL") ? atoi(getenv("CHPL2D_PRESEL")) : 0;
    const int g_psync = getenv("CHPL2D_PSYNC") ? atoi(getenv("CHPL2D_PSYNC")) : 0;
    std::vector<std::array<cudaEvent_t, 4>> evReadyU(npan + 1), evUpdC(npan + 1);
    for (int i = 0; i <= npan; i++) {
        CHECK_CUDA(cudaEventCreateWithFlags(&evReadyW[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evUpd[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evXg0[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evD2s[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evSel[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&evCand[i], cudaEventDisableTiming));
        for (int c = 0; c < 8; c++)
            CHECK_CUDA(cudaEventCreateWithFlags(&evWc[i][c],
                                                cudaEventDisableTiming));
        for (int c = 0; c < 4; c++) {
            CHECK_CUDA(cudaEventCreateWithFlags(&evReadyU[i][c],
                                                cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&evUpdC[i][c],
                                                cudaEventDisableTiming));
        }
    }
    /* chunk column ranges of iteration k for THIS rank: {urgent, rest0,
     * rest1} as (start, count) local column pairs */
    /* chunk ranges of iteration k: c0 = panel(k+1) columns, c1 = panel(k+2)
     * columns, c2 = mid up to the CONSTANT split S* = nloc/2, c3 = the tail.
     * The two panel chunks keep the pivot/exchange spine gated only by tiny
     * GEMMs; the constant split keeps mid/tail boundaries nested across
     * iterations so chainB(k+1) exchange groups always cover the columns
     * each U12 part of k+1 needs. */
    const long long Sstar = (nloc + 1) / 2;
    auto pcols_of = [&](int j) -> long long { /* local cols of panel j */
        if (j >= npan || qc != j % Q) return 0;
        int jbp = (int)std::min((long long)NB, (long long)N - (long long)j * NB);
        return lcols_below((long long)j * NB + jbp, NB, qc, ntot) -
               lcols_below((long long)j * NB, NB, qc, ntot);
    };
    auto chunks_of = [&](int k, long long rng[4][2]) {
        const long long gr0 = (long long)k * NB;
        const int jb = (int)std::min((long long)NB, (long long)N - gr0);
        const long long tc0 = lcols_below(gr0 + jb, NB, qc, ntot);
        const long long u1 = pcols_of(k + 1), u2 = pcols_of(k + 2);
        long long S = tc0 + u1 + u2 + (nloc - tc0 - u1 - u2 + 1) / 2;
        if (S > nloc) S = nloc;
        rng[0][0] = tc0;            rng[0][1] = u1;
        rng[1][0] = tc0 + u1;       rng[1][1] = u2;
        rng[2][0] = tc0 + u1 + u2;  rng[2][1] = S - (tc0 + u1 + u2);
        rng[3][0] = S;              rng[3][1] = nloc - S;
    };
    const int timing = getenv("CHPL2D_TIMING") ? atoi(getenv("CHPL2D_TIMING")) : 0;
    std::vector<cudaEvent_t> tU0(npan), tU1(npan), tB0(npan + 1), tB1(npan + 1),
                             tX0(npan + 1), tX1(npan + 1);
    /* timing==2: s0 fine events [9]/iter + sP chain segments [10]/iter */
    std::vector<std::array<cudaEvent_t, 9>> tS0(timing >= 2 ? npan : 0);
    std::vector<std::array<cudaEvent_t, 10>> tSP(timing >= 2 ? npan + 1 : 0);
    if (timing >= 2) {
        for (int i = 0; i < npan; i++)
            for (auto &e : tS0[i]) CHECK_CUDA(cudaEventCreate(&e));
        for (int i = 0; i <= npan; i++)
            for (auto &e : tSP[i]) CHECK_CUDA(cudaEventCreate(&e));
    }
    auto rec2 = [&](std::vector<std::array<cudaEvent_t, 10>> &v, int k, int j,
                    cudaStream_t st) {
        if (timing >= 2 && k >= 0) CHECK_CUDA(cudaEventRecord(v[k][j], st));
    };
    if (timing)
        for (int i = 0; i <= npan; i++) {
            if (i < npan) { CHECK_CUDA(cudaEventCreate(&tU0[i]));
                            CHECK_CUDA(cudaEventCreate(&tU1[i])); }
            CHECK_CUDA(cudaEventCreate(&tB0[i]));
            CHECK_CUDA(cudaEventCreate(&tB1[i]));
            CHECK_CUDA(cudaEventCreate(&tX0[i]));
            CHECK_CUDA(cudaEventCreate(&tX1[i]));
        }

    /* panel chain part A: tournament factorization of panel k (on sP).
     * Host-synchronous pivot bookkeeping is fine: the trailing update of
     * iteration k-1 is already queued on s0 and keeps the GPU busy. */
    const long long mloc0 = lrows_below(N, NB, 0, N);
    const long long mloc1 = lrows_below(N, NB, 1, N);
    auto chainA = [&](int k) {
        const long long gr0 = (long long)k * NB;
        const int jb = (int)std::min((long long)NB, (long long)N - gr0);
        const int kq = k % Q;
        const long long plr0 = lrows_below(gr0, NB, pr, N);
        const long long mlk = mloc - plr0;
        const long long pc0 = lcols_below(gr0, NB, qc, ntot);
        if (k > 0) CHECK_CUDA(cudaStreamWaitEvent(sP, evUpdC[k - 1][0], 0));
        /* s0 may read d_pkg[k-2 parity] (fallback M) until update k-2 ends */
        if (k >= 2) CHECK_CUDA(cudaStreamWaitEvent(sP, evUpdC[k - 2][3], 0));
        rec2(tSP, k, 0, sP);
        /* candidate counts are geometric: no communication needed */
        const int nc0 = (int)std::max(0LL, std::min((long long)jb,
                            mloc0 - lrows_below(gr0, NB, 0, N)));
        const int nc1 = (int)std::max(0LL, std::min((long long)jb,
                            mloc1 - lrows_below(gr0, NB, 1, N)));
        const int myncand = pr == 0 ? nc0 : nc1;
        if (qc == kq && myncand > 0) {
            /* bisect modes 6-8: preselect ran partially; use inline results
             * (wait evSel first so the sSel getrf isn't mid-flight on d_Wf) */
            if (presel && k >= 1 && g_psync >= 6)
                CHECK_CUDA(cudaStreamWaitEvent(sP, evSel[k], 0));
            if (!presel || k == 0 || g_psync >= 6) {
                {
                    dim3 gg(128, jb);
                    d2s_kernel<<<gg, 256, 0, sP>>>(d_A + pc0 * lda + plr0,
                                                   lda, (float *)d_Wf, mlk,
                                                   mlk, jb);
                }
                CHECK_CUSOLVER(cusolverDnSgetrf(cs, (int)mlk, jb,
                                                (float *)d_Wf, (int)mlk,
                                                (float *)d_work, d_ipiv,
                                                d_info));
                /* probe: artificial chain delay after the selection getrf to
                 * measure how much of it is exposed on the critical path */
                static const long long dly_us =
                    getenv("CHPL2D_DLY") ? atoll(getenv("CHPL2D_DLY")) : 0;
                if (dly_us > 0)
                    spin_kernel<<<1, 1, 0, sP>>>(dly_us * 1980);
                resolve_kernel<<<1, 256, 0, sP>>>(d_ipiv, myncand, (int)mlk,
                                                  d_occ, d_top);
                prep_cand_kernel<<<(myncand + 255) / 256, 256, 0, sP>>>(
                    d_top, myncand, plr0, d_grow, d_rows + 6 * NB,
                    pr == 0 ? d_prov0 : d_prov1);
            } else {
                /* selection ran one panel early on sSel (stale by one
                 * urgent update + exchange); values gathered fresh below */
                CHECK_CUDA(cudaStreamWaitEvent(sP, evSel[k], 0));
                CHECK_CUDA(cudaMemcpyAsync(d_rows + 6 * NB, d_rows2,
                                           (size_t)myncand * 4,
                                           cudaMemcpyDeviceToDevice, sP));
                CHECK_CUDA(cudaMemcpyAsync(pr == 0 ? d_prov0 : d_prov1,
                                           d_provP, (size_t)myncand * 4,
                                           cudaMemcpyDeviceToDevice, sP));
                static const int psync4 =
                    getenv("CHPL2D_PSYNC") ? atoi(getenv("CHPL2D_PSYNC")) : 0;
                if (psync4 == 4) {
                    /* bisect: recompute selection inline on sP and OVERWRITE
                     * the presel results; isolates result-vs-side-effect */
                    dim3 gg(128, jb);
                    d2s_kernel<<<gg, 256, 0, sP>>>(d_A + pc0 * lda + plr0,
                                                   lda, (float *)d_Wf, mlk,
                                                   mlk, jb);
                    CHECK_CUSOLVER(cusolverDnSgetrf(cs, (int)mlk, jb,
                                                    (float *)d_Wf, (int)mlk,
                                                    (float *)d_work, d_ipiv,
                                                    d_info));
                    resolve_kernel<<<1, 256, 0, sP>>>(d_ipiv, myncand,
                                                      (int)mlk, d_occ, d_top);
                    prep_cand_kernel<<<(myncand + 255) / 256, 256, 0, sP>>>(
                        d_top, myncand, plr0, d_grow, d_rows + 6 * NB,
                        pr == 0 ? d_prov0 : d_prov1);
                }
                static const int pdump3a =
                    getenv("CHPL2D_PDUMP") ? atoi(getenv("CHPL2D_PDUMP")) : 0;
                if (pdump3a == 3) {
                    dim3 gg(128, jb);
                    d2s_kernel<<<gg, 256, 0, sP>>>(d_A + pc0 * lda + plr0,
                                                   lda, (float *)d_Wf, mlk,
                                                   mlk, jb);
                    CHECK_CUDA(cudaMemsetAsync(d_tmp, 0, 8, sP));
                    sumf_kernel<<<256, 256, 0, sP>>>((float *)d_Wf,
                                                     (size_t)mlk * jb, d_tmp);
                    double hs;
                    CHECK_CUDA(cudaMemcpyAsync(&hs, d_tmp, 8,
                                               cudaMemcpyDeviceToHost, sP));
                    CHECK_CUDA(cudaStreamSynchronize(sP));
                    fprintf(stderr, "[pw r%d k%d] chainA=%.17g\n", g_rank, k,
                            hs);
                    fflush(stderr);
                }
            }
            static const int pdump =
                getenv("CHPL2D_PDUMP") ? atoi(getenv("CHPL2D_PDUMP")) : 0;
            if (pdump) {
                CHECK_CUDA(cudaStreamSynchronize(sP));
                std::vector<int> h(myncand);
                CHECK_CUDA(cudaMemcpy(h.data(), d_rows + 6 * NB, myncand * 4,
                                      cudaMemcpyDeviceToHost));
                long long s = 0;
                for (int i = 0; i < myncand; i++) s += h[i];
                fprintf(stderr, "[r%d k%d] candsum=%lld first=%d,%d,%d\n",
                        g_rank, k, s, h[0], h[1], h[2]);
                fflush(stderr);
            }
            dim3 blk(32, 8), grd((myncand + 31) / 32, (unsigned)((jb + 7) / 8));
            grow_gather_kernel<<<grd, blk, 0, sP>>>(
                d_A, lda, d_rows + 6 * NB, myncand, (size_t)pc0, jb,
                d_cand + (pr ? (size_t)NB * NB : 0));
        }
        rec2(tSP, k, 1, sP);
        const int b = k & 1;
        if (qc == kq && nc1 > 0) {
            CHECK_NCCL(ncclGroupStart());
            if (pr == 1) {
                CHECK_NCCL(ncclSend(d_cand + (size_t)NB * NB, (size_t)nc1 * jb,
                                    ncclDouble, 0, ncol, sP));
                CHECK_NCCL(ncclSend(d_prov1, nc1, ncclInt32, 0, ncol, sP));
            } else {
                CHECK_NCCL(ncclRecv(d_cand + (size_t)NB * NB, (size_t)nc1 * jb,
                                    ncclDouble, 1, ncol, sP));
                CHECK_NCCL(ncclRecv(d_prov1, nc1, ncclInt32, 1, ncol, sP));
            }
            CHECK_NCCL(ncclGroupEnd());
        }
        rec2(tSP, k, 2, sP);
        if (qc == kq && pr == 0) {
            int nst = nc0 + nc1;
            /* fp32 stacked getrf selects the winners; the diagonal block is
             * then re-gathered from the fp64 originals and factored square
             * in fp64 (its internal pivoting is composed into the winner
             * order) */
            {
                dim3 gg(64, jb);
                d2s_kernel<<<gg, 256, 0, sP>>>(d_cand, (size_t)nc0, d_sel,
                                               (size_t)nst, nc0, jb);
                if (nc1 > 0)
                    d2s_kernel<<<gg, 256, 0, sP>>>(d_cand + (size_t)NB * NB,
                                                   (size_t)nc1, d_sel + nc0,
                                                   (size_t)nst, nc1, jb);
            }
            CHECK_CUSOLVER(cusolverDnSgetrf(cs, nst, jb, d_sel, nst,
                                            (float *)d_work, d_ipiv, d_info));
            resolve_kernel<<<1, 256, 0, sP>>>(d_ipiv, jb, nst, d_occ, d_top);
            winmap_kernel<<<(jb + 255) / 256, 256, 0, sP>>>(
                d_top, jb, nc0, d_prov0, d_prov1, d_rows + 9 * NB);
            {
                dim3 blk(32, 8), grd((jb + 31) / 32, 64);
                win_gather_kernel<<<grd, blk, 0, sP>>>(
                    d_top, jb, nc0, nc1, d_cand, d_cand + (size_t)NB * NB,
                    d_pkg[b]);
            }
            CHECK_CUSOLVER(cusolverDnDgetrf(cs, jb, jb, d_pkg[b], jb, d_work,
                                            d_ipiv, d_info));
            resolve_kernel<<<1, 256, 0, sP>>>(d_ipiv, jb, jb, d_occ,
                                              d_top + NB);
            compose_win_kernel<<<(jb + 255) / 256, 256, 0, sP>>>(
                d_top + NB, d_rows + 9 * NB, jb, d_win);
            seteye_kernel<<<(jb * jb + 255) / 256, 256, 0, sP>>>(
                d_pkg[b] + (size_t)jb * jb, jb);
            CHECK_CUBLAS(cublasDtrsm(hbP, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_N, CUBLAS_DIAG_UNIT, jb, jb,
                                     &done_, d_pkg[b], jb,
                                     d_pkg[b] + (size_t)jb * jb, jb));
            seteye_kernel<<<(jb * jb + 255) / 256, 256, 0, sP>>>(
                d_pkg[b] + (size_t)2 * jb * jb, jb);
            CHECK_CUBLAS(cublasDtrsm(hbP, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, jb, jb,
                                     &done_, d_pkg[b], jb,
                                     d_pkg[b] + (size_t)2 * jb * jb, jb));
        }
        /* all consumers of d_rows+6NB / d_prov (grow_gather, prov send,
         * winmap) are behind this point: safe for the next preselect */
        CHECK_CUDA(cudaEventRecord(evCand[k], sP));
        rec2(tSP, k, 3, sP);
        CHECK_NCCL(ncclBroadcast(d_win, d_win, jb, ncclInt32, 2 * kq, nworld,
                                 sP));
        rec2(tSP, k, 4, sP);
        static const int pdump2 =
            getenv("CHPL2D_PDUMP") ? atoi(getenv("CHPL2D_PDUMP")) : 0;
        if (pdump2 == 2 && g_rank == 0) {
            CHECK_CUDA(cudaStreamSynchronize(sP));
            std::vector<int> h(jb);
            CHECK_CUDA(cudaMemcpy(h.data(), d_win, jb * 4,
                                  cudaMemcpyDeviceToHost));
            long long s = 0;
            for (int i = 0; i < jb; i++) s += h[i];
            fprintf(stderr, "[win k%d] sum=%lld first=%d,%d,%d,%d\n", k, s,
                    h[0], h[1], h[2], h[3]);
            fflush(stderr);
        }
    };

    /* early selection: called from inside chainB(k-1) right after the
     * urgent chunk (emit chunk0), at which point panel k's columns are
     * fully exchanged AND urgent-updated — i.e. bit-identical to what the
     * inline chainA(k) selection would read.  The snapshot (d2s) stays on
     * sP for ordering; the tall fp32 getrf runs on sSel, overlapping the
     * rest of chainB(k-1) (exch g1-3 + U staging) instead of sitting on
     * the panel-to-panel critical chain. */
    auto preselect = [&](int k) {
        if (!presel || k < 1 || k >= npan) return;
        const long long gr0 = (long long)k * NB;
        const int jb = (int)std::min((long long)NB, (long long)N - gr0);
        const int kq = k % Q;
        const long long plr0 = lrows_below(gr0, NB, pr, N);
        const long long mlk = mloc - plr0;
        const long long pc0 = lcols_below(gr0, NB, qc, ntot);
        const int nc0 = (int)std::max(0LL, std::min((long long)jb,
                            mloc0 - lrows_below(gr0, NB, 0, N)));
        const int nc1 = (int)std::max(0LL, std::min((long long)jb,
                            mloc1 - lrows_below(gr0, NB, 1, N)));
        const int myncand = pr == 0 ? nc0 : nc1;
        if (qc != kq || myncand <= 0) return;
        {
            dim3 gg(128, jb);
            d2s_kernel<<<gg, 256, 0, sP>>>(d_A + pc0 * lda + plr0, lda,
                                           (float *)d_Wf, mlk, mlk, jb);
        }
        static const int pdump3 =
            getenv("CHPL2D_PDUMP") ? atoi(getenv("CHPL2D_PDUMP")) : 0;
        if (pdump3 == 3) {
            CHECK_CUDA(cudaMemsetAsync(d_tmp, 0, 8, sP));
            sumf_kernel<<<256, 256, 0, sP>>>((float *)d_Wf,
                                             (size_t)mlk * jb, d_tmp);
            double hs;
            CHECK_CUDA(cudaMemcpyAsync(&hs, d_tmp, 8, cudaMemcpyDeviceToHost,
                                       sP));
            CHECK_CUDA(cudaStreamSynchronize(sP));
            fprintf(stderr, "[pw r%d k%d] presel=%.17g\n", g_rank, k, hs);
            fflush(stderr);
        }
        CHECK_CUDA(cudaEventRecord(evD2s[k], sP));
        if (g_psync == 6) { CHECK_CUDA(cudaEventRecord(evSel[k], sP)); return; }
        CHECK_CUDA(cudaStreamWaitEvent(sSel, evD2s[k], 0));
        /* previous consumers of d_rows+6NB / d_prov must be done */
        CHECK_CUDA(cudaStreamWaitEvent(sSel, evCand[k - 1], 0));
        CHECK_CUSOLVER(cusolverDnSgetrf(cs2, (int)mlk, jb, (float *)d_Wf,
                                        (int)mlk, d_work2, d_ipiv2, d_info2));
        if (g_psync == 7) { CHECK_CUDA(cudaEventRecord(evSel[k], sSel)); return; }
        resolve_kernel<<<1, 256, 0, sSel>>>(d_ipiv2, myncand, (int)mlk,
                                            d_occ2, d_top2);
        if (g_psync == 8) { CHECK_CUDA(cudaEventRecord(evSel[k], sSel)); return; }
        prep_cand_kernel<<<(myncand + 255) / 256, 256, 0, sSel>>>(
            d_top2, myncand, plr0, d_grow, d_rows2, d_provP);
        CHECK_CUDA(cudaEventRecord(evSel[k], sSel));
    };

    /* panel chain part B: row exchange, diagonal write-back, L21 + row
     * broadcast, U12 + column broadcast.  Must run after the trailing update
     * of the previous iteration (it touches all trailing rows/columns). */
    auto chainB = [&](int k) {
        const long long gr0 = (long long)k * NB;
        const int jb = (int)std::min((long long)NB, (long long)N - gr0);
        const int kq = k % Q, kp = k % P;
        const long long pc0 = lcols_below(gr0, NB, qc, ntot);
        const long long tc0 = lcols_below(gr0 + jb, NB, qc, ntot);
        const long long ntc = nloc - tc0;
        const int b = k & 1;
        if (timing) CHECK_CUDA(cudaEventRecord(tB0[k], sP));

        /* net permutation + exchange lists, built on device from d_win */
        buildlists_kernel<<<1, 256, 40 * NB + 1024, sP>>>(d_win, gr0, jb, NB,
                                                          pr, d_rows, d_cnt);

        /* one exchange pass over a local column range (row counts live on the
         * device; grids are worst-case-sized, NCCL messages fixed at NB rows) */
        auto exch_range = [&](long long c0a, long long nca) {
            const int xpartner = (1 - pr) + 2 * qc;
            for (long long t0 = 0; t0 < nca; t0 += XT) {
                long long c0 = c0a + t0, nc = std::min(XT, nca - t0);
                exch_seq++;
                const int sslot = (int)(exch_seq & 1); /* 0 or 1 */
                dim3 blk(32, 8);
                unsigned gy = (unsigned)std::min((nc + 7) / 8, (long long)64);
                dim3 g2NB((2 * NB + 31) / 32, gy), g1NB((NB + 31) / 32, gy);
                /* partner must have consumed my previous same-parity tile */
                if (exch_seq > 2)
                    flag_wait_kernel<<<1, 1, 0, sP>>>(
                        peer[xpartner].flags + 7, nullptr, nullptr,
                        exch_seq - 2);
                /* pack rows leaving this rank into my send slot */
                gather_dyn_kernel<<<g1NB, blk, 0, sP>>>(
                    d_A, lda, d_rows + 2 * NB, d_cnt, 1, (size_t)c0, nc,
                    d_xpack + (size_t)sslot * NB * XT, (size_t)NB);
                /* local moves staged before any scatter */
                gather_dyn_kernel<<<g2NB, blk, 0, sP>>>(
                    d_A, lda, d_rows + 3 * NB, d_cnt, 2, (size_t)c0, nc,
                    d_xpack + (size_t)2 * NB * XT, (size_t)2 * NB);
                flag_set_kernel<<<1, 1, 0, sP>>>(d_flags + 6, exch_seq);
                /* partner's rows: scatter DIRECTLY from its send slot */
                flag_wait_kernel<<<1, 1, 0, sP>>>(peer[xpartner].flags + 6,
                                                  nullptr, nullptr, exch_seq);
                scatter_dyn_kernel<<<g2NB, blk, 0, sP>>>(
                    d_A, lda, d_rows + 5 * NB, d_cnt, 2, (size_t)c0, nc,
                    d_xpack + (size_t)2 * NB * XT, (size_t)2 * NB);
                scatter_dyn_kernel<<<g1NB, blk, 0, sP>>>(
                    d_A, lda, d_rows + 7 * NB, d_cnt, 3, (size_t)c0, nc,
                    peer[xpartner].xp + (size_t)sslot * NB * XT, (size_t)NB);
                flag_set_kernel<<<1, 1, 0, sP>>>(d_flags + 7, exch_seq);
            }
        };        long long prng[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        long long ptc0 = 0;
        if (k > 0) { chunks_of(k - 1, prng); ptc0 = prng[0][0]; }
        else { prng[0][0] = 0; prng[0][1] = nloc;
               for (int g = 1; g < 4; g++) { prng[g][0] = nloc; prng[g][1] = 0; }
               ptc0 = 0; }

        long long rng[4][2];
        chunks_of(k, rng);
        const long long dlr = lrows_below(gr0, NB, pr, N);
        const long long blr0 = lrows_below(gr0 + jb, NB, pr, N);
        const long long mlb = mloc - blr0;
        int cnext = 0; /* next U12 chunk to emit */
        auto emit_ready = [&](long long xpos) {
            while (cnext < 4 && rng[cnext][0] + rng[cnext][1] <= xpos) {
                long long c0 = rng[cnext][0], nc = rng[cnext][1];
                if (nc > 0) {
                    if (pr == kp) {
                        for (long long t0 = 0; t0 < nc; t0 += XT) {
                            long long nt2 = std::min(XT, nc - t0);
                            CHECK_CUDA(cudaMemcpy2DAsync(
                                d_stage, (size_t)jb * 8,
                                d_A + (c0 + t0) * lda + dlr, lda * 8,
                                (size_t)jb * 8, nt2,
                                cudaMemcpyDeviceToDevice, sP));
                            CHECK_CUBLAS(cublasDgemm(
                                hbP, CUBLAS_OP_N, CUBLAS_OP_N, jb, (int)nt2,
                                jb, &done_, d_pkg[b] + (size_t)jb * jb, jb,
                                d_stage, jb, &dzero,
                                d_A + (c0 + t0) * lda + dlr, (int)lda));
                            CHECK_CUDA(cudaMemcpy2DAsync(
                                d_U[b] + (size_t)(c0 + t0 - tc0) * jb,
                                (size_t)jb * 8, d_A + (c0 + t0) * lda + dlr,
                                lda * 8, (size_t)jb * 8, nt2,
                                cudaMemcpyDeviceToDevice, sP));
                        }
                        flag_set_kernel<<<1, 1, 0, sP>>>(
                            d_flags + 2, (long long)k * 4 + cnext + 1);
                    } else {
                        flag_wait_kernel<<<1, 1, 0, sP>>>(
                            peer[upartner].flags + 2, nullptr, nullptr,
                            (long long)k * 4 + cnext + 1);
                        CHECK_CUDA(cudaMemcpyAsync(
                            d_U[b] + (size_t)(c0 - tc0) * jb,
                            peer[upartner].U[b] + (size_t)(c0 - tc0) * jb,
                            (size_t)jb * nc * 8, cudaMemcpyDeviceToDevice,
                            sP));
                    }
                    /* per tile: M = inv(U11)*U12 then slice immediately (no
                     * persistent M buffer); s0 consumes the int8 slices */
                    if (oz_on && nc >= OZ_MINN) {
                        for (long long t0 = 0; t0 < nc; t0 += XT) {
                            long long nt2 = std::min(XT, nc - t0);
                            CHECK_CUBLAS(cublasDgemm(
                                hbP, CUBLAS_OP_N, CUBLAS_OP_N, jb, (int)nt2,
                                jb, &done_,
                                d_pkg[b] + (size_t)2 * jb * jb, jb,
                                d_U[b] + (size_t)(c0 + t0 - tc0) * jb, jb,
                                &dzero, d_Mt, jb));
                            ozcolscale_kernel<<<(unsigned)((nt2 + 255) / 256),
                                                256, 0, sP>>>(
                                d_Mt, (size_t)jb, jb, (int)nt2,
                                ozCinv + (c0 + t0 - tc0),
                                ozCsc + (c0 + t0 - tc0));
                            dim3 ggB(8, (unsigned)nt2);
                            ozsliceB_kernel<<<ggB, 256, 0, sP>>>(
                                d_Mt, jb, (int)nt2,
                                ozCinv + (c0 + t0 - tc0),
                                ozSB + (size_t)(c0 + t0 - tc0) * OZ_S * jb,
                                OZ_S, (size_t)OZ_S * jb);
                            if (cnext == 0 && mlb > 0) {
                                if (wce) {
                                    CHECK_CUBLAS(cublasDgemm(
                                        hbP, CUBLAS_OP_N, CUBLAS_OP_N,
                                        (int)mlb, (int)nt2, jb, &dmone,
                                        d_W[b], (int)mlb, d_Mt, jb, &done_,
                                        d_A + (c0 + t0) * lda + blr0,
                                        (int)lda));
                                } else {
                                    /* K-slab per W chunk (accumulating gemm
                                     * splits over K); wait each chunk */
                                    const int jcw = (jb + WCH - 1) / WCH;
                                    for (int cw = 0; cw < WCH; cw++) {
                                        const int j0 = cw * jcw;
                                        const int jn =
                                            std::min(jcw, jb - j0);
                                        if (jn <= 0) break;
                                        CHECK_CUDA(cudaStreamWaitEvent(
                                            sP, evWc[k][cw], 0));
                                        CHECK_CUBLAS(cublasDgemm(
                                            hbP, CUBLAS_OP_N, CUBLAS_OP_N,
                                            (int)mlb, (int)nt2, jn, &dmone,
                                            d_W[b] + (size_t)j0 * mlb,
                                            (int)mlb, d_Mt + j0, jb, &done_,
                                            d_A + (c0 + t0) * lda + blr0,
                                            (int)lda));
                                    }
                                }
                            }
                        }
                    } else if (cnext == 0) {
                        /* panel chunk: M for the sP-side panel update */
                        CHECK_CUBLAS(cublasDgemm(
                            hbP, CUBLAS_OP_N, CUBLAS_OP_N, jb, (int)nc, jb,
                            &done_, d_pkg[b] + (size_t)2 * jb * jb, jb,
                            d_U[b] + (size_t)(c0 - tc0) * jb, jb, &dzero,
                            d_Mfb + (size_t)b * NB * NB, jb));
                        if (mlb > 0) {
                            if (wce) {
                                CHECK_CUBLAS(cublasDgemm(
                                    hbP, CUBLAS_OP_N, CUBLAS_OP_N, (int)mlb,
                                    (int)nc, jb, &dmone, d_W[b], (int)mlb,
                                    d_Mfb + (size_t)b * NB * NB, jb, &done_,
                                    d_A + c0 * lda + blr0, (int)lda));
                            } else {
                                const int jcw = (jb + WCH - 1) / WCH;
                                for (int cw = 0; cw < WCH; cw++) {
                                    const int j0 = cw * jcw;
                                    const int jn = std::min(jcw, jb - j0);
                                    if (jn <= 0) break;
                                    CHECK_CUDA(cudaStreamWaitEvent(
                                        sP, evWc[k][cw], 0));
                                    CHECK_CUBLAS(cublasDgemm(
                                        hbP, CUBLAS_OP_N, CUBLAS_OP_N,
                                        (int)mlb, (int)nc, jn, &dmone,
                                        d_W[b] + (size_t)j0 * mlb, (int)mlb,
                                        d_Mfb + (size_t)b * NB * NB + j0, jb,
                                        &done_, d_A + c0 * lda + blr0,
                                        (int)lda));
                                }
                            }
                        }
                    }
                }
                flag_set_kernel<<<1, 1, 0, sP>>>(
                    d_flags + 3, (long long)k * 4 + cnext + 1);
                if (cnext == 0)
                    CHECK_CUDA(cudaEventRecord(evUpdC[k][0], sP));
                CHECK_CUDA(cudaEventRecord(evReadyU[k][cnext], sP));
                cnext++;
            }
        };

        /* left (already-factored) columns are never read again (U-only
         * back-substitution; later exchanges only touch rows >= gr0(k')),
         * so they are not exchanged at all */
        /* urgent group of the previous update (contains this panel's cols) */
        if (k > 0) CHECK_CUDA(cudaStreamWaitEvent(sP, evUpdC[k - 1][0], 0));
        exch_range(prng[0][0], prng[0][1]);

        rec2(tSP, k, 5, sP);
        /* W = exchanged A21 (L21 folded into the update via M) */
        /* s0 must be done with the k-2 content of W/M (same buffer parity;
         * the sP-side evUpdC[k-1][0] no longer implies s0 progress) */
        if (k >= 2) CHECK_CUDA(cudaStreamWaitEvent(sP, evUpdC[k - 2][3], 0));
        if (wce) {
            /* wait until row peers consumed the k-2 W of this parity */
            if (k >= 2)
                for (int t = 0; t < Q - 1; t += 3)
                    flag_wait_kernel<<<1, 1, 0, sP>>>(
                        peer[rowpeer[t]].flags + 1,
                        t + 1 < Q - 1 ? peer[rowpeer[t + 1]].flags + 1
                                      : nullptr,
                        t + 2 < Q - 1 ? peer[rowpeer[t + 2]].flags + 1
                                      : nullptr,
                        (long long)k - 1);
            if (qc == kq) {
                if (mlb > 0)
                    CHECK_CUDA(cudaMemcpy2DAsync(
                        d_W[b], mlb * 8, d_A + pc0 * lda + blr0, lda * 8,
                        mlb * 8, jb, cudaMemcpyDeviceToDevice, sP));
                flag_set_kernel<<<1, 1, 0, sP>>>(d_flags + 0,
                                                 (long long)k + 1);
            } else {
                const int wroot = pr + 2 * kq;
                flag_wait_kernel<<<1, 1, 0, sP>>>(peer[wroot].flags + 0,
                                                  nullptr, nullptr,
                                                  (long long)k + 1);
                if (mlb > 0)
                    CHECK_CUDA(cudaMemcpyAsync(d_W[b], peer[wroot].W[b],
                                               (size_t)mlb * jb * 8,
                                               cudaMemcpyDeviceToDevice,
                                               sP));
            }
            flag_set_kernel<<<1, 1, 0, sP>>>(d_flags + 1, (long long)k + 1);
            CHECK_CUDA(cudaEventRecord(evReadyW[k], sP));
        } else {
            /* multi-node: W broadcast over the nrow communicator (NCCL) on
             * its own stream, pipelined in column chunks; the urgent-chunk
             * gemms consume K-slabs per chunk so the critical path only pays
             * the first chunk's transfer latency.  Host throttling at the
             * end of each iteration keeps the multi-comm queue shallow. */
            CHECK_CUDA(cudaEventRecord(evXg0[k], sP));
            CHECK_CUDA(cudaStreamWaitEvent(sW, evXg0[k], 0));
            if (k >= 2)
                CHECK_CUDA(cudaStreamWaitEvent(sW, evUpdC[k - 2][3], 0));
            const int jcw = (jb + WCH - 1) / WCH;
            for (int cw = 0; cw < WCH; cw++) {
                const int j0 = cw * jcw;
                const int jn = std::min(jcw, jb - j0);
                if (jn <= 0) break;
                if (qc == kq && mlb > 0)
                    CHECK_CUDA(cudaMemcpy2DAsync(
                        d_W[b] + (size_t)j0 * mlb, mlb * 8,
                        d_A + (pc0 + j0) * lda + blr0, lda * 8, mlb * 8, jn,
                        cudaMemcpyDeviceToDevice, sW));
                if (mlb > 0)
                    CHECK_NCCL(ncclBroadcast(d_W[b] + (size_t)j0 * mlb,
                                             d_W[b] + (size_t)j0 * mlb,
                                             (size_t)mlb * jn, ncclDouble,
                                             kq, nrow, sW));
                CHECK_CUDA(cudaEventRecord(evWc[k][cw], sW));
            }
            CHECK_CUDA(cudaEventRecord(evReadyW[k], sW));
        }
        rec2(tSP, k, 6, sP);
        /* partner must have consumed the k-2 slices of this U buffer */
        if (k >= 2)
            flag_wait_kernel<<<1, 1, 0, sP>>>(peer[upartner].flags + 3,
                                              nullptr, nullptr,
                                              (long long)(k - 2) * 4 + 4);
        /* inv(L11) + inv(U11) pieces for U12 staging and the local M GEMMs */
        CHECK_NCCL(ncclBroadcast(d_pkg[b] + (size_t)jb * jb,
                                 d_pkg[b] + (size_t)jb * jb,
                                 (size_t)2 * jb * jb, ncclDouble, 2 * kq,
                                 nworld, sP));
        emit_ready(prng[0][0] + prng[0][1]);

        /* remaining exchange groups, each followed immediately by the U12
         * parts whose columns are now fully exchanged */
        for (int g = 1; g < 4; g++) {
            if (k > 0) CHECK_CUDA(cudaStreamWaitEvent(sP, evUpdC[k - 1][g], 0));
            exch_range(prng[g][0], prng[g][1]);
            emit_ready(prng[g][0] + prng[g][1]);
            /* after g==1 the urgent chunk (panel k+1's cols) has been
             * exchanged AND its update emitted on sP: exact-data early
             * selection for the next panel overlaps g2/g3 + staging */
            if (g == 1) {
                preselect(k + 1);
                static const int psync =
                    getenv("CHPL2D_PSYNC") ? atoi(getenv("CHPL2D_PSYNC")) : 0;
                if (presel && psync == 1)
                    CHECK_CUDA(cudaDeviceSynchronize());
                if (presel && psync == 2 && k + 1 < npan)
                    CHECK_CUDA(cudaStreamWaitEvent(sP, evSel[k + 1], 0));
                /* latent-race probe: pure sP delay at this spot, no presel */
                if (psync == 5)
                    spin_kernel<<<1, 1, 0, sP>>>(5LL * 1000 * 1980);
            }
            if (g < 3) rec2(tSP, k, 6 + g, sP);
        }
        if (timing) CHECK_CUDA(cudaEventRecord(tX1[k], sP));
        emit_ready(nloc);
        /* factored diagonal block: ship to the owner column pair + write
         * back (must land before chainB(k+1) exchanges touch these rows) */
        if (qc == kq)
            CHECK_NCCL(ncclBroadcast(d_pkg[b], d_pkg[b], (size_t)jb * jb,
                                     ncclDouble, 0, ncol, sP));
        if (pr == kp && qc == kq)
            CHECK_CUDA(cudaMemcpy2DAsync(d_A + pc0 * lda +
                                             lrows_below(gr0, NB, pr, N),
                                         lda * 8, d_pkg[b], (size_t)jb * 8,
                                         (size_t)jb * 8, jb,
                                         cudaMemcpyDeviceToDevice, sP));
        if (timing) CHECK_CUDA(cudaEventRecord(tB1[k], sP));
        rec2(tSP, k, 9, sP);
    };

    /* ---- prologue: chain for panel 0 ---- */
    chainA(0);
    chainB(0);

    /* ---- main loop with depth-1 lookahead + chunked pipeline ---- */
    const int prog = getenv("CHPL2D_PROG") ? atoi(getenv("CHPL2D_PROG")) : 0;
    for (int k = 0; k < npan; k++) {
        const long long gr0 = (long long)k * NB;
        const int jb = (int)std::min((long long)NB, (long long)N - gr0);
        const long long tc0 = lcols_below(gr0 + jb, NB, qc, ntot);
        const long long blr0 = lrows_below(gr0 + jb, NB, pr, N);
        const long long mlb = mloc - blr0;
        const int b = k & 1;
        if (prog) { fprintf(stderr, "[r%d] enter k=%d\n", g_rank, k); fflush(stderr); }
        long long rng[4][2];
        chunks_of(k, rng);

        if (timing >= 2) CHECK_CUDA(cudaEventRecord(tS0[k][0], s0));
        CHECK_CUDA(cudaStreamWaitEvent(s0, evReadyW[k], 0));
        if (timing) CHECK_CUDA(cudaEventRecord(tU0[k], s0));
        const int use_oz = oz_on && mlb > 0;
        const int use_g8 = g8_on && mlb > 0;
        bool g8_first = true; /* first g8 call of a panel quantizes W;
                                 later tiles reuse it via skip_scalA */
        if (use_oz) {
            ozrowscale_kernel<<<(unsigned)((mlb + 255) / 256), 256, 0, s0>>>(
                d_W[b], (size_t)mlb, (int)mlb, jb, ozRinv, ozRsc);
            dim3 blk(32, 8), grd((unsigned)((mlb + 31) / 32),
                                 (unsigned)((jb + 31) / 32));
            ozsliceA_kernel<<<grd, blk, 0, s0>>>(d_W[b], (size_t)mlb,
                                                 (int)mlb, jb, ozRinv, ozSA,
                                                 OZ_S);
        }
        for (int c = 1; c < 4; c++) {
            CHECK_CUDA(cudaStreamWaitEvent(s0, evReadyU[k][c], 0));
            if (timing >= 2) CHECK_CUDA(cudaEventRecord(tS0[k][1 + 2 * c], s0));
            long long c0 = rng[c][0], nc = rng[c][1];
            if (mlb > 0 && nc > 0) {
                if (use_oz && nc >= OZ_MINN) {
                    /* B slices were produced tile-by-tile on sP with M */
                    oz_gemm(mlb, nc, jb, d_A + c0 * lda + blr0, c0 - tc0, s0);
                } else if (use_g8 && nc >= OZ_MINN) {
                    /* GEMMul8 path: M per tile on s0 (fp64, like fallback),
                     * then the CRT-emulated C -= W*M with W quantized only
                     * on the panel's first call */
                    for (long long t0 = 0; t0 < nc;) {
                        long long nt2 = std::min(G8_NT, nc - t0);
                        if (nc - t0 - nt2 > 0 && nc - t0 - nt2 < 1024)
                            nt2 = nc - t0; /* absorb small column tail */
                        CHECK_CUBLAS(cublasDgemm(
                            hb, CUBLAS_OP_N, CUBLAS_OP_N, jb, (int)nt2, jb,
                            &done_, d_pkg[b] + (size_t)2 * jb * jb, jb,
                            d_U[b] + (size_t)(c0 + t0 - tc0) * jb, jb,
                            &dzero, d_Mg8, jb));
                        gemmul8::gemm<double, gemmul8::Backend::INT8>(
                            hbG8, CUBLAS_OP_N, CUBLAS_OP_N, (size_t)mlb,
                            (size_t)nt2, (size_t)jb, &dmone, d_W[b],
                            (size_t)mlb, d_Mg8, (size_t)jb, &done_,
                            d_A + (c0 + t0) * lda + blr0, (size_t)lda,
                            G8_MODULI, G8_FAST != 0, g8_work, nullptr,
                            nullptr, /*enable_skip_scalA=*/true,
                            /*enable_skip_scalB=*/false,
                            /*skip_scalA=*/!g8_first,
                            /*skip_scalB=*/false);
                        g8_first = false;
                        t0 += nt2;
                    }
                } else {
                    /* fp64 fallback: M computed here on s0 (self-consistent,
                     * no cross-stream buffer aliasing), tiled by XT */
                    for (long long t0 = 0; t0 < nc; t0 += XT) {
                        long long nt2 = std::min(XT, nc - t0);
                        CHECK_CUBLAS(cublasDgemm(
                            hb, CUBLAS_OP_N, CUBLAS_OP_N, jb, (int)nt2, jb,
                            &done_, d_pkg[b] + (size_t)2 * jb * jb, jb,
                            d_U[b] + (size_t)(c0 + t0 - tc0) * jb, jb,
                            &dzero, d_Ms, jb));
                        CHECK_CUBLAS(cublasDgemm(
                            hb, CUBLAS_OP_N, CUBLAS_OP_N, (int)mlb, (int)nt2,
                            jb, &dmone, d_W[b], (int)mlb, d_Ms, jb, &done_,
                            d_A + (c0 + t0) * lda + blr0, (int)lda));
                    }
                }
            }
            CHECK_CUDA(cudaEventRecord(evUpdC[k][c], s0));
            if (timing >= 2) CHECK_CUDA(cudaEventRecord(tS0[k][2 + 2 * c], s0));
        }
        if (timing >= 2) { CHECK_CUDA(cudaEventRecord(tS0[k][1], s0));
                           CHECK_CUDA(cudaEventRecord(tS0[k][2], s0)); }
        if (timing) CHECK_CUDA(cudaEventRecord(tU1[k], s0));
        CHECK_CUDA(cudaEventRecord(evUpd[k], s0));
        /* multi-node: keep the multi-comm NCCL queue shallow (deep enqueue
         * across nrow/nworld/ncol deadlocks); free — iteration k+1's device
         * work is gated on W(k) via evUpdC[k][0] anyway */
        if (!wce) CHECK_CUDA(cudaEventSynchronize(evReadyW[k]));
        if (prog) {
            int spin = 0;
            while (cudaEventQuery(evReadyW[k]) == cudaErrorNotReady &&
                   ++spin < 100)
                usleep(100000);
            if (spin >= 100) {
                fprintf(stderr,
                        "[r%d k%d] STALL W:%d Xg0:%d U1:%d U2:%d U3:%d "
                        "UpdPrev:%d\n",
                        g_rank, k,
                        cudaEventQuery(evReadyW[k]) == cudaSuccess,
                        cudaEventQuery(evXg0[k]) == cudaSuccess,
                        cudaEventQuery(evReadyU[k][1]) == cudaSuccess,
                        cudaEventQuery(evReadyU[k][2]) == cudaSuccess,
                        cudaEventQuery(evReadyU[k][3]) == cudaSuccess,
                        k >= 2 ? (cudaEventQuery(evUpdC[k - 2][3]) ==
                                  cudaSuccess)
                               : -1);
                fflush(stderr);
            }
        }

        if (k + 1 < npan) {
            chainA(k + 1);
            chainB(k + 1);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double t_loop = MPI_Wtime();
    if (timing) {
        float ms; double sU_busy = 0, sB = 0, sX = 0, sGap = 0;
        for (int i = 0; i < npan; i++) {
            CHECK_CUDA(cudaEventElapsedTime(&ms, tU0[i], tU1[i])); sU_busy += ms;
            if (i > 0) { CHECK_CUDA(cudaEventElapsedTime(&ms, tU1[i-1], tU0[i])); sGap += ms; }
            CHECK_CUDA(cudaEventElapsedTime(&ms, tB0[i], tX1[i])); sX += ms;
            CHECK_CUDA(cudaEventElapsedTime(&ms, tB0[i], tB1[i])); sB += ms;
        }
        printf("[2dtiming] rank %d  s0_gemm=%.2fs s0_gap=%.2fs chainB=%.2fs (exch=%.2fs)\n",
               g_rank, sU_busy / 1e3, sGap / 1e3, sB / 1e3, sX / 1e3);
        fflush(stdout);
    }
    if (timing >= 2) {
        float ms;
        /* s0: waits vs gemms, split into halves of the run */
        for (int half = 0; half < 2; half++) {
            double wW = 0, w[4] = {0, 0, 0, 0}, gm[4] = {0, 0, 0, 0}, gap = 0;
            int k0 = half * npan / 2, k1 = (half + 1) * npan / 2;
            for (int i = k0; i < k1; i++) {
                CHECK_CUDA(cudaEventElapsedTime(&ms, tS0[i][0], tS0[i][1])); wW += ms;
                for (int c = 0; c < 4; c++) {
                    if (c > 0) { CHECK_CUDA(cudaEventElapsedTime(&ms, tS0[i][2 * c], tS0[i][1 + 2 * c])); w[c] += ms; }
                    CHECK_CUDA(cudaEventElapsedTime(&ms, tS0[i][1 + 2 * c], tS0[i][2 + 2 * c])); gm[c] += ms;
                }
                if (i > k0) { CHECK_CUDA(cudaEventElapsedTime(&ms, tS0[i - 1][8], tS0[i][0])); gap += ms; }
            }
            printf("[2dfine] r%d h%d s0: waitWU0=%.2f wU1=%.2f wU2=%.2f wU3=%.2f | g0=%.2f g1=%.2f g2=%.2f g3=%.2f | gap=%.2f\n",
                   g_rank, half, wW / 1e3, w[1] / 1e3, w[2] / 1e3, w[3] / 1e3,
                   gm[0] / 1e3, gm[1] / 1e3, gm[2] / 1e3, gm[3] / 1e3, gap / 1e3);
        }
        for (int half = 0; half < 2; half++) {
            double sg[9] = {0};
            int k0 = half * npan / 2, k1 = (half + 1) * npan / 2;
            for (int i = k0; i < k1; i++)
                for (int j = 0; j < 9; j++) {
                    CHECK_CUDA(cudaEventElapsedTime(&ms, tSP[i][j], tSP[i][j + 1]));
                    sg[j] += ms;
                }
            printf("[2dfine] r%d h%d sP: getrf1=%.2f candx=%.2f getrf2=%.2f wpbc=%.2f g0x=%.2f L21W=%.2f g1U01=%.2f g2U2=%.2f g3U3=%.2f\n",
                   g_rank, half, sg[0] / 1e3, sg[1] / 1e3, sg[2] / 1e3,
                   sg[3] / 1e3, sg[4] / 1e3, sg[5] / 1e3, sg[6] / 1e3,
                   sg[7] / 1e3, sg[8] / 1e3);
        }
        fflush(stdout);
    }

    /* --------------- back substitution: U x = y ------------------- */
    setzero_kernel<<<((int)mloc + 255) / 256, 256, 0, s0>>>(d_acc, mloc);
    setzero_kernel<<<(N + 255) / 256, 256, 0, s0>>>(d_x, N);
    {
        long long bblk = (long long)N / NB;
        if (qc == (int)(bblk % Q)) {
            long long bcl = lcols_below((long long)N, NB, qc, ntot);
            CHECK_CUDA(cudaMemcpyAsync(d_acc, d_A + bcl * lda, mloc * 8,
                                       cudaMemcpyDeviceToDevice, s0));
        }
    }
    /* CE-flag back-substitution: the per-panel NCCL reduce+bcast latency
     * chain (~0.9s over 170 panels) is replaced by copy-engine pulls with
     * monotone flag handshakes (flags[4]=iteration progress, [5]=x ready) */
    for (int j = npan - 1; j >= 0; j--) {
        long long gr0 = (long long)j * NB;
        int jb = (int)std::min((long long)NB, (long long)N - gr0);
        int jq = j % Q, jp = j % P;
        const long long stepv = (long long)(npan - j);
        if (!wce && pr == jp) {
            /* multi-node: acc reduction over nrow via NCCL; the root's
             * d_tmp must first be released by its ncol partner */
            if (qc == jq)
                flag_wait_kernel<<<1, 1, 0, s0>>>(
                    peer[(1 - jp) + P * jq].flags + 4, nullptr, nullptr,
                    stepv - 1);
            long long dlr = lrows_below(gr0, NB, jp, N);
            CHECK_NCCL(ncclReduce(d_acc + dlr, d_tmp, jb, ncclDouble,
                                  ncclSum, jq, nrow, s0));
        }
        if (pr == jp && qc == jq) {
            long long dlr = lrows_below(gr0, NB, jp, N);
            long long cl = lcols_below(gr0, NB, qc, ntot);
            if (wce) {
                /* nrow mates (and the ncol partner, for d_tmp reuse) must
                 * be past iteration j+1 */
                int rp[7]; { int t = 0;
                    for (int q2 = 0; q2 < Q; q2++)
                        if (q2 != jq) rp[t++] = jp + P * q2; }
                for (int t = 0; t < Q - 1; t += 3)
                    flag_wait_kernel<<<1, 1, 0, s0>>>(
                        peer[rp[t]].flags + 4,
                        t + 1 < Q - 1 ? peer[rp[t + 1]].flags + 4 : nullptr,
                        t + 2 < Q - 1 ? peer[rp[t + 2]].flags + 4 : nullptr,
                        stepv - 1);
                flag_wait_kernel<<<1, 1, 0, s0>>>(
                    peer[(1 - jp) + P * jq].flags + 4, nullptr, nullptr,
                    stepv - 1);
                for (int o = 0; o < Q - 1; o++)
                    CHECK_CUDA(cudaMemcpyAsync(d_tmp3 + (size_t)o * NB,
                                               peer[rp[o]].acc + dlr,
                                               (size_t)jb * 8,
                                               cudaMemcpyDeviceToDevice,
                                               s0));
                addacc_kernel<<<(jb + 255) / 256, 256, 0, s0>>>(
                    d_tmp, d_acc + dlr, d_tmp3, NB, jb, Q - 1);
            } else {
                flag_wait_kernel<<<1, 1, 0, s0>>>(
                    peer[(1 - jp) + P * jq].flags + 4, nullptr, nullptr,
                    stepv - 1);
            }
            CHECK_CUBLAS(cublasDtrsv(hb, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                     CUBLAS_DIAG_NON_UNIT, jb,
                                     d_A + cl * lda + dlr, (int)lda, d_tmp, 1));
            CHECK_CUDA(cudaMemcpyAsync(d_x + gr0, d_tmp, (size_t)jb * 8,
                                       cudaMemcpyDeviceToDevice, s0));
            flag_set_kernel<<<1, 1, 0, s0>>>(d_flags + 5, stepv);
        }
        if (qc == jq) {
            if (pr != jp) {
                const int owner = jp + P * jq;
                flag_wait_kernel<<<1, 1, 0, s0>>>(peer[owner].flags + 5,
                                                  nullptr, nullptr, stepv);
                CHECK_CUDA(cudaMemcpyAsync(d_tmp, peer[owner].tmp,
                                           (size_t)jb * 8,
                                           cudaMemcpyDeviceToDevice, s0));
            }
            long long cl = lcols_below(gr0, NB, qc, ntot);
            long long mabove = lrows_below(gr0, NB, pr, N);
            if (mabove > 0)
                CHECK_CUBLAS(cublasDgemv(hb, CUBLAS_OP_N, (int)mabove, jb,
                                         &dmone, d_A + cl * lda, (int)lda,
                                         d_tmp, 1, &done_, d_acc, 1));
        }
        flag_set_kernel<<<1, 1, 0, s0>>>(d_flags + 4, stepv);
    }
    CHECK_NCCL(ncclAllReduce(d_x, d_x, N, ncclDouble, ncclSum, nworld, s0));
    CHECK_CUDA(cudaStreamSynchronize(s0));
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    double t1 = MPI_Wtime();
    if (timing && !g_rank)
        printf("[epilogue] loop_end->t1 = %.3f s\n", t1 - t_loop);
    time_t ts1 = time(NULL);
    double gflops = (2.0 / 3.0 * (double)N * N * N + 1.5 * (double)N * N) /
                    (t1 - t0) / 1e9;

    /* --------------- verification -------------------------------- */
    double *d_ax = d_acc;
    setzero_kernel<<<((int)mloc + 255) / 256, 256, 0, s0>>>(d_ax, mloc);
    setzero_kernel<<<((int)mloc + 255) / 256, 256, 0, s0>>>(d_rs, mloc);
    /* NB-aligned chunks (one column block each -> contiguous global cols
     * for the x slice); d_W[0] (mloc x NB doubles, idle after the
     * factorization) is the regen buffer */
    for (long long c0 = 0; c0 < nloc; c0 += NB) {
        int nc = (int)std::min((long long)NB, nloc - c0);
        int ncm = nc;
        if (h_gcol[c0 + nc - 1] == N) ncm = nc - 1;
        if (ncm <= 0) continue;
        dim3 gg(64, ncm);
        matgen2d_kernel<<<gg, 256, 0, s0>>>(d_W[0], (size_t)mloc, (int)mloc,
                                            d_grow, d_gcol + c0, ncm, ntot,
                                            seed);
        CHECK_CUBLAS(cublasDgemv(hb, CUBLAS_OP_N, (int)mloc, ncm, &done_,
                                 d_W[0], (int)mloc, d_x + h_gcol[c0], 1,
                                 &done_, d_ax, 1));
        rowsum_kernel<<<((int)mloc + 255) / 256, 256, 0, s0>>>(
            d_W[0], (size_t)mloc, (int)mloc, ncm, d_rs);
    }
    {
        long long bc = N; long long *d_bc;
        CHECK_CUDA(cudaMalloc(&d_bc, 8));
        CHECK_CUDA(cudaMemcpy(d_bc, &bc, 8, cudaMemcpyHostToDevice));
        dim3 gg(64, 1);
        matgen2d_kernel<<<gg, 256, 0, s0>>>(d_b, (size_t)mloc, (int)mloc,
                                            d_grow, d_bc, 1, ntot, seed);
        CHECK_CUDA(cudaStreamSynchronize(s0));
        CHECK_CUDA(cudaFree(d_bc));
    }
    CHECK_NCCL(ncclAllReduce(d_ax, d_ax, mloc, ncclDouble, ncclSum, nrow, s0));
    CHECK_NCCL(ncclAllReduce(d_rs, d_rs, mloc, ncclDouble, ncclSum, nrow, s0));
    CHECK_CUDA(cudaStreamSynchronize(s0));
    axpby_res_kernel<<<((int)mloc + 255) / 256, 256, 0, s0>>>(d_b, d_ax, d_r,
                                                              (int)mloc);
    CHECK_CUDA(cudaStreamSynchronize(s0));

    double nr = dev_absmax(hb, d_r, mloc);
    double nbv = dev_absmax(hb, d_b, mloc);
    double na = dev_absmax(hb, d_rs, mloc);
    double nx = dev_absmax(hb, d_x, N);
    double loc[3] = {nr, nbv, na}, glob[3];
    CHECK_MPI(MPI_Allreduce(loc, glob, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    double eps = DBL_EPSILON / 2.0;
    double resid = glob[0] / (eps * (glob[2] * nx + glob[1]) * (double)N);

    if (!g_rank) {
        char b1[64], b2[64]; struct tm tmv;
        localtime_r(&ts0, &tmv); strftime(b1, 64, "%a %b %e %H:%M:%S %Y", &tmv);
        localtime_r(&ts1, &tmv); strftime(b2, 64, "%a %b %e %H:%M:%S %Y", &tmv);
        printf("================================================================================\n");
        printf("T/V                N    NB     P     Q         Time          Gflops (   per GPU)\n");
        printf("--------------------------------------------------------------------------------\n");
        printf("%-4s%16d %5d %5d %5d %12.2f %15.3e (%10.3e)\n", "WC2", N, NB,
               P, Q, t1 - t0, gflops, gflops / (P * Q));
        printf("\nHPL_pdgesv() start time %s\nHPL_pdgesv() end time   %s\n\n",
               b1, b2);
        printf("--------------------------------------------------------------------------------\n");
        printf("||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)= %16.12f ...... %s\n",
               resid, resid < thresh ? "PASSED" : "FAILED");
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
