/*
 * hpl_single.cu — single-process, single-GPU HPL solver, NO MPI/NCCL/IPC.
 *
 * Purpose: measure OUR Ozaki-I (base-128 digit slicing, fused-diagonal
 * cublasLt int8 GEMM) trailing-update scheme's TRUE single-GPU ceiling,
 * free of every distributed cost our production chpl2d.cu/chpl4x4.cu
 * carries (tournament pivoting across ranks, CE/IPC pulls, NCCL
 * broadcasts, preselect). This is the fair apples-to-apples counterpart
 * to a teammate's single-GPU hpl_ga.cu (which uses GEMMUL8/Ozaki-II).
 *
 * Structure deliberately mirrors hpl_ga.cu's A3-agg two-level blocking
 * (exact partial-pivoting LU, NB panel width, aggregated trailing update
 * at K=AGG*NB) and its exact residual/timing/report format, so the two
 * numbers are directly comparable. The trailing-update BACKEND is what
 * differs: OUR Ozaki-I fused-diagonal int8 scheme instead of GEMMUL8's
 * Ozaki-II CRT scheme. Panel factorization, pivoting, row-swaps, and the
 * A3-agg block structure are new clean-room code (same well-known
 * algorithm, no distributed machinery — single rank never needs
 * tournament pivoting, cusolverDnXgetrf gives exact pivots directly).
 *
 * Kernels ozrowscale/ozcolscale/ozsliceA/ozsliceB/ozrecon are copied
 * verbatim from src/chpl4x4.cu (they are already rank-independent,
 * single-GPU-only logic — no MPI/IPC touches them there either).
 *
 * A is column-major on GPU, N x (N+1), ld = N; b is stored as column N
 * (classic HPL trick: b rides through swaps/solves/updates automatically).
 *
 * CLI:  hpl_single N            one run, configured via HPLS_* env vars
 *       hpl_single --selftest   two small runs (native, ozaki); prints
 *                                SELFTEST_PASS / SELFTEST_FAIL
 * Env:  HPLS_NB=2048 HPLS_AGG=6 HPLS_NATIVE_TAIL=8192 HPLS_OZ_NT=1024
 *       HPLS_OZ_ALGO=3 HPLS_SEED=42 HPLS_BACKEND=native|ozaki
 * Exit: 0 pass, 1 fail/error, 4 memory planner abort.
 */

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cusolverDn.h>

#include <gemmul8.hpp> /* RIKEN Ozaki-II CRT emulation; minimal lib built by
                          scripts/build_gemmul8_min.sh (double/INT8/gemm only) */

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e_ = (call);                                          \
        if (e_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "[hpl_single] CUDA error %d (%s) at "    \
                         "%s:%d\n", (int)e_, cudaGetErrorString(e_),      \
                         __FILE__, __LINE__);                             \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)
#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t e_ = (call);                                       \
        if (e_ != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "[hpl_single] cuBLAS error %d at "       \
                         "%s:%d\n", (int)e_, __FILE__, __LINE__);         \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)
#define CUSOLVER_CHECK(call)                                               \
    do {                                                                   \
        cusolverStatus_t e_ = (call);                                     \
        if (e_ != CUSOLVER_STATUS_SUCCESS) {                              \
            std::fprintf(stderr, "[hpl_single] cusolver error %d at "     \
                         "%s:%d\n", (int)e_, __FILE__, __LINE__);         \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

/* ------------------------------------------------------------------ */
/* Matgen / residual kernels (same convention as hpl_ga.cu)            */
/* ------------------------------------------------------------------ */
__device__ __forceinline__ double hs_rand(uint64_t seed, uint64_t gi,
                                          uint64_t gj, uint64_t keyN)
{
    curandStatePhilox4_32_10_t st;
    curand_init((unsigned long long)seed,
                (unsigned long long)(gi * (keyN + 1) + gj), 0ULL, &st);
    return curand_uniform_double(&st) - 0.5;
}

__global__ void matgen_kernel(double *A, int64_t nrows, int64_t col0,
                              int64_t ncols, int64_t ld, uint64_t seed,
                              uint64_t keyN)
{
    const int64_t total = nrows * ncols;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int64_t i = idx % nrows;
        const int64_t j = col0 + idx / nrows;
        A[i + j * ld] = hs_rand(seed, (uint64_t)i, (uint64_t)j, keyN);
    }
}

__global__ void row_abs_sum_kernel(const double *A, int64_t nrows, int64_t n,
                                   int64_t ld, double *out)
{
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < nrows; i += stride) {
        double s = 0.0;
        for (int64_t j = 0; j < n; ++j) s += fabs(A[i + j * ld]);
        out[i] = s;
    }
}

__global__ void row_gather_kernel(double *__restrict__ buf,
                                  const double *__restrict__ A,
                                  const int *__restrict__ src, int64_t np,
                                  int64_t ld, int64_t col0, int64_t nc)
{
    const int64_t total = np * nc;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int64_t t = idx % np, c = idx / np;
        buf[idx] = A[(int64_t)src[t] + (col0 + c) * ld];
    }
}

__global__ void row_scatter_kernel(double *__restrict__ A,
                                   const double *__restrict__ buf,
                                   const int *__restrict__ dst, int64_t np,
                                   int64_t ld, int64_t col0, int64_t nc)
{
    const int64_t total = np * nc;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int64_t t = idx % np, c = idx / np;
        A[(int64_t)dst[t] + (col0 + c) * ld] = buf[idx];
    }
}

static inline int grid1d(int64_t work, int block)
{
    int64_t g = (work + block - 1) / block;
    if (g > 65535 * 16) g = 65535 * 16;
    if (g < 1) g = 1;
    return (int)g;
}

/* ------------------------------------------------------------------ */
/* Ozaki-I kernels (verbatim from src/chpl4x4.cu; already rank-free)   */
/* ------------------------------------------------------------------ */
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
    size_t ldS = (size_t)k * s;
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

__global__ void ozsliceB_kernel(const double *B, size_t ldb, int k, int n,
                                const double *cinv, signed char *S, int s,
                                size_t ldS)
{
    int j = blockIdx.y;
    if (j >= n) return;
    double ci = cinv[j];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k;
         i += gridDim.x * blockDim.x) {
        double e = B[(size_t)j * ldb + i] * ci;
        signed char *dst = S + (size_t)j * ldS + i;
        for (int p = 0; p < s; p++) {
            double a = rint(e * 128.0);
            if (a > 127.0) a = 127.0;
            e = e * 128.0 - a;
            dst[(size_t)(s - 1 - p) * k] = (signed char)a;
        }
    }
}

__global__ void ozrecon_kernel(double *C, size_t ldc, int m, int n,
                               const int *D0, const int *D1, const int *D2,
                               const int *D3, const int *D4, const int *D5,
                               int ndiag, const double *rsc,
                               const double *csc)
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

/* ------------------------------------------------------------------ */
/* Config                                                              */
/* ------------------------------------------------------------------ */
enum Backend { BK_NATIVE = 0, BK_OZAKI = 1, BK_GEMMUL8 = 2 };
static const int OZ_S = 6, OZ_D = 5; /* champion defaults: 47.7-bit */

struct Cfg {
    int64_t N = 0, NB = 2048, AGG = 6, K = 12288;
    Backend backend = BK_NATIVE;
    int64_t oz_nt = 1024;
    int oz_algo = 3;
    int64_t native_tail = 8192;
    /* gemmul8 backend knobs (teammate-matched defaults) */
    int g8_moduli = 14;
    int g8_fast = 0;
    int64_t g8_chunk = 16384;
    int g8_skipA = 1;
    uint64_t seed = 42;
    const char *tag() const
    {
        return backend == BK_NATIVE  ? "native"
               : backend == BK_OZAKI ? "ozaki"
                                     : "gemmul8";
    }
};

static int64_t env_i64(const char *name, int64_t dflt)
{
    const char *v = std::getenv(name);
    return v && *v ? std::atoll(v) : dflt;
}

/* ------------------------------------------------------------------ */
/* Phase clock (same design as hpl_ga.cu)                              */
/* ------------------------------------------------------------------ */
enum Phase { P_FACT = 0, P_INNER, P_LASWP, P_WTRSM, P_WGEMM, P_SOLVE, P_NPHASE };
static const char *phase_name[P_NPHASE] = {"panel_fact", "inner_update",
                                           "laswp", "wide_trsm", "wide_gemm",
                                           "back_solve"};
struct PhaseClock {
    struct Rec { cudaEvent_t a, b; int ph; };
    std::vector<Rec> recs;
    int cur = -1;
    cudaEvent_t pend{};
    static cudaEvent_t mk() { cudaEvent_t e; CUDA_CHECK(cudaEventCreate(&e)); return e; }
    void tic(int ph) { cur = ph; pend = mk(); CUDA_CHECK(cudaEventRecord(pend, 0)); }
    void toc() { cudaEvent_t e = mk(); CUDA_CHECK(cudaEventRecord(e, 0));
                 recs.push_back({pend, e, cur}); cur = -1; }
    void sums(double out[P_NPHASE]) {
        for (int p = 0; p < P_NPHASE; ++p) out[p] = 0.0;
        for (const Rec &r : recs) { float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, r.a, r.b)); out[r.ph] += ms / 1e3; }
    }
    void destroy() { for (Rec &r : recs) { CUDA_CHECK(cudaEventDestroy(r.a));
                     CUDA_CHECK(cudaEventDestroy(r.b)); } recs.clear(); }
};

/* ------------------------------------------------------------------ */
/* Solver                                                              */
/* ------------------------------------------------------------------ */
struct Solver {
    Cfg cfg;
    int64_t N = 0, ld = 0, ncols = 0;

    cublasHandle_t hb{};
    cusolverDnHandle_t hs{};
    cusolverDnParams_t hp{};
    cublasLtHandle_t ozLt{};
    cublasLtMatmulPreference_t ozPref{};

    double *d_A = nullptr;
    int64_t *d_ipiv = nullptr;
    int *d_info = nullptr;
    void *d_fwork = nullptr;
    void *h_fwork = nullptr;
    size_t fwork_d = 0, fwork_h = 0;
    double *d_swapbuf = nullptr;
    size_t swap_elems = 0;
    int *d_src = nullptr, *d_dst = nullptr;

    /* Ozaki-I workspace (sized for the FIRST/largest block: mt<=N, K) */
    signed char *ozSA = nullptr, *ozSB = nullptr;
    int *ozDg[2][6] = {};
    double *ozRinv = nullptr, *ozRsc = nullptr, *ozCinv = nullptr, *ozCsc = nullptr;
    void *ozWs = nullptr;
    size_t ozWsz = 256u << 20;
    cudaStream_t sOz = nullptr;
    cudaEvent_t evOzGemm[2], evOzRecon[2];

    /* gemmul8 (Ozaki-II) workspace */
    void *d_g8work = nullptr;
    double tg8[4] = {0, 0, 0, 0}; /* internal s: scale/int8-gemm/mod/CRT+unscale */
    int64_t n_g8 = 0;

    std::vector<int64_t> ipiv_g;
    std::vector<int64_t> h_ipiv;
    std::vector<int> rowsrc, srcs, dsts;

    PhaseClock clk;
    int64_t n_oz = 0, n_native = 0, max_pairs = 0;

    double *Aat(int64_t i, int64_t j) { return d_A + i + j * ld; }

    void plan_and_alloc()
    {
        N = cfg.N; ld = N; ncols = N + 1;
        size_t freeB = 0, totB = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeB, &totB));

        const size_t bytes_A = (size_t)ld * ncols * sizeof(double);
        swap_elems = std::max<size_t>((size_t)2 * cfg.K * 1024,
                                      (size_t)4 * 1024 * 1024);
        const size_t bytes_swap = swap_elems * sizeof(double);
        const int64_t mt_max = N - cfg.NB; /* first block's trailing rows */
        size_t bytes_oz = 0;
        if (cfg.backend == BK_OZAKI && mt_max > 0) {
            const size_t ldS = (size_t)OZ_S * cfg.K;
            bytes_oz = (size_t)OZ_S * mt_max * cfg.K /* ozSA */
                     + ldS * (size_t)cfg.oz_nt         /* ozSB: one n-tile only */
                     + (size_t)12 * mt_max * cfg.oz_nt * 4 /* ozDg x12 */
                     + (size_t)2 * (mt_max + ncols) * 8;   /* scale arrays */
        }
        size_t bytes_g8 = 0;
        const int64_t m_wide_max = N - cfg.K; /* largest emulated m_trail */
        if (cfg.backend == BK_GEMMUL8 && m_wide_max > 0) {
            /* Sized for the largest call: m = N-K, n = chunk+1024 (small
             * column tails are absorbed into the previous chunk in
             * update_trailing, so no call exceeds this), k = K, with the
             * skip_scalA slice reserved when A-reuse is enabled. */
            bytes_g8 = gemmul8::workSize<false, gemmul8::Backend::INT8>(
                (size_t)m_wide_max, (size_t)(cfg.g8_chunk + 1024),
                (size_t)cfg.K, cfg.g8_moduli, cfg.g8_skipA != 0, false);
        }
        const size_t bytes_misc = (size_t)(2 * N + ncols) * sizeof(double) +
            (size_t)cfg.NB * sizeof(int64_t) + (size_t)4 * 2 * cfg.K * sizeof(int);
        const size_t getrf_allow = (size_t)1 << 30;
        const size_t reserve = (size_t)512 << 20;
        const size_t need = bytes_A + bytes_oz + bytes_g8 + bytes_swap +
                            bytes_misc + getrf_allow + reserve;

        const double gib = 1073741824.0;
        std::printf(
            "MEM PLAN: matrix %.2f GiB | oz work %.2f GiB | g8 work %.2f GiB"
            " | laswp buf %.2f GiB | misc %.2f GiB | getrf allowance %.2f GiB"
            " | reserve %.2f GiB\n"
            "MEM PLAN: total %.2f GiB vs free %.2f GiB (total dev %.2f GiB)"
            " -> %s\n",
            bytes_A / gib, bytes_oz / gib, bytes_g8 / gib, bytes_swap / gib,
            bytes_misc / gib, getrf_allow / gib, reserve / gib, need / gib,
            freeB / gib, totB / gib, need <= freeB ? "OK" : "OVER BUDGET");
        std::fflush(stdout);
        if (need > freeB) {
            std::fprintf(stderr, "[hpl_single] memory plan exceeds free HBM"
                         " by %.2f GiB: reduce N or K (=AGG*NB)\n",
                         (need - freeB) / gib);
            std::exit(4);
        }

        CUBLAS_CHECK(cublasCreate(&hb));
        CUSOLVER_CHECK(cusolverDnCreate(&hs));
        CUSOLVER_CHECK(cusolverDnCreateParams(&hp));
        {
            void *w = nullptr;
            if (cudaMalloc(&w, 256u << 20) == cudaSuccess)
                CUBLAS_CHECK(cublasSetWorkspace(hb, w, 256u << 20));
        }

        CUDA_CHECK(cudaMalloc((void **)&d_A, bytes_A));
        CUDA_CHECK(cudaMalloc((void **)&d_ipiv, (size_t)cfg.NB * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc((void **)&d_info, sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_swapbuf, bytes_swap));
        CUDA_CHECK(cudaMalloc((void **)&d_src, (size_t)2 * cfg.K * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_dst, (size_t)2 * cfg.K * sizeof(int)));

        if (cfg.backend == BK_OZAKI && mt_max > 0) {
            CUDA_CHECK(cudaMalloc(&ozSA, (size_t)OZ_S * mt_max * cfg.K));
            CUDA_CHECK(cudaMalloc(&ozSB, (size_t)OZ_S * cfg.K * cfg.oz_nt));
            for (int ss = 0; ss < 2; ss++)
                for (int d = 0; d < 6; d++)
                    CUDA_CHECK(cudaMalloc(&ozDg[ss][d],
                                          (size_t)mt_max * cfg.oz_nt * 4));
            CUDA_CHECK(cudaMalloc(&ozRinv, mt_max * 8));
            CUDA_CHECK(cudaMalloc(&ozRsc, mt_max * 8));
            CUDA_CHECK(cudaMalloc(&ozCinv, ncols * 8));
            CUDA_CHECK(cudaMalloc(&ozCsc, ncols * 8));
            /* Deliberately the NULL/default stream (sOz = 0), NOT a separate
             * non-blocking stream: getrf/trsm/laswp for L21 and U12 all run
             * on the default stream via hb/hs (never given an explicit
             * stream), and a cudaStreamNonBlocking stream has NO implicit
             * ordering against the default stream. A separate stream here
             * previously let oz_gemm read L21/U12 before those writes
             * landed (silent race -> ~1e10 residual). No pipelining is
             * attempted in this single-GPU driver, so there is no overlap
             * benefit lost by keeping everything on one stream. */
            sOz = 0;
            for (int ss = 0; ss < 2; ss++) {
                CUDA_CHECK(cudaEventCreateWithFlags(&evOzGemm[ss], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&evOzRecon[ss], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(evOzRecon[ss], 0));
            }
            CUBLAS_CHECK(cublasLtCreate(&ozLt));
            CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&ozPref));
            CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(ozPref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ozWsz, sizeof(ozWsz)));
            CUDA_CHECK(cudaMalloc(&ozWs, ozWsz));
        }

        if (cfg.backend == BK_GEMMUL8 && bytes_g8 > 0)
            CUDA_CHECK(cudaMalloc(&d_g8work, bytes_g8));

        CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(
            hs, hp, N, cfg.NB, CUDA_R_64F, d_A, ld, CUDA_R_64F, &fwork_d, &fwork_h));
        std::printf("MEM PLAN: getrf workspace actual dev %.3f GiB host %.3f GiB\n",
                    fwork_d / gib, fwork_h / gib);
        CUDA_CHECK(cudaMalloc(&d_fwork, fwork_d > 0 ? fwork_d : 8));
        h_fwork = std::malloc(fwork_h > 0 ? fwork_h : 8);
        if (!h_fwork) { std::fprintf(stderr, "[hpl_single] host workspace alloc failed\n"); std::exit(1); }

        ipiv_g.assign((size_t)N, 0);
        h_ipiv.resize((size_t)cfg.NB);
        rowsrc.reserve((size_t)N);
        srcs.reserve((size_t)2 * cfg.K);
        dsts.reserve((size_t)2 * cfg.K);
    }

    void destroy()
    {
        clk.destroy();
        std::free(h_fwork);
        CUDA_CHECK(cudaFree(d_fwork));
        if (d_g8work) CUDA_CHECK(cudaFree(d_g8work));
        if (ozWs) CUDA_CHECK(cudaFree(ozWs));
        if (ozSA) CUDA_CHECK(cudaFree(ozSA));
        if (ozSB) CUDA_CHECK(cudaFree(ozSB));
        for (int ss = 0; ss < 2; ss++)
            for (int d = 0; d < 6; d++) if (ozDg[ss][d]) CUDA_CHECK(cudaFree(ozDg[ss][d]));
        if (ozRinv) CUDA_CHECK(cudaFree(ozRinv));
        if (ozRsc) CUDA_CHECK(cudaFree(ozRsc));
        if (ozCinv) CUDA_CHECK(cudaFree(ozCinv));
        if (ozCsc) CUDA_CHECK(cudaFree(ozCsc));
        CUDA_CHECK(cudaFree(d_dst));
        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_swapbuf));
        CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFree(d_ipiv));
        CUDA_CHECK(cudaFree(d_A));
        CUSOLVER_CHECK(cusolverDnDestroyParams(hp));
        CUSOLVER_CHECK(cusolverDnDestroy(hs));
        CUBLAS_CHECK(cublasDestroy(hb));
    }

    void apply_swaps(int64_t base, int64_t ns, int64_t col0, int64_t nc)
    {
        if (ns <= 0 || nc <= 0) return;
        const int64_t mr = N - base;
        rowsrc.resize((size_t)mr);
        std::iota(rowsrc.begin(), rowsrc.end(), 0);
        for (int64_t i = 0; i < ns; ++i) {
            const int64_t p = ipiv_g[(size_t)(base + i)] - base;
            std::swap(rowsrc[(size_t)i], rowsrc[(size_t)p]);
        }
        srcs.clear(); dsts.clear();
        for (int64_t r = 0; r < mr; ++r)
            if (rowsrc[(size_t)r] != r) {
                dsts.push_back((int)(base + r));
                srcs.push_back((int)(base + rowsrc[(size_t)r]));
            }
        const int64_t np = (int64_t)srcs.size();
        if (np == 0) return;
        max_pairs = std::max(max_pairs, np);
        CUDA_CHECK(cudaMemcpy(d_src, srcs.data(), (size_t)np * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dst, dsts.data(), (size_t)np * sizeof(int), cudaMemcpyHostToDevice));
        const int64_t cchunk = std::max<int64_t>((int64_t)(swap_elems / (size_t)np), 1);
        const int blk = 256;
        for (int64_t c = col0; c < col0 + nc; c += cchunk) {
            const int64_t ncc = std::min(cchunk, col0 + nc - c);
            row_gather_kernel<<<grid1d(np * ncc, blk), blk>>>(d_swapbuf, d_A, d_src, np, ld, c, ncc);
            row_scatter_kernel<<<grid1d(np * ncc, blk), blk>>>(d_A, d_swapbuf, d_dst, np, ld, c, ncc);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    /* Ozaki-I fused-diagonal emulated GEMM: C(m x n) -= A(m x kk) * B(kk x n).
     * A is L21 (Aat(J+KJ,J)), B is U12 (Aat(J,J+KJ)) — both are VIEWS into
     * the shared d_A matrix with real leading dimension `ld`, not compact
     * kk-row buffers, so every slicer must be given the true ld. A is
     * sliced once (kk wide, the full aggregated K) since it is reused
     * across all n-tiles; B is column-scaled once over the full n but
     * SLICED per n-tile (into a small reusable ozSB buffer) both to keep
     * ozsliceB_kernel's gridDim.y bounded for large n and to bound ozSB's
     * footprint to one tile instead of the whole trailing width. Safe to
     * reuse the same ozSB buffer across tiles: every kernel/matmul here
     * runs on the single stream sOz, so tile t+1's slice write is ordered
     * strictly after tile t's matmul read of the same buffer. */
    void oz_gemm(int64_t m, int64_t n, int64_t kk, double *Ap, double *Bp,
                double *Cp)
    {
        dim3 rsblk(256), rsgrd((unsigned)((m + 255) / 256));
        ozrowscale_kernel<<<rsgrd, rsblk, 0, sOz>>>(Ap, (size_t)ld, (int)m,
                                                    (int)kk, ozRinv, ozRsc);
        {
            dim3 blk(32, 8), grd((unsigned)((m + 31) / 32), (unsigned)((kk + 31) / 32));
            ozsliceA_kernel<<<grd, blk, 0, sOz>>>(Ap, (size_t)ld, (int)m,
                                                  (int)kk, ozRinv, ozSA, OZ_S);
        }
        ozcolscale_kernel<<<(unsigned)((n + 255) / 256), 256, 0, sOz>>>(
            Bp, (size_t)ld, (int)kk, (int)n, ozCinv, ozCsc);

        cublasLtMatmulDesc_t opI;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&opI, CUBLAS_COMPUTE_32I, CUDA_R_32I));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opI, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opI, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        const size_t ldS = (size_t)OZ_S * kk;
        cublasLtMatrixLayout_t laD[6], lbD[6], lcD[6], lbDt[6], lcDt[6];
        cublasLtMatmulHeuristicResult_t hD[6][8], hDt[6][8];
        int64_t ntail = (n % cfg.oz_nt) && (n > cfg.oz_nt) ? n % cfg.oz_nt : 0;
        for (int d = 0; d <= OZ_D; d++) {
            long long kd = (long long)(std::min(d, OZ_S - 1) + 1) * kk;
            int nh = 0;
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&laD[d], CUDA_R_8I, kd, m, ldS));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lbD[d], CUDA_R_8I, kd,
                std::min(n, cfg.oz_nt), ldS));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lcD[d], CUDA_R_32I, m,
                std::min(n, cfg.oz_nt), m));
            CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(ozLt, opI, laD[d], lbD[d],
                lcD[d], lcD[d], ozPref, 8, hD[d], &nh));
            if (!nh) { std::fprintf(stderr, "[hpl_single] no int8 algo d=%d\n", d); std::exit(1); }
            if (nh > cfg.oz_algo) hD[d][0] = hD[d][cfg.oz_algo];
            if (ntail) {
                CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lbDt[d], CUDA_R_8I, kd, ntail, ldS));
                CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lcDt[d], CUDA_R_32I, m, ntail, m));
                CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(ozLt, opI, laD[d], lbDt[d],
                    lcDt[d], lcDt[d], ozPref, 8, hDt[d], &nh));
                if (!nh) { std::fprintf(stderr, "[hpl_single] no tail algo d=%d\n", d); std::exit(1); }
                if (nh > cfg.oz_algo) hDt[d][0] = hDt[d][cfg.oz_algo];
            }
        }
        for (int64_t t0 = 0; t0 < n; t0 += cfg.oz_nt) {
            int64_t nt = std::min(cfg.oz_nt, n - t0);
            int istail = (nt != cfg.oz_nt && n > cfg.oz_nt);
            const int set = (int)((t0 / cfg.oz_nt) & 1);
            CUDA_CHECK(cudaStreamWaitEvent(sOz, evOzRecon[set], 0));
            {
                dim3 gg(8, (unsigned)nt);
                ozsliceB_kernel<<<gg, 256, 0, sOz>>>(Bp + (size_t)t0 * ld,
                    (size_t)ld, (int)kk, (int)nt, ozCinv + t0, ozSB, OZ_S, ldS);
            }
            for (int d = 0; d <= OZ_D; d++) {
                int ia = 1, ib = 0;
                CUBLAS_CHECK(cublasLtMatmul(ozLt, opI, &ia, ozSA, laD[d],
                    ozSB + (size_t)(OZ_S - 1 - d) * kk, istail ? lbDt[d] : lbD[d],
                    &ib, ozDg[set][d], istail ? lcDt[d] : lcD[d], ozDg[set][d],
                    istail ? lcDt[d] : lcD[d],
                    istail ? &hDt[d][0].algo : &hD[d][0].algo, ozWs, ozWsz, sOz));
            }
            CUDA_CHECK(cudaEventRecord(evOzGemm[set], sOz));
            {
                dim3 blk(64, 4), grd((unsigned)((m + 63) / 64), (unsigned)((nt + 3) / 4));
                ozrecon_kernel<<<grd, blk, 0, sOz>>>(
                    Cp + (size_t)t0 * ld, (size_t)ld, (int)m, (int)nt,
                    ozDg[set][0], ozDg[set][1], ozDg[set][2], ozDg[set][3],
                    ozDg[set][4], ozDg[set][5], OZ_D + 1, ozRsc, ozCsc + t0);
            }
            CUDA_CHECK(cudaEventRecord(evOzRecon[set], sOz));
        }
        for (int d = 0; d <= OZ_D; d++) {
            cublasLtMatrixLayoutDestroy(laD[d]);
            cublasLtMatrixLayoutDestroy(lbD[d]);
            cublasLtMatrixLayoutDestroy(lcD[d]);
            if (ntail) { cublasLtMatrixLayoutDestroy(lbDt[d]); cublasLtMatrixLayoutDestroy(lcDt[d]); }
        }
        cublasLtMatmulDescDestroy(opI);
        CUDA_CHECK(cudaStreamSynchronize(sOz)); /* single-stream driver: keep it simple at A1 */
    }

    void update_trailing(int64_t J, int64_t KJ, int64_t mt, int64_t nt)
    {
        const double one = 1.0, mone = -1.0;
        const bool emul = cfg.backend != BK_NATIVE && KJ == cfg.K && mt >= cfg.native_tail;
        double *Lp = Aat(J + KJ, J);
        if (!emul) {
            clk.tic(P_WGEMM);
            CUBLAS_CHECK(cublasDgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, (int)mt, (int)nt,
                (int)KJ, &mone, Lp, (int)ld, Aat(J, J + KJ), (int)ld, &one,
                Aat(J + KJ, J + KJ), (int)ld));
            ++n_native;
            clk.toc();
        } else if (cfg.backend == BK_OZAKI) {
            clk.tic(P_WGEMM);
            oz_gemm(mt, nt, KJ, Lp, Aat(J, J + KJ), Aat(J + KJ, J + KJ));
            ++n_oz;
            clk.toc();
        } else {
            /* n-chunked gemmul8 update; the A (=L21) low-precision slices
             * are identical for every chunk of this block, so chunks 2..
             * reuse them via skip_scalA. Column tails < 1024 are absorbed
             * into the previous chunk (workspace sized for chunk+1024),
             * avoiding degenerate-n calls. */
            for (int64_t c = 0; c < nt;) {
                int64_t nc = std::min(cfg.g8_chunk, nt - c);
                if (nt - c - nc > 0 && nt - c - nc < 1024) nc = nt - c;
                clk.tic(P_WGEMM);
                std::vector<double> t = gemmul8::gemm<double, gemmul8::Backend::INT8>(
                    hb, CUBLAS_OP_N, CUBLAS_OP_N, (size_t)mt, (size_t)nc,
                    (size_t)KJ, &mone, Lp, (size_t)ld, Aat(J, J + KJ + c),
                    (size_t)ld, &one, Aat(J + KJ, J + KJ + c), (size_t)ld,
                    cfg.g8_moduli, cfg.g8_fast != 0, d_g8work, nullptr, nullptr,
                    /*enable_skip_scalA=*/cfg.g8_skipA != 0,
                    /*enable_skip_scalB=*/false,
                    /*skip_scalA=*/cfg.g8_skipA != 0 && c > 0,
                    /*skip_scalB=*/false);
                clk.toc();
                for (size_t i = 0; i < 4 && i < t.size(); ++i) tg8[i] += t[i];
                ++n_g8;
                c += nc;
            }
        }
    }

    void factorize_and_solve()
    {
        const double one = 1.0, mone = -1.0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int64_t J = 0; J < N; J += cfg.K) {
            const int64_t KJ = std::min(cfg.K, N - J);
            for (int64_t jp = J; jp < J + KJ; jp += cfg.NB) {
                const int64_t nb = std::min(cfg.NB, J + KJ - jp);
                const int64_t mp = N - jp;
                clk.tic(P_FACT);
                CUSOLVER_CHECK(cusolverDnXgetrf(hs, hp, mp, nb, CUDA_R_64F,
                    Aat(jp, jp), ld, d_ipiv, CUDA_R_64F, d_fwork, fwork_d,
                    h_fwork, fwork_h, d_info));
                clk.toc();
                int info = 0;
                CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
                if (info != 0) {
                    std::fprintf(stderr, "[hpl_single] getrf info=%d at jp=%" PRId64 "\n", info, jp);
                    std::exit(1);
                }
                CUDA_CHECK(cudaMemcpy(h_ipiv.data(), d_ipiv, (size_t)nb * sizeof(int64_t), cudaMemcpyDeviceToHost));
                for (int64_t i = 0; i < nb; ++i) ipiv_g[(size_t)(jp + i)] = jp + h_ipiv[(size_t)i] - 1;

                clk.tic(P_LASWP);
                apply_swaps(jp, nb, J, jp - J);
                apply_swaps(jp, nb, jp + nb, J + KJ - (jp + nb));
                clk.toc();

                const int64_t nr = J + KJ - (jp + nb);
                if (nr > 0) {
                    clk.tic(P_INNER);
                    CUBLAS_CHECK(cublasDtrsm(hb, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N, CUBLAS_DIAG_UNIT, (int)nb, (int)nr, &one,
                        Aat(jp, jp), (int)ld, Aat(jp, jp + nb), (int)ld));
                    const int64_t mi = N - (jp + nb);
                    if (mi > 0)
                        CUBLAS_CHECK(cublasDgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, (int)mi, (int)nr,
                            (int)nb, &mone, Aat(jp + nb, jp), (int)ld, Aat(jp, jp + nb), (int)ld,
                            &one, Aat(jp + nb, jp + nb), (int)ld));
                    clk.toc();
                }
            }
            clk.tic(P_LASWP);
            apply_swaps(J, KJ, 0, J);
            apply_swaps(J, KJ, J + KJ, ncols - (J + KJ));
            clk.toc();

            const int64_t nt = ncols - (J + KJ);
            if (nt > 0) {
                clk.tic(P_WTRSM);
                CUBLAS_CHECK(cublasDtrsm(hb, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N, CUBLAS_DIAG_UNIT, (int)KJ, (int)nt, &one, Aat(J, J),
                    (int)ld, Aat(J, J + KJ), (int)ld));
                clk.toc();
                const int64_t mt = N - (J + KJ);
                if (mt > 0) update_trailing(J, KJ, mt, nt);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double dN = (double)N, dJ = (double)(J + KJ);
            const double fdone = (2.0 / 3.0) * (dN * dN * dN - (dN - dJ) * (dN - dJ) * (dN - dJ));
            std::printf("PROGRESS: J=%-8" PRId64 " KJ=%-6" PRId64 " elapsed %8.2f s  running %8.1f GF\n",
                        J, KJ, el, fdone / el / 1e9);
            std::fflush(stdout);
        }
        clk.tic(P_SOLVE);
        CUBLAS_CHECK(cublasDtrsm(hb, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, (int)N, 1, &one, d_A, (int)ld, Aat(0, N), (int)ld));
        clk.toc();
    }
};

/* ------------------------------------------------------------------ */
/* run_case / report (mirrors hpl_ga.cu's format for direct comparison)*/
/* ------------------------------------------------------------------ */
static int run_case(const Cfg &cfg)
{
    std::printf(
        "==========================================================\n"
        "hpl_single: N=%" PRId64 " NB=%" PRId64 " AGG=%" PRId64 " K=%" PRId64
        " backend=%s oz_nt=%" PRId64 " oz_algo=%d native_tail=%" PRId64
        " seed=%" PRIu64 "\n"
        "==========================================================\n",
        cfg.N, cfg.NB, cfg.AGG, cfg.K, cfg.tag(), cfg.oz_nt, cfg.oz_algo,
        cfg.native_tail, cfg.seed);
    std::fflush(stdout);

    Solver s; s.cfg = cfg; s.plan_and_alloc();
    const int64_t N = s.N, ld = s.ld;
    const int blk = 256;

    auto tg0 = std::chrono::steady_clock::now();
    matgen_kernel<<<grid1d(N * s.ncols, blk), blk>>>(s.d_A, N, 0, s.ncols, ld, cfg.seed, (uint64_t)N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    std::printf("MATGEN: %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tg0).count());
    std::fflush(stdout);

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    s.factorize_and_solve();
    CUDA_CHECK(cudaDeviceSynchronize());
    const double tSolve = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    double *d_x = nullptr, *d_y = nullptr, *d_rs = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_x, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_y, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_rs, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x, s.Aat(0, N), (size_t)N * sizeof(double), cudaMemcpyDeviceToDevice));
    std::vector<double> h_x((size_t)N);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));

    matgen_kernel<<<grid1d(N * s.ncols, blk), blk>>>(s.d_A, N, 0, s.ncols, ld, cfg.seed, (uint64_t)N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(d_y, 0, (size_t)N * sizeof(double)));
    const double one = 1.0;
    for (int64_t c = 0; c < N; c += 32768) {
        const int64_t nc = std::min<int64_t>(32768, N - c);
        CUBLAS_CHECK(cublasDgemv(s.hb, CUBLAS_OP_N, (int)N, (int)nc, &one,
            s.Aat(0, c), (int)ld, d_x + c, 1, &one, d_y, 1));
    }
    row_abs_sum_kernel<<<grid1d(N, blk), blk>>>(s.d_A, N, N, ld, d_rs);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_y((size_t)N), h_rs((size_t)N), h_b((size_t)N);
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rs.data(), d_rs, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), s.Aat(0, N), (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));

    double resid0 = 0.0, AnormI = 0.0, XnormI = 0.0, BnormI = 0.0;
    for (int64_t i = 0; i < N; ++i) {
        const double r = std::fabs(h_y[(size_t)i] - h_b[(size_t)i]);
        if (r > resid0) resid0 = r;
        if (h_rs[(size_t)i] > AnormI) AnormI = h_rs[(size_t)i];
        const double ax = std::fabs(h_x[(size_t)i]);
        if (ax > XnormI) XnormI = ax;
        const double ab = std::fabs(h_b[(size_t)i]);
        if (ab > BnormI) BnormI = ab;
    }
    const double eps = std::ldexp(1.0, -53);
    const double resid1 = resid0 / (eps * (AnormI * XnormI + BnormI) * (double)N);
    const bool pass = resid1 < 16.0;

    const double dN = (double)N;
    const double gflops = ((2.0 / 3.0) * dN * dN * dN + (3.0 / 2.0) * dN * dN) / tSolve / 1.0e9;
    std::printf(
        "==========================================================\n"
        "T/V                   N    NB   AGG      K           Time                 Gflops\n"
        "----------------------------------------------------------\n"
        "%-16s%8" PRId64 " %5" PRId64 " %5" PRId64 " %6" PRId64 " %14.2f     %18.4e\n"
        "----------------------------------------------------------\n"
        "||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)= %13.7f ...... %s\n",
        cfg.backend == BK_OZAKI     ? "HPLS_ozaki"
        : cfg.backend == BK_GEMMUL8 ? "HPLS_gemmul8"
                                    : "HPLS_native",
        N, cfg.NB, cfg.AGG, cfg.K,
        tSolve, gflops, resid1, pass ? "PASSED" : "FAILED");

    double ph[P_NPHASE]; s.clk.sums(ph);
    double sum = 0.0; for (int p = 0; p < P_NPHASE; ++p) sum += ph[p];
    std::printf("----------------------------------------------------------\nPHASE              sec     %%wall\n");
    for (int p = 0; p < P_NPHASE; ++p)
        std::printf("%-14s%9.3f   %6.2f%%\n", phase_name[p], ph[p], 100.0 * ph[p] / tSolve);
    std::printf("%-14s%9.3f   %6.2f%%   (gap = host bookkeeping/launch)\n", "sum_phases", sum, 100.0 * sum / tSolve);
    std::printf("wide_gemm calls: ozaki=%" PRId64 " gemmul8=%" PRId64 " native=%" PRId64
                " | laswp max pairs=%" PRId64 "\n",
                s.n_oz, s.n_g8, s.n_native, s.max_pairs);
    if (s.n_g8 > 0)
        std::printf("gemmul8 internal: scale %.3f s | int8-gemm %.3f s | "
                    "mod %.3f s | CRT+unscale %.3f s (moduli=%d fast=%d "
                    "chunk=%" PRId64 " skipA=%d)\n",
                    s.tg8[0], s.tg8[1], s.tg8[2], s.tg8[3], cfg.g8_moduli,
                    cfg.g8_fast, cfg.g8_chunk, cfg.g8_skipA);
    std::printf("==========================================================\n");
    std::fflush(stdout);

    CUDA_CHECK(cudaFree(d_rs)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_x));
    s.destroy();
    return pass ? 0 : 1;
}

static Cfg cfg_from_env(int64_t N)
{
    Cfg c;
    c.N = N;
    c.NB = env_i64("HPLS_NB", 2048);
    c.AGG = env_i64("HPLS_AGG", 6);
    c.K = c.NB * c.AGG;
    c.oz_nt = env_i64("HPLS_OZ_NT", 1024);
    c.oz_algo = (int)env_i64("HPLS_OZ_ALGO", 3);
    c.native_tail = env_i64("HPLS_NATIVE_TAIL", 8192);
    c.g8_moduli = (int)env_i64("HPLS_G8_MODULI", 14);
    c.g8_fast = (int)env_i64("HPLS_G8_FAST", 0);
    c.g8_chunk = env_i64("HPLS_G8_CHUNK", 16384);
    c.g8_skipA = (int)env_i64("HPLS_G8_SKIPA", 1);
    c.seed = (uint64_t)env_i64("HPLS_SEED", 42);
    const char *b = std::getenv("HPLS_BACKEND");
    if (b && std::strcmp(b, "ozaki") == 0) c.backend = BK_OZAKI;
    else if (b && std::strcmp(b, "gemmul8") == 0) c.backend = BK_GEMMUL8;
    else if (b && std::strcmp(b, "native") != 0) {
        std::fprintf(stderr, "[hpl_single] unknown HPLS_BACKEND '%s'\n", b);
        std::exit(1);
    }
    if (c.N <= 0 || c.NB <= 0 || c.AGG <= 0 || c.NB > c.N) {
        std::fprintf(stderr, "[hpl_single] bad N/NB/AGG\n");
        std::exit(1);
    }
    return c;
}

/* Isolate oz_gemm from the whole LU solver: place A(m x kk), B(kk x n),
 * C(m x n) as disjoint views into one Solver's d_A (same addressing oz_gemm
 * uses in production), run oz_gemm on the real view, run cublasDgemm on a
 * compact copy of the same inputs, and diff. */
static int gemmtest(int64_t m, int64_t kk, int64_t n, int64_t oz_nt = -1)
{
    Cfg c; c.N = kk + std::max(m, n) + 8; c.NB = std::min<int64_t>(kk, c.N - 1);
    c.AGG = 1; c.K = kk; c.backend = BK_OZAKI;
    c.oz_nt = oz_nt > 0 ? oz_nt : std::min<int64_t>(256, n); c.native_tail = 0; c.seed = 7;
    std::printf("GEMMTEST: m=%" PRId64 " k=%" PRId64 " n=%" PRId64 " (N=%" PRId64 " oz_nt=%" PRId64 ")\n",
                m, kk, n, c.N, c.oz_nt);

    Solver s; s.cfg = c; s.plan_and_alloc();
    const int64_t ld = s.ld, blk = 256;
    matgen_kernel<<<grid1d(s.N * s.ncols, blk), blk>>>(s.d_A, s.N, 0, s.ncols, ld, c.seed, (uint64_t)s.N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    double *Ap = s.Aat(kk, 0), *Bp = s.Aat(0, kk), *Cp = s.Aat(kk, kk);
    double *d_Cref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Cref, (size_t)m * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy2D(d_Cref, (size_t)m * sizeof(double), Cp, ld * sizeof(double),
                            (size_t)m * sizeof(double), (size_t)n, cudaMemcpyDeviceToDevice));

    const double one = 1.0, mone = -1.0;
    CUBLAS_CHECK(cublasDgemm(s.hb, CUBLAS_OP_N, CUBLAS_OP_N, (int)m, (int)n, (int)kk,
        &mone, Ap, (int)ld, Bp, (int)ld, &one, d_Cref, (int)m));

    s.oz_gemm(m, n, kk, Ap, Bp, Cp);

    std::vector<double> h_ref((size_t)m * n), h_oz((size_t)m * n);
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_Cref, (size_t)m * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_oz.data(), (size_t)m * sizeof(double), Cp, ld * sizeof(double),
                            (size_t)m * sizeof(double), (size_t)n, cudaMemcpyDeviceToHost));

    double maxabs = 0.0, maxrel = 0.0, sumabsref = 0.0;
    int64_t maxi = -1, maxj = -1;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i) {
            const double r = h_ref[(size_t)j * m + i], o = h_oz[(size_t)j * m + i];
            const double e = std::fabs(r - o);
            sumabsref += std::fabs(r);
            if (e > maxabs) { maxabs = e; maxi = i; maxj = j; }
            const double rel = e / (std::fabs(r) + 1e-12);
            if (rel > maxrel) maxrel = rel;
        }
    std::printf("GEMMTEST: max_abs_err=%.6e max_rel_err=%.6e at (i=%" PRId64 ",j=%" PRId64 ")"
                " ref[that]=%.6e oz[that]=%.6e mean|ref|=%.3e\n",
                maxabs, maxrel, maxi, maxj,
                maxi >= 0 ? h_ref[(size_t)maxj * m + maxi] : 0.0,
                maxi >= 0 ? h_oz[(size_t)maxj * m + maxi] : 0.0,
                sumabsref / (double)(m * n));
    const bool ok = maxrel < 1e-2; /* int8 emulation: expect ~1e-6..1e-4, not O(1) */
    std::printf("GEMMTEST_%s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_Cref));
    s.destroy();
    return ok ? 0 : 1;
}

static int selftest()
{
    Cfg c1; c1.N = 8192; c1.NB = 512; c1.AGG = 4; c1.K = 2048; c1.backend = BK_NATIVE; c1.seed = 42;
    Cfg c2 = c1; c2.backend = BK_OZAKI; c2.oz_nt = 512; c2.native_tail = 0;
    Cfg c3 = c1; c3.backend = BK_GEMMUL8; c3.native_tail = 0;
    c3.g8_chunk = 3072; /* forces multi-chunk => exercises skip_scalA */
    std::printf("SELFTEST run 1/3: native\n");
    const int r1 = run_case(c1);
    std::printf("SELFTEST run 2/3: ozaki\n");
    const int r2 = run_case(c2);
    std::printf("SELFTEST run 3/3: gemmul8 (moduli=14 accurate, chunked)\n");
    const int r3 = run_case(c3);
    const bool ok = (r1 == 0) && (r2 == 0) && (r3 == 0);
    std::printf("SELFTEST_%s (native=%s ozaki=%s gemmul8=%s)\n", ok ? "PASS" : "FAIL",
                r1 == 0 ? "PASS" : "FAIL", r2 == 0 ? "PASS" : "FAIL",
                r3 == 0 ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--selftest") == 0) return selftest();
    if ((argc == 5 || argc == 6) && std::strcmp(argv[1], "--gemmtest") == 0)
        return gemmtest(std::atoll(argv[2]), std::atoll(argv[3]), std::atoll(argv[4]),
                        argc == 6 ? std::atoll(argv[5]) : -1);
    if (argc != 2) {
        std::fprintf(stderr, "usage: hpl_single N | hpl_single --selftest | "
                     "hpl_single --gemmtest m k n [oz_nt]\n"
                     "env: HPLS_NB HPLS_AGG HPLS_BACKEND(native|ozaki|gemmul8) "
                     "HPLS_OZ_NT HPLS_OZ_ALGO HPLS_NATIVE_TAIL HPLS_SEED "
                     "HPLS_G8_MODULI HPLS_G8_FAST HPLS_G8_CHUNK HPLS_G8_SKIPA\n");
        return 1;
    }
    return run_case(cfg_from_env(std::atoll(argv[1])));
}
