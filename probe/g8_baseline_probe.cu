/* Baseline: gemmul8 3.1.0's actual production INT8/CRT DGEMM call
 * (same call shape as chpl4x4.cu:2336-2346, barrier mode), timed at
 * champion-config shapes: k=NB=4096, n=G8_NT=3072, m swept over the
 * range mlb takes at 1n/2n/4n champion configs.  This is the number
 * EmuGEMM's int8_crt_dgemm_nt (+ its own slice/CRT kernels) needs to
 * beat to be worth integrating. */
#include <gemmul8.hpp>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do{auto e=(x); if(e){printf("cuda err %d @%d\n",(int)e,__LINE__);exit(1);}}while(0)
#define CB(x) do{auto e=(x); if(e){printf("cublas err %d @%d\n",(int)e,__LINE__);exit(1);}}while(0)

int main(int argc, char** argv) {
    int num_moduli = argc > 1 ? atoi(argv[1]) : 11;
    bool fastmode  = argc > 2 ? atoi(argv[2]) != 0 : true;
    int k = 4096;          /* NB, champion */
    int n = 3072;           /* G8_NT, champion */
    int ms[] = {8192, 16384, 32768, 57600, 115200}; /* mlb range: 1n P2Q4 mloc~ N/2, 2n P4Q4 mloc~N/4 etc */

    cublasHandle_t h; CB(cublasCreate(&h));
    { void *w; CK(cudaMalloc(&w, 64u<<20)); CB(cublasSetWorkspace(h, w, 64u<<20)); }

    double *A, *B, *C; /* A = W panel [m,k] fp64, B = M tile [k,n] fp64, C = trailing [m,n] fp64 (accumulate) */
    size_t maxm = 115200;
    CK(cudaMalloc(&A, maxm * (size_t)k * 8));
    CK(cudaMalloc(&B, (size_t)k * n * 8));
    CK(cudaMalloc(&C, maxm * (size_t)n * 8));
    CK(cudaMemset(A, 0, maxm*(size_t)k*8));
    CK(cudaMemset(B, 0, (size_t)k*n*8));
    CK(cudaMemset(C, 0, maxm*(size_t)n*8));
    /* fill with something non-trivial via curand-free approach: use cudaMemset patterns +
     * a simple init kernel isn't needed for TIMING (data-independent perf); skip for speed. */

    double done_ = 1.0, dmone = -1.0;

    printf("gemmul8 3.1.0 baseline: moduli=%d fastmode=%d k=%d n=%d\n", num_moduli, fastmode, k, n);
    printf("%8s | %10s %8s %8s\n", "m", "ms/call", "TFLOPS", "GB-equiv-out(mod_hi2mid x N)");
    for (int m : ms) {
        size_t wsA=0, wsB=0;
        size_t wsz = gemmul8::workSize<false, gemmul8::Backend::INT8>(
            (size_t)m, (size_t)n, (size_t)k, num_moduli, true, false, &wsA, &wsB);
        void *work; CK(cudaMalloc(&work, wsz + 256));

        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        int iters = 5, warmup = 2;
        for (int i = 0; i < warmup; i++) {
            gemmul8::gemm<double, gemmul8::Backend::INT8>(
                h, CUBLAS_OP_N, CUBLAS_OP_N, (size_t)m, (size_t)n, (size_t)k,
                &dmone, A, (size_t)m, B, (size_t)k, &done_, C, (size_t)m,
                num_moduli, fastmode, work, nullptr, nullptr,
                true, false, /*skip_scalA=*/i>0, false);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; i++) {
            gemmul8::gemm<double, gemmul8::Backend::INT8>(
                h, CUBLAS_OP_N, CUBLAS_OP_N, (size_t)m, (size_t)n, (size_t)k,
                &dmone, A, (size_t)m, B, (size_t)k, &done_, C, (size_t)m,
                num_moduli, fastmode, work, nullptr, nullptr,
                true, false, /*skip_scalA=*/true, false);
        }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms_; CK(cudaEventElapsedTime(&ms_, e0, e1));
        ms_ /= iters;
        double tflops = 2.0 * m * n * k / (ms_ / 1e3) / 1e12;
        printf("%8d | %10.4f %8.2f\n", m, ms_, tflops);
        cudaFree(work);
    }
    return 0;
}
