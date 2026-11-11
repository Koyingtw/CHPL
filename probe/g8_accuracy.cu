#include <gemmul8.hpp>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#define CK(x) do{auto e=(x); if(e){printf("cuda err %d @%d\n",(int)e,__LINE__);exit(1);}}while(0)
#define CB(x) do{auto e=(x); if(e){printf("cublas err %d @%d\n",(int)e,__LINE__);exit(1);}}while(0)

__global__ void shift_kernel(double* x, size_t n) {
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] -= 0.5;
}

int main(){
    int M=8192, N=3072, K=4096, num_moduli=11; bool fastmode=true;
    cublasHandle_t h; CB(cublasCreate(&h));
    { void *w; CK(cudaMalloc(&w, 64u<<20)); CB(cublasSetWorkspace(h, w, 64u<<20)); }
    double *A,*B,*C,*Cref;
    CK(cudaMalloc(&A,(size_t)M*K*8)); CK(cudaMalloc(&B,(size_t)K*N*8));
    CK(cudaMalloc(&C,(size_t)M*N*8)); CK(cudaMalloc(&Cref,(size_t)M*N*8));
    curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42ULL);
    curandGenerateUniformDouble(gen, A, (size_t)M*K);
    curandGenerateUniformDouble(gen, B, (size_t)K*N);
    shift_kernel<<<((size_t)M*K+255)/256,256>>>(A,(size_t)M*K);
    shift_kernel<<<((size_t)K*N+255)/256,256>>>(B,(size_t)K*N);
    CK(cudaMemset(C,0,(size_t)M*N*8));
    double done_=1.0, dmone=-1.0, dzero=0.0;
    CB(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &done_, A, M, B, K, &dzero, Cref, M));
    size_t wsA=0,wsB=0;
    size_t wsz = gemmul8::workSize<false, gemmul8::Backend::INT8>((size_t)M,(size_t)N,(size_t)K,num_moduli,true,false,&wsA,&wsB);
    void *work; CK(cudaMalloc(&work, wsz+256));
    gemmul8::gemm<double, gemmul8::Backend::INT8>(h,CUBLAS_OP_N,CUBLAS_OP_N,(size_t)M,(size_t)N,(size_t)K,&dmone,A,(size_t)M,B,(size_t)K,&done_,C,(size_t)M,num_moduli,fastmode,work,nullptr,nullptr,true,false,false,false);
    // C currently holds (-A*B + 0) since beta applied to C which was memset 0; so C = -A*B -> negate
    std::vector<double> hC(M*(size_t)N), hRef(M*(size_t)N);
    CK(cudaMemcpy(hC.data(), C, M*(size_t)N*8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hRef.data(), Cref, M*(size_t)N*8, cudaMemcpyDeviceToHost));
    double num=0, den=0;
    for (size_t i=0;i<(size_t)M*N;i++){ double d = (-hC[i]) - hRef[i]; num += d*d; den += hRef[i]*hRef[i]; }
    double relerr = std::sqrt(num/den);
    double effbits = relerr>0 ? -std::log2(relerr) : 1e9;
    printf("gemmul8 3.1.0 accuracy: M=%d N=%d K=%d moduli=%d fastmode=%d -> rel_err=%.3e eff_bits=%.2f\n",
           M,N,K,num_moduli,fastmode,relerr,effbits);
    return 0;
}
