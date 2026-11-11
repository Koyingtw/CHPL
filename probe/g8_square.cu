#include <gemmul8.hpp>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do{auto e=(x); if(e){printf("cuda err %d @%d\n",(int)e,__LINE__);exit(1);}}while(0)
#define CB(x) do{auto e=(x); if(e){printf("cublas err %d @%d\n",(int)e,__LINE__);exit(1);}}while(0)
int main(){
    cublasHandle_t h; CB(cublasCreate(&h));
    { void *w; CK(cudaMalloc(&w, 64u<<20)); CB(cublasSetWorkspace(h, w, 64u<<20)); }
    int Ks[] = {2048,4096,8192};
    int M=8192,N=8192, num_moduli=12; bool fastmode=true;
    double done_=1.0, dmone=-1.0;
    for (int k : Ks) {
        double *A,*B,*C;
        CK(cudaMalloc(&A,(size_t)M*k*8)); CK(cudaMalloc(&B,(size_t)k*N*8)); CK(cudaMalloc(&C,(size_t)M*N*8));
        CK(cudaMemset(A,0,(size_t)M*k*8)); CK(cudaMemset(B,0,(size_t)k*N*8)); CK(cudaMemset(C,0,(size_t)M*N*8));
        size_t wsA=0,wsB=0;
        size_t wsz = gemmul8::workSize<false, gemmul8::Backend::INT8>((size_t)M,(size_t)N,(size_t)k,num_moduli,true,false,&wsA,&wsB);
        void *work; CK(cudaMalloc(&work, wsz+256));
        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        int iters=5, warmup=2;
        for (int i=0;i<warmup;i++)
            gemmul8::gemm<double, gemmul8::Backend::INT8>(h,CUBLAS_OP_N,CUBLAS_OP_N,(size_t)M,(size_t)N,(size_t)k,&dmone,A,(size_t)M,B,(size_t)k,&done_,C,(size_t)M,num_moduli,fastmode,work,nullptr,nullptr,true,false,i>0,false);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i=0;i<iters;i++)
            gemmul8::gemm<double, gemmul8::Backend::INT8>(h,CUBLAS_OP_N,CUBLAS_OP_N,(size_t)M,(size_t)N,(size_t)k,&dmone,A,(size_t)M,B,(size_t)k,&done_,C,(size_t)M,num_moduli,fastmode,work,nullptr,nullptr,true,false,true,false);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
        double tflops = 2.0*M*N*k/(ms/1e3)/1e12;
        printf("gemmul8 3.1.0 square M=N=%d K=%d moduli=%d: %.4f ms, %.2f TFLOPS\n", M,k,num_moduli,ms,tflops);
        cudaFree(A);cudaFree(B);cudaFree(C);cudaFree(work);
    }
}
