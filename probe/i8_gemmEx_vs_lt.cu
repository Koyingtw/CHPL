/* Does a TUNED cublasLt int8 GEMM beat cublasGemmEx(CUBLAS_GEMM_DEFAULT) --
 * the path GEMMul8's cuBLAS-handle entry actually uses -- at CHPL's production
 * trailing-update shapes?  TN, int8 -> int32.  This is the fast.cu premise
 * ("beat cuBLAS's default GEMM") mapped onto OUR int8 shape. */
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do{auto e=(x); if(e){printf("cuda err %d @%d\n",(int)e,__LINE__);exit(1);}}while(0)
#define CB(x) do{auto e=(x); if(e){printf("cublas err %d @%d\n",(int)e,__LINE__);exit(1);}}while(0)

static float bench_gemmEx(cublasHandle_t h, int m,int n,int k,
                          const int8_t*A,const int8_t*B,int32_t*C,int iters){
    int a=1,b=0; cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    // warmup
    for(int i=0;i<3;i++) CB(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&a,
        A,CUDA_R_8I,k, B,CUDA_R_8I,k, &b, C,CUDA_R_32I,m,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    CK(cudaEventRecord(e0));
    for(int i=0;i<iters;i++) CB(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&a,
        A,CUDA_R_8I,k, B,CUDA_R_8I,k, &b, C,CUDA_R_32I,m,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); return ms/iters;
}

static float bench_lt(cublasLtHandle_t lt,int m,int n,int k,
                      const int8_t*A,const int8_t*B,int32_t*C,void*ws,size_t wsz,int iters,int*algo_idx){
    cublasLtMatmulDesc_t op; CB(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I));
    cublasOperation_t ta=CUBLAS_OP_T, tb=CUBLAS_OP_N;
    CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    cublasLtMatrixLayout_t la,lb,lc;
    CB(cublasLtMatrixLayoutCreate(&la,CUDA_R_8I,k,m,k));
    CB(cublasLtMatrixLayoutCreate(&lb,CUDA_R_8I,k,n,k));
    CB(cublasLtMatrixLayoutCreate(&lc,CUDA_R_32I,m,n,m));
    cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
    CB(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz)));
    cublasLtMatmulHeuristicResult_t hr[16]; int nh=0;
    CB(cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,lc,lc,pref,16,hr,&nh));
    if(!nh){printf("no lt algo for %dx%dx%d\n",m,n,k); return -1;}
    int a=1,b=0; cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float best=1e30f; int bi=0;
    for(int j=0;j<nh;j++){
        // warmup this algo
        if(cublasLtMatmul(lt,op,&a,A,la,B,lb,&b,C,lc,C,lc,&hr[j].algo,ws,wsz,0)!=CUBLAS_STATUS_SUCCESS) continue;
        CK(cudaEventRecord(e0));
        for(int i=0;i<iters;i++) cublasLtMatmul(lt,op,&a,A,la,B,lb,&b,C,lc,C,lc,&hr[j].algo,ws,wsz,0);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
        if(ms<best){best=ms; bi=j;}
    }
    *algo_idx=bi;
    return best;
}

int main(){
    cublasHandle_t h; CB(cublasCreate(&h));
    cublasLtHandle_t lt; CB(cublasLtCreate(&lt));
    size_t wsz=256u<<20; void*ws; CK(cudaMalloc(&ws,wsz));
    // production TN int8 shapes: k=NB=4096; n = G8 tile (~3072); m sweeps mloc-ish tiles
    struct S{int m,n,k;} shapes[]={
        {8192,3072,4096},{16384,3072,4096},{32768,3072,4096},
        {65536,3072,4096},{115200,3072,4096},
        {8192,2048,4096},{16384,1024,4096},
    };
    printf("%7s %6s %6s | %10s %10s | %8s  %s\n","m","n","k","gemmEx ms","lt-best ms","speedup","(lt algo idx)");
    for(auto&s:shapes){
        size_t szA=(size_t)s.m*s.k, szB=(size_t)s.n*s.k, szC=(size_t)s.m*s.n;
        int8_t*A,*B; int32_t*C; CK(cudaMalloc(&A,szA)); CK(cudaMalloc(&B,szB)); CK(cudaMalloc(&C,szC*4));
        CK(cudaMemset(A,1,szA)); CK(cudaMemset(B,1,szB));
        int iters=20, ai=-1;
        float tex=bench_gemmEx(h,s.m,s.n,s.k,A,B,C,iters);
        float tlt=bench_lt(lt,s.m,s.n,s.k,A,B,C,ws,wsz,iters,&ai);
        double tops_ex=2.0*s.m*s.n*s.k/(tex/1e3)/1e12;
        printf("%7d %6d %6d | %10.3f %10.3f | %7.3fx  idx=%d  (gemmEx %.0f TOPS)\n",
               s.m,s.n,s.k,tex,tlt,tex/tlt,ai,tops_ex);
        cudaFree(A);cudaFree(B);cudaFree(C);
    }
    return 0;
}
