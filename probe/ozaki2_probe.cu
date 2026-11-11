/* Self-written Ozaki-II (CRT/modular) DGEMM emulation probe.
 * C = A(m x k) * B(k x n), fp64, via t coprime moduli of int8 GEMMs + Garner
 * CRT reconstruction.  Compares TF-equiv + accuracy vs cublasDgemm.
 *
 * Math: scale A per-row and B per-col by powers of 2 so |A*scaleA|<2^s.
 *   A_int = round(A*scaleA) in [-2^s,2^s].  For each modulus M_r:
 *     Amod_r = A_int mod M_r  (int8, 0..M_r-1),  Bmod_r likewise.
 *     Cint_r = Amod_r^T-GEMM Bmod_r   (int32, TN)
 *     P mod M_r = Cint_r mod M_r      (P = exact integer inner product)
 *   prod(M_r) > 2*k*2^(2s) => Garner recovers signed P exactly.
 *   C = P / (scaleA_i * scaleB_j).
 *
 * Usage: oz2own [m n k]
 */
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>

#define CK(x) do{ auto e=(x); if(e){printf("ERR %s:%d code=%d\n",__FILE__,__LINE__,(int)e); exit(1);} }while(0)
#define CB(x) do{ cublasStatus_t e=(x); if(e){printf("CUBLAS %s:%d %d\n",__FILE__,__LINE__,(int)e); exit(1);} }while(0)

static const int TMAX = 24;

/* fill with reproducible pseudo-random in [-1,1) */
__global__ void fill_kernel(double *p, size_t n, unsigned seed){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){
        unsigned x=(unsigned)(i*2654435761u)^seed; x^=x<<13; x^=x>>17; x^=x<<5;
        p[i]=((double)x/4294967296.0-0.5)*2.0;
    }
}
/* per-row (A, m x k col-major lda) power-of-2 scale so |A*s|<2^s_bits */
__global__ void rowscale_kernel(const double*A,size_t lda,int m,int k,int sbits,double*scale){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=m) return;
    double mx=0; for(int j=0;j<k;j++){double v=fabs(A[(size_t)j*lda+i]); if(v>mx)mx=v;}
    int e = mx>0 ? (int)ceil(log2(mx)) : 0;
    scale[i]=exp2((double)(sbits-1-e));   /* |A*scale| < 2^(sbits-1) */
}
__global__ void colscale_kernel(const double*B,size_t ldb,int k,int n,int sbits,double*scale){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=n) return;
    double mx=0; for(int i=0;i<k;i++){double v=fabs(B[(size_t)j*ldb+i]); if(v>mx)mx=v;}
    int e = mx>0 ? (int)ceil(log2(mx)) : 0;
    scale[j]=exp2((double)(sbits-1-e));
}
/* quantize A (m x k) -> Amod stored as k x m (A^T, ld=k) int8 for TN gemm, for modulus M */
__global__ void quantA_kernel(const double*A,size_t lda,int m,int k,const double*scale,int M,int8_t*Amod){
    int i=blockIdx.x*blockDim.x+threadIdx.x;      /* row of A */
    int j=blockIdx.y*blockDim.y+threadIdx.y;      /* col of A = contraction */
    if(i>=m||j>=k) return;
    long long ai=llround(A[(size_t)j*lda+i]*scale[i]);
    int r=(int)(((ai%M)+M)%M);
    Amod[(size_t)i*k+j]=(int8_t)r;                /* A^T: [i*k + j] */
}
__global__ void quantB_kernel(const double*B,size_t ldb,int k,int n,const double*scale,int M,int8_t*Bmod){
    int i=blockIdx.x*blockDim.x+threadIdx.x;      /* row of B = contraction */
    int j=blockIdx.y*blockDim.y+threadIdx.y;      /* col of B */
    if(i>=k||j>=n) return;
    long long bi=llround(B[(size_t)j*ldb+i]*scale[j]);
    int r=(int)(((bi%M)+M)%M);
    Bmod[(size_t)j*k+i]=(int8_t)r;                /* col-major k x n */
}
/* Fractional-CRT reconstruct (no bignum division): P/prod = frac(sum_r res_r*c_r),
 * c_r = (y_r/M_r) as double-double, Neumaier-compensated sum for ~full accuracy.
 * chi/clo: hi+lo of y_r/M_r.  prodDD = prod as (phi,plo) doubles.
 * C = signed_frac * prod / (scaleA_i*scaleB_j). */
__device__ __forceinline__ void twosum(double a,double b,double&s,double&e){
    s=a+b; double bb=s-a; e=(a-(s-bb))+(b-bb);
}
__device__ __forceinline__ void twoprod(double a,double b,double&p,double&e){
    p=a*b; e=fma(a,b,-p);
}
/* Full double-double fractional-CRT recon: S=sum res_r*(chi+clo) in DD,
 * frac(S) keeps ~53 bits below the integer part -> P/prod to ~fp64.
 * C = frac * prod / (sA*sB), prod as (phi,plo) DD. */
__global__ void recon_kernel(const int*const*Cint,int m,int n,int t,
                             const int*M,const double*chi,const double*clo,
                             double phi,double plo,
                             const double*scaleA,const double*scaleB,double*C,size_t ldc){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=m) return;
    for(int j=blockIdx.y*blockDim.y+threadIdx.y;j<n;j+=gridDim.y*blockDim.y){
        double Shi=0.0,Slo=0.0;
        for(int r=0;r<t;r++){
            int Mr=M[r];
            int res=Cint[r][(size_t)j*m+i]%Mr; if(res<0)res+=Mr;
            double ph,pe; twoprod((double)res,chi[r],ph,pe);
            pe=fma((double)res,clo[r],pe);
            double s1,e1; twosum(Shi,ph,s1,e1);
            Shi=s1; Slo+=e1+pe;
        }
        /* normalize + take signed fractional part in DD */
        double s2,e2; twosum(Shi,Slo,s2,e2); Shi=s2; Slo=e2;
        double ni=nearbyint(Shi);
        double fhi,fe; twosum(Shi,-ni,fhi,fe); double flo=fe+Slo;
        /* P = (fhi+flo)*(phi+plo) in DD, then /(sA*sB) */
        double ph,pe; twoprod(fhi,phi,ph,pe);
        pe=fma(fhi,plo,pe); pe=fma(flo,phi,pe);
        double P=ph+pe;
        C[(size_t)j*ldc+i]=P/(scaleA[i]*scaleB[j]);
    }
}
__global__ void maxdiff_kernel(const double*a,const double*b,size_t n,double*out){
    __shared__ double sh[256]; double mx=0;
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){
        double d=fabs(a[i]-b[i]); if(d>mx)mx=d; }
    sh[threadIdx.x]=mx; __syncthreads();
    for(int o=128;o;o>>=1){ if(threadIdx.x<o&&sh[threadIdx.x+o]>sh[threadIdx.x])sh[threadIdx.x]=sh[threadIdx.x+o]; __syncthreads(); }
    if(!threadIdx.x){ unsigned long long*u=(unsigned long long*)out; atomicMax(u,__double_as_longlong(sh[0])); }
}

static long long modinv(long long a,long long m){ long long g=m,x=0,x1=1,a0=a%m; if(a0<0)a0+=m;
    long long r=a0; while(r>1){ long long q=r/g; long long t=g; g=r-q*g; r=t; t=x; x=x1-q*x; x1=t; } return (x1%m+m)%m; }

int main(int argc,char**argv){
    size_t m=124416,n=1024,k=2048;
    if(argc>3){m=atoll(argv[1]);n=atoll(argv[2]);k=atoll(argv[3]);}
    printf("Ozaki-II own  m=%zu n=%zu k=%zu\n",m,n,k);
    cublasHandle_t h; CB(cublasCreate(&h));
    cublasLtHandle_t ltH; CB(cublasLtCreate(&ltH));
    void*ltWs; CK(cudaMalloc(&ltWs,(size_t)256<<20));
    double *A,*B,*C,*Cr; CK(cudaMalloc(&A,m*k*8)); CK(cudaMalloc(&B,k*n*8));
    CK(cudaMalloc(&C,m*n*8)); CK(cudaMalloc(&Cr,m*n*8));
    fill_kernel<<<2048,256>>>(A,m*k,1); fill_kernel<<<2048,256>>>(B,k*n,2);
    double one=1,zero=0;
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    /* fp64 ref */
    CB(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&one,A,m,B,k,&zero,Cr,m));
    CK(cudaEventRecord(e0));
    for(int r=0;r<3;r++) CB(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&one,A,m,B,k,&zero,Cr,m));
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1));
    printf("cublasDgemm   %.3f ms  %.1f TF\n",ms/3,2.0*m*n*k/(ms/3e3)/1e12);

    double*scaleA; CK(cudaMalloc(&scaleA,m*8)); double*scaleB; CK(cudaMalloc(&scaleB,n*8));
    int8_t*Amod[TMAX],*Bmod[TMAX]; int*Cint[TMAX];
    for(int r=0;r<TMAX;r++){ Amod[r]=Bmod[r]=nullptr; Cint[r]=nullptr; }
    const int*hCintPtr[TMAX]; const int**dCintPtr; CK(cudaMalloc(&dCintPtr,TMAX*sizeof(int*)));
    int *dM; double*dChi,*dClo; CK(cudaMalloc(&dM,TMAX*4)); CK(cudaMalloc(&dChi,TMAX*8)); CK(cudaMalloc(&dClo,TMAX*8));
    /* descending primes < 127, pairwise coprime */
    int primes[]={127,113,109,107,103,101,97,89,83,79,73,71,67,61,59,53,47,43,41,37,31,29,23,19};

    int sset[]={25,33,41,49}; for(int si=0;si<4;si++){ int sbits=sset[si];
      int tset[]={9,11,13,15,18,20}; for(int ti=0;ti<6;ti++){ int t=tset[ti];
        std::vector<int> M(primes,primes+t);
        /* Garner constants */
        unsigned __int128 prod=1; for(int r=0;r<t;r++) prod*=(unsigned)M[r];
        /* direct-CRT weights: N_r=prod/M_r, y_r=inv(N_r mod M_r), c_r=y_r/M_r */
        std::vector<double> chi(t),clo(t);
        for(int r=0;r<t;r++){
            unsigned __int128 Nr=prod/(unsigned)M[r];
            long long Nmod=(long long)(Nr%(unsigned __int128)M[r]);
            long long y=modinv(Nmod,M[r]);
            /* c_r = y/M_r as double-double */
            double hi=(double)y/(double)M[r];
            double lo=fma((double)y,1.0/(double)M[r],-hi);  /* residual */
            /* refine lo with exact remainder: y/M = hi + (y - hi*M)/M */
            double rem=fma(-hi,(double)M[r],(double)y);
            lo=rem/(double)M[r];
            chi[r]=hi; clo[r]=lo;
        }
        /* prod as double-double phi+plo (phi=nearest double, plo=residual) */
        double phi=(double)(unsigned long long)(prod>>64)*18446744073709551616.0
                 +(double)(unsigned long long)prod;
        double plo;
        { unsigned __int128 pr=(unsigned __int128)phi; /* phi rounds to <2^128 */
          if(pr<=prod){ plo=(double)(long long)(unsigned long long)(prod-pr); }
          else        { plo=-(double)(long long)(unsigned long long)(pr-prod); } }
        CK(cudaMemcpy(dM,M.data(),t*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dChi,chi.data(),t*8,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dClo,clo.data(),t*8,cudaMemcpyHostToDevice));
        for(int r=0;r<t;r++){ if(!Amod[r]){ CK(cudaMalloc(&Amod[r],k*m)); CK(cudaMalloc(&Bmod[r],k*n));
                                            CK(cudaMalloc(&Cint[r],m*n*4)); }
            hCintPtr[r]=Cint[r]; }
        CK(cudaMemcpy(dCintPtr,hCintPtr,t*sizeof(int*),cudaMemcpyHostToDevice));

        rowscale_kernel<<<(m+127)/128,128>>>(A,m,m,k,sbits,scaleA);
        colscale_kernel<<<(n+127)/128,128>>>(B,k,k,n,sbits,scaleB);
        dim3 bl(16,16);
        auto quant=[&](){ for(int r=0;r<t;r++){
            quantA_kernel<<<dim3((m+15)/16,(k+15)/16),bl>>>(A,m,m,k,scaleA,M[r],Amod[r]);
            quantB_kernel<<<dim3((k+15)/16,(n+15)/16),bl>>>(B,k,k,n,scaleB,M[r],Bmod[r]); } };
        /* cublasLt int8 TN with heuristic algo (shape fixed across moduli) */
        cublasLtMatmulDesc_t opI; CB(cublasLtMatmulDescCreate(&opI,CUBLAS_COMPUTE_32I,CUDA_R_32I));
        cublasOperation_t ta=CUBLAS_OP_T, tb=CUBLAS_OP_N;
        CB(cublasLtMatmulDescSetAttribute(opI,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
        CB(cublasLtMatmulDescSetAttribute(opI,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
        cublasLtMatrixLayout_t la,lb,lc;
        CB(cublasLtMatrixLayoutCreate(&la,CUDA_R_8I,k,m,k));
        CB(cublasLtMatrixLayoutCreate(&lb,CUDA_R_8I,k,n,k));
        CB(cublasLtMatrixLayoutCreate(&lc,CUDA_R_32I,m,n,m));
        cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
        size_t lwz=(size_t)256<<20; CB(cublasLtMatmulPreferenceSetAttribute(pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&lwz,sizeof(lwz)));
        cublasLtMatmulHeuristicResult_t hr[8]; int nh=0;
        CB(cublasLtMatmulAlgoGetHeuristic(ltH,opI,la,lb,lc,lc,pref,8,hr,&nh));
        if(!nh){ printf("no int8 algo\n"); return 1; }
        auto gemms=[&](){ int io=1,zo=0; for(int r=0;r<t;r++)
            CB(cublasLtMatmul(ltH,opI,&io,Amod[r],la,Bmod[r],lb,&zo,
               Cint[r],lc,Cint[r],lc,&hr[0].algo,ltWs,lwz,0)); };
        auto recon=[&](){ dim3 rb(32,8); recon_kernel<<<dim3((m+31)/32,32),rb>>>(
            dCintPtr,m,n,t,dM,dChi,dClo,phi,plo,scaleA,scaleB,C,m); };
        /* correctness */
        quant(); gemms(); recon(); CK(cudaDeviceSynchronize());
        double*d_md; CK(cudaMalloc(&d_md,8)); CK(cudaMemset(d_md,0,8));
        maxdiff_kernel<<<512,256>>>(C,Cr,m*n,d_md); double md; CK(cudaMemcpy(&md,d_md,8,cudaMemcpyDeviceToHost)); cudaFree(d_md);
        /* timing: full (quant+gemm+recon) and gemm+recon only (A/B reused) */
        CK(cudaEventRecord(e0)); for(int it=0;it<3;it++){quant();gemms();recon();} CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float msf; CK(cudaEventElapsedTime(&msf,e0,e1));
        CK(cudaEventRecord(e0)); for(int it=0;it<3;it++){gemms();recon();} CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float msg; CK(cudaEventElapsedTime(&msg,e0,e1));
        double flops=2.0*m*n*k;
        printf("s=%d t=%2d  full %.2fms(%6.1fTF)  gemm+recon %.2fms(%6.1fTF)  maxdiff=%.3e\n",
               sbits,t,msf/3,flops/(msf/3e3)/1e12,msg/3,flops/(msg/3e3)/1e12,md);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatmulDescDestroy(opI);
      }
    }
    return 0;
}
