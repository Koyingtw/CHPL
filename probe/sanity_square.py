import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
import bench_utils as bu

for K in [2048, 4096, 8192]:
    M = N = 8192
    A = (torch.rand(M, K, dtype=torch.float64, device="cuda") - 0.5)
    B = (torch.rand(K, N, dtype=torch.float64, device="cuda") - 0.5)
    C = bu.crt_kernel_dgemm(A, B, num_moduli=12)
    torch.cuda.synchronize()
    iters=5
    t0=time.perf_counter()
    for _ in range(iters): C = bu.crt_kernel_dgemm(A, B, num_moduli=12)
    torch.cuda.synchronize()
    t1=time.perf_counter()
    ms=(t1-t0)/iters*1e3
    tflops=2.0*M*N*K/(ms/1e3)/1e12
    print(f"EmuGEMM square M=N={M} K={K} moduli=12: {ms:.4f} ms, {tflops:.2f} TFLOPS")
