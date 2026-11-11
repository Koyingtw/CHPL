#!/usr/bin/env python
"""
EmuGEMM (Emu-II / crt_kernel_dgemm) vs gemmul8 3.1.0 baseline (g8_baseline_probe),
at CHPL production shapes: K=NB=4096, N=G8_NT=3072, M swept.

Prerequisites: git@github.com:NTHU-SC/EmuGEMM.git built into a conda env (see
environment.yml + install.sh step 1 only -- `pip install -e EmuGEMM/`), then copy
EmuGEMM/EmuGEMM/benchmarks/bench_utils.py next to this script (its BSD-3 license
allows vendoring; not duplicated here to avoid tracking a second copy of upstream
code). See RESULTS.md / docs/EXPERIMENTS.md "EmuGEMM vs gemmul8 3.1.0" entry for
the env path used originally ($WORK/conda_envs/emu_gemm) and results.

Reports:
  - wall time / TFLOPS for EmuGEMM's full pipeline (fused slice + s serial
    per-modulus GEMM+mod kernels + fused CRT merge), at num_moduli=11
    (matching CHPL2D_G8_MODULI champion config).
  - relative Frobenius-norm accuracy vs a true FP64 cuBLAS reference,
    to compare against gemmul8 3.1.0's HPL-validated residual behavior.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
import bench_utils as bu

K = 4096
N = 3072
Ms = [8192, 16384, 32768, 57600, 115200]
NUM_MODULI = 11

device = "cuda"
torch.manual_seed(0)

def make_hpl_like(m, k, scale=1.0):
    # HPL-ish matrices: uniform [-0.5,0.5), moderate magnitude (matches CHPL's
    # trailing-update operands after prior scaling stages -- magnitude itself
    # doesn't matter for a relative-error accuracy check).
    return (torch.rand(m, k, dtype=torch.float64, device=device) - 0.5) * scale

print(f"EmuGEMM crt_kernel_dgemm vs gemmul8 3.1.0: K={K} N={N} moduli={NUM_MODULI}")
print(f"{'M':>8} | {'ms/call':>10} {'TFLOPS':>8} | {'rel_err':>10} {'eff_bits':>9}")

for m in Ms:
    A = make_hpl_like(m, K)
    B = make_hpl_like(K, N)

    # warmup (JIT compiles + caches cubin on first call per shape/config)
    C = bu.crt_kernel_dgemm(A, B, num_moduli=NUM_MODULI)
    torch.cuda.synchronize()

    iters = 5
    t0 = time.perf_counter()
    for _ in range(iters):
        C = bu.crt_kernel_dgemm(A, B, num_moduli=NUM_MODULI)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) / iters * 1e3
    tflops = 2.0 * m * N * K / (ms / 1e3) / 1e12

    # accuracy vs true FP64 reference
    C_ref = A @ B
    rel_err = (torch.linalg.norm(C - C_ref) / torch.linalg.norm(C_ref)).item()
    eff_bits = -torch.log2(torch.tensor(rel_err)).item() if rel_err > 0 else float('inf')

    print(f"{m:8d} | {ms:10.4f} {tflops:8.2f} | {rel_err:10.3e} {eff_bits:9.2f}")
