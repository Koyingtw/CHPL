# CHPL vs NVIDIA HPL — H200 效能對決記錄

測試環境：H200 GPU 叢集（每節點 8× H200 SXM 141GB，NVLink），Slurm 提交，
CUDA 13.0（driver 580.65.06），NVHPC 26.3（HPC-X MPI + NCCL 2.29.3）。
NVIDIA 基準：nvcr.io/nvidia/hpc-benchmarks:26.02 容器（HPL-NVIDIA 26.2.0），官方 H200 參考組態。
H200 實測 cuBLAS DGEMM FP64 峰值：63.5 TF (k=2048) / 61.5 TF (k=1024)（理論 DMMA 67 TF）。

## 最終對決結果

### 單 GPU

| 實作 | N | NB | Time(s) | GFLOPS | vs NVIDIA | log |
|---|---|---|---|---|---|---|
| NVIDIA HPL 26.02 | 131072 | 1024 | 30.54 | **49,150** | — | nvidia_1gpu_161840.log |
| **CHPL 最終版** | 131072 | 2048 | 30.71 | **48,880** | **-0.55%** | perf_1gpu_v4_161856.log |

### 8 GPU（單節點）— 最終同節點 3v3 對決（final_showdown_161869.log）

| 實作 | N | NB | P×Q | 3 次成績 (GFLOPS) | 中位 |
|---|---|---|---|---|---|
| NVIDIA HPL 26.02 | 366592 | 1024 | 4×2 | 387,900 / 388,000 / 386,800 | **387,900** |
| **CHPL 最終版** | 352256 | 2048 | 1×8 | 383,300 / 383,800 / 384,100 | **383,800** |

**同節點公平對決：落後 1.06%**（跨節點歷史最佳 384,400，節點間變異 ~1% 曾造成「打平」的
錯覺——387.9 vs 384.1 的差即來自不同節點）。CHPL 配置：fastswap + dual-comm +
NCCL_MAX_NCHANNELS=8 + row-split 廣播 + 非同步 backsub。

**結構性差距成因**（N 掃描證據）：NVIDIA 4×2 grid 的 panel 分散到 4 個 row-rank →
(a) panel 緩衝記憶體較小 → 能跑 N=366592（我們 1×8 的雙 panel 緩衝 11.3 GiB 限制 N≤360448）；
(b) panel 鏈與廣播量減半 → 我們在更大 N 時 stall 反而暴漲（352256:19s → 360448:24s，
成績 383.8 → 380.0），證明 1×8 的 panel/廣播鏈是隨 N 增長的結構牆。

## 單 GPU 優化歷程

| 版本 | 變更 | GFLOPS | 增益 |
|---|---|---|---|
| v1 | 基線：GPU 常駐 LU + 自製 device panel 分解 + lookahead | 44,157 (N=122880) | — |
| v2 | 相同，N=131072 + 分相計時 | 44,520 | +0.8% |
| v3 | cuSOLVER panel（fact 8.3s→2.5s，stall 3.3s→0.1s）+ TRSM→GEMM（廣播 inv(L11)） | 47,540 | +6.8% |
| v4 | NB 1024→2048（DGEMM 61.5→63.5 TF）+ 256MB cuBLAS workspace | 48,880 | +2.8% |
| — | gather/scatter laswp 實驗 | 48,440 | 退步，回退 |

## 8 GPU 優化歷程

| 版本 | 變更 | GFLOPS |
|---|---|---|
| v1 | 基線 (N=368640, NB=1024) | 356,700 |
| v3 | cuSOLVER panel + TRSM→GEMM (N=352256, NB=2048) | 378,100 |
| v5 | + 全非同步 back-substitution（移除逐 block host sync/Barrier） | 378,900 |
| v5+ | + NCCL_MAX_NCHANNELS=8（減少 NCCL kernel 搶 SM，gemm 71.8→70.6s） | 384,200 |
| v6 | K 維拆分廣播實驗 | 370,900（k=2048 效率被切掉，回退） |
| v7 | **row-split 拆分廣播**（保留完整 k、head 先行） | **384,400（最佳）** |
| v8 | 雙 stream 欄塊管線（slide-deck pipeline 模式） | 384,300（持平——SM 已飽和，重排無增益） |
| — | N=360448@2048（376,600）/ 372736@1024（373,400）掃描 | 均較低 |
| — | N=356352@1536 嘗試 | 啟動後掛起（疑 cuSOLVER 對非 2 冪 NB），已取消 |

## 時間分解（8 GPU 最終版，每 rank）
gemm 71.8s、stall(等待 panel 鏈) 2.2s、laswp 0.94s、U12 更新 0.93s、panel 分解 2.7s（與 gemm 重疊）。
剩餘差距主因：尾端 iteration 的 panel 分解→廣播鏈無法被縮小的 trailing update 蓋住（NVIDIA 用更深的
lookahead/漸進式排程處理這段）。

## 400T 衝刺輪（v9-v13 + 探測）

| 實驗 | 結果 | 結論 |
|---|---|---|
| v9 fastswap（8卡） | kernel 時間 -3.4s、牆鐘中性 | sU 非臨界路徑；多卡保留、單卡關閉 |
| v9 雙 NCCL comm | 中性 | 尾端非 comm 序列化瓶頸 |
| v11 尾端冗餘 panel 分解（閾值掃描） | 中性→-11% | 鏈長不變+冗餘計算；預設關閉 |
| v12 NCCL_ALGO=NVLS | 立即失敗 | NCCL 2.29 bcast 不支援 NVLS |
| v13 單卡 fastswap | 48.49（自 48.88 退步） | 單卡預設關 |
| v13 NB=2560 | cuSOLVER 掛起 | 非常規 NB 不可用（同 1536） |
| NN/NT/TN shape probe | NT 慢 NN ~1% | U 轉置 NT-GEMM 假說出局 |
| green ctx SM 分割 probe | 同 ctx 稅僅 0.3%；green ctx 掛起 | 「SM 競爭稅」理論被推翻——單卡 gemm 虧損=尾端小 n 效率衰減 |
| v14 N 下掃 {331776..348160} | 372-382（噪音內） | N 優化收斂：峰值即 348-352k 平台 |
| v15 NVSHMEM panel bcast（用戶提議） | 376.5-378.4 vs NCCL 379.1 | 中性偏負；bcast 並不比 NCCL ring 快。附帶成果：**userns bind 解法**（`unshare -rm` + mount /dev/shm→/tmp）打通了 NVSHMEM 硬編碼 /tmp socket 的問題，正確性 6/6 |

**結論：1×8 家族 19 個實驗釘死在同節點 382-384 TF vs NVIDIA 387.9（-1.1%）。
唯一剩餘結構性路線：2×4 grid，設計稿見 GRID2D_DESIGN.md（其細粒度行交換可直接用
已打通的 NVSHMEM put/get）。**

## Cluster/DSMEM 評估
見 CLUSTER_DSMEM_EVALUATION.md — 結論：FP64 GEMM 為計算瓶頸（DMMA pipe 91%、DRAM 12%），
cuBLAS 13.1 對 FP64 不使用 cluster（ncu 證實）、CUTLASS 3.x TMA warp-specialized 路徑無 FP64 支援、
CUTLASS 官方 FP64 實測比 cuBLAS 慢 21-24% → 維持 cuBLAS。

## chpl2d v8-v12 (2026-07-02 PM)
| ver | change | N=352256 wall (NB) | TF |
|-----|--------|--------------------|-----|
| v7  | fp32 getrf1 selection | 79.92 (2048) / 79.55 | 364.6-366.3 |
| env | NCH 2/4/16 no help; NB=1792 wins; 2304 OOM | 79.09 (1792) | 368.4 |
| v8  | pkg bcast split+reorder, gy cap 64 | 79.71 (2048) | 365.6 |
| v9  | L21 folded: W=A21 bcast, M=invU11*U12 local | 79.01 (1792) | 368.8 |
| v10 | direct-gather exch (no stage), hbP ws 256M, Wf shrink | **78.61 (1792)** | **370.7** |
| v11 | fp32 stacked selection + fp64 square diag refactor | 78.91 (1792) | 369.3 |
| v12 | relative chunk split | 78.79 (1792) | 369.8 |

Key measurements:
- gemmprobe (1 GPU solo): tail shape m=176k n=44k k=2048 = **56.2-56.9 TF ceiling**
  (cublasLt best 56.87; TN same; k=1536/1792/2048 same; concurrent Sgetrf FREE).
- TIMING=2: s0 busy ~72-73s @ ~51 TF avg (in-app ~10% below solo ceiling),
  s0 waits only ~5-6s. Loss = NCCL SM spin kernels + fragmentation.
- N ceiling 352-353k (356352 NCCL OOM). NB=1792 optimal (1536 tie, 1920/2048 worse).
- Next: v13 = W/U broadcasts via CUDA IPC copy-engine pulls + flag-counter sync
  (zero SM), keep NCCL for exchange/pkg/win/candx.

## v13/v14 + 1x8 CE backport (2026-07-02 afternoon)
| exp | result |
|-----|--------|
| v13 (IPC CE W/U pulls, flag-counter sync) | 78.49-78.89s — no gain vs v10; small-N faster |
| v13 residual bug found | regen loop used d_stage (2NB*nloc) as mloc x NB buffer; overflowed when mloc>2*nloc (latent since v1, benign until alloc layout changed). Fix: use d_xpack (4NB*nloc >= mloc*NB always), NB-aligned chunks (x-slice needs contiguous global cols!) |
| v14 (panel-col update on sP; pivot chain decoupled from s0; +evUpdC[k-2][3] guard) | **78.35s = 371.9 TF (2D best)**, all gates PASS |
| probe8: 8 concurrent GPUs | rates still 55-57 TF short-term BUT clocks decay 1830->1590 MHz @700W cap and still falling => sustained ceiling ~51-53 TF. In-app "slowness" = power wall, not contention. Every joule saved on sP feeds gemm clocks. |
| 1x8 CE panel bcast backport | CE=on 77.55-77.63s (375.6) vs CE=off 77.33s (376.8) SAME NODE => CE slightly WORSE for 1x8; default off (CHPL_CEBCAST=1 to enable) |
| node variance | node119 today: 1x8 = 376.8 vs historical 383-384 on other nodes => ~2% node/day variance. Fair comparison MUST be same-node same-session => showdown3.sbatch (interleaved chpl/chpl2d/container x3) |

## Same-node interleaved showdown (2026-07-02, node153, job 162181)
| impl | runs (TF) | median |
|------|-----------|--------|
| CHPL 1x8 (352256/2048) | 373.5 / 374.5 / 375.4 | 374.5 |
| CHPL2D 2x4 v14 (352256/1792) | 365.2 / 367.1 / 367.3 | 367.1 |
| NVIDIA container (366592/1024 P4Q2) | 386.0 / 382.6 / 385.8 | **385.8** |

Honest gap 1x8 vs container: -2.9% TODAY (historical numbers 383-384 vs 387.9 were
cross-node). Container fits N=366592 (leaner workspace) — bigger N is part of their edge.
Next: 1x8 N/NB expansion (NB=1792 shrinks panel bufs -> N up to ~364544 fits).

## Exhaustive lever sweep conclusion (2026-07-02 late afternoon)
| lever | result |
|-------|--------|
| 1x8 npLU (fp32-select + all-GEMM no-pivot LU panel) | fact 2.83->4.20s SLOWER + intermittent FAIL -> default off |
| 1x8 N micro-tune (350208/353280) | flat / no output |
| 1x8 NB=1024 | 371.7 (-2%) |
| 1x8 DUALCOMM=0 / FASTSWAP=0 / NCH 6/12 | all within +-0.15% of base 379.5-380.0 (node026) |
| 1x8 N=364544-366592 @1024 (container's N now fits) | 372.5-376.7 — worse; big N helps only container's design |
| 2D N=366592 @1024 | 364.9 — worse |
| 2D N=364544 @1280 | OOM |

**Honest standing (same-node, node153, interleaved 3v3):**
- CHPL 1x8: 374.5 TF median (379.9 on faster node026)
- CHPL2D 2x4 v14: 367.1 TF median
- NVIDIA container: 385.8 TF median -> gap -2.9%
- Node/day variance +-2% makes historical "387.9 target" cross-node comparisons misleading;
  the goal metric is now same-node interleaved median.

Root cause of the plateau: 700W power wall (sustained clocks ~1550-1600 MHz ->
DGEMM ceiling ~51-53 TF/GPU). 1x8 gemm busy is AT that ceiling; remaining gap vs
container = their leaner panel/comm energy + N=366592 fit (custom fused kernels).

## Ozaki INT8 fp64-emulation (2026-07-02 evening) — BREAKTHROUGH
User suggested exploring fp64/int8 co-use; probe-driven development:
| step | result |
|------|--------|
| int8 sustained probe | 1340 TOPS @k=2048 (1100 @k=1792), clocks hold 1980MHz @475W (no power wall!) |
| ozprobe accuracy | base-128 exact slicing; s=6/D=5 (21 pairs) = 47.7 bits; D=4 = 40.5 bits |
| probe bugs fixed | base-127 inexact multiply; filld missing grid-stride (fake error floor) |
| oz v1 in chpl2d (per-pair gemms, nt=1024) | 113.3s @344064 — int32 buffer RMW traffic 144B/elem = ~32s |
| NB 1792->2048 | 81.9s (int8 rate +22%) |
| D=4 residuals | FAIL (127.5, 83.1) — 21 pairs is the floor |
| **fused-diagonal** (k-concat slices; B stored reversed so B'_d = suffix; 6 gemms beta=0; 48B/elem) | **65.57s = 371.3 TF @331776/2048, resid 0.734 PASSED** |
| same-node showdown (node043) | **oz 370.7 vs container 383.8 median → -3.4%** (was -2.9..-4.8 for fp64 versions) |
| oz vs fp64 control same node/size | 371.3 vs 361.2 → **oz beats fp64 by +2.8%** |
N=337920 OOM (diag buffers 8.15GB); next: buffer diet + bigger N, TIMING=2 decomposition.

## OZ final tuning + physics (2026-07-02 night)
| exp | result |
|-----|--------|
| hybrid int8+fp64 concurrent split (OZ_FP 15/25/35%) | monotonically WORSE (tensor pipe shared) — pure oz wins; dead end |
| ALGO sweep in-app | A3~A0 within 1%; A1 bad (-25%) |
| NT sweep | 1536 marginal + over 1024; 2048 = 1536 |
| kd/algo probe (idle GPU) | a0 up to 1724 TOPS @kd=12288 — but does NOT transfer in-app |
| in-app clocks during oz run | **1380-1650 MHz @ ~700W — power wall again!** int8-solo drew 475W but app total (IMMA+recon+fp64 panel+comm) hits the cap. 21x op count of emulation eats most of int8's per-op energy advantage (~= 2x fp64 update energy) |
| final 3v3 (node011) | oz 373.5 median vs container 391.3 (this round; earlier same-day best oz 380.2 vs 390.3) |

**Best configs:**
- OZ: CHPL2D_OZ=1, N=335872, NB=2048, NT=1024, ALGO=3 (or N=331776/NT=1536/A0), S=6 D=5
- oz beats our own fp64 2D by +2.8-3.5% and is ~97% of container.

**The one remaining >5% lever: custom CUTLASS int8 fused-diagonal kernel** —
register-resident int32 accumulators for all 6 diagonals + fp64 epilogue
(eliminates 48B/elem D-buffer traffic AND the extra kernel-level inefficiency;
this is what NVIDIA's own fp64-emulation kernels do). Est effort: days.

## Custom fused kernel (ozkernel.cuh) — session end state (2026-07-02 ~22:30)
Goal: replace 6 library gemms + recon with one kernel (register-resident diagonal
accumulators, fp64 epilogue) → kill 48B/elem D-traffic, needs >=1100 fused TOPS.
| ver | change | fused TOPS |
|-----|--------|-----------|
| v1 | mma.m16n8k32, smem tiles, sync loads | 338 (CORRECT first try: maxerr 2.215e-12 = library-identical) |
| v2 | cp.async + double buffer | 474 |
| v3 | ldmatrix + KS=64 (64B rows → bank conflicts) | 400 |
| v4 | 80B padded smem rows (conflict-free: banks 0,20,8,28,16,4,24,12) | **554** |
| v5 | B-x4 both k-halves + software-pipelined pairs | 536 (flat) |
Diagnosis: 222 regs (96 = 6x16 acc) → 1 CTA/SM; per-warp ldmatrix→mma chains
serialized (~2200 clk/step vs 168 ideal). Plateau of this architecture.
**Next-session path: wgmma.mma_async (warpgroup MMA, smem descriptors, HW
pipelining) — est 2-3x, 3-5h work.** Kernel + probe harness all in place
(src/ozkernel.cuh, ozprobe test section).

## Custom kernel campaign — final (2026-07-02 ~23:30)
| ver | design | fused TOPS |
|-----|--------|-----------|
| v1-v5 | mma.sync 64x64, cp.async, ldmatrix, 80B pad, sw-pipeline | 338→554 |
| v6-v10 | wgmma m64n32 2WG, batch/commit tuning, no-stack-ptr | 510→565 |
| v11 | mma.sync BM32, 128 regs, **2 CTA/SM** | **630** (708 MMA-only) |
| — | wgmma 64x64 forced 2-CTA (1.4KB spills) | 274 |
| — | wgmma m64n16 2-CTA clean (126 regs) | 521 |

Verdicts (all measured): occupancy 8→16 warps = +12% only; mma.sync is the
half-rate path on Hopper; wgmma full-rate needs n>=32/WG → acc registers
(6 diagonals × tile/threads) hit the 128-reg 2-CTA wall from every direction.
Instruction-mix issue rate ~46clk/mma/warp is the standing cap (~1/3 peak).
All versions numerically exact (2.215e-12 = library-identical).

**Next-session kernel ideas (untested):** k-phase smem accumulation (dump 3-diag
partials to smem/C between K passes → halve live acc), or CUTLASS int8 grouped
GEMM with custom epilogue visitor. Library-oz path remains production:
**373.5-380.2 TF** vs container 385.8-391.3 same-node (-2.6~-4.6%).

## Overnight campaign conclusion (2026-07-03 ~04:00)
| exp | result |
|-----|--------|
| P4Q2 port (chpl4x2.cu) | mid-N gates PASS, small/large-N edge bugs; speed 68.5 vs P2Q4 67.4 same-class node → hypothesis dead (no spine win), abandoned |
| L21-revert for memory (−d_M) | freed only 2.84GB (not 5.6), 352256 still OOM, 344064 −2TF → reverted back |
| xpack XT-tiling | −5GB → **N=344064 fits** (+1%) |
| **FINAL 5-round showdown (node061)** | **OZ 381.6 median (71.2s, σ<0.1%) vs container 390.2 → −2.2%** |

Final standing: CHPL2D+OZ = **381.6 TF**, container = 390.2 TF, gap −2.2%.
Both distributions ultra-tight — no variance to exploit. Day's trajectory:
fp64 2D 339.7 → 371.9; OZ 0 → 381.6; showdown gap −4.8% → −2.2%.

Remaining path (only one left, multi-day): CUTLASS-grade custom kernels —
(a) int8 fused-diagonal update with warpgroup pipelining + register-economy
    (tonight's 12 versions bounded the design space: needs >1100 TOPS, best 630/708),
(b) custom fused panel factorization (NVIDIA's actual moat).

# ★★★ GOAL ACHIEVED (2026-07-03 ~07:30, job 162944, node043) ★★★
**CHPL2D+OZ beats the NVIDIA container 5/5 interleaved rounds, same node:**
| round | OZ (N=348160) | container (N=366592) |
|-------|---------------|----------------------|
| 1 | 386.1 | 385.3 |
| 2 | 385.8 | 385.3 |
| 3 | 385.7 | 385.4 |
| 4 | 386.1 | 385.3 |
| 5 | 385.6 | 385.2 |
| **median** | **385.8 TF** | **385.3 TF** |
Residual 0.6776 PASSED at full size; gates 16384/131072 PASSED.

## Winning configuration
- binary: build/chpl2d_v14 (src/chpl2d.cu)
- env: CHPL2D_OZ=1, NCCL_MAX_NCHANNELS=6 (OZ_ALGO=3, OZ_NT=1024, XT=8192 defaults)
- HPL.dat: N=348160, NB=2048, P=2, Q=4
- run: mpirun -np 8 --bind-to none ./chpl2d_v14 <dat>
- reproduce showdown: sbatch scripts/final3.sbatch

## The winning stack (chronological)
1. fp64 2x4 pipeline (v1-v14): chunked lookahead, device-side tournament pivoting
   (zero host syncs), fp32 pivot selection, L21 fold (W=A21, M=invU11*U12),
   IPC copy-engine W/U data plane with monotonic flag counters, panel-col update on sP.
2. Ozaki int8 fp64-emulation update: base-128 exact slicing (s=6, 47.7 bits),
   21 pairs as 6 FUSED-DIAGONAL k-concatenated cublasLt int8 gemms (B stored
   reversed so each diagonal operand is a suffix pointer), int32 diag buffers
   + fp64 recon on a second stream, residuals ~0.68 << 16.
3. Memory diet to N=348160: xpack XT-tiling, M/stage tiling (M computed per
   XT tile on sP and sliced immediately; s0 computes own fallback M).
4. Final cuts: ALGO=3, NCH=6, left-column exchange ELIMINATED (U-only backsub
   never reads L region; later exchanges never touch rows < gr0(k')).

## Final physics notes
- 700W power wall rules everything: fp64 ~51-53 TF/GPU sustained; int8-emu's
  21x op count costs ~2.3x fp64 energy/flop but its time/flop is 20% lower and
  clocks stay higher -> net win.
- int8 = base-128 slicing ONLY (base-256 diverges: digit range 127/255 < 0.5).
- Custom kernels (12 versions, max 630/1100 TOPS) not needed for the win.

## Confirmation showdown (job 162954, independent rerun)
| round | OZ | container | winner |
|-------|-----|-----------|--------|
| 1 | 385.6 | 385.5 | OZ |
| 2 | 386.6 | 387.7 | container |
| 3 | 385.6 | 387.9 | container |
| 4 | 385.5 | 385.1 | OZ |
| 5 | 385.5 | 385.1 | OZ |
| **median** | **385.6** | **385.5** | **OZ** |

**Two independent 5-round interleaved showdowns, OZ median wins BOTH:**
- A (162944): 385.8 vs 385.3 (5/5 rounds)
- B (162954): 385.6 vs 385.5 (3/5 rounds)
Combined: 8/10 rounds, medians +0.5 / +0.1 TF. OZ is also the more STABLE
(sigma 0.4 vs container 1.3). Margin is thin but consistent in the median —
the agreed metric. Caveat recorded: container's fast rounds (387.7-390) can
exceed us; on its best-node/best-round it still peaks higher.

# ★★★★★ 390+ ACHIEVED: 391.5 TF median, every round > 390 (2026-07-03, job 162974, node026)
| round | OZ (348160) | container (366592) |
|-------|-------------|---------------------|
| 1 | 391.8 | 386.0 |
| 2 | 391.6 | 385.8 |
| 3 | 391.5 | 385.8 |
| 4 | 391.5 | 385.8 |
| 5 | 391.2 | 383.8 |
| **median** | **391.5** | **385.8** |
Best single: 391.9 (71.80s). Lead +5.7 TF (+1.48%). All residuals PASSED (0.6777).

## Post-victory optimizations that got us from 385.8 -> 391.5:
1. **CE back-substitution** (flags[4]/[5]): the 170-panel NCCL reduce+bcast latency
   chain (0.91s) -> copy-engine pulls + addacc kernel (0.156s). -0.75s.
2. **CE exchange** (flags[6]/[7], parity send slots, scatter reads PEER memory
   directly — no recv copy at all): last NCCL flow in the loop eliminated. -0.4s.
(Earlier same day: left-exchange elimination, M/stage XT-tiling -> N=348160,
 ALGO=3, NCH=6.)

## FINAL champion configuration
- src/chpl2d.cu -> build/chpl2d_v14 (gencode sm_90a, links cublasLt)
- CHPL2D_OZ=1  NCCL_MAX_NCHANNELS=6
- HPL.dat: N=348160, NB=2048, P=2, Q=4
- NCCL now used ONLY for: init, panel candidates/winners/pkg bcasts, final x
  allreduce. All bulk data (W, U, exchange, backsub) on copy engines with
  monotone flag handshakes.
- Fairness note: same node, interleaved rounds, each side its own best-tuned
  parameters (standard HPL methodology; container runs NVIDIA's tuned dat).

## 2026-07-03: 2-node campaign
- Our 16-rank port (P2Q8, hybrid W-NCCL/CE): gates PASS at 16384/65536.
- Our 2-node v14: N=487424/NT768/NCH6 = **719.7 TF** PASSED (45.0 TF/GPU).
  N=491520/NT512 = 586.8 (NT=512 kills int8 gemm eff). 483328/NT1024 OOM (line 687 ozWs).
  NCH=8: 704.8 (worse). NCH=16 / NT896 / 479232-NT1024: silent fail (grep ate errors; NT896 = NCCL "unhandled cuda error" @1004).
- Container 2-node sweep (full HPL.dat format required — abbreviated 13-line rejected):
  N=507904 P4Q4: 775.1 | 516096 P4Q4: 776.3 | P8Q2: 772.2 | P2Q8: 770.8
  **524288 P4Q4 NB1024: 781.8 TF = container ceiling** (532480+/NB2048: OOM on A_dev).
- Gap: 719.7 vs 781.8 (-62 TF). W bcast (nrow, up to 4GB/panel over IB) sits on the
  sP panel-to-panel critical path: exch g0(k) -> Wbcast(k) -> urgent gemm(k) -> chainA(k+1).
  Exposure ~8.8s of 107.26s.
- v15w (W bcast on separate sW stream): DEADLOCKS at first panel (16 ranks, N=16384).
  Also has a latent correctness bug: emit chunk0 gemms read d_W[b] on sP w/o evReadyW wait.
  Debug job 162993 (PROG probe + NCCL_DEBUG=WARN + LAUNCH_MODE=GROUP) in flight.
- Plan: chunked W pipeline — bcast W in R row-chunks on sW, per-chunk events,
  urgent gemm consumes per chunk on sP (row-parallel in the update), sliceA per chunk on s0.
  Critical path pays chunk1 latency only. evXg0 covers sP reads parity; evUpdC[k-2][3] covers s0.

### 2-node tuning progression (ours)
- v15w chunked W pipeline (sW stream + WCH col-chunks + K-slab urgent gemms + host throttle
  cudaEventSynchronize(evReadyW[k]) fixing multi-comm NCCL deadlock): correct, but == v14 perf.
  WCH 2/4/6/8 insensitive → W bcast was NOT the exposed cost.
- NT effect at 2-node shape (N=417792): NT768=701.1 vs NT1024=735.7 (+4.9%!) → NT768 was the loss.
- **N=475136 NT=1024 NCH6 WCH4 = 759.2 TF PASSED** ← our 2-node best so far
  (477184/479232 NT1024: NCCL "unhandled cuda error" (async mem pressure); 481280: hard OOM @693)
- NCH4: 755.8; WCH8: 757.7 — both worse.
- Container refinement: 526336=775.7, 528384=775.8 → ceiling stays **781.8 @ 524288 P4Q4 NB1024**.
- Timing @475136/1024: s0_gemm=84s busy, s0_gap=9.2s, chainB(sP)=30-33s (mostly hidden).
  [2dfine]: getrf1 (fp32 stacked cusolver getrf, chainA) ~290ms per owned panel on critical path.
- Gap to container: 759.2 vs 781.8 = -2.9%.

### nt4/nt5/nt6 probes (all N=475136 NT=1024 unless noted)
- XT=16384: 758.8 (nc); 477184+NCCL_BUFFSIZE=2M: still dies; ALGO=2: 743.9; ALGO=1: 573.0
- 471040/NT1280: 556.0 (NT must be 512-multiple w/ good Lt algos; 1280 catastrophic)
- 466944/NT1536: 734.5; NCH10: 761.2; NCH5: 756.8
- NCH8: 761.4/761.3 (repeatable, beats NCH6 at NT1024 — reversed vs NT768)
- **NCH8 + CUDA_DEVICE_MAX_CONNECTIONS=32: 763.1 TF ← our 2-node best**
- IB QPS=4: 760.5
### Official 2-node showdown submitted (job 163010)
- Ours: N=475136 NB2048 P2Q8, OZ NT1024 WCH4 NCH8 CONN32 (~763)
- Container: N=524288 NB1024 P4Q4 (~782)
- Structural levers left (not attempted, multi-day): P4Q4 rework of tournament/CE
  (halves getrf height + W payload, widens nloc, frees ~9GB → N~493k),
  custom tall fp32 getrf, depth-2 lookahead.

## OFFICIAL 2-node showdown (job 163010, nodes node038/039, interleaved 5 rounds)
- Ours  (N=475136 NB2048 P2Q8 OZ NT1024 WCH4 NCH8 CONN32): 747.2, 763.0, 764.6, 762.8, 763.0 → median 763.0 TF (all PASSED, res 0.5517)
- Container (N=524288 NB1024 P4Q4):                        777.3, 777.4, 783.0, 777.2, 777.3 → median 777.3 TF (all PASSED)
- Result: container wins 2-node by 1.84% (median). Single-node remains OUR win (391.5 v 385.8).
- Note: interleaved container median (777.3) < its standalone 781.8 (power/thermal interplay).

## PRESELECT breakthrough (2026-07-03 late)
- Delay probe: getrf-on-chain sensitivity slope ~0.5 (50/100/200ms injections).
- Early selection ("preselect"): tall fp32 selection getrf for panel k+1 launched from inside
  chainB(k) right after g==1 (urgent chunk emitted → panel k+1 cols exact), getrf on dedicated
  sSel stream + cs2 handle + private d_work2/d_ipiv2/d_occ2/d_top2, overlapping chainB tail.
- Debug saga: v1 stale (2e7 FAIL), v2/v3 wrong g-placement (prng=chunks_of(k-1)! urgent emits at
  g==1 not g==0), v4 exact-at-16384 but FAIL npan>=64 even under full serialization.
  Bisect (PSYNC modes): results-overwritten still FAILED; pure-delay PASSED; d2s/getrf/resolve
  steps PASSED; only prep_cand's write to SHARED d_rows+6NB/d_prov from sSel poisoned runs.
- Fix: prep_cand → private d_rows2/d_provP; chainA copies to canonical buffers on sP after
  evSel wait. Bit-identical residuals restored (0.8648@131072, 0.5517@475136).
- **475136 PRESEL=1: 778.2/778.0 TF (vs 765.3 PRESEL=0 same job) — above container interleaved
  median 777.3.** Micro-tune batch 163105 (NCH 6/10, WCH 2/6, TIMING) before official rerun.

## ★★ OFFICIAL 2-NODE VICTORY (job 163113, node038/039, interleaved 5 rounds) ★★
- Ours  (N=475136 NB2048 P2Q8, OZ NT1024 WCH4 PRESEL1 NCH8 CONN32):
  778.7, 777.6, 778.7, 777.7, 777.4  → median **777.7 TF**  (res 0.5517 PASSED ×5)
- Container (N=524288 NB1024 P4Q4):
  777.8, 777.1, 777.3, 777.3, 776.8  → median **777.3 TF**  (PASSED ×5)
- **Median win +0.4 TF; round-by-round 5/5** (778.7v777.8, 777.6v777.1, 778.7v777.3,
  777.7v777.3, 777.4v776.8).
- Both goals now met: single-node 391.5 v 385.8 (5/5), two-node 777.7 v 777.3 (5/5).
- Decisive final lever: preselect (early tall fp32 selection getrf on sSel with private
  output buffers) — took s0_gap 9.2s→~7s, +15 TF over the no-presel official 763.0.

## N-squeeze to 479232 (answering 「N 不能再設大一點嗎」)
- Freed ~1.2GB: CHPL2D_WS=64 (cublas pools 640→192MB), CHPL2D_XT=6144 (xpack family −0.24GB),
  NCCL_BUFFSIZE=2M. NT=1024 kept (896 = 715 TF, terrible).
- N=479232 fits: **783.9-784.2 TF solo** (> container solo ceiling 781.8). 477184=777.5 (worse
  panel balance), 481280 still dies (NCCL async mem) — 479232 is the ceiling for P2Q8.
- Env re-sweep at 479232: BUF4M 782.9, NCH10 781.9, CONN64 781.3, XT8192 783.9 → converged.

## Official showdown #2 (job 163254, node042/043, N=479232 config)
- Ours:      784.0, 781.7, 781.5, 781.6, 784.2 → median 781.7 (sigma ~1.3)
- Container: 780.1, 786.5, 783.4, 780.4, 786.6 → median 783.4 (sigma ~3.2!)
- 2/5 rounds, median -1.7. NODE-PAIR VARIANCE: on 038/039 container ran 777.3 (we won 5/5);
  on 042/043 it runs 783.4. Our margin sits inside cross-node variance (~0.8%).
- Record: 1 win (038/039), 1 loss (042/043). Robust win needs +1-2% → P4Q4 rework
  (est +2-4%) or stage-2 getrf off-chain (~+1%, hard: needs fresh candidates).

## Official showdown #3 (job 163277, N=479232 config)
- Ours:      782.0, 765.7(outlier), 781.9, 782.0, 781.8 → median 781.9
- Container: 787.5, 780.2, 783.9, 783.9, 783.9         → median 783.9
- 0/5, median -2.0. Tally across 3 showdowns: 1 WIN (slow pair 038/039), 2 LOSS (fast pair(s)).
- Pattern: container's N=524288 gains more from fast/cool nodes (fp64 clocks) — ~777 on 038/039
  vs ~784 on 042/043-class; ours is stable ~778(old cfg)/782(new cfg) interleaved everywhere.
- To win on ALL node pairs need +0.3-0.5% more: stage-2 is truly serial (winners gate ALL of
  chainB — no overlap possible); P4Q4 remains the decisive lever.

## P4Q4 REWORK (2026-07-04, src/chpl4x4.cu, binary chpl4x4)
- Full P=4 generalization landed in ONE session: star-topology CE exchange (all inter-rank
  rows route via diag-owner row kp; per-peer lists, xpack [sslot][dst_pr] regions), 4-way
  tournament (all cand slots gathered to pr0, winmap4/win_gather4 prefix kernels, d_provA
  packed), U pull from uroot=kp+P*qc, 3-peer backsub guards, d_lrb(P), P*kq roots.
- P4Q2 single-node gates: ALL PASS first build (OZ=0/1 16384, OZ=1 65536/131072).
- **P4Q4 2-node first flight: N=493568 NT=1024 = 779.2 TF PASSED** (gates clean).
  NT=1536 OOM (ozWs). vs P2Q8 champion 783.9 solo @479232.
- presel@P4: 1 FAIL then 3/3 PASS (flaky race, unlike P2's deterministic bug; PSYNC=1/4/6/8
  all pass). Deferred — P4 chain already has half-height getrf.

## ★★★ P4Q4 DECISIVE VICTORY (job 163448, official showdown #4, 5 rounds) ★★★
- Ours (chpl4x4, N=497664 NB2048 P4Q4, OZ NT1024 WCH4 PRESEL1 WS64 XT6144 BUF2M NCH8 CONN32):
  791.6, 789.1, 790.6, 789.4, 789.6 → median **789.6 TF**  (res 0.4773 PASSED ×5)
- Container (N=524288 NB1024 P4Q4): 785.6, 786.9, 786.2, 786.1, 786.8 → median 786.2
  (container's best-ever showing — fast node pair — and still lost every round)
- **5/5 rounds; median +3.4; COMPLETE SEPARATION: our worst round (789.1) > their best (786.9).**
- P4Q4 tuning summary: 497664/NT1024 best (783.3 no-presel); NT1536/2048 lose (744-748);
  499712 worse; WCH8 worse. PRESEL adds +6.3 (783.3→789.6): half-height getrf off the chain.
- presel race at P4: 8/8 clean at production N (1 flake at 131072 early); residual self-certifies.
- Showdown tally: #1 W 777.7v777.3 (P2Q8-475136) | #2 L 781.7v783.4 | #3 L 781.9v783.9
  (P2Q8-479232) | #4 **W 789.6v786.2 (P4Q4)** — architecture rework closed the node-variance gap.

## SINGLE-NODE 394.5-TARGET CAMPAIGN (2026-07-04, beat Dell XE9680 published 8xH200 number)
- Lever: PRESEL applied to the single-node champion for the first time (developed post-391.5).
- P2Q4 348160 on node011: PRESEL=0 = 396.6; PRESEL=1 = 403.5-404.6 (+1.7% causal).
- Bigger N dead (350208+ NCCL async mem death even with WS64/XT6144 diet).
- P4Q2 single-node dead: 381.1-386.3 (4-way exchange + doubled nloc staging > getrf halving at Q=2).
- NCH: 4=403.9, 8=403.7, 6=404.4 → 6 confirmed.
- **PROOF (job 163888, 5 rounds @NCH6): 404.6, 403.2, 393.4(dip), 404.6, 403.3 → median 403.3 TF**
  All PASSED, residual bit-identical 0.6777 (presel exact; 0 flakes in 8 runs).
- vs Dell XE9680 394.5: +8.8 TF (+2.2%). Even PRESEL=0 on this node (396.6) clears it.
- NEW single-node champion config: chpl2d_v15w, N=348160 NB2048 P2Q4, CHPL2D_OZ=1
  CHPL2D_PRESEL=1 NCCL_MAX_NCHANNELS=6.

## 800-TARGET CAMPAIGN (2026-07-04): 789.6 -> 792.2 median, wall identified
- PKGPRE (pkg bcast prefetch on dedicated comm nworld2+sK): built, correct, REJECTED (-5 TF:
  extra comm SM contention > chain saving). Env CHPL2D_PKGPRE stays 0.
- NCH=4 (with NCCL_MIN_NCHANNELS=1): +2.2 TF vs NCH8 — fewer NCCL SMs feed the OZ gemm.
  NCH 3/2 lose (W too slow); NCH5/WCH6/WCH8/NTHREADS256/XT8192/CONN64 all neutral.
- **FINAL PROOF (job 163903, 5 rounds): 792.3/792.2/791.8/792.3/792.0 → median 792.2 TF**
  (sigma 0.2!). Config: chpl4x4 N=497664 P4Q4 NT1024 WCH4 PRESEL1 WS64 XT6144 BUF2M
  NCH=4 MIN=1 CONN32.
- The 800 wall: s0_gemm floor ~99.7s = 96% of 103.7s runtime (51.5 TF/GPU OZ rate);
  zero-gap theoretical ~812; gap 2.5-3s is stage-2-serial + skew. Next real lever =
  custom CUTLASS fused int8 kernel (prior attempts 630/1100 TOPS needed, days of work).
- Single-node campaign same day: PRESEL applied to 1-node champion -> 403.3 median
  (beats Dell XE9680's published 394.5 by +2.2%).

## 800-CAMPAIGN FINAL (2026-07-04): plateau 792.2, all avenues exhausted
- Production ALGD sweep (uniform idx 0-7 @full N): default idx3 = 792.2 WINS; others
  780.6/590.6/769.5/581.6/761.8/714.0/778.0. Isolated startup-bench best (5,0,0,0,7,7)
  loses -17 TF in production — recon/NCCL bandwidth contention flips algo ranking;
  ONLY full-run benchmarking is valid for algo choice (startup bench kept, default off).
- Clock telemetry: 1980MHz held (45/56 samples) — NOT power-limited at 2 nodes.
- urgent-chunk fp64→int8: recomputed cost = only ~0.6s/GPU total (owner-column rotates) — not worth.
- 499712@NCH4: 788.1 (N ceiling stays 497664). NB=4096: gates PASS (code NB-generic) but
  458752 full run = 716.0 vs same-node control 792.7 — k-doubling loses to N/buffer costs.
- FINAL: 792.2 median (sigma 0.2), 99.0% of 800. Wall = cublasLt int8 production rate
  (~1300 eff TOPS vs 1689 isolated = 20-25% contention loss from int32 D-traffic + NCCL).
  Remaining path: custom fused int8 kernel w/ register accumulators + fp64 epilogue
  (kills 48B/elem D-writes; could recover contention loss → ~850 ceiling; multi-day).

## Container FP64-emulation probe (2026-07-05, job 164838)
- nvcr 26.02 SHIPS Ozaki-family fp64 emulation (GEMMul8; HPL_EMULATE_DOUBLE_PRECISION=1/2 =
  Ozaki-I/II) — but it is BLACKWELL-GATED: on H200 (SM90) it prints
  "HPL WARNING: ... supported only on SM100 or newer" and falls back to native fp64
  (verified empirically: residual 3.6e-4 = fp64-class, normal fp64 perf).
- Implications: (1) all our showdowns were fair — container was at its true H200 best;
  (2) our implementation delivers Ozaki-scheme HPL on Hopper where NVIDIA's own does not run.

## ENERGY DUEL (2026-07-05, job 164863, node121/122, 2 rounds each interleaved, GPU-only 1Hz)
- STEADY-STATE POWER: both sides at the 700W cap in production!
  CHPL 687-693 W/GPU (busy throughout, median≈steady); container 694-698 W/GPU
  (median 350-551 — long low-power init/verify phases).
  → The old "int8 draws only 475W" was from ISOLATED gemm probes; full HPL (int8 + fp64
  panel/backsub/slicing mix) also saturates the cap. The advantage shows in energy/flop.
- ENERGY PER SOLVE: CHPL 0.931 MJ (0.259 kWh) @N=497664 vs container 1.244 MJ (0.346 kWh)
  @N=524288.
- PER-FLOP EFFICIENCY (full-run integrated): CHPL 88.2 GFLOPS/W vs container 77.2
  → **CHPL +14.3% more compute per joule** at the same power cap (int8 ops cheaper per
  effective flop; clocks hold 1980MHz vs fp64's ~1500MHz at cap).
- In-window naive GFLOPS/W (whole-run avg power) misleads: container's longer init dilutes
  its average power — always integrate energy and normalize per flop.

## TDP-scaling question (2026-07-05, job 164866): nvidia-smi -pl = Insufficient Permissions
- Cluster locks 700W (no user control). Analysis from measured data:
  at 700W container is DEEP power-bound (fp64 clocks 1980->~1500), CHPL sits at the
  clock-cap/power-cap boundary (1980MHz @ 687-693W). Speed ratio +0.8% but energy ratio
  +14.3% -> under a tighter cap both go fully power-bound and the speed ratio converges
  toward the energy-efficiency ratio: predicted CHPL advantage ~+13-14% at 500W
  (perf ~ (TDP - ~85W static) x flops/dyn-J). Supporting: container's node-cooling
  sensitivity sigma 3.2 vs ours 1.3 TF.

## WI8 int8 W-transport VERDICT (2026-07-05, jobs 164871/164874/164875)
- Implemented CHPL2D_WI8=1: root row-chunk slices W on sW, bcasts 6xint8+scales (75% of
  fp64 bytes), receivers feed digit gemms from slice buffer + rebuild 42-bit W (ozrecW).
- CORRECT (residual 0.4710 vs 0.4684 baseline, PASSED) but PERF-NEUTRAL: same-node A/B
  @487424: 783.0/786.0 (off) vs 781.8/782.8 (on). W bcast was already fully hidden by the
  WCH pipeline; +1.5GB transport buffers push N down → net negative vs champion.
- Keep for >2-node scaling (W bcast grows with node count) — not for the 2-node config.
- Also: earlier "gate +10%" was node-pair artifact (same-node gate control identical).

## OZAKI-II own implementation (2026-07-05, src/oz2own.cu)
- Wrote self-contained Ozaki-II (CRT/modular): scale->int quant per modulus->int8 TN gemm
  ->Garner CRT reconstruct (__int128). GEMMul8 lib build was stuck (full complex/fp8 template
  matrix), so built our own tailored to our shapes.
- Accuracy scales cleanly with (s bits, t moduli): s=25/t=9 -> 2.3e-6 (24-bit);
  s=33/t=11 -> 2.98e-13 (~fp64 noise floor, EXCEEDS scheme-I's 47-bit) using only 11 moduli.
- k-volume at t=11: 11*2048 = 22528 vs scheme-I 6-fused-diagonal 21*2048 = 43008 -> 52%
  int8 tensor work at matched-or-better accuracy. Theoretical GEMM ~2x.
- Login-GPU small-shape: gemm+recon 11-14 TF (recon-bound at small n). H200 real-shape
  probe (job 164891) pending: the question is whether recon (t int32 planes + 128-bit
  Garner, ~2x scheme-I D-traffic) eats the int8 savings at panel widths n>=4096.

## OZAKI-II breakthrough (2026-07-05): double-double fractional-CRT recon
- Two fixes made scheme-II viable: (1) cublasLt int8 (cublasGemmEx picked 174-TOPS algo!),
  (2) double-double (two-sum/two-prod Dekker) fractional-CRT reconstruction — full fp64
  accuracy WITHOUT __int128 bignum division.
- s=33 t=11: maxdiff 2.4e-13 (matches fp64 noise floor, EXCEEDS scheme-I 47-bit) using only
  11 moduli (k-volume 11*2048 vs scheme-I fused 21*2048 = 52%).
- Login-GPU: gemm+recon 35 TF @ full accuracy. H200 measurement pending (job re-run).

## OZAKI-II FINAL VERDICT (2026-07-05): scheme-I fused-diagonal is near-optimal for HPL accuracy
- Built correct full-accuracy Ozaki-II (cublasLt int8 + double-double fractional-CRT recon).
- H200 n=4096 FULL accuracy (4.4e-13, exceeds scheme-I 47-bit):
    Ozaki-II s=33 t=11 (tight): gemm+recon 62.9 TF, 11 int32 planes
    Ozaki-II s=33 t=13 (safe margin): 55.0 TF, 13 planes
    scheme-I fused-diagonal: ~62 TF isolated / 51.5 production, 6 planes, 47-bit
  => TIED at tight-t, SLOWER at safe-t. Ozaki-II fast only at low accuracy (s=25 t=9 = 78 TF
     @ 24-bit) which HPL@N=500k cannot use (40-bit D=4 already FAILED historically).
- WHY scheme-I wins: fused-diagonal concatenates slices along k (6 GEMMs, growing kd=(d+1)*2048,
  cover 21 pairs) and reads only 6 int32 planes. Ozaki-II (CRT) needs 11-13 full-k GEMMs + as
  many int32 planes for the same accuracy -> more HBM traffic (worse in contended production) +
  more quantization. Our specialization beats the general scheme for HPL's accuracy tier.
- DECISION: NO integration. scheme-I stays champion (792.2). Probe kept at src/oz2own.cu.
- Bonus: int8 transport (WI8) already tested neutral at 2 nodes; scheme-II wouldn't change that.

## CUSTOM FUSED KERNEL BATTLE — CONCLUSIVE LOSS (2026-07-05, job 164918/164920)
- Re-measured all custom int8 kernels (src/ozkernel.cuh) vs cublasLt at BOTH n=8192 and the
  production n=1024 (OZ_NT), m=32768 k=2048, H200 1980MHz.
- n=8192: library scheme-I 843 TOPS-eff vs best-custom (oz2 bm32 mma.sync) 704 MMA-only/620 recon.
- n=1024 (production shape): library 674 TOPS-eff (32.14 TF-eq) vs oz2 bm32 707 MMA-only but
  ONLY 630 with fused recon (29.98 TF-eq) — STILL below library.
- All wgmma variants worse (n32=285, wg16=514) than mma.sync bm32 — Hopper hand-wgmma
  underperforms; the grouped-diagonal-n64 idea would at best reach ~700-800 MMA-only, still
  losing after the epilogue cost (MMA-only 707 -> 630 with recon = -11%).
- VERDICT: cublasLt's int8 GEMM scheduling beats any achievable hand-written kernel here; its
  advantage exceeds the custom kernel's D-traffic-elimination benefit. Confirms the prior
  12-version campaign's plateau (~630-704). NO integration.
- FINAL: 792.2 TF (2-node) / 403.3 (1-node) is the genuine ceiling of the GEMM-emulation
  approach on H200 with cublasLt. Beating it needs hardware NVIDIA doesn't beat either
  (Blackwell fp64-emulation is SM100+ only). Both scheme-II and custom-kernel avenues closed
  with measured negative results.

## STAGE-2 PRESELECT (2026-07-05, the last documented structural lever) — +0.34% REAL
- Delay probe confirmed stage-2 (final winner selection on pr==0) is exposed on the sP chain
  (slope ~1.4). Moved the FULL tournament — grow_gather + ncol candidate exchange + stacked
  fp32 getrf + inv-piece trsms — into preselect on sSel during chainB(k-1), off the critical
  chain. Private buffers (d_candP/d_selP/d_pkgP[2]/d_win2[2]/d_provAP), dedicated cs3/hb3
  handles; chainA fast-path (p2fast) copies d_win2/d_pkgP -> canonical. Env CHPL2D_PRESEL2.
- BUG found+fixed: owner-column ranks with myncand==0 at end-panels early-returned before the
  ncol collective -> mismatch DEADLOCK (hung at k=6-7). Fix: they stay to participate, only
  skip stage-1. (ncol is same-node in P4Q4 and idle during chainB, so no multi-comm deadlock.)
- Correctness: residual bit-identical to inline (0.869@131072, 0.477328744971@497664) —
  deterministic tournament, same winners.
- Interleaved 4-round A/B @497664: P2=0 median 786.35 vs P2=1 median 789.0 = **+2.65 TF
  (+0.34%)**; 3/4 rounds +2.7..3.0, one noise round. Smaller than the ~1% estimate because
  stage-2 on sSel serializes behind stage-1's tall getrf + contends with the main GEMM for
  SMs (the critical-path saving is real but partly offset).
- ADOPTED into champion config (CHPL2D_PRESEL2=1). New 2-node best ~789 median on this pair
  (was 786.4); on the good node pair (792.2) expect ~795.

## OFFICIAL SHOWDOWN with PRESEL2 champion (job 165011, 5 rounds interleaved)
- Ours (chpl4x4 N=497664 PRESEL2=1 NCH4 MIN1 WCH4 WS64 XT6144 BUF2M CONN32):
  783.0, 783.9, 781.8, 785.2, 782.2 → median 783.0 (res 0.4773 PASSED ×5, bit-identical)
- Container (524288 P4Q4): 771.8, 776.2, 770.5, 777.2, 770.2 → median 771.8
- 5/5, complete separation (our worst 781.8 > container best 777.2), +11.2 TF (+1.45%).
  (slower node pair — both sides lower absolute; good-pair best remains ~789-792.)
- PRESEL2 (stage-2 preselect) is now in the champion config; the last documented structural
  lever is filled (+0.34% measured, correctness bit-identical).

## WI8 MULTI-NODE SCALING TEST (2026-07-06): hypothesis REJECTED — gets WORSE, not better
- Infra generalization (kept, useful regardless of WI8 result): chpl4x4.cu was hardcoded to
  8-or-16 ranks (samenode[16], PeerBufs peer[16], H all[16], rowpeer[7], rp[7], d_tmp3 fixed
  7*NB). Generalized all to dynamic g_size/Q (std::vector sized g_size or Q-1). Rank-count
  guard now accepts any multiple of P=4 that's also a multiple of 8 (node-major rank layout
  assumption). FIRST-EVER >2-node test of chpl4x4: 4 nodes / 32 GPUs (P4Q8) — all correctness
  gates PASSED on first build (OZ, PRESEL, PRESEL2, WI8 all combined, residual clean).
- Fixed baseline for the on/off comparison: N=688128 (confirmed working at 4 nodes, not
  pushed to the OOM ceiling — 704512/700416 died, per user instruction "不需要找到可用最大
  N，只要在同樣的基準下找到 WI8 啟用跟不啟用的差異就好了").
- Interleaved 4-round A/B, 4 nodes, full champion config (PRESEL/PRESEL2 on):
    WI8=0: 1593, 1594, 1593, 1593 TF -> median 1593.3 (near-zero variance)
    WI8=1: 1567, 1566, 1566, 1567 TF -> median 1566.6 (near-zero variance)
    Delta: -26.7 TF = **-1.7%**
  vs the 2-node baseline (N=487424): WI8 was -0.2~-0.4% (mildly negative).
  TREND: -0.3% (2 nodes) -> -1.7% (4 nodes). Gets WORSE, opposite of the hypothesis.
- MECHANISM (found in code, chpl4x4.cu ~1868-1915): the WI8 path issues TWO separate
  ncclBroadcast calls per row-chunk (int8 slice payload + a SEPARATE fp64 row-scale array),
  vs ONE call in the plain fp64 path. At P=4 fixed, the nrow communicator (fixed pr, varying
  qc) spans MORE PHYSICAL NODES as node count grows (2 nodes -> nrow crosses 2 nodes; 4 nodes
  -> nrow crosses 4 nodes, 2 ranks/node each). More physical nodes in the ring/tree means more
  real IB hops per collective call. WI8 trades -25% bytes for 2x the per-chunk NCCL call count;
  as the topology grows, the fixed per-call/hop latency tax dominates over the bandwidth saved,
  flipping the sign further negative. This is the OPPOSITE of "more W traffic = more benefit
  from compression" — the real driver at scale is call/hop count, not raw bytes.
- CONCLUSION: WI8 should NOT be adopted at ANY node count tested (2 or 4); the effect gets
  worse with scale, not better. Champion config (WI8=0) is correct and untouched. If ever
  revisited, the fix would be to MERGE the int8-payload and row-scale broadcasts into a single
  NCCL call (pack them contiguously) to remove the doubled call count — untested.

## WI8V2 merged-broadcast FIX ATTEMPT (2026-07-06): FAILED, made it WORSE
- Hypothesis from the WI8 scaling test: 2 separate ncclBroadcast calls/chunk (int8 payload +
  row-scale array) doubled the per-hop NCCL overhead as node count grew. Fix: merge into ONE
  call by packing digit-bytes + a trailing scale double into a single "wire" row format.
  New kernels oz_wire_pack_kernel (root: slices W row-chunk INTO the wire buffer, embedding
  rsc at a padded tail offset) + oz_wire_unpack_kernel (non-root: unpacks wire -> tight ozSAt
  for the digit-GEMMs + reconstructs fp64 d_W in one pass). Persistent per-row scale kept in
  ozRs2[2] (parity-buffered -- reusing the single global ozRsc would race across the
  panel pipeline, since chainB(k+1)'s sW writes could clobber it before s0(k) finishes
  reading it; ozRinv stays a safe single global scratch since it's produce-then-immediately-
  consume within one chunk on the SAME stream). CHPL2D_WI8V2=1 (implies WI8=1).
- CORRECTNESS: confirmed bit-identical to WI8v1 at N=65536 (residual 0.222299897851 exactly
  matches) -- the merge is mathematically sound, root still exact-slices, non-root gets the
  same quantized approximation, just delivered via 1 broadcast instead of 2.
- PERFORMANCE: WORSE than v1, not better. Interleaved 4-round A/B @ N=688128, 4 nodes:
    WI8=0:    1594 TF median (near-zero variance)
    WI8=1(v1): 1566.6 TF (-1.7%, the original regression)
    WI8V2=1:  1499 TF (-5.96%, MORE THAN 3x WORSE than v1)
- ROOT CAUSE OF THE FIX'S FAILURE: to merge the broadcasts, root now runs TWO full passes
  over the row-chunk (ozsliceA_kernel for the tight ozSAt copy the GEMM needs, PLUS
  oz_wire_pack_kernel re-slicing the SAME data into the padded wire format) instead of one;
  non-root's unpack kernel does MORE work than v1's reconstruct-only kernel (copies all 6
  digit planes AND reconstructs fp64, vs v1's broadcast writing ozSAt in-place with zero
  extra copy). At ~250MB/chunk payload sizes, NCCL broadcast is BANDWIDTH-bound, not
  call-count-bound -- the saved small-message call's latency was negligible next to the
  added full-array kernel passes' compute cost. The original root-cause theory (call count
  drives the v1 regression) is therefore UNCONFIRMED; the fix attempted here doesn't
  isolate/test it cleanly since it necessarily added compute to achieve the merge.
- CONCLUSION: WI8V2 should NOT be adopted; it's strictly worse than both WI8=0 and WI8=1(v1)
  at every tested config. A genuine fix (if the call-count theory is even right) would need
  to merge the broadcast WITHOUT adding extra compute -- e.g., embedding the padding directly
  into ozSAt's OWN persistent layout (changing the digit-GEMM's cublasLt leading dimension to
  match), which was deliberately avoided here as high-risk (touches the hottest GEMM path).
  That higher-risk path remains untested. Champion (WI8=0, WI8V2=0) unaffected.

## WI8V2 cheap-repack retry (2026-07-06): user's catch was right, but STILL didn't help
- User correctly identified that oz_wire_pack_kernel redundantly RE-DERIVED the sliced digits
  from d_W (re-reading fp64, redoing 6x rint/scale ops) instead of just COPYING what
  ozsliceA_kernel had ALREADY computed into ozSAt. Fixed: new oz_wire_repack_kernel does a
  pure byte-copy from ozSAt[b]+r0*ldSw (already sliced) into the padded wire row + writes the
  already-computed ozRs2[b]+r0 scale -- zero re-derivation, zero arithmetic.
- Gate: residual STILL bit-identical (0.222299897851 @ N=65536) -- confirms the copy is
  mathematically exact, as expected.
- Interleaved 4-round A/B @ N=688128, 4 nodes: WI8=0 median ~1564 TF vs WI8V2(cheap) median
  ~1456 TF = **-6.9%** -- essentially UNCHANGED from the "buggy" redundant-recompute version
  (-5.96%), actually marginally worse. Removing the redundant arithmetic did NOT recover the
  regression.
- REVISED DIAGNOSIS: the recompute wasn't the dominant cost. The real problem is STRUCTURAL:
  merging via a separate "wire" staging buffer requires EXTRA memory traffic that v1 (which
  broadcasts the final buffer in place with zero extra copies) doesn't pay:
    root:     writes ozSAt (needed regardless) PLUS writes the wire buffer (extra, even as a
              cheap copy) PLUS the memcpy2D+rowscale (needed regardless)
    non-root: v1 broadcasts DIRECTLY into ozSAt in place (reconstruct-only kernel, one extra
              read pass, one W write) -- v2 broadcasts into a SEPARATE wire buffer, requiring
              an UNPACK step that writes BOTH ozSAt (extra, absent in v1) AND W (needed
              regardless)
  Also noted (secondary factor, not separately isolated): the repack/unpack kernels map row
  index i to the warp's fast (threadIdx.x) lane, while the ORIGINAL ozsliceA_kernel/
  ozrecW_kernel map the COLUMN j to that lane -- i-as-fast-lane means adjacent threads in a
  warp touch memory ldSw(~12KB) apart, uncoalesced, whereas the established convention keeps
  32 consecutive bytes-per-row coalesced. This likely compounds the structural extra-copy
  cost but wasn't isolated as its own experiment (diminishing returns; the extra-copy
  structural cost alone plausibly explains the bulk of the loss).
- CONCLUSION (revised, final): the "merge via separate wire buffer" family of fixes is
  FUNDAMENTALLY more expensive in GPU memory traffic than v1's broadcast-in-place, regardless
  of arithmetic/coalescing tuning -- extra full-array copies on BOTH root and non-root are
  structurally required to keep ozSAt's tight layout untouched while merging the NCCL calls.
  The only way to truly avoid the extra copies is the HIGH-RISK direct-ozSAt-padding approach
  (changing the digit-GEMM's cublasLt leading dimension), still untested, still risky.
  STOPPING further WI8 fix iterations here -- two independent attempts (redundant-recompute,
  cheap-copy) both failed for a now well-understood structural reason. WI8=0/WI8V2=0 remains
  correct at every tested scale (2 and 4 nodes). Champion untouched throughout.

## W-broadcast delay-probe: theoretical ceiling for WI8-style transport wins (2026-07-06)
- User asked: is "no perf gain from int8 transport" theoretically reasonable? Ran a direct
  delay-probe (CHPL2D_DLYW, spin_kernel injected right after the LAST W-broadcast chunk
  completes, plain fp64 path) at N=688128, 4 nodes, champion config, to measure exposure.
- Results (336 panels, N/NB): baseline 138.76/138.80s (noise floor ~0.04s); DLYW=500us
  +0.08s (slope~0.48, noise-level); DLYW=1000us +0.07s (slope~0.21, noise-level, non-
  monotonic vs 500us -> both effectively indistinguishable from zero at n=1 sample);
  DLYW=2000us +0.59s (slope~0.88, CLEARLY above noise -- substantial exposure).
- INTERPRETATION: not a clean "fully hidden" (slope 0) nor "fully exposed" (slope 1) -- a
  THRESHOLD/slack-absorption pattern: small perturbations (hundreds of us/panel) get mostly
  absorbed by concurrent pipeline work (exchange, stage-2, U staging have some scheduling
  slack), but once delay exceeds that slack budget, it becomes ~88% exposed.
- WHY THIS EXPLAINS WI8's CONSISTENT FAILURE: int8's theoretical bandwidth savings (-25% of
  the W payload) for a per-panel fp64 broadcast likely on the order of hundreds of us --
  squarely in the "mostly absorbed, thin-to-noise-level exposure" regime measured here. So
  even a THEORETICALLY PERFECT, zero-overhead int8 transport would only recover a small
  fraction of its face-value 25% savings in wall-clock time (the slack absorbs most of it).
  Any REAL implementation's overhead (extra small-message broadcast in v1; extra full-array
  copies in v2) is larger than this already-thin theoretical ceiling, guaranteeing a net
  loss -- exactly the escalating pattern observed (WI8v1 mild loss, WI8v2 attempts larger
  loss, in proportion to how much extra overhead each added).
- CONCLUSION: Amdahl's-law style reasoning is confirmed, refined by data: not "zero
  theoretical benefit" but "a benefit so thin (falls in the pipeline's absorption regime)
  that no realistic implementation overhead can come out ahead." This is the first-
  principles closure of the WI8 investigation -- delay-probing the TARGET before
  implementing a transport optimization would have predicted this outcome without
  needing 3 failed implementation attempts. Lesson for future transport-layer ideas:
  delay-probe first, implement only if the probe shows exposure comparable to or larger
  than the realistic implementation overhead.

## Single-GPU ceiling probe: hpl_single.cu vs teammate's 63.1 TF (2026-07-08)
- Context: a teammate shared hpl_ga.cu, a single-GPU HPL solver using GEMMUL8 (RIKEN's
  Ozaki-II / CRT-based fp64 emulation library) claiming 63.1 TFLOPS. Its architecture:
  factor panels at small NB via cusolverDnXgetrf directly (no tournament pivoting -- a
  single rank has no cross-rank pivot-candidate consensus problem), but AGGREGATE the
  expensive trailing-update GEMM to run once per AGG panels at K=AGG*NB, amortizing the
  emulation scheme's fixed per-call overhead over more useful work.
- Built src/hpl_single.cu: a standalone, zero-MPI/NCCL/IPC single-GPU solver reusing OUR
  production Ozaki-I fused-diagonal kernels (ozrowscale/ozcolscale/ozsliceA/ozsliceB/
  ozrecon, copied verbatim from chpl4x4.cu) inside the SAME AGG-block two-level-blocking
  structure, to give a fair, apples-to-apples measurement of our own scheme's true
  single-GPU ceiling, free of every distributed-only cost (tournament pivoting, CE/IPC,
  NCCL broadcasts, preselect) our production code carries. Build: scripts/build_single.sh.
  Run: scripts/single_selftest.sbatch (correctness), scripts/single_gemmtest.sbatch
  (isolated oz_gemm-only correctness harness), scripts/single_perf.sbatch (performance).
- Two bugs found and fixed while bringing this up (see memory file
  hpl-cuda-project.md for full detail):
  1. ozsliceB_kernel hardcoded its input's leading dimension to equal the contraction
     width k -- valid in the distributed code (B is always a compact dedicated buffer
     there) but WRONG here, where B is a view into the shared N x (N+1) matrix with a
     real ld=N. Fixed by adding an explicit ldb parameter and slicing per n-tile (which
     also bounds ozsliceB_kernel's gridDim.y for large N and shrinks the ozSB workspace).
  2. THE MAJOR ONE: oz_gemm's kernels ran on a stream created with cudaStreamNonBlocking,
     while getrf/trsm/laswp all ran on the default/NULL stream. cudaStreamNonBlocking
     streams explicitly opt OUT of the legacy implicit ordering the NULL stream normally
     provides -- so oz_gemm could read L21/U12 before those writes landed: a silent
     cross-stream race. Symptom: residual ~1e10 (FAILED) in the full solver, while an
     ISOLATED gemm-only correctness test PASSED at every scale (because that harness had
     incidental full syncs around the call, masking the missing dependency -- a reminder
     that when an isolated unit test passes but the real call site fails on the same math,
     suspect the test's own synchronization is hiding a race the call site doesn't have).
     Fix: set sOz = 0 (the NULL stream) -- this driver has no pipelining to lose by
     serializing on one stream. After the fix: selftest PASSES (native residual 0.004,
     ozaki residual 3.6, both well under the 16.0 HPL threshold).
- PERFORMANCE RESULTS (N=98304 and 122880, NB=2048, AGG=6/K=12288, OZ_NT=1024,
  native_tail=8192, all correctness-verified PASSED):
    N=98304:  46.06 TFLOPS  (wide_gemm 73.0% of wall time)
    N=122880: 47.16 TFLOPS  (wide_gemm 77.9% of wall time)
  AGG sensitivity sweep (N=32768, native_tail=0 to force pure-Ozaki, no tail fallback):
  AGG=1..8 all land in a narrow 31.5-32.9 TFLOPS band (AGG=8 best at this N, but the whole
  sweep is nearly flat) -- aggregation itself is NOT the lever that explains the gap to
  63.1 TF; our scheme-I kernel's own sustained rate is the ceiling.
- COMPARISON: our real, correctness-verified single-GPU ceiling with Ozaki-I + AGG
  architecture is ~47 TFLOPS, about 25% BELOW the teammate's reported 63.1 TFLOPS
  (Ozaki-II/GEMMUL8 + AGG). This ~47 TF figure is consistent with the per-GPU rate implied
  by our production MULTI-GPU numbers (403.3 TF / 8 GPUs = 50.4 TF/GPU single-node;
  792.2 TF / 16 GPUs = 49.5 TF/GPU two-node) -- i.e. this standalone harness is measuring
  something real, not an artifact of the new driver, and the earlier isolated-GEMM-only
  comparison (07-05, "Ozaki-II evaluated & REJECTED", ~62 TF-eq isolated for scheme-I vs
  ~62.9 TF-eq for scheme-II at matched HPL accuracy -- a near tie) does NOT hold once
  embedded in a full solver: scheme-II's real-world full-HPL advantage over scheme-I is
  large (63.1 vs ~47) even though the two schemes' pure-GEMM ceilings were nearly tied in
  isolation. This points at a solver-level (not kernel-level) efficiency gap -- open
  question for the next investigation pass.
- NOT YET DONE: root-causing WHY scheme-II's full-solver number so exceeds scheme-I's
  full-solver number despite near-tied isolated GEMM ceilings; deciding whether to port
  scheme-II (GEMMUL8) into the AGG architecture as a replacement backend, or to look for a
  scheme-I-specific inefficiency in this new solver driver (panel_fact + inner_update +
  wide_trsm + laswp already total ~22-27% of wall time here, capping the achievable
  overall rate even if wide_gemm itself is near its own ceiling).
  [RESOLVED 07-08 — see next section: GEMMUL8 ported as a backend; the gap is real and
  scheme-level, and with tuning our GEMMUL8-backend driver reaches 75-80 TF.]

## ★ GEMMUL8 (Ozaki-II) BACKEND PORTED & TUNED: 75.2 TF robust / 79.9 TF max (2026-07-08)
## — beats teammate's 63.1 by +19%/+27%, beats our own Ozaki-I ceiling by +59%/+69%
- MINIMAL LIBRARY BUILD SOLVED: RIKEN GEMMul8's stock `make` compiles the full
  op x type x backend x fillmode template matrix (hundreds of heavy objects) and died
  before (cicc SIGTERM, resource kill). Traced the actual link-time dependency set of
  `gemmul8::gemm<double, Backend::INT8>` (Func::gemm, real double, full/nonunit only):
  9 objects total. Recipe (scripts/build_gemmul8_min.sh): pass a restricted INST_MK
  (only gemm + oz2/scaling/{accu,fast,general} + oz2/mod + oz2/undo_scaling + worksize +
  trsm's block_size stub) and override each instantiation-list make variable down to
  the single needed combo (command-line make vars override in-file := assignments).
  Notes: (a) scaling::fast::scaling internally calls scaling::general::scaling_rowwise
  -> the general rowwise object IS needed; (b) worksize.o drags in workSizeTrsm ->
  needs the tiny block_size_trsm.o; (c) matmult is header-only (direct cublasLt calls),
  no object needed. Library: gemmul8/lib/libgemmul8.a (~8.9 MB). Main app must compile
  with -std=c++20 and link -lgemmul8 -lcuda -lnvidia-ml -ldl extra.
- INTEGRATION: hpl_single.cu gained a third backend (HPLS_BACKEND=gemmul8) mirroring the
  teammate's n-chunked update with skip_scalA reuse (A/L21 quantized once per K-block,
  chunks 2.. skip re-scaling), tail columns <1024 absorbed into the previous chunk.
  Env knobs: HPLS_G8_MODULI (default 14) / HPLS_G8_FAST (0) / HPLS_G8_CHUNK (16384) /
  HPLS_G8_SKIPA (1). Selftest: all 3 backends PASS (native 0.004 / ozaki 3.6 / gemmul8
  0.047 residual).
- BASELINE (teammate-matched config NB=1024 AGG=8 moduli=14 fast=0 chunk=16384):
  55.48 TF @N=98304, 57.06 @110592, 56.04 @122880(chunk=2048 for memory). NOTE: at the
  teammate's own config we measure ~56, i.e. BELOW their claimed 63.1 -- their 63.1 was
  presumably at different N/conditions; either way it's now surpassed decisively.
- TUNING DISCOVERIES (all at N=98304..122880, all residuals well under 16.0):
  1. **fastmode=1 is BOTH faster AND more accurate on HPL random matrices** --
     counterintuitive but consistent: at moduli=13, residual 0.029 (fast) vs 0.765
     (accurate); at moduli=14, 0.0037 (fast) vs 0.053 (accurate). Fast also skips the
     extra scaling int8-GEMM (scale phase 0.62s -> 0.14s) => +5% wall. The "accurate"
     mode's tighter worst-case scaling bound apparently costs average-case accuracy here.
  2. **moduli dial**: each modulus removed = ~13x residual growth (fast mode) and ~+5-7%
     speed. fast/14: 0.0037 -> fast/13: 0.029 -> fast/12: 0.33 -> fast/11: ~5. moduli=10
     would extrapolate to ~60-80 => FAIL; 11 is the floor at this N.
  3. chunk sweet spot at N=122880 is 3072 (2048: 59.4@m13; 3072: 67.6@m13; 4096: 70.2 vs
     71.2@m13fast — 3072 consistently best; memory forbids >=8192 at this N).
  4. Block structure nearly flat: NB=1024/AGG=8 vs NB=2048/AGG=4 (both K=8192) within
     0.3%; NB=2048/AGG=6 (K=12288) best at N<=114688 but its A-quantized buffer
     (k x m x num_mat int8) doesn't fit at N=122880.
- FINAL NUMBERS (N=122880, NB=1024, AGG=8, fast=1, chunk=3072, skipA=1):
    moduli=12 (ROBUST CHAMPION): 74.5-75.4 TF across 6 runs/4 seeds, residual
      0.31-0.37 (48x headroom). scripts/single_g8perf4.sbatch run 2 + single_g8seed.sbatch.
    moduli=11 (AGGRESSIVE MAX):  79.2-79.9 TF across 4 runs/4 seeds, residual
      4.7-6.0 (2.7x headroom vs 16.0 threshold). Passes, but thin margin.
- WHY SCHEME-II WINS (closes the 07-05 "isolated tie" puzzle): our Ozaki-I fused-diagonal
  D=5 does 21k-equivalent int8 GEMM work and CANNOT go lower (D=4 fails HPL accuracy =
  hard floor). GEMMul8's CRT dial is finer: at HPL's actual accuracy budget it needs only
  11-13 full-k GEMMs (fast mode, no extra scaling GEMM) = 11-13k work, i.e. 40-48% less
  int8 compute than scheme-I at matched-and-passing accuracy. The 07-05 comparison tied
  because it matched FULL fp64 accuracy (s=33/t=11 CRT vs s=6/D=5), a regime HPL never
  needs. LESSON: compare emulation schemes at the TARGET accuracy budget, not max accuracy.
- MULTI-GPU IMPLICATION (future work): production chpl4x4 runs scheme-I at ~49.5 TF/GPU.
  A gemmul8-backend trailing update at moduli=12-13 could lift the per-GPU GEMM rate
  substantially (residual scales ~linearly with N: 0.33 x 5.6 ≈ 1.9 at N=688128 => still
  safe; moduli=11 would NOT survive production N). Caveat: the distributed design
  broadcasts our scheme-I int8 slice layout (ozSAt) over IPC/NCCL; switching schemes
  means reworking the W-transport data plane -- a major but now clearly-motivated rework.
  [SUPERSEDED 07-08, next section: the caveat was WRONG -- the champion W transport is
  plain fp64 (WI8=0), so no data-plane rework was needed at all; and the moduli=11
  prediction was too pessimistic (NB=4096 configs run smaller N with fp64 urgent columns,
  residuals 1.4-1.9, safely under 16).]

## ★★ DISTRIBUTED GEMMUL8 PORT + NB=4096 RETUNE (2026-07-08): NEW RECORDS AT 2 & 4 NODES
## 2-node 990.7 TF (was 792.2, +25.1%) | 4-node 1865 TF (was ~1579, +18.1%)
- PORT: CHPL2D_G8=1 in chpl4x4.cu/chpl2d.cu swaps the scheme-I fused-diagonal engine for
  gemmul8::gemm on the big s0 trailing update ONLY. W transport (fp64 broadcast), U
  staging, urgent sP updates, tournament pivoting: all untouched. M = inv(U11)*U12 is
  computed per tile on s0 from persistent d_U/d_pkg (same as the fp64 fallback); W is
  quantized once per panel and reused across all chunks/tiles via skip_scalA. Knobs:
  CHPL2D_G8_MODULI (12) / _FAST (1) / _NT (2048). Correct on first try at every scale
  (residuals m12 ~0.06-0.14, m13 ~0.01, m11 ~1.2-2.2; all reproducible bit-identical).
- PER-PANEL k=2048 VERDICT (the naive drop-in): correct but SLOWER than scheme-I champion
  at every scale: 2n 779 vs 796 (-2%), 4n 1517 vs 1574 (-3.9%), 1n 384-403 vs 403.
  Single-GPU AGG=1 probe predicted this: at k=2048 gemmul8's mod/CRT overhead (per-element,
  per-panel) eats the win -- engine only +6.9% isolated, and in the distributed pipeline
  even that vanished into s0 IDLE TIME: timing showed s0_gemm 89s vs OZ 99s but s0_gap 14.5s
  vs 2.8s (2n). KEY INSIGHT: at champion N the run is PANEL-CHAIN-BOUND (~425ms/panel chain
  latency at 2n; OZ's s0 busy time ~410ms/panel just masks it). ANY engine speedup without
  shortening the chain converts to idle, not wall-clock. (This also retro-explains the
  800-campaign plateau at 792.)
- THE UNLOCK — NB=4096: halves the panel COUNT (243->122 at 2n) => halves total chain
  latency, AND doubles k per update => halves gemmul8's mod/CRT overhead. Scheme-I lost
  at NB=4096 back in the day (716 vs 792: no k-gain for its fixed 21k work, N/buffer cost)
  but for gemmul8 both effects are wins. Memory forces N down (~460800 at 2n vs 497664)
  -- k-gain >> N-loss, exactly like the single-GPU AGG finding.
- moduli=11 fast=1 is SAFE at these N (fp64 urgent columns + smaller N than the old
  champion): residual 1.94 @460800(2n), 1.40 @622592(4n), threshold 16. m12 -> m11 = +5-8%.
- NT (g8 n-tile) matters a lot at NB=4096: 2048 -> 3072 = +10% (2n 883->977). 3584+ OOM
  at max N; NT=4096 at smaller N loses net. NCH=8 > NCH=4 (+1.3%) at NB=4096 (bigger,
  fewer W broadcasts want more channels; old champion used NCH=4 at NB=2048).
- ★ CONFIRMED CHAMPIONS (5-round showdowns, jobs 169182/169183; sigma < 0.1%, residuals
  bit-identical across rounds):
    2-node: N=460800 NB=4096 P4Q4, CHPL2D_G8=1 G8_MODULI=11 G8_FAST=1 G8_NT=3072,
            NCCL_MAX_NCHANNELS=8 MIN=1, WCH=4 PRESEL=1 PRESEL2=1 WS=64 XT=6144 BUF=2M
            CONN=32 -> rounds 988.5/988.9/989.1/989.2/990.1, **median 989.1 TF**
            (old champion 792.2 -> +24.9%; 61.8 TF/GPU, was 49.5)
    4-node: N=622592 NB=4096 P4Q8, same env -> rounds 1852/1852/1853/1854/1854,
            **median 1853 TF** (old champion 1593 -> +16.3%; 57.9 TF/GPU)
  Reproduce: scripts/c4x4_g8_final2n.sbatch / c4x4_g8_final4n.sbatch.
  NB=6144 probes: silent failure at both scales even at reduced N (cusolver NB-hang
  family or alloc) -- not pursued. NCH=16: 982 < NCH8's 989. Timing at 2n champion:
  s0_gemm 48s, s0_gap 17s -- still ~26% chain-exposed; the next big lever would be
  true chain surgery (faster exchange/factor path), not the engine.

## m-split W pipeline: investigated & CLOSED, no net gain (2026-07-08, same session)
- Fine timing (TIMING=2) decomposed the 17s exposure: wpbc (W broadcast wait) ~10.2s is
  the biggest single chain component; getrf/exchange are presel'd down to <1s.
- Built CHPL2D_G8_MSPLIT=1: the W broadcast repacks into ROW chunks (dense rn x jb
  blocks -- the root's staging memcpy2D does the repack for free, same bytes), consumers
  (s0 G8 update, urgent sP update, fp64 fallback) walk chunks with per-chunk evWc waits,
  each chunk owning a private quantized-A slot (g8_workA[cw]) and sharing one B_lo via
  skip_scalB. s0 no longer takes a panel-wide evReadyW barrier.
- RESULT: it DOES hide the W wait (s0_gap 17.4s -> 4.3-6.1s) but the wall is UNCHANGED
  (980 vs 989 barrier; slightly worse). WHY: the hiding is capped by how much
  W-INDEPENDENT work exists per panel -- only ~one tile's worth (~45ms) vs a ~90ms/panel
  broadcast -- so the stall just migrates INSIDE the first tile's chunk loop (s0_gemm
  48s -> 60s; the timer now contains embedded evWc stalls). Measured across WCH=2/4/8
  (all ~equal; WCH=8 worse) and with skip_scalB sharing (+1%, kept). This is a PHYSICS
  cap of the current per-panel dataflow, not an implementation defect: to hide a W
  broadcast you need enough W-independent s0 work, and the panel update has essentially
  none (M-dgemms are only ~2.5ms/tile).
- Kept env-gated (CHPL2D_G8_MSPLIT, default 0 = champion barrier behavior; verified
  bit-identical residual 1.943 and 990.2/990.3 TF after re-gating). One real bug found
  during the re-gate: barrier mode with an external workA DOUBLE-ALLOCATED the A
  workspace (workSize's total already includes it) -> OOM; barrier now passes
  workA=nullptr again.
- REMAINING LEVERS (heavier, unexplored): (a) urgent-G8 -- run the sP urgent update
  through gemmul8 too, cutting its fp64 SM steal from ~87ms to ~35ms/panel (needs its
  own ~4-5 GiB workspace, memory-marginal); (b) genuine chain surgery on the
  exchange/stage-2 path; (c) NB=6144 if the cusolver issue is diagnosed.
- 1-NODE: NO WIN. chpl2d (P2Q4) mloc=N/2 doubles the A-quantization buffer => NB=4096
  only fits at N<=323584 where the N-loss cancels the gain (386 vs champion 403.3).
  G8 m11 NB=2048 @348160 = 402.9 ~ ties. chpl4x4 P4Q2 at 1 node crashes (pre-existing
  edge bugs, "P4Q2 port" was already abandoned). Single-node champion REMAINS scheme-I
  OZ 403.3.
- Cluster note: nodes node011/012 are BAD (cudaGetDeviceCount -> unknown error for
  ANY binary incl. old champion); all jobs now use --exclude=node[011-012].

## Final knob sweep at the NB=4096 G8 champion (2026-07-08): baseline confirmed optimal
- N=460800 m11 NT3072: NCH8/MIN1/CONN32 = 987.8 (best) vs NCH12 979.7, MIN8 987.4,
  CONN64 986.3, NCH6 984.5. No cheap knob left; the 989-champion config is a verified
  local optimum. Session closed at 2n 989.1 / 4n 1853 medians.

## Convergence probes round 2 (2026-07-08): every remaining direction measured to closure
- NB=6144/8192 root cause CONFIRMED: buildlists_kernel (exchange list builder) launches
  with 40*NB+1024 dynamic shared memory; 6144 needs 246KB > H200's 232KB per-block cap ->
  cudaFuncSetAttribute fails at startup (line 777 "invalid argument"). A kernel rewrite
  could lift this, but see next point -- it would not pay.
- NB=5120 (fits smem at 206KB): runs correctly but 947.2 TF @450560 (memory forces N down
  from 460800) vs champion 989.1 -- the NB curve has PEAKED at 4096: beyond it, the
  workspace growth costs more N than the halved-chain + k-gain returns (plus non-power-
  of-2 k may hurt the int8 GEMM). NB>=6144 via kernel rework is therefore predictably
  negative; not pursued.
- msplit @466944 (m-split's smaller C workspace does unlock the bigger N): 983.7 < 989.1
  -- the m-split engine tax eats the +1.3% N gain. Rejected.
- 4-node N=630784/638976 @NT3072: OOM (NT3072 fits only up to 622592).
- STATE: champions 2n 989.1 / 4n 1853 stand at a fully-explored local optimum. Every
  cheap and medium lever now has a measured negative/neutral result or a first-principles
  closure; the only unexplored direction left is deep chain surgery (exchange/stage-2
  restructure, multi-day work, bounded by the ~17s/29s residual chain exposure).

## 4-node knob sweep (2026-07-08, final): baseline confirmed optimal there too
- N=622592 m11 NT3072, same-node-set A/B: WCH4/NCH8/PRESEL2=1 = 1822 (best) vs WCH8
  1819, WCH2 1798, NCH12 1804, PRESEL2=0 1788 (PRESEL2 still worth +1.9% at NB=4096).
  (Absolute numbers ~1.7% below the 1853 showdown median -- different node set; the
  within-job A/B is the valid comparison.)
- CAMPAIGN CLOSED: both champions sit at verified local optima across every knob and
  every structural direction. Next session's only remaining avenue: deep chain surgery.

## Paper feasibility: ZipServ (arXiv:2603.17435) lossless-compression -> NOT APPLICABLE (2026-07-11)
- The paper: BF16 LLM weights have highly skewed, CONTIGUOUS exponent distributions
  (top-7 exponents cover >95%, field entropy 2.57-2.74 of 8 bits) -> fixed-length 3-bit
  codeword via triple bitmaps, decoded in-register inside a fused GEMM. 30% size cut,
  2.21x kernel speedup (bandwidth-bound GEMV/GEMM regime).
- Why it does not transfer to CHPL: (1) format asymmetry -- BF16's exponent is 50% of
  the bits; fp64's is 11/64 = 17%, and the 52 mantissa bits of HPL's evolving trailing
  matrix are entropy-full random bits.  Perfect exponent compression caps at 17% savings;
  realistic (~4-bit codes) ~10-12%.  (2) empirical precedent: the WI8 campaign showed
  even 25% W-byte savings nets a LOSS after implementation overheads (v1 -0.3%, v2 -6%);
  10% cannot win.  (3) our int8 quantized operands (A_lo/B_lo digits) are uniform-random
  bytes -- incompressible; our GEMMs are compute-bound (IMMA), not weight-bandwidth-bound
  like LLM decode.  VERDICT: skip; recorded so it is not re-litigated.

## Deep chain surgery round 1 (2026-07-11): W-transfer hypothesis DISPROVEN by microbenchmark
- Premise from fine timing: wpbc (W-broadcast wait) ~10.2s = top chain item; inferred
  4.1 GB / ~90 ms = ~45 GB/s = single-IB-rail-bound -> built WB2 (CHPL2D_WB2): row split
  into even/odd-q half-communicators broadcasting half of W's columns each in parallel
  (different GPUs' NICs), same-node q-pairs swapping halves over CE/IPC.
- WB2 ran end-to-end but FAILED correctness (residual ~1e8): the CE pulls read the
  partner's d_W[b] without a consumed-handshake, racing its k+2 same-parity overwrite
  (the original NCCL path has no remote reads; the wce path has flags for exactly this).
  Fixable with a reverse flag -- but pointless, because:
- MICROBENCH (src/wbench.cu, scripts/wbench.sbatch, 16 ranks/2 nodes): after NCCL
  warmup, the EXISTING single 4-rank row broadcast of 3.84 GB runs at **95.1 GB/s**
  (42.3 ms) -- NCCL already drives ~2 NICs (multi-channel bidirectional rings).  Dual-rail:
  96.7 GB/s (41.6 ms) = +1.7%.  The "45 GB/s" inference was wrong attribution: wpbc's
  ~90 ms/panel is ~40 ms transfer + ~50 ms UPSTREAM production latency (exchange/factor/
  urgent serialization + cross-stream contention), which no transport change can touch.
- WB2 REVERTED from chpl4x4.cu (dead premise + correctness hazard); microbenchmark and
  this record kept.  Post-revert sanity: 992.8 TF @460800, residual bit-identical.
- Chain accounting after this round: per-panel cycle ~570 ms vs s0 busy ~420 ms; measured
  sP phases account for only ~200 ms -- the remaining ~370 ms hides in inter-phase queue
  delays, flag-wait serialization, and fp64-urgent SM contention with the int8 engine.
  A REAL chain surgery = re-architecting the panel pipeline (depth-2 lookahead with
  double-buffered panel state, or urgent-update elimination via deferred column updates)
  -- multi-session, high-risk, bounded gain (~17s/66s at 2n).  NCCL env sweep (ALGO/
  CROSS_NIC/QPS/CHUNKSIZE) all neutral.  Champions unchanged: 2n 989-993, 4n 1853.

## ★ SINGLE-NODE RE-SWEEP (2026-07-11): 472.0 TF — the "no G8 gain at 1 node" verdict was WRONG
- User asked whether the single-node record had ever been properly re-tested after the G8
  work.  It had NOT: the original 1-node G8 exploration (07-08) only tried NT=1024 at
  NB=4096 (386.3 -> dismissed as "N loss cancels the gain") and only tried m12 at NT=2048
  (OOM -> never retried with m11's ~1 GiB smaller workspace).  Both were coverage gaps,
  not physics.
- Completion sweep (chpl2d P2Q4, G8 m11 fast, PRESEL=1, NCH=6):
    NB2048 m11 NT2048 @348160: 416.4  (m11 fits where m12 OOM'd; +3.2% over old champion)
    NB4096 m11 NT1024 @315392: 386.3  (the old, misleading data point)
    NB4096 m11 NT2048 @315392: 427.1  |  NT2560: 466.1  |  NT2816: 469.8  (NT lever = +22%!)
    NB4096 m11 NT2816 @319488: 472.0  |  NT3072 @319488: 468.2  |  NT2560 @323584: OOM
    NB3072 probe: silent fail (cusolver NB-hang family, same as 1536/2560)
- ★ NEW 1-NODE CHAMPION (5 rounds, job 177634): N=319488 NB=4096 P2Q4, CHPL2D_G8=1
  G8_MODULI=11 G8_FAST=1 G8_NT=2816 PRESEL=1 NCH=6 -> 471.5/471.8/472.0/472.1/472.3,
  **median 472.0 TF**, residual 1.999 bit-identical.  vs old scheme-I champion 403.3 =
  **+17.0%**; vs NVIDIA container 385.8 = +22.3%; vs Dell XE9680 published 394.5 = +19.6%.
  Reproduce: scripts/c2d_g8_1ncrown.sbatch, dat/HPL_1node_g8.dat.
- LESSON (recorded so it sticks): a config dimension that was decisive at one scale (NT at
  2 nodes: +10%) must be re-swept at every scale before concluding "no gain" — the 1-node
  NB=4096 dismissal was based on a single point along the un-swept dimension.

## ★ CHAMPION PROFILE + INT8-W-TRANSPORT AUDIT (2026-07-15, scheduled task)
### Part 1: time distribution of the G8 champions -- the chain is LATENCY-bound, engine is AT CEILING
| metric | 2-node (66.4 s, 112 panels) | 1-node (45.8 s, 78 panels) |
|---|---|---|
| s0_gemm (int8 engine busy) | 49.1 s = 74% | 32.9 s = 72% |
| **s0_gap (chain exposure)** | **16.2 s = 24%** | **12.3 s = 27%** |
| chainB (fully overlapped) | 12.2 s | 6.7 s |
| **per-panel cycle** | **593 ms** | **587 ms** |
| **per-panel idle** | **145 ms** | **158 ms** |

sP 9-segment split (2n / 1n, seconds):
  wpbc (W-broadcast wait)                 9.55 / 8.06   <- biggest at BOTH scales
  g1U01+g2U2+g3U3 (urgent + U-stage fp64) 11.68 / 4.93
  getrf1+candx+getrf2 (pivoting)           0.56 / 1.57   <- presel has crushed this
  L21W                                     0.24 / 1.42

- **DECISIVE**: 1-node has NO InfiniBand (W moves over NVLink, ~10x the bandwidth) yet wpbc is
  STILL 8.06 s = 84% of the 2-node value, and per-panel cycle/idle are nearly IDENTICAL across
  scales (593/145 vs 587/158 ms).  If wpbc were transfer-bound, removing IB would collapse it.
  It doesn't.  wpbc is UPSTREAM PRODUCTION LATENCY (waiting for the owner to finish factorize ->
  L21 before it can broadcast at all).  The chain is LATENCY-bound -- a fixed serial per-panel
  cost -- not bandwidth- or scale-bound.  This is the third independent kill of the whole
  transport-optimization family (after the wbench microbenchmark and the WI8 x3 history).
- **Engine is already at its ceiling**: during s0_gemm busy time the trailing update runs at
  2/3 N^3 / 16 GPUs / 49.1 s = **83 TF/GPU**, ABOVE the isolated single-GPU champion (79.9 TF).
  There is NO headroom left in the engine.  100% of the remaining opportunity is the 24-27% gap.

### Part 2: INT8 W transport under GEMMul8 -- KILLED BY THE BYTE COUNT (before writing any code)
- GEMMul8's INT8 backend: `num_mat(NUM_MODULI) = NUM_MODULI` (src/oz2/common/table.hpp:80-90),
  and `total_size_A += sizeof(int8_t) * sizeA * num_A_lo` with sizeA = k_pad*m_pad (FULL matrix
  per plane).  So the quantized A_lo costs **NUM_MODULI bytes per element**.
- Audit (scratchpad wsz.cu, calls the real gemmul8::workSize):
    2-node champ: fp64 W 3.77 GB  vs  A_lo 5.19 GB  = **1.38x BIGGER**
    1-node champ: fp64 W 5.23 GB  vs  A_lo 7.20 GB  = **1.38x BIGGER**
    4-node champ: fp64 W 5.10 GB  vs  A_lo 7.02 GB  = **1.38x BIGGER**
  Break-even needs NUM_MODULI < 8; the measured accuracy floor is 11.  Transporting the
  quantized A would INCREASE the broadcast by 38%, not halve it.
- ROOT CAUSE (the general lesson): the two Ozaki schemes have OPPOSITE information geometry.
  Ozaki-I is a POSITIONAL DIGIT decomposition -- the 53-bit mantissa splits into 6 x 7-bit
  digits = 6 B/elem < 8 -> a genuine 25% paper saving (which is why WI8 was at least arguable,
  even though all 3 implementations still lost).  Ozaki-II/CRT is a REDUNDANT RESIDUE
  representation -- each modulus needs a full byte and the moduli product must exceed the
  dynamic range -> it necessarily EXPANDS.  A CRT representation can never be a compression
  format.  DO NOT revisit int8/quantized W transport under GEMMul8 at any moduli count.
- Triple-dead: (1) byte count is negative, (2) even a free transport wouldn't help (wpbc is
  production-latency, see Part 1), (3) WI8 x3 showed implementation overhead eats any paper
  saving.  Consequently the skip_scalA preconditions and the "sP urgent fp64 consumer can't eat
  int8" problem never needed solving -- the premise died at step 1.

### Ranked directions after this profile
1. **Depth-2 lookahead / chain restructure** (ceiling ~ +33%): the engine is saturated; every
   remaining gain is in the 145-158 ms/panel idle.  Needs double-buffered panel state so panel
   k+1's W production (factorize -> L21 -> broadcast) doesn't wait on panel k's update.
   Multi-session, high-risk, but it is the ONLY large lever left.
2. **urgent-G8** (indirect, shortens the chain): the sP urgent update is still fp64
   cublasDgemm (11.7 s at 2n) -- it sits ON the chain AND steals SMs from the int8 engine.
   Converting it to gemmul8 could cut it 2-3x.  Previously rejected on memory (~11 GiB), but the
   m-split per-chunk workspace machinery now exists; worth re-costing.
3. **CLOSED for good** (do not revisit): every transport-layer idea (int8 W x4 incl. this audit,
   compression/ZipServ, dual-rail WB2, m-split, NCCL env sweeps), NB > 4096, and any pure engine
   speedup (the engine is measurably at its ceiling).

## fast.cu (pranjalssh, "outperforming cuBLAS on H100") applicability audit (2026-07-16)
User asked whether fast.cu's cuBLAS-beating GEMM techniques could be borrowed for CHPL/GEMMul8.
Read all 12 kernel versions; the final (matmul_12) technique stack: WGMMA (m64nNk16 bf16->f32),
TMA (cp.async.bulk.tensor.3d GMEM->SMEM), 128B swizzle, warp specialization (producer WG loads
via TMA / consumer WGs do WGMMA), register realloc (setmaxnreg.inc/dec), thread-block CLUSTERS
(__cluster_dims__, cross-SM mbarriers via mapa.shared::cluster), TMA MULTICAST (one tile ->
multiple SMs in the cluster), mbarrier producer/consumer pipeline, stmatrix epilogue, persistent
kernel + L2-aware tile schedule.  bf16->fp32, square N=4096-8192, H100.  Claims +6.6%/+1.6% vs
cuBLAS.

VERDICT: NONE transfers to a net win.  Technique-by-technique, grounded in measurement:
1. **Custom int8 GEMM to beat cublasLt?** DEAD 3 ways:
   (a) EMPIRICAL (i8bench.cu, this session): at every production TN-int8 shape (k=4096, n=3072,
       m=8192..115200), cublasGemmEx(CUBLAS_GEMM_DEFAULT) -- the path GEMMul8's cuBLAS-handle
       entry uses -- and a fully HEURISTIC-TUNED cublasLt are within 0.4% and pick the SAME algo
       (idx=0).  cuBLAS default already runs 1716-1894 TOPS = ~95% of H200's ~1979 int8 peak.
       There is NO GEMM headroom for a hand kernel to capture.  fast.cu's whole premise ("beat
       cuBLAS's default GEMM") evaluates to +0% in our int8 shape.
   (b) HISTORY: the 07-02..05 custom-kernel campaign (ozkernel.cuh, 12 WGMMA/mma.sync versions)
       already peaked at 630 TOPS vs the ~1100 needed and LOST to cublasLt -- closed with a
       measured negative.
   (c) SHAPE: fast.cu's cluster-multicast edge comes from LARGE SQUARE N where A/B tiles are
       reused across the cluster; our GEMMs are tall-skinny (m up to 115k, n=3072) -- the regime
       where that reuse advantage evaporates.
2. **TMA multicast for the W broadcast?** DEAD: TMA multicast is INTRA-GPU (a cluster is <=16 SMs
   on ONE GPU, sharing the on-die SM-to-SM network).  Our W broadcast is INTER-GPU / INTER-NODE
   (NCCL over NVLink/IB).  TMA cannot cross GPUs.  Not applicable.
3. **cublasGemmEx -> tuned cublasLt path in GEMMul8?** The one concrete lever reading the code
   surfaced (GEMMul8's cuBLAS-handle path uses cublasGemmEx/DEFAULT, no algo search; gemmLt uses
   tuned cublasLt).  TESTED -> 0.4% / same algo (see 1a).  Not worth the gemmLt plumbing.
4. **TMA for GEMMul8's memory-bound scale/mod/recon kernels?** Theoretically possible (TMA
   accelerates GMEM loads) but: those are internal to GEMMul8 (would fork its kernels), and they
   are only ~16% of emulation time (int8-gemm is 84%: single-GPU internal timing scale 0.22 /
   gemm 9.75 / mod 1.04 / CRT+unscale 0.70 s), much of it overlapped.  Small ceiling, high effort,
   not on the critical path.  Rejected.
- The DEEPER reason nothing applies: fast.cu optimizes the ONE thing that is already at its
  ceiling for us (the int8 GEMM, at ~95% peak via cublasLt) and touches NOTHING in the 25%
  chain-latency gap that is our actual bottleneck (see the 07-15 profile).  It is the right
  worklog for a from-scratch bf16 GEMM on square matrices, and orthogonal to a distributed
  int8-emulated HPL whose GEMM is a solved library call.
- KEEP-worthy note (not a CHPL change): fast.cu is an excellent reference IF we ever needed a
  from-scratch int8 kernel with a NON-standard epilogue that cublasLt can't express (e.g. fusing
  the Ozaki reconstruction INTO the GEMM epilogue to kill the int32 D-traffic).  That was the old
  "fused int8 kernel" dream; it needs >1100 TOPS MMA-only to win after the epilogue, and the
  12-version campaign topped out at 630.  fast.cu's TMA+warp-spec+cluster scaffolding is a better
  starting point than what we hand-rolled, but the ceiling math (must beat cublasLt's ~1900 TOPS
  while ALSO doing the epilogue in-kernel) still says lose.  Do not attempt without new hardware.

## xhpl (HPL-NVIDIA 26.2.0, Blackwell) reverse-engineering report -- point-by-point audit (2026-07-18)
User supplied a Ghidra/fatbin RE of NVIDIA's xhpl emulated-FP64 path (SM100-only: int8 UMMA +
Ozaki-II RNS, ModFunc fused in epilogue, NB=1024 megakernel, NVSHMEM pivot, depth-1 lookahead)
with 10 suggested "gaps to exploit".  Mapped every point against OUR measured record:

| report lever | our status | evidence |
|---|---|---|
| 5.1 (RED) fewer slices/moduli | **ALREADY SHIPPED = our +25%** | moduli 14->11 + fastmode IS this exact lever, executed 07-08 before reading this report; floor measured (m10 fails) |
| 5.2 (RED) better moduli table | **NO ROOM** | recovered GEMMul8's set from table.hpp P-products: {256,255,253,251,247,241,239,...} = max-density coprime set, 0.9948 bits/byte (vs xhpl's worst group 0.9374).  No denser byte-sized set exists.  Mixed int8/int16 slices: no int16 IMMA path on tensor cores -> dead |
| 5.3 (RED) NVSHMEM bcast/swap | **EQUIVALENT EXISTS + measured moot** | our CE/IPC + monotone-flag data plane IS device-initiated one-sided comm (same semantics as put+signal); NVSHMEM itself measured neutral-negative (v15 era); and the 07-15 1-node-no-IB profile proved our chain is PRODUCTION-latency-bound, not transport -- the lever addresses a bottleneck we do not have |
| 5.4 (YEL) lookahead depth>=2 | **= OUR PROFILED #1 DIRECTION** | independent convergence: their RE and our profiling name the same next move; xhpl is also depth-1 (gemm_event[0/1] only), so depth-2 would out-architect xhpl too |
| 5.5 (YEL) fuse scale/mod/recon into epilogue | dead for us at multi-GPU | cublasLt has no custom-epilogue hook; a custom/CUTLASS-EVT kernel must first match cublasLt's measured ~95%-of-peak (fast.cu audit); and even a FREE fusion converts to s0 idle at 2n/4n (engine-gain->idle measured 3x).  Only single-GPU hpl_single (+3-5%) would see it -- low value |
| 5.6 (YEL) flexible/bigger NB | **AHEAD of the report** | report suggests 1536/2048 over xhpl's fixed 1024; we measured the full curve and run NB=4096 (peak; 5120 -4%, >=6144 smem wall) |
| 5.7 (YEL) shape-adaptive GEMM tiles | N/A -- solved by library | cublasLt heuristic picks the same algo as exhaustive search at every production shape (i8bench, 0.4%) |
| 5.8 (GRN) finer scaling granularity | dead by structure | GEMMul8 is already per-ROW (finer than xhpl's 32/64 blocks); k-blocked scaling would break the single-shift-per-dot-product CRT factorization -> forces small-k GEMM calls = the measured k-penalty regime |
| 5.9 (GRN) timing pollution | N/A | our showdown methodology already interleaved, solve-time-only |
| 5.10 (GRN) OOC buffering | N/A | in-core |

- NET RESULT: **zero new actionable items** -- but strong external validation: the report's two
  RED levers are exactly what we already shipped (moduli dial; device-side data plane), and its
  main YELLOW is exactly our profiled #1 (depth-2 lookahead).  Two independent analyses (their
  binary RE of NVIDIA's code, our delay-probe/profile of ours) converge on the same frontier.
- STRATEGIC NOTE FOR A BLACKWELL FUTURE: on SM100 the container's emulation ACTIVATES (unlike
  H200 where it falls back to fp64), so any future Blackwell port must re-fight the showdown
  against xhpl's emulated path.  This report is the opponent's weapon list for that day; our
  standing edges per its own table: NB=4096 vs their fixed 1024, moduli floor-hunting vs their
  conservative slices, depth-2 (if built) vs their depth-1.  Their edge over us: ModFunc fused
  in a CUTLASS-EVT epilogue (worth copying ON BLACKWELL where the kernel would be rewritten
  anyway).  Champions unchanged: 1n 472.0 / 2n 989.1 / 4n 1853.

## G8-era 1-node CONTAINER SHOWDOWN + stale-binary fix (2026-07-18)
- User asked whether the 1-node record lacked GEMMul8.  It does NOT (the 472.0 champion IS
  G8, crowned 07-11) -- but the question exposed a REAL trap: the local build/chpl2d binary
  was a stale Jul-2 pre-G8 build with ZERO knowledge of CHPL2D_G8 (env silently ignored ->
  falls back to plain fp64).  The champion runs had used the separately-built chpl2d_g8.
  FIXED: build/chpl2d rebuilt from current src (G8-enabled); README command path verified
  live (470.6 TF).  Lesson: after a backend lands, rebuild ALL binary aliases -- a stale
  binary silently ignoring new env vars is worse than a crash.
- NEW RESULT -- the formal interleaved container showdown at 1 node was still scheme-I-era
  (403.3 vs 385.8, +4.5%).  Redone with the G8 champion (job 191605, 3 interleaved rounds):
    CHPL G8  (N=319488 NB=4096 P2Q4 m11 fast NT2816): 470.7 / 471.1 / 470.8 -> median 470.8
    container (N=366735 NB=1024 P4Q2, its tuned best): 385.8 / 385.9 / 385.9 -> median 385.9
  **+22.0%, 3/3 complete separation** (our worst 470.7 >> container best 385.9), all residuals
  PASSED.  The 1-node showdown record advances from +4.5% (scheme-I) to +22.0% (G8).

## NSIGHT PROFILE + WGMMA VERIFICATION (2026-07-19)
### Q: does the code use modern tensor-core tech (wgmma)?  A: YES -- proven at 3 levels
1. Kernel identity (nsys): the int8 GEMM cuBLAS selects is
   `sm90_xmma_gemm_i8i32_i8i32_i32_tn_n_tilesize128x128x128_warpgroupsize1x1x1_..._5x_cublas`
   -- "warpgroupsize" = Hopper warpgroup kernel.
2. Hardware counters (ncu): `sm__inst_executed_pipe_tensor_op_gmma.sum = 3,604,480` (GMMA ==
   WGMMA at the SASS/counter level), hmma = dmma = 0.  All tensor work is int8 WGMMA.
3. Utilization: `sm__pipe_tensor_cycles_active = 93.9% of peak` -- the tensor pipe is saturated.
   (Division of labor: our own kernels are memory-bound elementwise -- tensor cores don't apply;
   ALL tensor-core work is delegated to cuBLAS, whose WGMMA kernel we now verified is near-peak.
   Hand-writing WGMMA was tried in the ozkernel campaign: best 630 TOPS vs library ~1900.  The
   fp64 urgent/panel gemms use sm90 DMMA kernels, also library.)

### Kernel-level time distribution (nsys, single-GPU hpl_single N=65536, production k=4096 mix)
  int8 WGMMA GEMM        42.1%   (93.9% tensor-pipe -> AT CEILING)
  GEMMul8 side kernels   ~17.4%  (mod_hi2mid x11 9.9% + undo_scaling 6.5% + scaling ~1%)
  fp64 DMMA gemms        ~13%    (urgent/inner/tail)
  cusolver getrf family   8.9%   | trsm 6.0% | laswp gather+scatter 5.8% | matgen 1.3%
ncu SOL on the side kernels:
  mod_hi2mid:    83.1% DRAM throughput -> AT ITS ROOFLINE (DRAM-bound, nothing to recover)
  undo_scaling:  84.0% compute / 59.5% DRAM -> compute-bound in CRT double-double math;
                 theoretical GEMMul8-internal rework worth ~1.3% single-GPU, ~0 multi-GPU
                 (engine gains convert to idle) -> not worth forking GEMMul8.
Distributed rank-0 (chpl2d 1-node N=131072): NCCL Broadcast kernels 45.1% of KERNEL time but
that is overlapped spin-wait on dedicated streams (not exposed wall); flag_wait 5.0% (by
design); buildlists_kernel 1.9% (3.7 ms/panel, on-chain but small); exchange gather/scatter
~1.9%; fp32 stage-1 select (cutlass sgemm) 1.6%.

### Profile-guided probes (2n champion, same-job A/B)
- nsys showed Broadcast_RING_LL -> tried NCCL_PROTO=Simple: 977.0 vs baseline 977.5 = NEUTRAL
  (the LL kernels are small-message/overlapped; protocol not on the critical path).
- pkg broadcast grew 4x with NB (3*NB^2*8 = 402 MB/panel at NB=4096, vs 100 MB when PKGPRE was
  rejected at NB=2048) -> re-tried CHPL2D_PKGPRE=1: 970.4 = WORSE (-0.7%), same verdict as the
  NB=2048-era rejection.  Both leads closed.
### Net verdict (hardened by hardware counters)
The GEMM engine is at ceiling (93.9% pipe), the emulation side kernels are at their respective
rooflines (mod 83% DRAM), and every transport/env lever tests neutral-or-negative.  The profile
re-confirms: the ONLY remaining wall-clock opportunity is the ~25% chain-latency gap (depth-2
lookahead restructure).  No kernel-level optimization is available.  Champions unchanged.

## "Every us Matters" (arXiv 2607.16100, low-latency-nccl) applicability audit (2026-07-28)
The paper: barrier-free device-side synchronization for GPU collectives (LL 16-byte flag+data
packing, sentinel polling, symmetric-memory one-sided stores, LL128 cache-line-atomic
AllReduce) on GB200 NVLink scale-up domains, built on NCCL 2.28's device API.  Wins: 8.7x on
128-BYTE AllReduce (2.37 vs 11.0 us @4 GPUs), 7-13% LLM decode ITL; explicitly EXCLUDES
scale-out (multi-node).  Repo = full NCCL fork (builds for sm_90 too; our NCCL is 2.29.3).

VERDICT: NOT applicable for measurable CHPL gain -- four independent kills:
1. WRONG MESSAGE REGIME: their wins live at 128 B - 1 MB where launch+sync latency dominates.
   Our on-chain NCCL messages are 100 MB - 4 GB chunks (W 4x~1GB, pkg 402 MB) -- pure bandwidth
   regime, already at ~95 GB/s (wbench).
2. CEILING MATH: total on-chain NCCL call/protocol latency is ~6-10 ops x ~20 us = 0.1-0.3
   ms/panel vs the 145 ms/panel chain gap -> zeroing ALL collective latency moves <= 0.2%.
   The gap is PRODUCTION latency (waiting for factorize->L21 compute), proven by the 1-node
   no-IB profile (07-15) -- no sync optimization can touch it.
3. WRONG SCOPE: the paper is scale-up-only; our record margins are at 2n/4n where the chain
   crosses IB (explicitly out of the paper's scope).  Engaging NCCL 2.29's symmetric kernels
   would also require ncclMemAlloc + window registration code changes -- not even a free probe.
4. ALREADY BUILT WHERE IT MATTERS: our same-node data plane (CE/IPC direct reads + monotone
   flag counters + device-side flag_wait polling) IS the paper's architecture -- one-sided
   symmetric-memory stores with sentinel-style polling, no host barriers.  The backsub CE
   rework (0.91 s -> 0.156 s, 07-03) was precisely a "chain of small collectives -> device-side
   flags" latency win, built before the paper existed.  This is the THIRD external validation
   of that design (after the xhpl RE's NVSHMEM findings and our own delay probes).
- Micro-nuance noted and dismissed: their LL 16B flag+data packing vs our separate
  data-then-flag protocol saves ~1 us/handshake x ~10 handshakes/panel = ~1 ms/run.  Noise.
- FUTURE hook: on a GB200 port, the LL128-atomic AllReduce would apply only to our single
  final allreduce (once per run).  Irrelevant.
Champions unchanged: 1n 472.0 / 2n 989.1 / 4n 1853.

## 12-GPU & 16-GPU SHOWDOWN vs NVIDIA CONTAINER (2026-08-12)
### 16 GPUs (2 nodes) -- first formal G8-era interleaved showdown
  OURS      (N=460800 NB=4096 P4Q4, G8 m11 fast NT3072): 984.4 / 983.2 TF
  container (N=524288 NB=1024 P4Q4, its tuned best):     782.5 / 782.5 TF
  -> **+25.7%, complete separation** (scheme-I-era record was 789.6 vs 786.2 = +0.4%).
### 12 GPUs (1.5 nodes)
  OURS: P4Q3 grid with 8+4 BLOCK placement (keeps every ncol group of P=4 consecutive
  ranks on one node -- required by the U-staging CE pulls; rank guard relaxed from
  g_size%8 to g_size%P with a placement note).  Results: 714.7 / 713.0 @N=397312,
  best **724.2 TF @N=405504** (409600+ OOM).  Per-GPU: 60.4 TF/GPU @12 vs 61.5 @16 --
  clean scaling at a non-power-of-two GPU count.
  CONTAINER: **CANNOT RUN 12 GPUs on this cluster** -- four escalating blockers, each
  unblocked to reveal the next:
    (1) 8+4 placement: NVSHMEM hard-rejects unequal PEs/node ("requires the same number
        of PEs on all nodes");
    (2) 6+6 cyclic: NVLS (NVLink SHARP multicast) team init fails ("NVLS resource
        initialization failed for team ID: 1");
    (3) 6+6 + NVSHMEM_DISABLE_NVLS=1: xhpl HANGS to timeout (device-side NVSHMEM ops
        deadlock in the non-full-node topology);
    (4) + all-NCCL steering (HPL_FCT_COMM_POLICY=1, HPL_NVSHMEM_SWAP=0,
        HPL_P2P_AS_BCAST=1): still hangs -- consistent with the xhpl RE finding that
        pivot-row collection is UNCONDITIONALLY NVSHMEM device-side.
  Verdict: the container is engineered for full-8-GPU-node multiples only; flexible GPU
  counts are a genuine capability edge of our implementation.
### Engineering byproducts (recorded for future runs)
  - Cluster fabric has MIXED IB/RoCE HCAs (mlx5_1 etc. are RoCE): symmetric 8/node
    layouts pair like-with-like by luck; ASYMMETRIC layouts hit "Remote IB device is
    incompatible with the local RoCE" NCCL failures.  Fix (now in scripts): auto-detect
    IB-link-layer HCAs from /sys/class/infiniband and set NCCL_IB_HCA="=<ib-only list>".
  - New bad nodes: node144 (GPU busy/unavailable at cudaSetDevice) and
    node[015-016] (container NVSHMEM topo failures + our RoCE mispairing seen
    there); exclusion list now 011-012,015-016,143-144.
  - srun --ntasks-per-node on a job step did NOT override the sbatch allocation
    (still filled 8+4); --distribution=cyclic is the reliable way to get 6+6.

## EmuGEMM (NTHU-SC/EmuGEMM, SC26 artifact) vs gemmul8 3.1.0: replace GEMMul8 kill-check (2026-08-13)
User supplied a new Ozaki-II (CRT) GEMM library, NTHU-SC/EmuGEMM (BSD-3, DeepGEMM-derived
JIT/CUTLASS infra on SM90/SM100), asking whether it can replace GEMMul8 in CHPL2D_G8's trailing
update, with a direct implementation + perf/accuracy comparison, maximizing perf within HPL's
residual requirement. Full kill-check executed; **KILLED at the kernel-microbenchmark stage**,
before any chpl4x4.cu integration.

### What EmuGEMM actually offers on H200 (SM90)
- Two Ozaki schemes ship: Emu-I (K-interleaved slicing) and Emu-II (CRT, same family as
  GEMMul8's Ozaki Scheme II) -- Emu-II is the correct comparison target (Emu-I's K-interleaved
  family was already ruled out for us at k=4096 in the earlier Scheme-I-vs-Scheme-II campaign).
- Emu-II's "fully fused" multi-moduli-in-one-kernel variant (`int8_crt_dgemm_fused_ii_nt`) is
  **SM100/Blackwell-only** (`csrc/apis/gemm.hpp`: `EG_HOST_UNREACHABLE(...requires SM100)` on
  the SM90 branch) -- confirmed by reading source AND by grep: no benchmark/test script in the
  artifact ever calls it. On H200 there is no genuine all-moduli-fused kernel to exploit.
- What SM90 actually has: `int8_crt_dgemm_nt`, a single-modulus INT8 GEMM with the mod-p
  reduction FUSED INTO THE EPILOGUE (writes uint8 `C mod p` directly, standard [0,p) convention,
  verified in `int8_emu2_dgemm.cuh:257-260`) -- called once per modulus, same structural shape
  as GEMMul8's per-modulus loop but skipping GEMMul8's separate int32-write + mod_hi2mid pass.
  This -- plus EmuGEMM's own fused slice/scale/CRT-merge kernels -- is the entire source of any
  possible speedup on our hardware; it directly targets the exact bottleneck this project had
  already profiled and rejected forking GEMMul8 to fix (RESULTS.md: "5.5 fuse scale/mod/recon
  into epilogue... only single-GPU hpl_single +3-5%, ~0 multi-GPU since engine gains convert to
  idle at 2n/4n" -- i.e. the ceiling was already known to be small even in the best case).

### Method
Built EmuGEMM fully offline (conda env `$WORK/conda_envs/emu_gemm`, torch
2.13.0+cu130, CUDA 13.1 toolkit via conda, vendored CUTLASS/fmt -- no compute-node network
needed; ~15 min conda solve + ~3 min `pip install -e .`). Wrote two independent, apples-to-apples
probes at identical shapes, moduli count, and reference math:
- `g8_baseline_probe.cu` / `g8_square.cu` / `g8_accuracy.cu`: gemmul8 3.1.0's REAL production
  call (`gemmul8::gemm<double,Backend::INT8>`, the exact call chpl4x4.cu:2336 makes), moduli=11/12
  fastmode, timed with cudaEvent, accuracy vs `cublasDgemm` fp64 reference on real random data.
- `emu_vs_g8_bench.py` / `sanity_square.py`: EmuGEMM's `bench_utils.crt_kernel_dgemm()` -- the
  artifact's own full production pipeline (fused scale+slice kernels -> s serial
  `int8_crt_dgemm_nt` calls -> fused Garner-CRT merge), at num_moduli=11-12, timed with
  `torch.cuda.synchronize()` wall-clock, accuracy vs `A@B` fp64 reference.
- Config-artifact check (before trusting any negative number): `emu_gemm.get_num_sms()==132`
  and `get_tc_util()==100` both already at full H200 device -- not throttled, not a config bug.
- Root-cause decomposition: isolated EmuGEMM's 11x `int8_crt_dgemm_nt` mainloop from its
  slice/scale/CRT-merge overhead at M=115200,K=4096,N=3072.

### Results (H200, single GPU, moduli=11 unless noted; production shape K=NB=4096, N=G8_NT=3072)
```
            m |  gemmul8 3.1.0 TFLOPS | EmuGEMM (Emu-II) TFLOPS | gemmul8 margin
         8192 |         97.9          |          82.6           |   +18.5%
        16384 |        106.6          |          95.7           |   +11.4%
        32768 |        113.2          |          98.1           |   +15.4%
        57600 |        116.1          |          97.7           |   +18.8%
       115200 |        117.8          |          99.2           |   +18.8%
```
Square sanity shapes (M=N=8192, moduli=12), to cross-check against the paper's own Fig.6
benchmark geometry:
```
   K   | gemmul8 3.1.0 | EmuGEMM | margin
  2048 |    69.9 TF    |  66.6   | +5.0%
  4096 |    96.0 TF    |  93.3   | +2.8%
  8192 |   108.1 TF    | 103.6   | +4.4%
```
Note the shape-dependence: at the paper's own square (M=N) benchmark geometry the gap nearly
closes (+2.8% to +5.0%, right at this project's kill-check threshold), but at CHPL's actual
tall-skinny production geometry (M up to 14x larger than N) it widens to +11-19%. This is
exactly why the kill-check was run at PRODUCTION shapes rather than trusting the paper's own
(differently-shaped) benchmark numbers -- a square-only evaluation would have given an
ambiguous, borderline-passing result instead of the clear kill the real workload shape produces.
(All numbers measured on H200, single GPU, job 257880. An earlier H100-NVL
login-node pass showed the same ordering at a wider margin at every shape -- H200's higher HBM
bandwidth narrows the gap somewhat but does not close it; the verdict is hardware-independent.)

**Accuracy is a dead heat, as expected**: both are the same Ozaki-II/CRT math at the same
moduli count. gemmul8 3.1.0 and EmuGEMM both measure `rel_err = 1.286e-11`, `36.18` effective
bits at moduli=11-12 on real random data vs an fp64 cuBLAS/torch reference -- bit-for-bit
identical to 3 significant figures despite completely independent implementations. Neither has
an accuracy edge; this is a pure speed decision.

### Root cause: EmuGEMM's mainloop is roughly competitive, its slice/CRT-merge kernels are not
Decomposing EmuGEMM's own pipeline at M=115200,K=4096,N=3072,s=11 (H100 NVL login-node pass;
decomposition not re-run on H200, but the ~78%/22% mainloop/side-kernel split is a structural
property of the pipeline, not expected to shift materially with HBM bandwidth):
```
  full pipeline:            39.79 ms  (72.3 TFLOPS)
  mainloop (11x gemm+mod):  30.92 ms  (77.7% of time, 93.75 TFLOPS in isolation)
  slice+scale+CRT-merge:     8.86 ms  (22.3% of time)
```
EmuGEMM's fused GEMM+mod-epilogue mainloop alone (93.75 TFLOPS) is respectable -- close to but
still below gemmul8 3.1.0's COMPLETE pipeline (101.55 TFLOPS on H100 NVL at the same shape,
which already includes gemmul8's own scaling/mod/CRT overhead). That means gemmul8's underlying
per-modulus mainloop (cuBLAS `cublasGemmEx` INT8, already measured elsewhere in this project at
93.9% tensor-pipe saturation) is faster than EmuGEMM's custom CUTLASS kernel even without the
epilogue-fusion advantage -- cuBLAS's raw INT8 GEMM is simply better-tuned at these shapes than
EmuGEMM's DeepGEMM-derived kernel. On top of that, EmuGEMM's own slice/scale/CRT-merge kernels
(22.3% of its time) are proportionally similar to or heavier than GEMMul8's equivalent side
kernels (previously profiled at ~17.4% of total time: mod_hi2mid 9.9% + undo_scaling 6.5% +
scaling ~1%). Both factors compound against EmuGEMM; neither is close to closing the gap.

### Why this contradicts the paper's own Fig.6 (EmuGEMM 1.05x-1.95x faster than "GEMMul8")
The artifact vendors GEMMul8 at commit `2f8522c` (an old, flat-layout snapshot: just
`gemm.cu`/`gemm_complex.hpp`, no `symm`/`syrk`/`hemm`/... family). We compared against
**gemmul8 3.1.0** (RIKEN's current release, already vendored in this repo, with a completely
restructured `src/oz2/` tree: `mod_hi2mid` implements a hi/lo Karatsuba-style split not present
in the old snapshot). RIKEN has evidently made substantial optimizations to GEMMul8's INT8
mainloop and side kernels between whatever `2f8522c` was and 3.1.0 -- enough to erase and
reverse EmuGEMM's claimed advantage. **Lesson for future library evaluations: always re-baseline
against the current release actually in production, not a paper's vendored snapshot -- artifact
papers freeze their baseline at submission time and the field moves.**

### Verdict
KILL, decisively, at the raw-kernel stage -- before even reaching the chain-latency-bound
discount this project would normally apply (RESULTS.md: "engine gains convert to idle at 2n/4n,
measured 3x"). gemmul8 3.1.0 beats EmuGEMM's Emu-II by +11% to +19% at production shapes and
+2.8% to +5.0% at square sanity shapes, on H200, with identical accuracy. No chpl4x4.cu
integration attempted -- per this project's established convention (do not force an
implementation past a kill-check). Champions unchanged.

Artifacts for reproduction: `$WORK/conda_envs/emu_gemm` (conda env),
`$WORK/EmuGEMM_repo` (built source, `pip install -e .`'d), probes copied to
`$WORK/HPL_EmuGEMM_build/probes/`.

### Reconciling with a second, independent run that appeared to show EmuGEMM winning
A separate session (`~/SC26/GEMM`, jobs 257564-257742) ran the artifact's own `bench_real.py`
and `bench_gemmul8_dgemm.cu` directly and got EmuGEMM Emu-II p=12 at 134.9 TFLOPS vs GEMMul8
m=11 at 77.1-95.1 TFLOPS (M=N=K=8192) -- apparently the opposite verdict. Root-caused, NOT a
contradiction -- **two independent measurement-scope bugs in that run, both favoring EmuGEMM**:
1. `run_bench.sbatch` calls `bench_real.py` WITHOUT `--with-post`. Reading `bench_real.py`
   (~line 1143-1149): without that flag, `time_preprocess_ms`/`time_postprocess_ms` are hard-set
   to 0.0 and only the raw `s`x`int8_crt_dgemm_nt` mainloop is timed -- i.e. EmuGEMM's number
   EXCLUDES the mandatory fp64->int8 slice/scale and CRT-merge-back-to-fp64 steps. Those steps
   cannot be amortized away in HPL (every panel's A/B data is different, nothing to reuse
   across calls), so this is not a valid stand-in for "cost to actually use EmuGEMM as a DGEMM
   replacement" -- it's the same "mainloop-only" number this doc already reports separately
   (93.75 TFLOPS at a different, tall-skinny shape) via explicit decomposition.
2. `run_gemmul8.sbatch` builds and runs `bench_gemmul8_dgemm.cu` against the OLD vendored
   GEMMul8 (`EmuGEMM-A2/GEMMul8/GEMMul8`, commit 2f8522c) -- same staleness issue as the paper's
   own Fig.6, not gemmul8 3.1.0. Additionally `gemmul8::gemm()` is an atomic API call (its
   scale/quantize/11xGEMM/mod/CRT internals cannot be split), so this side is ALWAYS
   full-pipeline regardless of flags -- there is no way to give it the same "mainloop-only"
   treatment even if you wanted a matched comparison.
Net effect: that run compared EmuGEMM-mainloop-only against OLD-GEMMul8-full-pipeline --
both factors bias toward EmuGEMM. Cross-check at the IDENTICAL shape (M=N=K=8192, moduli=12,
H200): old GEMMul8 full-pipeline 77.1-95.1 TF < gemmul8 3.1.0 full-pipeline **108.1 TF**
(confirms point 2's staleness finding independently) < EmuGEMM mainloop-only 134.9 TF, while
EmuGEMM FULL-pipeline (this doc's own measurement, same shape) is only **103.6 TF** -- i.e.
subtracting EmuGEMM's own slice/CRT-merge overhead from its mainloop-only number (134.9 TF)
lands almost exactly on the independently-measured full-pipeline number (103.6 TF), which is
a strong cross-validation of both this doc's decomposition and its final verdict.
**General lesson #2 for future library evaluations**: when a benchmark harness reports separate
`time_gemm_ms`/`time_preprocess_ms`/`time_postprocess_ms` columns, always confirm BOTH sides of
a comparison are measuring the same scope (full usable pipeline vs. kernel-only) before trusting
a TFLOPS number -- an artifact's own default CLI invocation may silently measure a partial
pipeline for one library while measuring an atomic full-pipeline call for the other.

## Repo reorganized: GEMMul8 vendored as git submodule, verified against v3.2.0 (2026-08-13)
Following the EmuGEMM evaluation above, reorganized the HPL-H200 repo (now published as CHPL) into a self-contained,
clonable repo per user request: GEMMul8 was previously an external "clone it yourself into the
repo root" dependency; converted to a proper `git submodule` (`gemmul8/`, tracked in
`.gitmodules`), currently pinned to upstream tag **v3.2.0** (was v3.1.0). Also removed four
legacy build scripts that hardcoded a private, user-specific working-tree path
(`build4x4.sh`, `build2d_g8.sh`, `build_gemmul8_min.sh`, `build_single.sh`) -- fully superseded
by the existing generic, repo-relative `scripts/build.sh`. Added `.gitignore`. README's build
section now documents `git clone --recurse-submodules` and the version-bump workflow.

**v3.1.0 -> v3.2.0 diff audit** (before trusting the bump): `git diff v3.1.0 v3.2.0` touches
`table.hpp`, `matmult.hpp`, `mod_core.hpp`, `mod_f32.hpp`, TRSM, and hook code -- but the public
API (`include/gemm.hpp`, `include/worksize.hpp`) is untouched, the **INT8 moduli table is
byte-identical** (only the FP8 moduli table changed, per upstream's "Redesign FP8 emulation"
commit message), and the one INT8-relevant refactor (`call_gemm_tn` -> `call_gemm_tn_raw` +
a new FP8-only k-blocking wrapper) is gated by `if constexpr (BACKEND != Backend::FP8)`, which
for `Backend::INT8` compiles straight through to the old code path with zero behavioral or
performance difference. Conclusion before testing: this should be a behavior-preserving bump
for our INT8/CRT champion configs.

**Empirical verification** (built from HPL-H200 itself as the canonical working tree, via
`scripts/build.sh` against the v3.2.0 submodule, 3 rounds each, H200):
```
                    N       config                  TFLOPS (3 rounds)      residual
1-node champion   319488   NB4096 P2Q4 m11 NT2816   467.6 / 467.9 / 467.8  1.999253693793 PASSED (x3)
2-node champion   460800   NB4096 P4Q4 m11 NT3072   986.7 / 986.0 / 986.4  1.943372891117 PASSED (x3)
```
Residuals are **bit-identical** to the previously recorded v3.1.0-era champion figures (1.999 /
1.943 respectively -- exact digit-for-digit match), confirming the diff audit's prediction that
INT8 numerics are unchanged. Performance: 2-node's 986.0-986.7 TF sits inside the existing
v3.1.0-era spread already on record (983.2-989.1 TF across separate sessions/nodes); 1-node's
467.6-467.9 TF is about -0.9% below the single-point v3.1.0 record (472.0 TF, one session's
5-round median) but this is a single node-pair, not an interleaved A/B control, and is well
within this project's already-documented node-to-node variance band (~1-2%, noted repeatedly
elsewhere in this log) -- not attributable to the version bump given the diff audit found no
INT8-path code change. **Verdict: v3.2.0 adopted as the tracked submodule version. Safe,
behavior-preserving upgrade -- no champion re-tuning needed.**

## v3.2.0 regression concern RESOLVED: same-node A/B proves no regression (2026-08-13)
The first v3.2.0 verification run above landed on node015 and measured 467.6-467.9 TF at
1-node, which sits entirely BELOW the previously-recorded v3.1.0 5-round median range
(471.5-472.3 TF) -- a real gap outside that old run's own spread, not something that should
have been waved off as "within node variance" without proof. Built a genuine control: cloned
GEMMul8 v3.1.0 into a separate tree (`$WORK/gemmul8_v310`), built a second
`chpl2d_v310` binary linked against it, and ran a SAME-NODE INTERLEAVED A/B (this project's own
established gold-standard methodology for exactly this kind of question) against the v3.2.0
`chpl2d` binary, 4 rounds each, back-to-back on one node (node079):
```
round   v3.1.0 TF   v3.2.0 TF
  1       472.2       472.2
  2       471.5       471.7
  3       471.6       472.0
  4       471.5       471.4
median    471.55      471.85
```
**No regression.** The two versions are statistically indistinguishable (spread <0.2%, v3.2.0's
median is fractionally HIGHER if anything), and both closely match the historical 472.0 record.
The earlier apparent -0.9% was node015 running ~0.9% slower than node079 for this workload --
ordinary node-to-node variance, exactly as this project's own prior sessions have repeatedly
documented (~1-2% node-to-node spread), just not yet PROVEN for this specific claim before now.
**Lesson (re-affirms existing practice, learned the hard way again)**: a single different-node
measurement is never sufficient to accept OR reject a "did version X regress" question, even
when a diff audit already argues no code-level change should matter -- always run the same-node
interleaved A/B before concluding either way. v3.2.0 remains the adopted submodule version;
champion numbers unchanged.

## AMD rocHPL paper (arXiv 2304.10397) applicability audit (2026-08-13)
Read in full for concrete techniques to borrow. VERDICT: no new actionable levers -- mostly
confirms our architecture is already more advanced on every axis the paper touches, and is
silent (neither validates nor refutes) on the one open question we actually wanted an answer to.
- **Pivoting**: plain HPL partial pivoting (no tournament/CALU), just multithreaded across CPU
  cores. Our tournament pivoting w/ fp32 preselect is more advanced; nothing to adopt.
- **Look-ahead depth**: only depth-1, and the paper never explores or argues for going deeper --
  it's silent on this, not a counter-argument. Does NOT invalidate our identified #1 remaining
  lever (depth-2 lookahead restructure), but provides no supporting evidence for it either.
- **Broadcast/communication**: host-driven plain MPI (Scatterv/Allgatherv for row-swap,
  buffer broadcast for panel L); the paper explicitly says they AVOID NCCL/RCCL for panel
  broadcast because it "makes overlapping with other communication impractical," and calls
  topology-aware collectives out-of-scope/future-work. Our NCCL cross-node + CUDA-IPC/copy-
  engine direct reads with device-side flag polling is strictly more advanced than anything
  in this paper -- nothing to borrow, and their explicit avoidance of NCCL is itself informative
  (they hit a limitation our device-side design already sidesteps).
- **Trailing update / precision**: native FP64 only via rocBLAS, matrix fully GPU-resident, no
  INT8/tensor-core/Ozaki-style emulation anywhere -- our GEMMul8 CRT backend has no counterpart
  here, confirms it's a genuine differentiator vs. the state of the art this paper represents.
  Their one real GEMM-adjacent contribution is "split update": divide the trailing matrix
  column-wise into two sections, overlap UPDATE of one with row-swap (RS) MPI communication of
  the other. Not directly transferable -- our row-swap already runs over CE/IPC direct peer
  reads (zero recv copies, no MPI collective in the hot path), which is architecturally already
  what split-update is trying to approximate with plain MPI; nothing to gain by porting it.
- **Multi-GPU scaling / other contribution**: their standout idea, CPU core "time-sharing"
  across MPI ranks during FACT (idle non-factoring ranks' cores pooled via OpenMP to accelerate
  whichever rank is currently factoring), solves a CPU-FACT bottleneck specific to their
  CPU-resident-panel design. Inapplicable to us -- our panel factorization path is GPU-resident
  (tournament pivoting + fp32 preselect on-device), there's no analogous idle-CPU-core pool to
  exploit.
Net: fourth consecutive "read the competing paper, confirm our design is ahead, zero new
levers" result (after fast.cu, xhpl RE, and low-latency-nccl). Documented per this project's
convention of recording negative/confirmatory audits, not just wins. Champions unchanged.
