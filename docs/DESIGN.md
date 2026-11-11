# CHPL 深度設計文件

本文件說明 `chpl2d.cu`（P=2）與 `chpl4x4.cu`（P=4）的內部設計。兩者共享同一套骨架，P=4 版是 P=2 版的推廣（星型交換、4 路 tournament）。

---

## 1. 程序網格與資料佈局

- **P×Q 二維 block-cyclic 網格**，區塊大小 NB=2048。
  `rank = P*qc + pr`（pr = 列座標最快變動）。
  - 列方向：全域列區塊 `i` 屬於 `pr = i % P`
  - 行方向：全域行區塊 `j` 屬於 `qc = j % Q`
- 每個 rank 的本地矩陣 `d_A`：`mloc × nloc`（`mloc ≈ N/P`，`nloc ≈ N/Q`），column-major，`lda` 對齊 128 再加 64 個 double 的偏移（避開 bank/TLB 病態）。
- 在兩節點 16 GPU 上，此映射保證**每個 ncol（同一 qc 的 P 個 rank）落在同一節點** → 所有行內通訊（tournament、交換、U、回代）都走 NVLink；只有列方向（nrow）跨節點。

### 通訊器

| 通訊器 | 成員 | 用途 |
|---|---|---|
| `nworld` | 全部 | pivot 中繼資料（d_win）、pkg 逆矩陣片、最終 allreduce |
| `nrow` | 同 pr 的 Q 個 rank | W（panel 列塊）跨行廣播；多節點回代 reduce |
| `ncol` | 同 qc 的 P 個 rank | tournament 候選交換、對角塊廣播 |

同節點的 rank 之間另外建立 **CUDA IPC 資料面**（見 §4），大宗流量不走 NCCL。

---

## 2. Panel 流水線（chunked depth-1 lookahead）

每個 panel k 的工作拆成三條鏈，靠 CUDA event 銜接、在三條 stream 上重疊：

- **chainA（sP stream）— tournament 選主元 + 面板分解**
  1. 各列 rank 對自己的候選（本地 panel 列塊）做 fp32 stacked getrf（僅為「排序」主元列，數值不進最終因子）
  2. 候選值 + 出處（全域列號）聚到 owner 行的 pr=0
  3. pr=0：把 P 組候選堆疊成 `nst×NB` fp32 再 getrf 一次 → 選出 NB 個贏家
  4. 以 fp64 從 d_A 重新取贏家列，做精確的 `NB×NB` fp64 getrf；算出 `inv(L11)`、`inv(U11)` 片（trsm）
  5. `d_win`（贏家全域列號）廣播全體
- **chainB（sP stream）— 交換 / W / U staging**
  1. `buildlists`：從 d_win 在 device 上建 net-permutation 交換列表
  2. 交換 g0（下一 panel 的行，最緊急）→ W 段 → pkg 廣播 → **emit chunk0（urgent update：用 W×M 更新下一 panel 的行）** → 交換 g1..g3 + U12 staging（`M = inv(U11)·U12` 逐 XT tile 算出後立即切片）
- **s0 stream — trailing update**
  等 `evReadyW[k]`、`evReadyU[k][c]`，對 chunk 1..3 做 Ozaki int8 GEMM 更新（chunk 0 已在 sP 上由 emit 做完）。

chunk 分割：`chunks_of(k)` 把本地 trailing 行分成
`[u1 = panel k+1 的行][u2 = panel k+2 的行][剩餘的前一半][後一半]`
——u1/u2 提前完成讓下一兩個 panel 的鏈能盡早啟動（等效 depth-2 的關鍵路徑效果）。

### 關鍵路徑

每 panel 的串行鏈為：
`urgent(k-1) → 交換 g0(k) → W(k) → pkg(k) → urgent(k) → chainA(k+1) → …`
所有優化的本質都是把東西**搬離這條鏈**（fp32 選主元、CE、preselect），或把鏈上的傳輸**與計算重疊**（W 列分塊 + K-slab）。

---

## 3. Ozaki int8 模擬 fp64 GEMM

### 3.1 切片數學

對縮放後的元素 `a ∈ [-1, 1)`，以 **base B=128** 拆解：

```
a ≈ Σ_{p=0..5} a_p · 128^(-p-1)，a_p ∈ [-128, 127]（int8）
```

- **為何 B=128（而非 256）**：切片餘項收斂需要 `D/(B-1) ≥ 0.5`（D = digit 上界）。
  int8 範圍不對稱（-128..127），B=256 時 `127/255 < 0.5` → 餘項在 +0.5 邊界**發散**，殘差必炸（實測所有 B=256 變體全部 FAILED）。B=128 時每片只乘 2 的冪，**切片本身零捨入誤差**。
- s=6 片、保留 `p+q ≤ 5` 的 21 個切片對 ⇒ 47.7 個有效位元。實測 N=50 萬時殘差 ~0.5（門檻 16）。
  這組參數是資訊理論上的下限——實測 D=4（15 對）殘差爆炸，s、D 不可再降。

### 3.2 融合對角線（library-only 的核心技巧）

21 個切片對 GEMM 按「對角線 d = p+q」分組：同一對角線的貢獻權重相同（`128^(-d-2)`）。
把 A 的切片沿 K 維**正序**串接、B 的切片**反序**串接，則第 d 條對角線的所有配對恰好構成一段連續的 K 區間——**一條對角線 = 一個 GEMM**（`kd = (d+1)·2048`，`beta=0`）：

- 21 個 GEMM → **6 個** cublasLt int8 GEMM
- int32 中間結果流量：144 → 48 B/元素
- 重建 kernel（`ozrecon`）讀 6 個 int32 平面 + 列/行縮放係數，寫回 fp64，與下一 tile 的 GEMM 以雙緩衝 `ozDg[2][6]` 重疊（重建約佔 GEMM 時間 5%，藏得住；**不可**改單緩衝——會暴露 ~9s）

### 3.3 縮放

列縮放（`ozRinv/ozRsc`，對 W）與行縮放（`ozCinv/ozCsc`，對 M）都是 2 的冪，維持精確性。C tile 以 `OZ_NT`（=1024）為寬度分塊處理。

### 3.4 功耗牆物理

H200 的 700W 上限使原生 fp64 GEMM 持續時脈僅 ~1450-1600 MHz（~51-53 TF/GPU）。
int8 路徑功耗 ~475W、時脈鎖滿 1980 MHz。模擬的 21 倍運算量在能量上與 fp64 打平，但**時間**上快 ~20%——這是本方法成立的物理基礎。

---

## 4. Copy-Engine 資料面（CUDA IPC + flag 協議）

同節點 rank 互開 `cudaIpcOpenMemHandle`（W、U、acc、tmp、xpack、flags），以**單調遞增的 long long flag 計數器**握手（無鎖、無重置、天然免 ABA）：

| flag | 語義 |
|---|---|
| 0 / 1 | W pack 完成序號 / W 拉取完成序號 |
| 2 / 3 | U staging 完成（k*4+chunk）/ U 消費完成 |
| 4 / 5 | 回代迭代進度 / x 片就緒 |
| 6 / 7 | 交換 pack 就緒序號 / 交換消費完成序號 |

典型模式（以 U 為例）：owner 列（pr==kp）逐 chunk staging 完成後 `flag_set(2, k*4+c+1)`；
其餘列 `flag_wait` 後用 `cudaMemcpyAsync` 直接從 owner 的 U 緩衝拉取。owner 重用同位緩衝前，等所有讀者的 flag 3。

### 交換（pivot 列交換）

`buildlists` kernel 從 d_win 建出 net permutation：贏家列 → 對角塊槽位；被擠出的對角列 → 贏家騰出的槽位（配對順序全 rank 一致，故 per-peer 列表兩端天然對齊）。

**P=4 的星型結構**（`chpl4x4.cu` 的核心洞察）：
- 贏家列的目的地一定在 owner 列（對角塊）；被擠出列的來源一定是 owner 列
- ⇒ **非 owner 列之間永遠沒有流量**，交換是以 owner 為中心的星型
- 每 rank 只需 per-peer 的 send/recv 列表；xpack 佈局 `[奇偶槽][目的 pr]`，接收方用 `scatter_dyn` 直讀發送方 IPC 緩衝
- 列數上界每 peer ≤ NB，訊息定長免 device→host 同步

### 回代

由後往前逐 panel：owner 行內以 CE 拉取部分內積累加（`d_acc`），owner rank 做 `trsv` 得 x 片、`flag_set(5)`，同行其他列拉 x 片做 `gemv` 消去。多節點時跨行的縮減改用 `ncclReduce`（nrow）。全程無 host 同步。

---

## 5. Preselect（提前選主元）

### 動機：延遲探針

在選主元 getrf 之後注入人工延遲（spin kernel），量測總時間對延遲的斜率：
**斜率 ≈ 0.5** ⇒ 每 1ms 的 getrf 有 0.5ms 直接暴露在關鍵鏈上（P=2 時 getrf ≈ 168ms/panel）。
這個「探針法」值得推廣：它直接量測任意程式段的關鍵路徑暴露係數，比 profiler 的時間分解更能回答「優化它值多少」。

### 設計

panel k+1 的行在 chainB(k) 的**交換 g1 + emit chunk0 之後就已是最終值**（與 inline 讀到的位元級相同），而 chainB(k) 之後還有 ~150-250ms 的尾段（g2/g3 交換、U staging）。因此：

1. 在 chainB(k) 的 g1 之後，於 **sP 上做 d2s 快照**（順序性保證資料一致，~3ms）
2. fp32 getrf 移到**專用 stream sSel**（獨立 cusolver handle + 私有 workspace），與 chainB 尾段重疊
3. chainA(k+1) 以 `evSel` 事件等待結果

### 血淚教訓：私有輸出緩衝

sSel 上的 `prep_cand` 若直接寫共享的候選列表/出處緩衝，會以難以重現的方式污染執行（P=2 時 100% 重現、P=4 時 ~1/9 flaky；二分法證明 d2s/getrf/resolve 皆無辜，唯獨 prep_cand 的共享寫入有毒）。
**修法**：prep_cand 寫私有緩衝（`d_rows2`/`d_provP`），chainA 在 `evSel` 之後於 sP 上以兩個小 D2D copy 搬到正規緩衝。修復後殘差與 inline 位元級相同。

效益：P=2 兩節點 +2%；P=4（getrf 高度已減半）仍有 **+0.8%**（783.3 → 789.6）。

---

## 6. 多節點（跨節點）設計

- `samenode[]`（由 `MPI_COMM_TYPE_SHARED` 建出）決定 IPC 開啟範圍；`wce` 旗標 = 「本列的所有行 peer 都同節點」
- **W 廣播**（唯一的跨節點大宗流量，P4Q4 時 ~2GB/panel）：
  - `wce=1`（單節點）：CE 拉取
  - `wce=0`：NCCL 廣播（nrow），**按行分塊**（`CHPL2D_WCH`，預設 4 塊）於專用 stream sW；
    urgent GEMM 以 **K-slab** 逐塊消費（GEMM 沿 K 可加法分割），關鍵路徑只付第一塊的延遲
- **host 節流**：多 communicator（nworld/nrow/ncol）深排隊會造成 NCCL 調度死鎖。
  每個迭代尾端 `cudaEventSynchronize(evReadyW[k])`——因為第 k+1 迭代的 device 工作本來就被 W(k) 閘住，此等待**可證明為零成本**，卻讓佇列保持淺層、死鎖消失。

---

## 7. 驗證

- 殘差 `||Ax-b||∞ / (ε·(||A||∞·||x||∞+||b||∞)·N) < 16.0`，矩陣以確定性種子重生成（不佔存分解前的副本）
- 輸出格式與 HPL-NVIDIA 相容（`WC2  N NB P Q time GFLOPS`）
- 所有數值近似（Ozaki 參數、fp32 選主元、stale/exact preselect）都以殘差門檻為最終仲裁——任何越界立即 FAILED，不存在靜默錯誤

---

## 8. 效能歷程摘要

| 里程碑 | 單節點 TF | 關鍵改動 |
|---|---|---|
| fp64 v3 | 339.7 | 2×4 網格 + tournament 初版 |
| fp64 v14 | 371.9 | chunk 流水線、device 側 pivot、L21 fold、CE W/U |
| + Ozaki int8 | 380 | base-128 s=6、融合對角線 |
| + XT 分塊/N=348160 | 385.8 | 記憶體瘦身放大 N |
| + 左列免交換、CE 回代、CE 交換 | **391.5** | NCCL 僅剩中繼資料 |

| 里程碑 | 兩節點 TF | 關鍵改動 |
|---|---|---|
| P2Q8 移植 | 719.7 | 混合 CE/NCCL、host 節流 |
| NT=1024（縮 N 換 NT） | 759.2 | int8 GEMM tile 寬度 +4.9% |
| 緩衝瘦身 → N=479232 | 783.9 | WS/XT/NCCL_BUFFSIZE 擠出 1.2GB |
| + Preselect | （中位 777.7→）| getrf 半暴露 → 移出鏈 |
| **P4Q4 重構 + Preselect** | **789.6** | 星型交換、getrf 高度減半、N=497664 |
