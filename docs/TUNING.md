# 調參指南

所有結論皆為 H200（141GB HBM3e）實測；換硬體請重掃。

## 1. 環境變數總表

### 演算法主開關

| 變數 | 預設 | 說明 |
|---|---|---|
| `CHPL2D_OZ` | 0 | **1 = 開啟 Ozaki int8 模擬 fp64**（必開，這是效能來源） |
| `CHPL2D_OZ_NT` | 1024 | int8 GEMM 的 C tile 寬度。**1024 是鐵律**（見 §3） |
| `CHPL2D_OZ_ALGO` | 3 | 選主元演算法。3 = 全高 fp32 stacked getrf（最佳）；1/2 為分塊 tournament 變體（實測 -2% 與 -24%） |
| `CHPL2D_PRESEL` | 0 | **1 = 提前選主元 stage-1**（單節點 +1.7%、兩節點 +0.8~2%）。有罕見競態，殘差檢驗會抓住 |
| `CHPL2D_PRESEL2` | 0 | **1 = 提前選主元 stage-2**（把最終贏家選擇 + 精確重分解也移到 sSel）。兩節點 +0.34%。需 PRESEL=1。正確性 bit-identical。P4Q4 專用（ncol 需同節點）|

### 資源/分塊

| 變數 | 預設 | 說明 |
|---|---|---|
| `CHPL2D_XT` | 8192 | staging / 交換的行 tile 寬。8192 與 6144 效能相同；**6144 省 ~0.24GB**（衝大 N 時用） |
| `CHPL2D_WCH` | 4 | 跨節點 W 廣播的行分塊數（2/4/6/8 實測無感，留 4） |
| `CHPL2D_WS` | 256 | cuBLAS workspace pool（MB）。**64 無效能損失**，省 ~0.45GB（衝大 N 時用） |

### NCCL / CUDA（多節點）

| 變數 | 冠軍值 | 說明 |
|---|---|---|
| `NCCL_MAX_NCHANNELS` | 單節點 6 / 兩節點 **4**（配 `NCCL_MIN_NCHANNELS=1`） | 通道數 = SM 佔用 vs 頻寬的折衷；越少的 channel 留給 GEMM 越多 SM，只要 W 廣播還藏得住就往下調（P4Q4 實測 4 最佳、3 起 W 藏不住） |
| `NCCL_BUFFSIZE` | 2097152 | 2MB 比預設 4MB **又省記憶體又快 +4 TF**（IB 傳輸分塊效應）；1MB 反而慢 |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 32 | 多 stream + NCCL 並發下 +0.3% |

### 除錯（正式跑請全關）

`CHPL2D_TIMING`（1=粗分解/2=細分解）、`CHPL2D_PROG`（逐 panel 進度+卡死診斷）、`CHPL2D_PSYNC`/`CHPL2D_PDUMP`/`CHPL2D_DLY`（preselect 二分工具、候選 checksum、關鍵鏈延遲探針）。

## 2. N 的選擇與記憶體預算

N 必須是 NB=2048 的倍數。**N 越大越快**（鏈延遲攤提），直到撞記憶體。每 GPU 預算（bytes，十進位）：

```
d_A      ≈ (N/P + 192) × (N/Q) × 8          ← 主項
mloc 項  = (N/P) × [ 2×NB×8 (W雙緩衝) + NB×4 (fp32面板) + 6×NB (切片A)
                     + 12×NT×4 (ozDg 雙緩衝×6對角線) ]
nloc 項  = (N/Q) × [ 2×NB×8 (U雙緩衝) + 6×NB (切片B) ]
固定項  ≈ 3~4.5 GB（xpack=(2P+2)×NB×XT×8、workspace、NCCL、context）
```

可用上限實測 ≈ 148-150 GB；**超限的症狀不一定是明確 OOM**——常見「NCCL unhandled cuda error」在尾端才爆（執行中的非同步配置失敗）。

實測甜蜜點：

| 配置 | 最佳 N | 備註 |
|---|---|---|
| 8 GPU, P2Q4 | 348160 | 單節點冠軍（+PRESEL = 403.3；350208 起必死） |
| 16 GPU, P2Q8 | 479232 | 需 WS=64/XT=6144/BUFFSIZE=2M；481280 必死 |
| 16 GPU, P4Q4 | **497664** | P=4 的 mloc 減半省出空間；499712 反而變慢 |

另注意 **panel 平衡**：N/NB mod Q 影響尾端負載（實測 477184 比更小的 475136 慢）。

## 3. NT（int8 GEMM tile 寬）——最重要的單一參數

| NT | 兩節點實測 TF | 結論 |
|---|---|---|
| 512 | 586.8 | 災難 |
| 768 | 719.7 | -4.9% vs 1024 |
| **1024** | **759.2+** | **鐵律** |
| 1280 | 556.0 | 災難（cublasLt 無好演算法） |
| 1536 | 734.5 / 744.9 | 差 |
| 2048 | 748.0 | 差 |

寬度效應來自 cublasLt 的演算法選擇品質，**不是** tile 數量攤提——所以「縮 N 換 NT=1024」永遠划算。

## 4. 網格選擇

- **P=2**（chpl2d）：單節點最佳。行內只有一對 rank，CE 協議最簡。
- **P=4**（chpl4x4）：兩節點最佳。選主元 getrf 高度減半（關鍵鏈短）、W 廣播量減半、記憶體省 ~9GB → N 可加大 4%。星型交換讓 4 路成本近似 2 路。
- 兩節點時 **ncol 必須同節點**：`rank = P*qc + pr` 配 `--ntasks-per-node=8` 天然滿足（P=4 時每節點恰好 2 個完整 ncol）。

## 5. 對決方法論（重要）

- **節點間差異 ±1~2%，會淹沒一切小勝差。** 唯一公平的比較：同一批節點、交錯執行、比中位數。
- NVIDIA 容器對節點快慢**特別敏感**（σ≈3 TF，因 fp64 功耗牆時脈行為）；本實作 int8 路徑 σ≈1.3 TF。在慢節點對上容器可低至 777、快節點對高至 787。
- 容器方的最佳參數要獨立掃描（我們掃出 N=524288/NB=1024/P4Q4；其 N 上限受限於它自己的 OOM）。
- 每輪都看殘差：PASSED 才算數。

## 6. 平台注意事項

- `LD_LIBRARY_PATH` 必須把正確的 CUDA lib64 放最前（cublas/cublasLt 版本混用會 segfault）
- 計算節點 `/tmp` 唯讀 → `TMPDIR=/dev/shm`
- 容器對決需 `srun --mpi=pmix` + `singularity -B /dev/shm:/tmp`（PMIX gds_ds12 lock 錯誤訊息無害）
- 多次 run 之間勿 free/realloc 大 GPU 緩衝（驅動層曾觀察到損壞）；本實作單次執行內持久配置
