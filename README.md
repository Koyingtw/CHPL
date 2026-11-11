# CHPL：超越 NVIDIA 官方容器的 H200 HPL 實作

從 netlib HPL-2.3 出發、針對 NVIDIA H200（Hopper, sm_90a）全面重寫的 CUDA HPL（High-Performance Linpack）實作。

- 曾在**單節點（8×H200）與兩節點（16×H200）的同節點交錯正式對決中，以中位數擊敗 NVIDIA HPC-Benchmarks 26.02 容器**（5/5 完全分離）。
- 之後換裝 **GEMMul8（Ozaki-II CRT）trailing-update 後端 + NB=4096 重調參**，兩節點再 **+25%**、四節點再 **+16%**，形成現行冠軍。
- 所有成績均通過標準 HPL 殘差檢驗 `||Ax-b||∞ / (ε·(||A||∞·||x||∞+||b||∞)·N) < 16.0`，且各冠軍配置經 5 輪重複驗證（σ < 0.1%）。

## 現行冠軍成績與重現方式

| 規模 | 成績（5 輪中位數） | 二進位 | 輸入檔 | 重現腳本 |
|---|---|---|---|---|
| **4 節點（32×H200）** | **1853 TF**（1852–1854，殘差 1.397） | `build/chpl4x4` | [`dat/HPL_4node_g8.dat`](dat/HPL_4node_g8.dat)（N=622592, NB=4096, P4Q8） | [`scripts/c4x4_g8_final4n.sbatch`](scripts/c4x4_g8_final4n.sbatch) |
| **2 節點（16×H200）** | **989.1 TF**（988.5–990.1，殘差 1.943） | `build/chpl4x4` | [`dat/HPL_2node_g8.dat`](dat/HPL_2node_g8.dat)（N=460800, NB=4096, P4Q4） | [`scripts/c4x4_g8_final2n.sbatch`](scripts/c4x4_g8_final2n.sbatch) |
| **1 節點（8×H200）** | **472.0 TF**（471.5–472.3，殘差 1.999；勝 Dell XE9680 公開值 394.5 達 +19.6%） | `build/chpl2d` | [`dat/HPL_1node_g8.dat`](dat/HPL_1node_g8.dat)（N=319488, NB=4096, P2Q4） | [`scripts/c2d_g8_1ncrown.sbatch`](scripts/c2d_g8_1ncrown.sbatch) |
| **單 GPU（1×H200）** | **75.2 TF** 穩健（moduli=12）/ **79.9 TF** 極限（moduli=11） | `build/hpl_single` | 免輸入檔（CLI 給 N） | [`scripts/single_g8seed.sbatch`](scripts/single_g8seed.sbatch) |

多 GPU 冠軍共通環境變數（G8 後端）：

```bash
CHPL2D_G8=1 CHPL2D_G8_MODULI=11 CHPL2D_G8_FAST=1 CHPL2D_G8_NT=3072 \
CHPL2D_WCH=4 CHPL2D_PRESEL=1 CHPL2D_PRESEL2=1 CHPL2D_WS=64 CHPL2D_XT=6144 \
NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=1 NCCL_BUFFSIZE=2097152 CUDA_DEVICE_MAX_CONNECTIONS=32
```

```bash
# 兩節點冠軍（16 GPU, P4Q4, N=460800, NB=4096）
mpirun -np 16 ./build/chpl4x4 dat/HPL_2node_g8.dat
# 四節點冠軍（32 GPU, P4Q8, N=622592, NB=4096）
mpirun -np 32 ./build/chpl4x4 dat/HPL_4node_g8.dat
# 單節點冠軍（8 GPU, P2Q4, N=319488, NB=4096；G8 但 NT=2816、NCH=6）
CHPL2D_G8_NT=2816 NCCL_MAX_NCHANNELS=6 mpirun -np 8 ./build/chpl2d dat/HPL_1node_g8.dat
# 單 GPU（backend 三選一：native / ozaki / gemmul8）
HPLS_NB=1024 HPLS_AGG=8 HPLS_BACKEND=gemmul8 HPLS_G8_MODULI=12 HPLS_G8_FAST=1 \
  HPLS_G8_CHUNK=3072 ./build/hpl_single 122880
```

殘差裕度備註：`CHPL2D_G8_MODULI=11` 在上述 N 的殘差為 1.4–2.2（門檻 16.0，多種子重複驗證）。
更保守可用 `MODULI=12`（殘差 ~0.1–0.4，兩節點約 −5%）。

## 與 NVIDIA 容器的正式對決（scheme-I 時代，歷史紀錄）

同一批節點、交錯執行、雙方各用自己調到最佳的參數。當時我方冠軍為 scheme-I（下表）；
現行 G8 冠軍又比當時的我方再快 25%/16%，對決結論只會更強。

**兩節點**：G8 時代重打（2026-08-12，交錯）：**CHPL 984.4/983.2 vs 容器 782.5/782.5，+25.7%**。
scheme-I 時代舊紀錄：789.6 vs 786.2（5/5 完全分離）。

**12 GPU（1.5 節點）**：CHPL **724.2 TF**（P4Q3、8+4 佈局、N=405504；60.4 TF/GPU，對比 16 GPU 的
61.5 幾乎線性）；**NVIDIA 容器無法以 12 GPU 執行**（NVSHMEM 要求各節點 PE 數相等、NVLS team
初始化失敗、關 NVLS 後裝置端 NVSHMEM 死鎖——四層阻擋全數記錄於實驗日誌）。彈性 GPU 數是本實作
的真實能力優勢。

**單節點**：G8 時代重打（2026-07-18，3 輪交錯）：**CHPL 470.8 vs 容器 385.9，+22.0%，3/3 完全分離**
（重現：[`scripts/c2d_g8_showdown1n.sbatch`](scripts/c2d_g8_showdown1n.sbatch)）。
scheme-I 時代舊紀錄：403.3 vs 385.8 vs Dell XE9680 公開值 394.5。

## 為什麼快：核心技術

0. **GEMMul8（Ozaki-II CRT）trailing-update 後端 + NB=4096**（現行冠軍的兩把鑰匙）
   把 trailing update 換成 RIKEN GEMMul8 的 CRT 模數式 int8 模擬（`CHPL2D_G8=1`），W 傳輸、
   pivoting、交換全部不動。關鍵洞察有三：
   - **在 HPL 的實際精度預算下比較方案**：scheme-I 的 D=5 是硬下限（21k 等效 int8 工作量，D=4 過不了殘差）；
     GEMMul8 的 moduli 旋鈕更細，moduli=11–12 即可通過（11–12 個全 k GEMM），**少做 40–48% int8 運算**。
   - **fastmode 在 HPL 隨機矩陣上又快又準**（m13 殘差 0.029 vs accurate 模式 0.765），還省一個縮放 GEMM。
   - **NB=4096 解鎖 chain**：實測系統在冠軍 N 下是 panel-chain-bound（~425ms/panel），單純引擎加速只會變成
     s0 空等。NB=4096 讓 panel 數砍半（chain 總量砍半）+ k 加倍（G8 的 mod/CRT 開銷減半）——scheme-I
     吃不到這紅利（工作量與 k 成正比），G8 可以。NB 曲線在 4096 見頂（5120 實測 −4%；≥6144 撞
     buildlists kernel 的 232KB shared-memory 硬體牆）。

1. **Ozaki-I int8 模擬 fp64 GEMM**（單節點冠軍與所有基礎設施的根基）
   每個 fp64 拆成 6 片 base-128 int8 精確切片（只乘 2 的冪、零捨入誤差），**融合對角線技巧**把
   21 個切片對 GEMM 縮成 6 個 cublasLt int8 GEMM（int32 流量 144→48 B/元素）。
   物理根基：fp64 與 int8 模擬同樣頂到 700W 功耗牆，但 int8 每 flop 能量低——時脈維持 1980 MHz
   （fp64 剩 ~1500），實測**每焦耳有效運算量比容器多 14.3%**（88.2 vs 77.2 GFLOPS/W）。

2. **Tournament pivoting（CALU）+ fp32 選主元**：候選挑選用 fp32 stacked getrf，主元列以 fp64 精確重分解——精度零妥協。

3. **Copy-Engine 資料面（CUDA IPC + 單調 flag 計數器）**：同節點 W/U 傳遞、pivot 交換、回代繞過 NCCL 直讀對方顯存；回代 0.91s→0.156s；已分解左側列完全不再交換。

4. **Preselect / Preselect-2**：延遲探針量出選主元與 stage-2 的鏈上暴露，提前一個 panel 到專用 stream 與 chainB 重疊（PRESEL2 在 NB=4096 下仍值 +1.9%）。

5. **P4Q4 星型交換**：pivot 交換流量天然以對角 owner row 為中心，4-way 交換用 per-peer 列表 + IPC 直讀，成本近乎 2-way。

6. **多節點混合通訊**：同節點 CE、跨節點 NCCL（列分塊 + K-slab 消費）；多 communicator 深排隊死鎖以 host 節流解決。
   實測 W 廣播已達 95 GB/s（NCCL 原生多 NIC）——傳輸層已無油水（壓縮/雙 rail/int8 傳輸皆以數據封死，見實驗日誌）。

## 版本說明

| 檔案 | 網格 | 適用 | 最佳成績 |
|---|---|---|---|
| `src/chpl4x4.cu` | **P=4**（4×4 於 16 GPU、4×8 於 32 GPU） | **兩節點/四節點冠軍** | G8 後端：**989.1**（2n）/ **1853**（4n）；scheme-I：792.2（2n） |
| `src/chpl2d.cu` | **P=2**（2×4 於 8 GPU） | **單節點冠軍** | G8 後端：**472.0**；scheme-I：403.3 |
| `src/hpl_single.cu` | 單 GPU | 單卡天花板量測（native/ozaki/gemmul8 三後端 + selftest/gemmtest） | **79.9**（gemmul8 m11）/ 47.2（ozaki）|
| `src/chpl.cu` | 1×8 單列網格 | 歷史參考 | 384.4 |
| `src/wbench.cu` | — | W 廣播微基準（單廣播 vs 雙 rail） | （診斷工具） |

冠軍成績表已用 GEMMul8 **v3.2.0**（submodule 目前釘選版本）重新驗證：2-node 986.0–986.7 TF、
殘差與 v3.1.0 時代紀錄逐位元相同（1.999253693793 / 1.943372891117，3/3 輪 PASSED）。1-node
初測落在舊紀錄範圍之外（-0.9%），未直接歸咎於節點差異，而是另外建了 v3.1.0 對照二進位、
在**同一節點**上交錯 A/B 4 輪驗證：v3.1.0 471.5–472.2 TF vs v3.2.0 471.4–472.2 TF，完全打平
（v3.2.0 中位數甚至略高），證實原本的落差純粹是節點間差異，不是版本回退——安全的無痛升級，
細節見 `docs/EXPERIMENTS.md`。

## 建置

需求：CUDA 13.x、NVHPC（提供 OpenMPI 與 NCCL）、H200（sm_90a）。[RIKEN GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) 已內嵌為 git submodule（`gemmul8/`，目前釘選 `v3.2.0`）。

```bash
git clone --recurse-submodules https://github.com/Koyingtw/CHPL.git
cd CHPL
# 若 clone 時忘記 --recurse-submodules：
#   git submodule update --init --recursive
bash scripts/build.sh   # 自動最小化建置 GEMMul8（僅 9 個目標檔）＋ chpl4x4 / chpl2d / hpl_single
```

`build.sh` 會在 `gemmul8/lib/libgemmul8.a` 不存在時自動觸發最小化建置；若已手動 `git submodule update`
到別的 GEMMul8 版本（例如測試新 release），直接重跑 `bash scripts/build.sh` 即可（若 `libgemmul8.a`
已存在但對應到舊版本原始碼，先 `rm gemmul8/lib/libgemmul8.a` 再重跑，確保重新編譯）。

更新 GEMMul8 版本：
```bash
cd gemmul8 && git fetch --tags && git checkout v3.2.0 && cd ..
rm -f gemmul8/lib/libgemmul8.a
bash scripts/build.sh
git add gemmul8 && git commit -m "Bump GEMMul8 to v3.2.0"
```

注意：**不要**直接 `make` GEMMul8 全量建置——它會編數百個模板組合並耗盡編譯資源；
`build.sh` 內建的最小化配方只編 `gemm<double,INT8>` 實際需要的 9 個目標檔。

## 執行 Slurm 腳本

`scripts/*.sbatch` 是原始實驗的腳本，已移除叢集專屬設定（計畫帳號、partition、壞節點清單），
請在 **repo 根目錄**提交，並自行帶入帳號與 partition：

```bash
mkdir -p logs
sbatch -A <your_account> -p <partition> scripts/c4x4_g8_final2n.sbatch
```

- 腳本以 `CHPL_ROOT`（預設為 `$SLURM_SUBMIT_DIR`）定位 repo，二進位位於 `$CHPL_ROOT/build/`。
- 與 NVIDIA 容器對決的腳本需要 `hpc-benchmarks:26.02` 的 Singularity 映像，路徑由 `NV_HPL_SIF`
  指定（預設 `$HOME/hpc-benchmarks-26.02.sif`）。
- `module load x86-nvhpc/26.3`、`/usr/local/cuda-13.0` 等模組名與路徑請依你的環境調整。

## 授權

本專案以 [BSD-3-Clause](LICENSE) 授權釋出。CHPL 是 netlib HPL-2.3 求解核心的 CUDA 改寫，
原 HPL 版權聲明見 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)；GEMMul8 submodule 依其自身授權。

## 輸入檔格式

縮減版 13 行 HPL.dat（見 `dat/`）：第 5–6 行 `1、N`；第 7–8 行 `1、NB`；第 11–12 行 `P、Q`（rank = P·qc + pr）；第 13 行殘差門檻 16.0。

## 文件

| 文件 | 內容 |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | 深度設計：程序網格、panel 流水線、Ozaki 切片數學、CE flag 協議、星型交換、preselect |
| [docs/TUNING.md](docs/TUNING.md) | 環境變數表、記憶體預算、調參方法論 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 完整實驗日誌（170+ 實驗、~60 個死胡同的原始記錄，含 G8 移植與 chain 手術全程，英文） |

## Ozaki-I vs Ozaki-II：一個重要的教訓

早期以「全 fp64 精度」比較兩方案的孤立 GEMM：scheme-I ~62 TF-eq vs scheme-II ~62.9 TF-eq，
近乎打平，故當時選擇 scheme-I。**後來在完整 solver 層級被推翻**：HPL 只需要殘差 < 16，
在這個精度預算下 scheme-II 的 CRT 旋鈕（moduli=11–13）比 scheme-I 的硬下限（D=5）少做
40–48% 的 int8 運算——單卡 47→75 TF、兩節點 792→989、四節點 1593→1853。

**教訓：比較數值模擬方案必須在目標精度預算下比，不是在最高精度下比。**
（單節點一度誤判為「G8 受記憶體限制無增益」——那是因為初掃只用了 NT=1024；
補掃 NT 維度後 G8 以 472.0 全面接管，scheme-I 退為歷史基線與基礎設施的設計語境。）

## 評估過但未採用：EmuGEMM（NTHU-SC/EmuGEMM，SC26 artifact）

2026-08-13 評估是否以新版 Ozaki-II（CRT）實作 EmuGEMM 取代 GEMMul8。結論：**在 kernel 層級的
效能對決就被淘汰**，未進入 chpl4x4.cu 整合。

- EmuGEMM 唯一真正「多 moduli 融合成單一 kernel」的變體（`int8_crt_dgemm_fused_ii_nt`）**僅支援
  SM100/Blackwell**；在 H200（SM90）上只有逐 modulus 呼叫、mod 化簡融合進 epilogue 的版本，
  結構上與 GEMMul8 現行的逐 modulus 迴圈相同量級。
- 在 H200 上以生產形狀（K=NB=4096, N=G8_NT=3072, moduli=11）實測：**gemmul8 3.1.0 全面勝出
  +11%〜+19%**（97.9–117.8 TF vs EmuGEMM 82.6–99.2 TF）；在論文自己採用的方形基準形狀
  （M=N=8192）差距縮小到 +2.8%〜+5.0%，但仍未反轉。精度打平（兩者 moduli=11 皆為
  36.18 有效位元、相對誤差 1.286e-11，同屬 Ozaki-II/CRT 數學）。
- 根因：EmuGEMM 論文 Fig.6 宣稱的 1.05–1.95x 優勢是相對於**舊版 vendored GEMMul8（commit
  2f8522c）**量測的；本專案實際使用的 gemmul8 3.1.0 是 RIKEN 之後大幅優化過的版本，優勢已被
  抹平甚至反轉。**教訓一：評估任何論文/artifact 宣稱優於某函式庫的說法時，務必用「目前生產中
  實際使用的版本」重新基準測試，而非信任論文凍結時的舊版本快照。**
- 另一組獨立測試（直接跑 artifact 自帶的 `bench_real.py`）曾得出「EmuGEMM 135 TF 完勝舊版
  GEMMul8 77–95 TF」，看似矛盾。已根因排除：`bench_real.py` 沒加 `--with-post` 時只計時
  純 GEMM mainloop，不含 EmuGEMM 必要的量化切片與 CRT 重建（這兩步在 HPL 裡無法省略，每個
  panel 資料都不同）；而 `gemmul8::gemm()` 是不可分割的原子呼叫，天生就是全流程計時——等於
  拿「EmuGEMM 只算核心」對「GEMMul8 全流程」，且 GEMMul8 那邊還是舊版。同形狀交叉驗證：
  舊版 GEMMul8 全流程 77–95 TF ＜ gemmul8 3.1.0 全流程 108 TF ＜ EmuGEMM 純 kernel 135 TF，
  而 EmuGEMM 全流程（扣掉量化+CRT 開銷後）量到 103.6 TF——數字完全對得上，兩邊測試互相印證。
  **教訓二：比較 benchmark harness report 出來的 TFLOPS 前，務必確認兩邊量的是同樣的範圍
  （完整可用管線 vs. 純 kernel），不要被拆分計時欄位的預設行為誤導。**

完整量測與拆解分析見 `docs/EXPERIMENTS.md`「EmuGEMM vs gemmul8 3.1.0」；重現腳本在
`probe/`（`g8_baseline_probe.cu`、`emu_vs_g8_bench.py` 等）。

## 已知事項

- **NB 與後端綁定**：G8 後端冠軍用 NB=4096（NB≥6144 撞 `buildlists_kernel` 的 232KB shared-memory
  硬體上限；NB=5120 實測較慢）。scheme-I（`CHPL2D_OZ=1`）用 NB=2048。
- **moduli 下限**：`CHPL2D_G8_MODULI=11` 是實測地板（殘差 1.4–1.9）；moduli=10 外插必掛。
  N 大幅高於本文配置時殘差近似線性成長，需自行重驗。
- **Preselect 罕見競態**：小 N 曾見一次殘差爆炸；生產規模乾淨。殘差檢驗會抓住壞 run（不會靜默錯誤）。
- **驅動記憶體行為**：多次 run 之間勿 free/realloc 大緩衝（觀察到驅動層損壞）；本實作單次執行內持久配置。
- 叢集相關參數：腳本已移除 `-A`/`-p`/`--exclude`，提交時以 `sbatch -A … -p …` 帶入；module 名與 CUDA 路徑請依環境修改腳本開頭（見「執行 Slurm 腳本」）。

## 能耗實測（兩節點、GPU-only、1Hz 積分；scheme-I 時代量測）

| | CHPL (N=497664) | NVIDIA 容器 (N=524288) |
|---|---|---|
| 穩態功率 | 687–693 W/GPU | 694–698 W/GPU |
| 每次求解能量 | **0.93 MJ（0.26 kWh）** | 1.24 MJ（0.35 kWh） |
| 能效 | **88.2 GFLOPS/W** | 77.2 GFLOPS/W |

G8 冠軍解題時間再縮 17–25%，能耗數字只會更好（尚未重新積分量測）。

## 與 NVIDIA 官方 FP64 模擬的關係

nvcr 26.02 容器內建同族技術（GEMMul8 / Ozaki，`HPL_EMULATE_DOUBLE_PRECISION`），但**僅支援
Blackwell（SM100+）**——在 H200（SM90）實測會退回原生 fp64。因此本專案所有對決中容器都處於
它在 H200 上的真實最佳狀態，且本實作等於把 Ozaki 型 HPL（含 GEMMul8 後端）帶到了 NVIDIA
官方尚未支援的 Hopper 世代。

## 測試環境

- 節點：8× NVIDIA H200 SXM（141 GB HBM3e、NVLink）/節點，節點間 InfiniBand（實測 NCCL 廣播 ~95 GB/s）
- 軟體：CUDA 13.0、NVHPC 26.3（OpenMPI + NCCL）、Slurm
- 對照組：`nvcr.io/nvidia/hpc-benchmarks:26.02`（xhpl，其最佳參數由掃描取得）
