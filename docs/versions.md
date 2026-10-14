# 版本比較

這個 repository 保留兩支獨立腳本，並以 Git commit 記錄修改。檔名與程式內的版本標記曾分別演進，閱讀時以實際程式為準。

| 項目 | `Unet.m`，目前主入口 | `Unet_V6.m`，歷史版本 |
| --- | --- | --- |
| 內文版本 | v5.8 | v5.4 |
| 網路 | 自訂 Attention U-Net；encoder 通道 16、32、64、128，bottleneck 256 | MATLAB `unetLayers` |
| 輸入 | 512 × 512 RGB | 512 × 512 RGB |
| 訓練模式 | 80/20 隨機切分、五折交叉驗證 | 80/20 隨機切分 |
| 學習率 | Adam，初始 0.001，每 10 epoch 乘 0.2 | Adam，固定 0.001 |
| Batch size | 4 | 4 |
| Epoch | 五折明訂 30；80/20 未指定 `MaxEpochs`，使用 MATLAB 預設 | 30 |
| 面積後處理 | 去除小物件、閉運算、填洞、條件式凸包、開運算 | 去除小物件、閉運算 |
| 單張分析 | 面積、信心分數與視窗疊圖 | 另含 ROI 及 FFT 分析 |
| FFT | 已移除 | 保留歷史探索流程 |

主程式的兩種訓練模式使用不同增強設定：80/20 模式有翻轉、旋轉、平移與 HSV 色彩擾動；五折模式目前只有翻轉與旋轉。因此不能將兩者視為完全相同的實驗設定。

## 已有 Git 修改紀錄

下表記錄 commit 中的修改，不代表每次修改都已完成效能比較。

| 日期 | Commit | 修改內容 |
| --- | --- | --- |
| 2026-03-17 | [`e9d82f6`](https://github.com/tsengwt1210/BioUS-Net/commit/e9d82f6) | 收錄舊版分割與特徵分析腳本 |
| 2026-03-18 | [`e6a271b`](https://github.com/tsengwt1210/BioUS-Net/commit/e6a271b) | 新增移除 ROI／FFT、加入區域填補的主腳本 |
| 2026-03-18 | [`da79e77`](https://github.com/tsengwt1210/BioUS-Net/commit/da79e77) | 增加五折訓練模式與驗證準確率彙整 |
| 2026-04-14 | [`1421878`](https://github.com/tsengwt1210/BioUS-Net/commit/1421878)、[`a69ed07`](https://github.com/tsengwt1210/BioUS-Net/commit/a69ed07) | 調整亮度、幾何增強、學習率與影像前處理 |
| 2026-05-20 | [`fe73e47`](https://github.com/tsengwt1210/BioUS-Net/commit/fe73e47) | 主分支加入 Attention 架構與色彩擾動，保留 RGB 輸入 |

另有 `testing` 分支，最後的既有 commit 為 2026-04-16 的 [`5f92b9f`](https://github.com/tsengwt1210/BioUS-Net/commit/5f92b9f)。它保留獨立的 Attention 實驗紀錄，與 `main` 並非同一條線性修改序列。

## 舊版 FFT 的解讀範圍

`Unet_V6.m` 從 ROI 灰階影像沿一個方向取平均，形成一維序列，再做去平均、加窗與 FFT。程式用 `time_per_pixel_s` 將像素間隔換算為時間，並讀取 1.05 MHz 與 4.08 MHz 附近的頻譜幅值。

這些是舊版的參數與計算流程。顯微鏡影像的像素本身是空間取樣；沒有額外的時間軸校正依據時，不能將上述值當成實際聲學頻率量測。閱讀目前的分割與面積流程請從 `Unet.m` 開始，舊版保留作為版本紀錄。
