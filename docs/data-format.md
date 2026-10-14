# 資料格式與輸出

## 影像與標註

目前流程以 RGB 影像及背景／單一目標類別為準。網路輸入為 512 × 512 × 3；標籤縮放使用 nearest-neighbor。不同原始長寬比會被直接縮放，不會等比例補邊。

訓練用 MAT 檔中需有 MATLAB `groundTruth` 物件。程式會尋找該物件，讀取：

- `DataSource.Source`：原始影像檔案路徑。移動資料後，需確認這些路徑仍有效。
- `LabelDefinitions.Name`：標籤名稱；程式會將 `background` 排在第一個位置。
- `LabelData`：各影像的多邊形座標，頂點格式為 `[x y]`。此匯出流程是依多邊形標註撰寫，不能直接把任意二值遮罩 MAT 檔當成輸入。

可一次選取多個 `groundTruth` MAT 檔，但標籤定義必須一致。資料內容由使用者自行準備，不隨 repository 提供。

## 遮罩與檔名配對

影像 `sample_001.png` 對應 `sample_001_mask.png`。遮罩尺寸與原圖相同，背景值為 `0`、目標值為 `255`。主程式先將標註轉為遮罩，再依影像主檔名尋找配對。

缺少對應遮罩的影像會被排除。若不同影像使用相同主檔名，例如 `sample_001.jpg` 與 `sample_001.png`，會指向相同遮罩檔，因此資料應使用唯一的主檔名。

原程式允許讀入多個類別名稱，但遮罩使用 `uint8` 並乘上 255，與多類別 `labelIDs` 的設定不一致。請先限制為背景加一個目標；正式擴充多類別前需一併修正遮罩編碼及讀取方式。

## 模型檔

訓練儲存內容為 `net`、`classNames`、`info`、`targetSize`。推論入口至少檢查 `net` 與 `classNames`，並從網路輸入層讀取影像尺寸。模型需與這兩支腳本預期的 MATLAB 網路物件相容。

檔名沿用原始設計：

- 單次訓練：`trainedUnet_YYYYMMDD.mat`。
- 五折訓練：`trainedUnet_CV_FoldN_YYYYMMDD_HHMM.mat`。

這些檔案寫入 MATLAB 目前資料夾。日期相同的單次訓練會使用同一檔名，保留多次實驗時應自行更名。

## 面積與後處理參數

| 參數 | 預設值 | 意義 |
| --- | --- | --- |
| `pixelSize_sq_um` | 1 | 每像素實際面積，必須依影像尺度校正 |
| `minAreaThreshold_px` | 30000 | 納入面積表格的最小物件像素面積 |
| `morphologyRadius` | 15 | 閉運算的圓盤半徑，單位為像素 |
| `maskEffectValue` | 50 | 亮化／暗化圖的效果強度 |

目前 `Unet.m` 會去除小區塊、閉運算、填洞，再對 Solidity 小於 0.9 的物件做凸包修補。這些操作會改變區域邊界與面積，尤其凸包可能填入原本的凹區。參數應配合原始影像大小與目標形狀調整。

## 批次輸出

主程式輸出 `prediction_results_<資料來源>_<類別>.xlsx`，每列是一個通過面積門檻的物件：

| 欄位 | 內容 |
| --- | --- |
| `ImageName` | 原始影像檔名 |
| `ObjectID` | 該張影像的連通物件編號 |
| `PixelArea_px` | 後處理後的像素面積 |
| `RealArea_um2` | 像素面積乘上每像素實際面積 |
| `MeanConfidence` | 該物件範圍內模型最大類別分數的平均值 |

`MeanConfidence` 不是經校準的正確機率，也不是面積誤差。後處理加入的像素仍使用原預測分數圖計算，因此不能用這一欄代替與標註的比較。

選擇產生影像時，每張原圖另輸出 `_prediction.png`、`_masked_bright.png` 與 `_masked_dark.png`。主程式單張分析以視窗顯示結果，沒有自動儲存這些分析視窗。沒有通過篩選的物件時，不會產生 Excel。

歷史版 `Unet_V6.m` 另有 ROI 與 FFT 欄位，見 [版本說明](versions.md)。
