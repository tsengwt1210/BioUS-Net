% 使用方式：複製為 config_local.m，再修改需要的設定。
% 兩支主程式會在預設參數之後載入本檔；config_local.m 不納入 Git。
% projectRoot 由主程式提供，不需填入使用者名稱或電腦磁碟路徑。

gTruthPath_default = fullfile(projectRoot, 'data', 'gTruth.mat');
imageFolderPath = fullfile(projectRoot, 'data', 'images');

% 保留主程式預設輸出位置：Unet.m -> outputs/attention，舊版 -> outputs/legacy。
% 如需自訂，取消下行註解：
% maskFolderPath = fullfile(projectRoot, 'outputs', 'my_run');

% 以實際影像校正結果填入每像素面積。1 僅為預設值，不能代替校正。
pixelSize_sq_um = 1;
minAreaThreshold_px = 30000;
morphologyRadius = 15;
maskEffectValue = 50;

% 僅 Unet_V6.m 使用。此為舊版參數範例，不代表顯微影像具時間刻度。
% 只有資料確實具有經校正的時間軸時才可解讀為 Hz。
% time_per_pixel_s = 2.5e-8;
% sampling_rate_hz = 1 / time_per_pixel_s;
% target_f1_hz = 1.05e6;
% target_f2_hz = 4.08e6;
