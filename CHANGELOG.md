# 變更紀錄

## 2026-10-14：補齊公開展示與操作文件

- 補上專題用途、版本比較、資料與輸出格式、執行步驟及指標限制。
- 將兩支腳本的個人電腦絕對路徑改成 repository 內的相對預設位置，加入選用的 `config_local.m` 與設定範本。
- 主程式與舊版的遮罩／分析輸出分別預設為 `outputs/attention/`、`outputs/legacy/`，啟動時建立輸出目錄。
- 修正舊版註解：改用 `predict` 後，標註及 datastore 流程仍使用 Computer Vision Toolbox。
- 新增 `check_setup.m` 與 `.gitignore`，協助檢查環境並排除本機資料、模型、輸出與設定。

這次整理未修改網路結構、增強方式、訓練分組或後處理演算法。更早的模型修改見 [版本比較](docs/versions.md) 與 Git commit 紀錄。
