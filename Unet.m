%% ==== U-Net 全流程訓練與圈選腳本 v5.6 (智能修補版) ====
% 版本說明：  2026.03.17
% v5.6: 導入智能後處理。利用 Solidity < 0.9 條件式啟動 Convex Hull 修補缺角，
%       並修復 labeloverlay 的 categorical bug，確保疊圖顯示為飽滿實心。
%       單張圖片分析改為跳出視窗顯示亮化/暗化/輪廓線，不再直接儲存。
% v5.5: 移除所有 ROI 擷取與 FFT 特徵提取功能，專注於 U-Net 分割與面積計算。

clear; clc; close all;
%% ===== Step 1: 全域參數設定 =====
gTruthPath_default = 'D:\專題\U-net\scripts\v3table_rebuilt.mat'; % 預設 gTruth 檔案路徑
imageFolderPath = 'D:\專題\U-net\images_2';       % 包含所有原始圖片的資料夾
maskFolderPath = 'D:\專題\U-net\masks\V6';    % 儲存產生出來的 mask 與結果圖

pixelSize_sq_um = 1 * 1; % 【重要】每像素代表的實際"面積" (例如 0.5um * 0.5um = 0.25)。
minAreaThreshold_px = 30000; % 面積計算時，小於此像素數的物件將被過濾
morphologyRadius = 15; % 形態學閉運算的半徑
maskEffectValue = 50;  % 亮度調整圖的效果強度。

fprintf('====================================================\n');
fprintf('    U-Net 全流程腳本 v5.6 - 智能修補版\n');
fprintf('====================================================\n');

%% ===== 主流程控制迴圈 =====
while true
    fprintf('\n===== 主選單 =====\n');
    fprintf('[1] 產生標籤遮罩 (Generate Masks from gTruth)\n');
    fprintf('[2] 訓練新模型 (Train a New Model)\n');
    fprintf('[3] 使用現有模型進行批次預測與分析 (Batch Predict with Existing Model)\n');
    fprintf('[4] 從單張圖片分析 (Analyze a Single Image)\n');
    fprintf('[5] 離開程式 (Exit)\n');
    
    choice = input('請輸入您的選擇 [1-5]: ', 's');
    
    switch choice
        case '1'
            %% --- 任務 1: 產生標籤遮罩 ---
            fprintf('\n--- [任務 1: 產生標籤遮罩] ---\n');
            [gTruth, classNames] = loadAndPrepareGTruth(gTruthPath_default);
            if isempty(gTruth), continue; end
            generateMasksFromGTruth(gTruth, classNames, maskFolderPath);
            
        case '2'
            %% --- 任務 2: 訓練新模型 ---
            fprintf('\n--- [任務 2: 訓練新模型] ---\n');
            disp('正在建立與準備訓練資料集 (採用檔名對齊)...');
            [~, classNames] = loadAndPrepareGTruth(gTruthPath_default); 
            
            [imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionAlignedSets(imageFolderPath, maskFolderPath, classNames, 0.8);
            if isempty(imdsTrain)
                disp('❌ 訓練集為空，請檢查圖片與遮罩檔名是否匹配 (_mask.png)。返回主選單。');
                continue;
            end
            fprintf('✅ 資料集切分完成: %d 訓練, %d 驗證\n', numel(imdsTrain.Files), numel(imdsVal.Files));
            targetSize = [512 512];
            augmenter = imageDataAugmenter('RandXReflection',true, 'RandYReflection',true, 'RandRotation',[-20, 20]);
            dsTrain = pixelLabelImageDatastore(imdsTrain, pxdsTrain, 'DataAugmentation', augmenter);
            dsVal = pixelLabelImageDatastore(imdsVal, pxdsVal);
            dsTrain = transform(dsTrain, @(data) resizeImageAndLabel(data, targetSize));
            dsVal = transform(dsVal, @(data) resizeImageAndLabel(data, targetSize));
            disp('✅ 資料集準備完成');
            
            inputSize = [targetSize, 3];
            lgraph = unetLayers(inputSize, numel(classNames));
            options = trainingOptions('adam', 'InitialLearnRate', 1e-3, 'MaxEpochs', 30, ...
                'MiniBatchSize', 4, 'Shuffle', 'every-epoch', 'ValidationData', dsVal, 'Plots', 'training-progress');
            
            disp('🚀 開始訓練 U-Net 模型...');
            [net, info] = trainNetwork(dsTrain, lgraph, options);
            disp('✅ U-Net 訓練完成');
            
            if lower(input('是否要儲存模型? (y/n) [y]: ', 's')) ~= 'n'
                dateStr = datestr(now, 'yyyymmdd');
                modelFileName = sprintf('trainedUnet_%s.mat', dateStr);
                save(modelFileName, 'net', 'classNames', 'info', 'targetSize');
                fprintf('✅ 模型已儲存為 %s\n', modelFileName);
            end
            
        case '3'
            %% --- 任務 3: 使用模型進行批次預測 ---
            fprintf('\n--- [任務 3: 使用模型進行批次預測與分析] ---\n');
            
            [file, path] = uigetfile('*.mat', '請選擇要載入的模型 .mat 檔案');
            if isequal(file, 0), disp('⚠️ 已取消選擇模型，返回主選單。'); continue; end
            
            fprintf('正在載入模型: %s\n', fullfile(path, file));
            loadedData = load(fullfile(path, file));
            if isfield(loadedData, 'net') && isfield(loadedData, 'classNames')
                net = loadedData.net;
                loadedClassNames = loadedData.classNames;
            else
                fprintf('❌ .mat 檔案中缺少變數 (net, classNames)。返回主選單。\n'); continue;
            end
            fprintf('✅ 模型載入成功。\n');
            
            nonBackgroundClasses = loadedClassNames(~strcmpi(loadedClassNames, 'background'));
            if isempty(nonBackgroundClasses), error('錯誤：找不到非 "background" 的目標類別。'); end
            [selection, ok] = listdlg('PromptString', {'選擇要分析的目標類別:'}, 'SelectionMode', 'single', 'ListString', nonBackgroundClasses);
            if ~ok, disp('⚠️ 已取消選擇類別，返回主選單。'); continue; end
            targetClassName = nonBackgroundClasses{selection};
            fprintf('🎯 已選擇分析目標: %s\n', targetClassName);
            
            generateImages = (lower(input('是否要產生預測疊圖 (prediction images)? (y/n) [y]: ', 's')) ~= 'n');
            fprintf('\n請選擇要預測的對象:\n [1] 驗證集 [2] 訓練集 [3] 新的圖片資料夾\n');
            predictChoice = input('請輸入您的選擇 [1-3]: ', 's');
            
            imdsToPredict = []; outputFileName = ''; description = '';
            switch predictChoice
                case {'1', '2'}
                    disp('準備資料集 (採用檔名對齊)...');
                    [imdsAll, ~] = buildAlignedDatastores(imageFolderPath, maskFolderPath, loadedClassNames);
                    rng('default'); 
                    [imdsTrain, imdsVal, ~, ~] = partitionImdsPxds(imdsAll, imdsAll, 0.8);
                    if predictChoice == '1'
                        imdsToPredict = imdsVal;
                        outputFileName = sprintf('prediction_results_validation_%s.xlsx', targetClassName);
                        description = '驗證集 (Validation Set)';
                    else
                        imdsToPredict = imdsTrain;
                        outputFileName = sprintf('prediction_results_training_%s.xlsx', targetClassName);
                        description = '訓練集 (Training Set)';
                    end
                case '3'
                    newImgFolder = uigetdir([], '選擇要圈選的新圖片資料夾');
                    if newImgFolder ~= 0
                        imdsToPredict = imageDatastore(newImgFolder);
                        outputFileName = sprintf('prediction_results_new_images_%s.xlsx', targetClassName);
                        description = ['新資料夾: ' newImgFolder];
                    else
                        disp('⚠️ 已取消選擇資料夾，返回主選單。'); continue;
                    end
                otherwise, disp('無效的選擇，返回主選單。'); continue;
            end
            
            if ~isempty(imdsToPredict.Files)
                calcAreasAndSave(imdsToPredict, description, net, loadedClassNames, targetClassName, pixelSize_sq_um, maskFolderPath, outputFileName, ...
                    minAreaThreshold_px, morphologyRadius, generateImages, maskEffectValue);
            else
                fprintf('⚠️ 在指定的路徑中找不到任何圖片，返回主選單。\n');
            end
            
        case '4'
            %% --- 任務 4: 從單張圖片分析 ---
            fprintf('\n--- [任務 4: 單張圖片分析] ---\n');
            
            [file, path] = uigetfile('*.mat', '請選擇要載入的模型 .mat 檔案');
            if isequal(file, 0), disp('⚠️ 已取消選擇模型，返回主選單。'); continue; end
            
            fprintf('正在載入模型: %s\n', fullfile(path, file));
            loadedData = load(fullfile(path, file));
            if isfield(loadedData, 'net') && isfield(loadedData, 'classNames')
                net = loadedData.net;
                loadedClassNames = loadedData.classNames;
            else
                fprintf('❌ .mat 檔案中缺少變數 (net, classNames)。返回主選單。\n'); continue;
            end
            fprintf('✅ 模型載入成功。\n');
            predictAndAnalyzeSingleImage(net, loadedClassNames, pixelSize_sq_um, maskFolderPath, ...
                    minAreaThreshold_px, morphologyRadius, maskEffectValue);
        case '5'
            %% --- 任務 5: 離開 ---
            fprintf('程式已結束。\n'); break;
            
        otherwise
            fprintf('無效的選擇，請重新輸入。\n');
    end
end

%% ===== 副函式 (Helper Functions) =====
function [gTruth, classNames] = loadAndPrepareGTruth(defaultPath)
    gTruth = []; classNames = [];
    [filenames, path] = uigetfile('*.mat', '選擇一個或多個 groundTruth .mat 檔案', defaultPath, 'MultiSelect', 'on');
    if isequal(filenames, 0), disp('⚠️ 已取消檔案選擇。'); return; end
    if ~iscell(filenames), filenames = {filenames}; end
    allSources = {}; allLabels = {}; labelDefs = [];
    for i = 1:numel(filenames)
        S = load(fullfile(path, filenames{i}));
        fn = fieldnames(S);
        currentGTruth = [];
        for j = 1:numel(fn)
            if isa(S.(fn{j}), 'groundTruth'), currentGTruth = S.(fn{j}); break; end
        end
        if isempty(currentGTruth), warning('檔案 %s 中沒有找到 groundTruth 物件，已跳過。', filenames{i}); continue; end
        if i == 1 || isempty(labelDefs), labelDefs = currentGTruth.LabelDefinitions;
        elseif ~isequal(labelDefs, currentGTruth.LabelDefinitions), error('❌ 檔案 "%s" 的標籤定義不一致。', filenames{i}); end
        allSources = [allSources; currentGTruth.DataSource.Source];
        allLabels = [allLabels; currentGTruth.LabelData];
    end
    if isempty(allSources), error('❌ 您選擇的檔案中都沒有有效的 groundTruth 物件。'); end
    gTruth = groundTruth(groundTruthDataSource(allSources), labelDefs, allLabels);
    fprintf('✅ 已成功合併 %d 個檔案，總計 %d 筆資料。\n', numel(filenames), height(gTruth.DataSource.Source));
    classNames = labelDefs.Name;
    if ~any(strcmpi(classNames, 'background')), classNames = ['background'; classNames];
    else, bgIdx = strcmpi(classNames, 'background'); classNames = [classNames(bgIdx); classNames(~bgIdx)]; end
    fprintf('✅ 類別設定完成: %s\n', strjoin(classNames, ', '));
end

function generateMasksFromGTruth(gTruth, classNames, maskFolder)
    fprintf('正在產生影像遮罩 (Masks)...\n');
    if ~exist(maskFolder, 'dir'), mkdir(maskFolder); end
    numImages = height(gTruth.DataSource.Source);
    try
        if isempty(gcp('nocreate')), parpool; end
        parfor i = 1:numImages, generate_mask_helper(i, gTruth, classNames, maskFolder); end
        fprintf('✅ 平行運算執行成功。\n');
    catch ME
        fprintf('⚠️ 平行運算失敗，轉為標準迴圈。錯誤: %s\n', ME.message);
        for i = 1:numImages, generate_mask_helper(i, gTruth, classNames, maskFolder); end
    end
    disp('✅ 已轉出 segmentation mask');
end

function generate_mask_helper(idx, gTruth, classNames, maskFolder)
    imgPath = gTruth.DataSource.Source{idx};
    imgInfo = imfinfo(imgPath);
    localMask = zeros(imgInfo.Height, imgInfo.Width, 'uint8');
    for lblIdx = 2:numel(classNames)
        className = classNames{lblIdx};
        if ismember(className, gTruth.LabelData.Properties.VariableNames)
            polygons = gTruth.LabelData{idx, className};
            if ~isempty(polygons) && iscell(polygons)
                for p = 1:numel(polygons)
                    polyXY = polygons{p};
                    if ~isempty(polyXY) && size(polyXY,1) > 2
                        tempBinaryMask = poly2mask(polyXY(:,1), polyXY(:,2), imgInfo.Height, imgInfo.Width);
                        localMask(tempBinaryMask) = (lblIdx - 1);
                    end
                end
            end
        end
    end
    [~, name, ~] = fileparts(imgPath);
    imwrite(localMask * 255, fullfile(maskFolder, [name '_mask.png']));
end

function [imds, pxds] = buildAlignedDatastores(imageFolder, maskFolder, classNames)
    imds = imageDatastore(imageFolder);
    maskFiles = cell(numel(imds.Files), 1);
    validIdx = false(numel(imds.Files), 1);
    for i = 1:numel(imds.Files)
        [~, fname, ~] = fileparts(imds.Files{i});
        expectedMaskPath = fullfile(maskFolder, [fname '_mask.png']);
        if isfile(expectedMaskPath)
            maskFiles{i} = expectedMaskPath;
            validIdx(i) = true;
        end
    end
    imds = subset(imds, validIdx);
    maskFiles = maskFiles(validIdx);
    if isempty(imds.Files), pxds = []; return; end
    if numel(classNames) ~= 2
        warning('偵測到多於1個非背景類別，請手動確認 labelIDs 的像素值設定！');
        labelIDs = 0:(numel(classNames)-1);
    else
        labelIDs = [0; 255]; 
    end
    pxds = pixelLabelDatastore(maskFiles, classNames, labelIDs);
end

function [imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionAlignedSets(imageFolder, maskFolder, classNames, trainRatio)
    [imds, pxds] = buildAlignedDatastores(imageFolder, maskFolder, classNames);
    if isempty(imds) || isempty(pxds)
        imdsTrain = []; imdsVal = [];
        pxdsTrain = []; pxdsVal = [];
        return;
    end
    idx = randperm(numel(imds.Files));
    numTrain = round(trainRatio * numel(imds.Files));
    trainIdx = idx(1:numTrain);
    valIdx = idx(numTrain+1:end);
    imdsTrain = subset(imds, trainIdx);
    imdsVal = subset(imds, valIdx);
    pxdsTrain = subset(pxds, trainIdx);
    pxdsVal = subset(pxds, valIdx);
end

function [imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionImdsPxds(imds, pxds, trainRatio)
    idx = randperm(numel(imds.Files));
    numTrain = round(trainRatio * numel(imds.Files));
    imdsTrain = subset(imds, idx(1:numTrain));
    imdsVal = subset(imds, idx(numTrain+1:end));
    pxdsTrain = subset(pxds, idx(1:numTrain));
    pxdsVal = subset(pxds, idx(numTrain+1:end));
end

function dataOut = resizeImageAndLabel(dataIn, targetSize)
    if istable(dataIn), localImage = dataIn{1, 1}{1}; localLabel = dataIn{1, 2}{1};
    elseif iscell(dataIn), localImage = dataIn{1}; localLabel = dataIn{2};
    else, error('Transform function received unexpected data type: %s', class(dataIn)); end
    dataOut = {imresize(localImage, targetSize), imresize(localLabel, targetSize, 'nearest')};
end

% =========================================================================
% === 單張圖片分析 (包含智能修補、輪廓線、雙視窗) ===
% =========================================================================
function predictAndAnalyzeSingleImage(net, classNames, pixelSize, resultFolder, minArea_px, morphRadius, maskEffectVal)
    [file, path] = uigetfile({'*.png;*.jpg;*.tif;*.bmp', 'Image Files'}, '請選擇一張要分析的圖片');
    if isequal(file, 0), disp('⚠️ 已取消選擇圖片。'); return; end
    
    imgPath = fullfile(path, file);
    fprintf('正在分析圖片: %s\n', imgPath);
    
    nonBackgroundClasses = classNames(~strcmpi(classNames, 'background'));
    if isempty(nonBackgroundClasses), error('錯誤：找不到任何非 "background" 的目標類別。'); end
    [selection, ok] = listdlg('PromptString', {'選擇要分析的目標類別:'}, 'SelectionMode', 'single', 'ListString', nonBackgroundClasses);
    if ~ok, disp('⚠️ 已取消選擇類別，返回。'); return; end
    targetClassName = nonBackgroundClasses{selection};
    fprintf('🎯 已選擇分析目標: %s\n', targetClassName);
    netInputSize = net.Layers(1).InputSize(1:2);
    se = strel('disk', morphRadius);
    originalImg = imread(imgPath);
    
    resizedImg = imresize(originalImg, netInputSize);
    
    scores_resized = predict(net, resizedImg);
    [confidenceMap_resized, predMask_indices_resized] = max(scores_resized, [], 3);
    
    predMask_indices_originalSize = imresize(predMask_indices_resized, [size(originalImg,1) size(originalImg,2)], 'nearest');
    confidenceMap_originalSize = imresize(confidenceMap_resized, [size(originalImg,1) size(originalImg,2)]);
    
    predMask_categorical = categorical(predMask_indices_originalSize, 1:numel(classNames), classNames);
    
    % --- 智能影像後處理流程 ---
    binaryMask = predMask_categorical == targetClassName;
    cleanMask = bwareaopen(binaryMask, 50); % 去除極小星塵雜訊
    closedMask = imclose(cleanMask, se);    % 閉運算
    filledMask = imfill(closedMask, 'holes'); % 填補孔洞
    
    % 智能凸包修補 (Solidity < 0.9)
    cc_temp = bwconncomp(filledMask);
    stats_temp = regionprops(cc_temp, 'Solidity', 'PixelIdxList');
    smartHullMask = filledMask;
    for k = 1:cc_temp.NumObjects
        if stats_temp(k).Solidity < 0.9
            singleObjMask = false(size(filledMask));
            singleObjMask(stats_temp(k).PixelIdxList) = true;
            singleHull = bwconvhull(singleObjMask, 'objects');
            smartHullMask = smartHullMask | singleHull;
        end
    end
    
    smoothMask = imopen(smartHullMask, strel('disk', 5)); % 邊緣平滑化
    closedMask = bwareaopen(smoothMask, round(minArea_px / 2)); % 最終嚴格面積過濾
    
    cc = bwconncomp(closedMask);
    
    fprintf('--------------------------------------------------\n');
    fprintf('               分析結果 (%s)\n', targetClassName);
    fprintf('--------------------------------------------------\n');
    
    if cc.NumObjects == 0
        fprintf('在圖片中未偵測到有效物件。\n');
    else
        stats = regionprops(cc, 'PixelIdxList');
        objCount = 0;
        for objIdx = 1:cc.NumObjects
            pixelArea = numel(stats(objIdx).PixelIdxList);
            if pixelArea < minArea_px, continue; end
            objCount = objCount + 1;
            mean_confidence = mean(confidenceMap_originalSize(stats(objIdx).PixelIdxList));
            fprintf('物件 ID: %d\n', objIdx);
            fprintf('  - 像素面積: %.0f px\n', pixelArea);
            fprintf('  - 實際面積: %.2f (um^2)\n', pixelArea * pixelSize);
            fprintf('  - 平均信心分數: %.4f (%.1f%%)\n', mean_confidence, mean_confidence*100);
            fprintf('\n');
        end
        if objCount == 0, fprintf('偵測到的物件均小於面積閾值 (%d px)，無有效物件。\n', minArea_px); end
    end
    fprintf('--------------------------------------------------\n');
    
    [~, name] = fileparts(file);
    
    % --- 修正 Categorical Bug，將完美遮罩轉回疊圖 ---
    bgIdx = find(strcmpi(classNames, 'background'), 1);
    if isempty(bgIdx), bgIdx = 1; end 
    targetIdx = find(strcmpi(classNames, targetClassName), 1);
    
    tempIndices = ones(size(originalImg,1), size(originalImg,2)) * bgIdx;
    tempIndices(closedMask) = targetIdx;
    tempCategoricalMask = categorical(tempIndices, 1:numel(classNames), classNames);
    
    overlayImg = labeloverlay(originalImg, tempCategoricalMask, 'Colormap', 'jet', 'Transparency', 0.4);
    
    % --- 視窗 1：總體分析報告 ---
    figure('Name', ['總體分析報告: ' file], 'Position', [100 100 1200 500]);
    subplot(1, 3, 1); imshow(originalImg); title('原始圖片');
    subplot(1, 3, 2); imshow(overlayImg); title('U-Net預測結果 (實心疊圖)');
    subplot(1, 3, 3); imagesc(confidenceMap_originalSize); colormap('jet'); colorbar; axis image; title('信心分數熱圖');
    
    if ~exist(resultFolder, 'dir'), mkdir(resultFolder); end
    imwrite(overlayImg, fullfile(resultFolder, [name '_prediction.png']));
    fprintf('  -> 預測疊圖已儲存: %s\n', [name '_prediction.png']);
    
    % --- 計算亮化、暗化與輪廓圖 ---
    effectMagnitude = abs(maskEffectVal);
    if size(originalImg, 3) == 1, originalImg = cat(3, originalImg, originalImg, originalImg); end
    tempImg = int16(originalImg);
    mask3D = repmat(closedMask, [1, 1, 3]);
    brightImg = uint8(min(255, max(0, tempImg + int16(mask3D) * effectMagnitude)));
    darkImg = uint8(min(255, max(0, tempImg - int16(mask3D) * effectMagnitude)));
    
    % 提取邊緣輪廓並畫上黃色 (R=255, G=255, B=0)
    perim = bwperim(closedMask);
    contourImg = originalImg;
    r = contourImg(:,:,1); g = contourImg(:,:,2); b = contourImg(:,:,3);
    r(perim) = 255; g(perim) = 255; b(perim) = 0; 
    contourImg(:,:,1) = r; contourImg(:,:,2) = g; contourImg(:,:,3) = b;
    
    % --- 視窗 2：效果圖與輪廓圖 ---
    figure('Name', ['效果與邊界分析: ' file], 'Position', [150 150 1350 450]);
    subplot(1, 3, 1); imshow(brightImg); title(sprintf('亮化效果圖 (+%d)', effectMagnitude));
    subplot(1, 3, 2); imshow(darkImg); title(sprintf('暗化效果圖 (-%d)', effectMagnitude));
    subplot(1, 3, 3); imshow(contourImg); title('物件邊界輪廓線 (Contour)');
    
    fprintf('✅ 分析完成，已顯示結果視窗。\n');
end

% =========================================================================
% === 批次推論與計算 (同步導入智能修補與 Categorical Bug 修復) ===
% =========================================================================
function calcAreasAndSave(imdsSet, description, net, classNames, targetClassName, pixelSize, resultFolder, outFile, minArea_px, morphRadius, generateImages, maskEffectVal)
    safe_description = strrep(description, '\', '/');
    disp(['正在對 ' description ' 進行推論與計算...']);
    netInputSize = net.Layers(1).InputSize(1:2);
    
    results = table('Size', [0, 5], 'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
        'VariableNames', {'ImageName', 'ObjectID', 'PixelArea_px', 'RealArea_um2', 'MeanConfidence'});
    se = strel('disk', morphRadius);
    
    h_wait = waitbar(0, ['正在初始化: ' safe_description '...']);
    
    % 預先找出背景與目標類別的 Index (用於修正 Categorical)
    bgIdx = find(strcmpi(classNames, 'background'), 1);
    if isempty(bgIdx), bgIdx = 1; end 
    targetIdx = find(strcmpi(classNames, targetClassName), 1);
    
    for i = 1:numel(imdsSet.Files)
        waitbar(i/numel(imdsSet.Files), h_wait, sprintf('處理中 %d / %d: %s', i, numel(imdsSet.Files), safe_description));
        originalImg = readimage(imdsSet, i);
        
        resizedImg = imresize(originalImg, netInputSize);
        
        scores_resized = predict(net, resizedImg);
        [confidenceMap_resized, predMask_indices_resized] = max(scores_resized, [], 3);
        
        predMask_indices_originalSize = imresize(predMask_indices_resized, [size(originalImg,1) size(originalImg,2)], 'nearest');
        confidenceMap_originalSize = imresize(confidenceMap_resized, [size(originalImg,1) size(originalImg,2)]);
        predMask_categorical = categorical(predMask_indices_originalSize, 1:numel(classNames), classNames);
        
        % --- 智能影像後處理流程 ---
        binaryMask = predMask_categorical == targetClassName;
        cleanMask = bwareaopen(binaryMask, 50); 
        closedMask_temp = imclose(cleanMask, se);
        filledMask = imfill(closedMask_temp, 'holes');
        
        % 智能凸包修補
        cc_temp = bwconncomp(filledMask);
        stats_temp = regionprops(cc_temp, 'Solidity', 'PixelIdxList');
        smartHullMask = filledMask;
        for k = 1:cc_temp.NumObjects
            if stats_temp(k).Solidity < 0.9
                singleObjMask = false(size(filledMask));
                singleObjMask(stats_temp(k).PixelIdxList) = true;
                singleHull = bwconvhull(singleObjMask, 'objects');
                smartHullMask = smartHullMask | singleHull;
            end
        end
        
        smoothMask = imopen(smartHullMask, strel('disk', 5)); 
        closedMask = bwareaopen(smoothMask, round(minArea_px / 2));
        
        cc = bwconncomp(closedMask);
        
        if cc.NumObjects > 0
            stats = regionprops(cc, 'PixelIdxList');
            for objIdx = 1:cc.NumObjects
                pixelArea = numel(stats(objIdx).PixelIdxList);
                if pixelArea < minArea_px, continue; end
                
                [~, fname, fext] = fileparts(imdsSet.Files{i});
                mean_confidence = mean(confidenceMap_originalSize(stats(objIdx).PixelIdxList));
                
                newRow = {string([fname, fext]), objIdx, pixelArea, pixelArea * pixelSize, mean_confidence};
                results = [results; newRow];
            end
        end
        
        if generateImages
            [~, name] = fileparts(imdsSet.Files{i});
            
            % --- 修正 Categorical Bug 進行疊圖 ---
            tempIndices = ones(size(originalImg,1), size(originalImg,2)) * bgIdx;
            tempIndices(closedMask) = targetIdx;
            tempCategoricalMask = categorical(tempIndices, 1:numel(classNames), classNames);
            
            overlayImg = labeloverlay(originalImg, tempCategoricalMask, 'Colormap', 'jet', 'Transparency', 0.4);
            imwrite(overlayImg, fullfile(resultFolder, [name '_prediction.png']));
            
            effectMagnitude = abs(maskEffectVal);
            if size(originalImg, 3) == 1, originalImg = cat(3, originalImg, originalImg, originalImg); end
            tempImg = int16(originalImg);
            mask3D = repmat(closedMask, [1, 1, 3]);
            brightImg = uint8(min(255, max(0, tempImg + int16(mask3D) * effectMagnitude)));
            darkImg = uint8(min(255, max(0, tempImg - int16(mask3D) * effectMagnitude)));
            imwrite(brightImg, fullfile(resultFolder, [name '_masked_bright.png']));
            imwrite(darkImg, fullfile(resultFolder, [name '_masked_dark.png']));
        end
    end
    
    close(h_wait);
    
    if ~isempty(results)
        writetable(results, fullfile(resultFolder, outFile));
        fprintf('✅ 分析完成（過濾 <%d px，連接半徑 %d px），結果已存為 %s\n', minArea_px, morphRadius, outFile);
    else
        fprintf('⚠️ 未偵測到有效物件，未產生 Excel 檔案。\n');
    end
end