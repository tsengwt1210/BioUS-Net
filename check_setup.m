function report = check_setup()
%CHECK_SETUP 顯示 MATLAB 版本與主程式會用到的函式，不載入資料或啟動訓練。
% 在 repository 根目錄執行：report = check_setup;
fprintf('MATLAB %s (%s)\n', version, version('-release'));
names = {'trainNetwork'; 'trainingOptions'; 'layerGraph'; ...
    'multiplicationLayer'; 'unetLayers'; 'groundTruth'; ...
    'groundTruthDataSource'; 'pixelLabelDatastore'; ...
    'pixelLabelImageDatastore'; 'imageDataAugmenter'; 'jitterColorHSV'; ...
    'poly2mask'; 'bwareaopen'; 'bwconvhull'; 'labeloverlay'};
locations = cellfun(@which, names, 'UniformOutput', false);
available = ~cellfun(@isempty, locations);
report = table(names, available, 'VariableNames', {'Function', 'Available'});
disp(report);
if all(available)
    fprintf('函式查詢通過；這不代表資料格式、GPU 或完整訓練已驗證。\n');
else
    fprintf('尚缺函式：%s\n', strjoin(names(~available), ', '));
    fprintf('請依 README 的工具箱需求檢查安裝與授權。\n');
end
fprintf('Parallel Computing Toolbox 可選：%d\n', ~isempty(which('parpool')));
end
