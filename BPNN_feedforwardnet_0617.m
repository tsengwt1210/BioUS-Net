clear all;
close all;
clc;

% === 使用者選擇資料組 ===
dataset_id = input('請輸入資料組別 (1 / 2 / 3)：');

if ~ismember(dataset_id, [1 2 3])
    error('輸入錯誤，請輸入 1、2 或 3');
end

% === 對應行數範圍 ===
start_row = (dataset_id - 1) * 8 + 1;
end_row = start_row + 6;

% === 讀取 Excel 資料 ===
data = readmatrix('test.xlsx');

if size(data,1) < end_row
    error('Excel 資料不足，請確認格式');
end

% === 解析資料格式 ===
% 資料格式：2個輸入變數 + 每組5筆面積資料，共5組（共7列）
P_base = data(start_row:start_row+1, 1:5);    % 2×5 輸入
T_matrix = data(start_row+2:start_row+6, 1:5);% 5×5 輸出（面積）

% === 展開為神經網路輸入格式 ===
num_groups = 5;
num_measurements = 5;

P_all = repelem(P_base, 1, num_measurements); % 2×25
T_all = reshape(T_matrix, 1, []);             % 1×25

% === 面積正規化 ===
T_max = max(T_all);
T_all = T_all / T_max;

% === 建立神經網路 ===
net = feedforwardnet([9,40], 'trainlm');  % 可以微調層數與神經元
net.inputs{1}.size = 2;
net = configure(net, P_all, T_all);

% 設定訓練參數
net.trainParam.epochs = 1000;
net.trainParam.goal = 0;

% 訓練網路
[net, tr] = train(net, P_all, T_all);

% 預測
out = net(P_all);

% === 圖表 ===
figure;
plot(T_all, '-rd', 'LineWidth', 1.5);
hold on;
plot(out, '-bo', 'LineWidth', 1.5);
xlabel('Data Index');
ylabel('Normalized Area');
legend('Target','NN output');
title(['Dataset ' num2str(dataset_id) ' Prediction vs. Target']);
%grid on;
