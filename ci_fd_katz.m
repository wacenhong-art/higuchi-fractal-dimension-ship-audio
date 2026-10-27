% 实验名称：无随机共振的 Katz 分形维数四类舰船声音分类
% 是否包含随机共振：否
% 分形维数算法：Katz_FD.m 中的 Katz Fractal Dimension
% 相对 ci_hfd.m 的修改：删除随机共振参数及 Rung_kutta_JDSR_k 调用，
% 以 Katz_FD 替换 CI_Higuchi；将输入转换为该函数要求的行向量。
clear; clc; close all;

%% 1. 载入四类舰船声音
project_path = fileparts(mfilename('fullpath'));
comparison_path = fullfile(project_path, '对比算法');
addpath(comparison_path);       % 引用原有 Katz_FD.m，不修改对比算法文件
cargo_wav = fullfile(project_path, 'DeepShip-main', 'Cargo', '27.wav');
passengership_wav = fullfile(project_path, 'DeepShip-main', 'Passengership', '1.wav');
tanker_wav = fullfile(project_path, 'DeepShip-main', 'Tanker', '2.wav');
tug_wav = fullfile(project_path, 'DeepShip-main', 'Tug', '9.wav');

wav_paths = {cargo_wav, passengership_wav, tanker_wav, tug_wav};
x_all = cell(1, 4);
for i = 1:4
    if exist(wav_paths{i}, 'file') ~= 2
        error('找不到 WAV 文件：%s', wav_paths{i});
    end
    [x_all{i}, ~] = audioread(wav_paths{i});
    if size(x_all{i}, 2) > 1
        x_all{i} = mean(x_all{i}, 2);     % 多声道取平均
    end
    if length(x_all{i}) < 200 * 2048
        error('WAV 长度不足 200 个 2048 点样本：%s', wav_paths{i});
    end
end

x1 = x_all{1};       % Cargo
x2 = x_all{2};       % Passengership
x3 = x_all{3};       % Tanker
x4 = x_all{4};       % Tug


%% 2. 计算四类 Katz 分形维数特征（不使用随机共振）
features = zeros(800, 1);
labels = zeros(800, 1);
x_all = {x1, x2, x3, x4};
for c = 1:4
    for i = 1:200
        idx = (i-1) * 2048 + (1:2048);
        x = x_all{c}(idx);
        features((c-1)*200 + i) = Katz_FD(x(:).');
        labels((c-1)*200 + i) = c;
    end
end

%% 3. KNN 分类
rng(42);
train_idx = false(800, 1);
for c = 1:4
    idx = (c-1)*200 + randperm(200, 100);
    train_idx(idx) = true;
end

X_train = features(train_idx);
Y_train = labels(train_idx);
X_test = features(~train_idx);
Y_test = labels(~train_idx);

model = fitcknn(X_train, Y_train, ...
    'NumNeighbors', 1, ...
    'Distance', 'euclidean');

Y_pred = predict(model, X_test);

%% 4. 输出结果
class_names = {'Cargo', 'Passengership', 'Tanker', 'Tug'};
for c = 1:4
    rate = mean(Y_pred(Y_test == c) == c) * 100;
    fprintf('%s 识别率 = %.2f%%\n', class_names{c}, rate);
end
fprintf('总体识别率 = %.2f%%\n', mean(Y_pred == Y_test) * 100);
confusion_matrix = confusionmat(Y_test, Y_pred, 'Order', 1:4);
disp('混淆矩阵（行=真实标签，列=预测标签）：');
disp(confusion_matrix);
