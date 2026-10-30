% 实验名称：无随机共振的传统 Higuchi 分形维数四类舰船声音分类
% 是否包含随机共振：否
% 分形维数算法：传统 Higuchi（Higuchi, 1988）
% 每个 2048 点片段在各粗粒化尺度上调用传统 Higuchi；不含随机共振。
clear; clc; close all;

%% 1. 载入四类舰船声音
project_path = fileparts(mfilename('fullpath'));
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


%% 2. 计算四类传统 Higuchi 特征（不使用随机共振）
kmax = 20;          % 常用尺度

tau = 1:10;
features = zeros(800, numel(tau));
labels = zeros(800, 1);
x_all = {x1, x2, x3, x4};
for c = 1:4
    for i = 1:200
        idx = (i-1) * 2048 + (1:2048);
        x = x_all{c}(idx);
        for scale_idx = 1:numel(tau)
            x_coarse = CoarseGrain(x, tau(scale_idx));
            fd = Traditional_Higuchi(x_coarse, kmax);
            if ~isreal(fd) || ~isfinite(fd)
                error('传统 Higuchi 特征不是有限实数：类别 %d，样本 %d，尺度 %d。', ...
                    c, i, tau(scale_idx));
            end
            features((c-1)*200 + i, scale_idx) = fd;
        end
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

% 保留既有线性索引：KNN 只使用第一尺度；十列均保留供 t-SNE 展示。
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


%% 5. 保存本次真实分类结果与混淆矩阵图
result_dir = fullfile(project_path, 'results_traditional_higuchi');
if ~isfolder(result_dir), mkdir(result_dir); end
f = figure('Visible', 'off', 'Color', 'w');
chart = confusionchart(confusion_matrix, class_names);
chart.Title = sprintf('Traditional Higuchi: %.2f%% (KNN %d-D)', ...
    mean(Y_pred == Y_test) * 100, size(X_train, 2));
exportgraphics(f, fullfile(result_dir, 'confusion_matrix.png'), 'Resolution', 300);
exportgraphics(f, fullfile(result_dir, 'confusion_matrix.pdf'), 'ContentType', 'vector');
close(f);
save(fullfile(result_dir, 'classification_result.mat'), 'features', 'labels', ...
    'train_idx', 'X_train', 'X_test', 'Y_train', 'Y_test', 'Y_pred', ...
    'confusion_matrix', 'class_names', 'kmax');

%% 传统 Higuchi 核心：不使用置信区间或离群值筛选
function D = Traditional_Higuchi(x, kmax)
    % Higuchi (1988), Eq. (1): all offsets, arithmetic mean, ordinary LS.
    % https://www.ism.ac.jp/~higuchi/index_e/papers/PhysicaD-1988.pdf
    % Nm counts increments, not points; the two k factors are required.
    validateattributes(x, {'numeric'}, {'vector', 'real', 'finite'});
    x = double(x(:));
    N = numel(x);
    validateattributes(kmax, {'numeric'}, ...
        {'scalar', 'integer', '>=', 2, '<=', floor(N / 2)});
    L = zeros(1, kmax);
    for k = 1:kmax
        Lm = zeros(1, k);
        for m = 1:k
            Nm = floor((N - m) / k);
            seq = x(m + (0:Nm) * k);
            Lm(m) = sum(abs(diff(seq))) * (N - 1) / (Nm * k^2);
        end
        L(k) = mean(Lm); % Retain every offset; no outlier rejection.
    end
    if any(L <= 0)
        error('TraditionalHiguchi:ZeroLength', ...
            'Higuchi 对数拟合要求所有 L(k)>0；不添加 eps 或删除拟合点。');
    end
    p = polyfit(log(1 ./ (1:kmax)), log(L), 1);
    D = p(1);
end

function y = CoarseGrain(x, tau)
    usable_length = floor(length(x) / tau) * tau;
    y = mean(reshape(x(1:usable_length), tau, []), 1).';
end
