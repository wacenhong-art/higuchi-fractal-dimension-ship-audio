% 实验名称：无随机共振的 10 尺度粗粒化 Katz 分形维数八类舰船声音分类
% 是否包含随机共振：否
% 分形维数算法：Katz_FD.m 中的原有 Katz Fractal Dimension
% 相对 ci_fd_katz.m 的修改：对每个 2048 点样本独立进行尺度 1:10 的
% 非重叠均值粗粒化；每个尺度沿用 Katz_FD，得到一条 10 维特征。
clear; clc; close all;

%% 1. 载入八类舰船声音
project_path = fileparts(mfilename('fullpath'));
comparison_path = fullfile(project_path, '对比算法');
% 源脚本优先使用 v2 内的对比算法目录；当前目录不存在时回退到项目原有目录。
if exist(fullfile(comparison_path, 'Katz_FD.m'), 'file') ~= 2
    comparison_path = fullfile(fileparts(project_path), '对比算法');
end
if exist(fullfile(comparison_path, 'Katz_FD.m'), 'file') ~= 2
    error('找不到原有 Katz_FD.m：%s', comparison_path);
end
addpath(comparison_path);       % 引用原有 Katz_FD.m，不修改对比算法文件
audio_dirs = { ...
    fullfile(project_path, '音频', '国家渡轮'), ...
    fullfile(project_path, '音频', '货轮'), ...
    fullfile(project_path, '音频', '螺旋桨呜呜呜'), ...
    fullfile(project_path, '音频', '舷外发动机发出呜呜声和加速'), ...
    fullfile(project_path, '音频', '游轮'), ...
    fullfile(project_path, '音频', '舷外发动机（60马力）在10节'), ...
    fullfile(project_path, '音频', '舷外发动机（60马力）在20节'), ...
    fullfile(project_path, '音频', '由小型柴油发动机驱动的过往船只的水下记录')};
num_classes = 8;
wav_paths = cell(1, num_classes);
for i = 1:5
    wav_paths{i} = FindSupportedAudioFile(audio_dirs{i});
end
wav_paths{6} = fullfile(audio_dirs{6}, '705da20f-b986-4e49-ae58-39f9f286e9ad.mp3');
wav_paths{7} = fullfile(audio_dirs{7}, '39f7562b-1d7a-4fe4-a969-6be98e7d4886.mp3');
wav_paths{8} = fullfile(audio_dirs{8}, 'cb678cb6-5627-4544-9fbb-c8e3884f0827.mp3');
x_all = cell(1, num_classes);
for i = 1:num_classes
    if exist(wav_paths{i}, 'file') ~= 2
        error('找不到音频文件：%s', wav_paths{i});
    end
    [x_all{i}, ~] = audioread(wav_paths{i});
    if size(x_all{i}, 2) > 1
        x_all{i} = mean(x_all{i}, 2);     % 多声道取平均
    end
    if length(x_all{i}) < 200 * 2048
        error('音频长度不足 200 个 2048 点样本：%s', wav_paths{i});
    end
end

x1 = x_all{1};       % 国家渡轮
x2 = x_all{2};       % 货轮
x3 = x_all{3};       % 螺旋桨呜呜呜
x4 = x_all{4};       % 舷外发动机发出呜呜声和加速
x5 = x_all{5};       % 游轮
x6 = x_all{6};       % 舷外发动机60马力10节
x7 = x_all{7};       % 舷外发动机60马力20节
x8 = x_all{8};       % 小型柴油发动机船只


%% 2. 计算八类 10 尺度粗粒化 Katz 分形维数特征（不使用随机共振）
numScales = 10;
scales = 1:numScales;
features = zeros(num_classes * 200, numScales);
labels = zeros(num_classes * 200, 1);
x_all = {x1, x2, x3, x4, x5, x6, x7, x8};
for c = 1:num_classes
    for i = 1:200
        idx = (i-1) * 2048 + (1:2048);
        x = x_all{c}(idx);
        if floor(length(x) / max(scales)) < 2
            error('样本过短，尺度 %d 的粗粒化序列不足 2 点（类别 %d，样本 %d）。', ...
                max(scales), c, i);
        end

        feature = zeros(1, numScales);
        for s = 1:numScales
            tau = scales(s);
            y_tau = CoarseGrainNonOverlapping(x, tau);
            feature(s) = Katz_FD(y_tau);
        end
        if ~isreal(feature) || any(~isfinite(feature))
            error('Katz 特征出现 NaN、Inf 或复数（类别 %d，样本 %d）。', c, i);
        end
        features((c-1)*200 + i, :) = feature;
        labels((c-1)*200 + i) = c;
    end
end
has_invalid_features = any(~isfinite(features(:))) || ~isreal(features);
if has_invalid_features
    error('特征矩阵存在 NaN、Inf 或复数，停止 KNN 分类。');
end

%% 3. KNN 分类
rng(42);
train_idx = false(num_classes * 200, 1);
for c = 1:num_classes
    idx = (c-1)*200 + randperm(200, 100);
    train_idx(idx) = true;
end

X_train = features(train_idx, :);
Y_train = labels(train_idx);
X_test = features(~train_idx, :);
Y_test = labels(~train_idx);

model = fitcknn(X_train, Y_train, ...
    'NumNeighbors', 1, ...
    'Distance', 'euclidean');

Y_pred = predict(model, X_test);

%% 4. 输出结果
fprintf('多尺度方法：10尺度Katz分形维数\n');
fprintf('尺度范围：%d至%d\n', min(scales), max(scales));
fprintf('每类样本数量：%d\n', 200);
fprintf('总样本数量：%d\n', num_classes * 200);
fprintf('特征矩阵维度：%d × %d\n', size(features, 1), size(features, 2));
if has_invalid_features
    fprintf('是否存在NaN或Inf：是\n');
else
    fprintf('是否存在NaN或Inf：否\n');
end

class_names = {'国家渡轮', '货轮', '螺旋桨呜呜呜', '舷外发动机发出呜呜声和加速', ...
    '游轮', '舷外发动机60马力10节', '舷外发动机60马力20节', '小型柴油发动机船只'};
for c = 1:num_classes
    rate = mean(Y_pred(Y_test == c) == c) * 100;
    fprintf('%s 识别率 = %.2f%%\n', class_names{c}, rate);
end
fprintf('总体识别率 = %.2f%%\n', mean(Y_pred == Y_test) * 100);
confusion_matrix = confusionmat(Y_test, Y_pred, 'Order', 1:num_classes);
disp('混淆矩阵（行=真实标签，列=预测标签）：');
disp(confusion_matrix);

%% ==============================
%%  对单条时间序列进行非重叠均值粗粒化
%% ==============================
function y_tau = CoarseGrainNonOverlapping(x, tau)
    x = x(:).';
    N = length(x);
    coarse_length = floor(N / tau);
    if coarse_length < 2
        error('粗粒化后序列长度不足 2，无法计算有效 Katz 分形维数。');
    end
    x = x(1:coarse_length * tau);     % 舍弃末尾不足 tau 的点
    y_tau = mean(reshape(x, tau, coarse_length), 1);
end

%% ==============================
%%  在指定类别文件夹中查找唯一的受支持音频文件
%% ==============================
function audio_file = FindSupportedAudioFile(audio_dir)
    supported_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.mp4', ...
        '.ogg', '.aif', '.aiff', '.au'};
    audio_files = [];
    for i = 1:length(supported_extensions)
        candidates = dir(fullfile(audio_dir, '**', ['*' supported_extensions{i}]));
        candidates = candidates(~[candidates.isdir]);
        if isempty(audio_files)
            audio_files = candidates;
        else
            audio_files = [audio_files; candidates]; %#ok<AGROW>
        end
    end
    if isempty(audio_files)
        error('未在文件夹中找到受支持的音频文件：%s', audio_dir);
    end
    if length(audio_files) ~= 1
        error('文件夹中必须恰有一个受支持的音频文件：%s', audio_dir);
    end
    audio_file = fullfile(audio_files(1).folder, audio_files(1).name);
end
