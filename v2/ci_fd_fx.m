% 实验名称：计盒维数五类舰船声音分类
% 分形维数算法：FX.m 中的计盒维数
clear; clc; close all;

%% 1. 载入五类舰船声音
project_path = fileparts(mfilename('fullpath'));
comparison_path = fullfile(fileparts(project_path), '对比算法');
if exist(fullfile(comparison_path, 'FX.m'), 'file') ~= 2
    error('找不到 FX.m：%s', comparison_path);
end
addpath(comparison_path);       % 引用原有 FX.m，不修改对比算法文件
audio_dirs = { ...
    fullfile(project_path, '音频', '国家渡轮'), ...
    fullfile(project_path, '音频', '货轮'), ...
    fullfile(project_path, '音频', '螺旋桨呜呜呜'), ...
    fullfile(project_path, '音频', '舷外发动机发出呜呜声和加速'), ...
    fullfile(project_path, '音频', '游轮')};
num_classes = 5;
wav_paths = cell(1, num_classes);
for i = 1:num_classes
    wav_paths{i} = FindSupportedAudioFile(audio_dirs{i});
end
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


%% 2. 计算五类计盒维数特征
scales = 1:10;
num_scales = length(scales);
features = zeros(num_classes * 200, num_scales);
labels = zeros(num_classes * 200, 1);
x_all = {x1, x2, x3, x4, x5};
for c = 1:num_classes
    for i = 1:200
        idx = (i-1) * 2048 + (1:2048);
        x = x_all{c}(idx);
        feature_idx = (c-1)*200 + i;
        for scale_idx = 1:num_scales
            tau = scales(scale_idx);
            if tau == 1
                x_coarse = x;
            else
                coarse_length = floor(length(x) / tau);
                x_coarse = mean(reshape(x(1:coarse_length*tau), tau, []), 1).';
            end

            cellmax = 2^floor(log2(length(x_coarse)));
            x_fx = x_coarse(1:cellmax);
            features(feature_idx, scale_idx) = FX(x_fx, cellmax);
            if ~isreal(features(feature_idx, scale_idx)) || ...
                    ~isfinite(features(feature_idx, scale_idx))
                error('FX 特征不是有限实数：类别 %d，样本 %d，尺度 %d。', ...
                    c, i, tau);
            end
        end
        labels(feature_idx) = c;
    end
end

%% 3. KNN 分类
rng(42);
train_idx = false(num_classes * 200, 1);
for c = 1:num_classes
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
class_names = {'国家渡轮', '货轮', '螺旋桨呜呜呜', '舷外发动机发出呜呜声和加速', '游轮'};
for c = 1:num_classes
    rate = mean(Y_pred(Y_test == c) == c) * 100;
    fprintf('%s 识别率 = %.2f%%\n', class_names{c}, rate);
end
fprintf('总体识别率 = %.2f%%\n', mean(Y_pred == Y_test) * 100);
confusion_matrix = confusionmat(Y_test, Y_pred, 'Order', 1:num_classes);
disp('混淆矩阵（行=真实标签，列=预测标签）：');
disp(confusion_matrix);

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
