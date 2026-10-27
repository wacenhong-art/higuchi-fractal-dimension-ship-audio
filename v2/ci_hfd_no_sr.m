% 实验名称：CI-Higuchi 分形维数五类舰船声音分类
% 分形维数算法：置信区间改进 Higuchi（CI-Higuchi）
clear; clc; close all;

%% 1. 载入五类舰船声音
project_path = fileparts(mfilename('fullpath'));
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


%% 2. 计算五类 CI-Higuchi 特征
kmax = 20;          % 常用尺度
alpha = 0.95;       % 95% 置信区间
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
            if length(x_coarse) < 2 * kmax
                error('Higuchi 输入长度不足：类别 %d，样本 %d，尺度 %d。', ...
                    c, i, tau);
            end
            features(feature_idx, scale_idx) = CI_Higuchi(x_coarse, kmax, alpha);
            if ~isreal(features(feature_idx, scale_idx)) || ...
                    ~isfinite(features(feature_idx, scale_idx))
                error('CI-Higuchi 特征不是有限实数：类别 %d，样本 %d，尺度 %d。', ...
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
%%  核心1：置信区间改进 Higuchi（你的创新）
%% ==============================
function [D] = CI_Higuchi(x, kmax, alpha)
    N = length(x);
    L = zeros(1, kmax);
    
    for k = 1:kmax
        Lk = zeros(1, k);
        for m = 1:k
            seq = x(m:k:N);
            n = length(seq);
            if n < 2
                Lk(m) = 0;
                continue;
            end
            sum_len = sum(abs(diff(seq)));
            coeff = (N - 1) / (n * k);
            Lk(m) = sum_len * coeff;
        end
        
        % ========== 置信区间剔除异常分形长度 ==========
        Lk_valid = RemoveOutliers_CI(Lk, alpha);
        if ~isempty(Lk_valid)
            L(k) = mean(Lk_valid);
        else
            L(k) = mean(Lk);
        end
    end
    
    logk = log(1:kmax);
    logL = log(L + eps);
    p = polyfit(logk, logL, 1);
    D = -p(1);
end

%% ==============================
%%  核心2：95% 置信区间剔除离群值
%% ==============================
function [y_valid] = RemoveOutliers_CI(y, alpha)
    y = y(y > 0);
    if length(y) < 3
        y_valid = y;
        return;
    end
    mu = mean(y);
    sig = std(y);
    n = length(y);
    t = tinv((1+alpha)/2, n-1);
    lower = mu - t*sig;
    upper = mu + t*sig;
    y_valid = y(y >= lower & y <= upper);
end

%% ==============================
%%  传统 Higuchi（用于对比）
%% ==============================
function [D] = Traditional_Higuchi(x, kmax)
    N = length(x);
    L = zeros(1, kmax);
    for k = 1:kmax
        Lk = zeros(1, k);
        for m = 1:k
            seq = x(m:k:N);
            n = length(seq);
            sum_len = sum(abs(diff(seq)));
            coeff = (N - 1) / (n * k);
            Lk(m) = sum_len * coeff;
        end
        L(k) = mean(Lk);
    end
    logk = log(1:kmax);
    logL = log(L + eps);
    p = polyfit(logk, logL, 1);
    D = -p(1);
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
