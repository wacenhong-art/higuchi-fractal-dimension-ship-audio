
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


%% 2. 随机共振参数（你自己填 a,b,h）
a = 10;     % 你自己的参数
b = 1;      % 你自己的参数
h = 0.01;   % 你自己的参数

%% 3. 计算四类 CI-Higuchi 特征
kmax = 20;          % 常用尺度
alpha = 0.95;       % 95% 置信区间

features = zeros(800, 1);
labels = zeros(800, 1);
x_all = {x1, x2, x3, x4};
for c = 1:4
    for i = 1:200
        idx = (i-1) * 2048 + (1:2048);
        x = x_all{c}(idx);
        sr_signal = Rung_kutta_JDSR_k(a, b, h, x);
        sr_signal = (sr_signal - mean(sr_signal)) / std(sr_signal);
        features((c-1)*200 + i) = CI_Higuchi(sr_signal, kmax, alpha);
        labels((c-1)*200 + i) = c;
    end
end

%% 4. KNN 分类
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

%% 5. 输出结果
class_names = {'Cargo', 'Passengership', 'Tanker', 'Tug'};
for c = 1:4
    rate = mean(Y_pred(Y_test == c) == c) * 100;
    fprintf('%s 识别率 = %.2f%%\n', class_names{c}, rate);
end
fprintf('总体识别率 = %.2f%%\n', mean(Y_pred == Y_test) * 100);
confusion_matrix = confusionmat(Y_test, Y_pred, 'Order', 1:4);
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
