% 实验名称：10 尺度 Dubuc 分形维数四类舰船声音分类
% 分形维数算法：同目录 Dubuc_FD.m 中的原有 Dubuc variation 方法
clear; clc; close all;

%% 1. 载入四类 DeepShip 舰船声音
project_path = fileparts(mfilename('fullpath'));
data_candidates = {fullfile(project_path, 'DeepShip-main'), ...
                   fullfile(fileparts(project_path), 'DeepShip-main')};
relative_wavs = {'Cargo', '27.wav'; ...
                 'Passengership', '1.wav'; ...
                 'Tanker', '2.wav'; ...
                 'Tug', '9.wav'};

data_path = '';
for candidate_idx = 1:numel(data_candidates)
    candidate = data_candidates{candidate_idx};
    has_all_wavs = all(cellfun(@(class_name, file_name) ...
        exist(fullfile(candidate, class_name, file_name), 'file') == 2, ...
        relative_wavs(:, 1), relative_wavs(:, 2)));
    if has_all_wavs
        data_path = candidate;
        break;
    end
end
if isempty(data_path)
    error('找不到包含四个指定 WAV 文件的 DeepShip-main 目录。');
end

cargo_wav = fullfile(data_path, 'Cargo', '27.wav');
passengership_wav = fullfile(data_path, 'Passengership', '1.wav');
tanker_wav = fullfile(data_path, 'Tanker', '2.wav');
tug_wav = fullfile(data_path, 'Tug', '9.wav');

wav_paths = {cargo_wav, passengership_wav, tanker_wav, tug_wav};
x_all = cell(1, 4);
for c = 1:4
    [x_all{c}, ~] = audioread(wav_paths{c});
    if size(x_all{c}, 2) > 1
        x_all{c} = mean(x_all{c}, 2);
    end
    if length(x_all{c}) < 200 * 2048
        error('WAV 长度不足 200 个 2048 点样本：%s', wav_paths{c});
    end
end

%% 2. 10 尺度粗粒化与 Dubuc 分形维数特征提取
tau = 1:10;
features = zeros(800, 10);
labels = zeros(800, 1);

for c = 1:4
    for i = 1:200
        idx = (i-1)*2048 + (1:2048);
        x = x_all{c}(idx);

        for scale_idx = 1:numel(tau)
            x_coarse = CoarseGrain(x, tau(scale_idx));
            fd = Dubuc_FD(x_coarse);

            if ~isscalar(fd) || ~isreal(fd) || ~isfinite(fd)
                error('Dubuc 特征无效：类别 %d，样本 %d，尺度 %d', ...
                    c, i, tau(scale_idx));
            end

            features((c-1)*200+i, scale_idx) = fd;
        end

        labels((c-1)*200+i) = c;
    end
end

%% 3. KNN 分类
rng(42);
train_idx = false(800, 1);
for c = 1:4
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
class_names = {'Cargo', 'Passengership', 'Tanker', 'Tug'};
fprintf('特征矩阵尺寸：%d x %d\n', size(features, 1), size(features, 2));
fprintf('训练集尺寸：%d x %d\n', size(X_train, 1), size(X_train, 2));
fprintf('测试集尺寸：%d x %d\n', size(X_test, 1), size(X_test, 2));

for c = 1:4
    rate = mean(Y_pred(Y_test == c) == c) * 100;
    fprintf('%s 识别率 = %.2f%%\n', class_names{c}, rate);
end
fprintf('总体识别率 = %.2f%%\n', mean(Y_pred == Y_test) * 100);

confusion_matrix = confusionmat(Y_test, Y_pred, 'Order', 1:4);
disp('混淆矩阵（行=真实标签，列=预测标签）：');
disp(confusion_matrix);


function y = CoarseGrain(x, tau)
    usable_length = floor(length(x) / tau) * tau;
    y = mean(reshape(x(1:usable_length), tau, []), 1).';
end
