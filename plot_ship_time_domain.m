%% 舰船声音原始时域波形可视化
% 本脚本独立运行；仅读取原始音频并绘制时域波形，不进行分段、特征计算、
% 随机共振、分类、归一化、滤波或其他会改变原始振幅的处理。

clear; clc;

%% 1. 路径与绘图公共设置
repo_path = fileparts(mfilename('fullpath'));
output_dir = fullfile(repo_path, '波形图');
if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

% 为中文类别标题指定 Windows 常见中文字体；不改变音频数据或采样率。
chinese_font = 'Microsoft YaHei';
max_plot_points = 200000;
% 当完整音频超过此点数时，仅对绘图副本等间隔抽点以避免渲染卡顿；
% 原始 x 保持不变，且抽点后的横坐标仍使用真实时间。

%% 2. Figure 1：DeepShip 四类指定音频
ship_files = {
    'Cargo',         '27.wav';
    'Passengership', '1.wav';
    'Tanker',        '2.wav';
    'Tug',           '9.wav'
};

% 以实际存在且同时包含四个指定 WAV 的目录为准，避免假定数据位置。
deepShip_candidates = {fullfile(repo_path, 'DeepShip-main'), ...
                       fullfile(repo_path, 'v3_ds', 'DeepShip-main')};
deepShip_path = '';
for candidate_idx = 1:numel(deepShip_candidates)
    candidate_path = deepShip_candidates{candidate_idx};
    required_wavs_exist = true;
    for class_idx = 1:size(ship_files, 1)
        required_wav = fullfile(candidate_path, ship_files{class_idx, 1}, ...
            ship_files{class_idx, 2});
        if exist(required_wav, 'file') ~= 2
            required_wavs_exist = false;
            break;
        end
    end
    if required_wavs_exist
        deepShip_path = candidate_path;
        break;
    end
end

fig1 = figure('Color', 'w', 'Name', 'DeepShip Time Domain Waveforms');
tl1 = tiledlayout(fig1, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl1, 'DeepShip Dataset - Time Domain Waveforms', 'FontWeight', 'bold');

if isempty(deepShip_path)
    warning('plot_ship_time_domain:DeepShipPathMissing', ...
        '未找到同时包含四个指定 WAV 文件的 DeepShip-main 目录。');
end

for class_idx = 1:size(ship_files, 1)
    ax = nexttile(tl1, class_idx);
    class_name = ship_files{class_idx, 1};
    file_name = ship_files{class_idx, 2};
    wav_path = fullfile(deepShip_path, class_name, file_name);

    if isempty(deepShip_path) || exist(wav_path, 'file') ~= 2
        warning('plot_ship_time_domain:DeepShipFileMissing', ...
            '找不到 DeepShip 音频：%s', wav_path);
        text(ax, 0.5, 0.5, {'缺失音频', [class_name, ' / ', file_name]}, ...
            'HorizontalAlignment', 'center', 'FontName', chinese_font);
        axis(ax, [0 1 0 1]);
    else
        try
            [x, fs] = audioread(wav_path);
            if size(x, 2) > 1
                x = mean(x, 2);  % 双声道转单声道仅为统一显示，不改变采样率。
            end
            x = x(:);
            t = (0:length(x)-1)' / fs;
            if numel(x) > max_plot_points
                plot_idx = round(linspace(1, numel(x), max_plot_points));
                plot(ax, t(plot_idx), x(plot_idx), 'LineWidth', 0.5);
            else
                plot(ax, t, x, 'LineWidth', 0.5);
            end
        catch ME
            warning('plot_ship_time_domain:DeepShipReadFailed', ...
                '无法读取 DeepShip 音频 %s：%s', wav_path, ME.message);
            text(ax, 0.5, 0.5, {'无法读取音频', file_name}, ...
                'HorizontalAlignment', 'center', 'FontName', chinese_font);
            axis(ax, [0 1 0 1]);
        end
    end
    grid(ax, 'on');
    xlabel(ax, 'Time (s)');
    ylabel(ax, 'Amplitude');
    title(ax, [class_name, ' - ', file_name], 'Interpreter', 'none');
end
exportgraphics(fig1, fullfile(output_dir, 'DeepShip_4classes_time_domain.png'), ...
    'Resolution', 300);

%% 3. Figure 2：自定义八类舰船声音
% 八类音频实际位于项目根目录下的 v2\音频，因此 project_path 指向 v2。
project_path = fullfile(repo_path, 'v2');
audio_dirs = {
    fullfile(project_path, '音频', '国家渡轮'), ...
    fullfile(project_path, '音频', '货轮'), ...
    fullfile(project_path, '音频', '螺旋桨呜呜呜'), ...
    fullfile(project_path, '音频', '舷外发动机发出呜呜声和加速'), ...
    fullfile(project_path, '音频', '游轮'), ...
    fullfile(project_path, '音频', '舷外发动机（60马力）在10节'), ...
    fullfile(project_path, '音频', '舷外发动机（60马力）在20节'), ...
    fullfile(project_path, '音频', '由小型柴油发动机驱动的过往船只的水下记录')
};
class_names = {'国家渡轮', '货轮', '螺旋桨呜呜呜', '舷外发动机发出呜呜声和加速', ...
               '游轮', '舷外发动机（60马力）在10节', '舷外发动机（60马力）在20节', ...
               '由小型柴油发动机驱动的过往船只的水下记录'};
audio_patterns = {'*.wav', '*.flac', '*.mp3', '*.m4a', '*.aac', '*.ogg', '*.wma', '*.mp4'};

fig2 = figure('Color', 'w', 'Name', 'Eight Ship Sound Classes Time Domain Waveforms');
tl2 = tiledlayout(fig2, 4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl2, 'Eight Ship Sound Classes - Time Domain Waveforms', 'FontWeight', 'bold');

for class_idx = 1:numel(class_names)
    ax = nexttile(tl2, class_idx);
    audio_dir = audio_dirs{class_idx};
    selected_path = '';
    x = [];
    fs = [];

    if exist(audio_dir, 'dir') ~= 7
        warning('plot_ship_time_domain:CustomDirectoryMissing', ...
            '类别“%s”的音频目录不存在：%s', class_names{class_idx}, audio_dir);
    else
        audio_files = dir(fullfile(audio_dir, audio_patterns{1}));
        for pattern_idx = 2:numel(audio_patterns)
            audio_files = [audio_files; dir(fullfile(audio_dir, audio_patterns{pattern_idx}))]; %#ok<AGROW>
        end
        [~, order] = sort({audio_files.name});
        audio_files = audio_files(order);

        % 按文件名顺序依次尝试，选取第一个可由 audioread 成功读取的文件。
        for file_idx = 1:numel(audio_files)
            candidate_path = fullfile(audio_dir, audio_files(file_idx).name);
            try
                [x, fs] = audioread(candidate_path);
                selected_path = candidate_path;
                break;
            catch ME
                warning('plot_ship_time_domain:CustomAudioReadFailed', ...
                    '无法读取候选音频 %s：%s', candidate_path, ME.message);
            end
        end

        if isempty(selected_path)
            warning('plot_ship_time_domain:CustomAudioMissing', ...
                '类别“%s”的目录没有可由 audioread 读取的音频：%s', ...
                class_names{class_idx}, audio_dir);
        else
            fprintf('类别“%s”实际选取的音频：%s\n', class_names{class_idx}, selected_path);
        end
    end

    if isempty(selected_path)
        text(ax, 0.5, 0.5, '缺失或无法读取音频', 'HorizontalAlignment', 'center', ...
            'FontName', chinese_font);
        axis(ax, [0 1 0 1]);
    else
        if size(x, 2) > 1
            x = mean(x, 2);  % 双声道转单声道仅为统一显示，不改变采样率。
        end
        x = x(:);
        t = (0:length(x)-1)' / fs;
        if numel(x) > max_plot_points
            plot_idx = round(linspace(1, numel(x), max_plot_points));
            plot(ax, t(plot_idx), x(plot_idx), 'LineWidth', 0.5);
        else
            plot(ax, t, x, 'LineWidth', 0.5);
        end
    end
    grid(ax, 'on');
    xlabel(ax, '时间 (s)', 'FontName', chinese_font);
    ylabel(ax, '幅值', 'FontName', chinese_font);
    title(ax, class_names{class_idx}, 'FontName', chinese_font, 'Interpreter', 'none');
end
exportgraphics(fig2, fullfile(output_dir, 'ShipSound_8classes_time_domain.png'), ...
    'Resolution', 300);
