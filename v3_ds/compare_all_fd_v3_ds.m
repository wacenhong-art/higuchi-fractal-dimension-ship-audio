% 按原始分类脚本整合：保留参数、特征索引、单次划分与分类输出。
% v3_ds；不修改原算法文件。运行后保存一张柱状图和一张 2x2 t-SNE 综合图。
% Higuchi 直接调用同目录已恢复为传统公式的分类脚本，不另复制算法。
% t-SNE 使用原脚本提取的全部十列特征；实际 KNN 维度会另行标明。
clear; clc; close all;

%% 1. 本数据集的原始分类入口
project_path = fileparts(mfilename('fullpath'));
original_scripts = {'ds_katz.m', 'ds_higuchi.m', 'ds_fx.m', 'ds_rs.m'};
dataset_name = 'v3_ds: DeepShip four classes';

%% 2. 直接运行原脚本，在各自局部工作区收集分类输出
RunComparison(project_path, original_scripts, dataset_name);

%% ============================== 局部函数 ==============================
function RunComparison(project_path, original_scripts, dataset_name)
    root_path = fileparts(project_path);
    previous_path = path;
    restore_path = onCleanup(@() path(previous_path)); %#ok<NASGU>
    % 只补齐现有辅助函数的搜索路径，不修改原脚本。
    addpath(fullfile(root_path, '对比算法'));
    addpath(fullfile(root_path, 'v2'));
    result_dir = fullfile(project_path, 'results_fd_comparison');
    if ~isfolder(result_dir), mkdir(result_dir); end

    algorithm_names = {'Katz', 'Higuchi', 'FX', 'R/S'};
    num_algorithms = numel(original_scripts);
    run_results = cell(1, num_algorithms);
    feature_data = cell(1, num_algorithms);
    accuracy_all_runs = zeros(num_algorithms, 1);
    knn_dimensions = zeros(num_algorithms, 1);
    config = struct('dataset', dataset_name, 'num_repeats', 1, ...
        'classification_seed', 42, 'tsne_seed', 1042, ...
        'tsne_perplexity', 30, 'tsne_standardize', false, ...
        'classification_protocol', 'Current scripts; Higuchi core corrected to Higuchi (1988)', ...
        'tsne_samples', 'All samples; all ten extracted coarse-graining features', ...
        'original_scripts', {original_scripts});

    for a = 1:num_algorithms
        original_path = fullfile(project_path, original_scripts{a});
        if ~isfile(original_path), error('原脚本不存在：%s', original_path); end
        fprintf('\n[%s] 正在运行原脚本 %s\n', dataset_name, original_scripts{a});
        original_log = evalc('result = RunOneOriginal(original_path);');
        fprintf('%s', original_log);
        log_file = fullfile(result_dir, [erase(original_scripts{a}, '.m') '_original.log']);
        fid = fopen(log_file, 'w', 'n', 'UTF-8');
        if fid < 0, error('无法写入日志：%s', log_file); end
        fprintf(fid, '%s', original_log);
        fclose(fid);

        assert(isreal(result.features) && all(isfinite(result.features(:))), ...
            '原始特征包含 NaN、Inf 或复数。');
        assert(size(result.features, 2) == 10, '预期原脚本提取十列特征。');
        assert(numel(result.Y_test) == numel(result.Y_pred), '测试标签与预测长度不符。');
        if a > 1
            assert(isequal(result.labels, run_results{1}.labels), '原脚本之间的样本标签不一致。');
            assert(isequal(result.train_idx, run_results{1}.train_idx), '原脚本之间的划分不一致。');
        end
        run_results{a} = result;
        feature_data{a} = result.features;
        accuracy_all_runs(a) = 100 * mean(result.Y_pred == result.Y_test);
        knn_dimensions(a) = size(result.X_train, 2);
        matrix_accuracy = 100 * trace(result.confusion_matrix) / sum(result.confusion_matrix(:));
        assert(abs(accuracy_all_runs(a) - matrix_accuracy) < 1e-10, '混淆矩阵与准确率不一致。');
    end

    %% 3. 原脚本单次准确率：不伪造重复实验标准差
    labels = run_results{1}.labels;
    class_names = run_results{1}.class_names;
    result_table = table(string(algorithm_names(:)), accuracy_all_runs, ...
        nan(num_algorithms, 1), ones(num_algorithms, 1), knn_dimensions, ...
        'VariableNames', {'Algorithm', 'MeanAccuracyPercent', ...
        'StdAccuracyPercent', 'NumRepeats', 'KNNFeatureDimensions'});
    writetable(result_table, fullfile(result_dir, 'accuracy_results.csv'));
    save(fullfile(result_dir, 'feature_data.mat'), 'run_results', 'feature_data', ...
        'labels', 'class_names', 'algorithm_names', 'accuracy_all_runs', ...
        'knn_dimensions', 'config');

    %% 4. 准确率柱状图
    PlotAccuracy(result_table, dataset_name, result_dir);

    %% 5. t-SNE 展示全部样本的十维特征；二维坐标不参与分类
    [tsne_coordinates, tsne_success] = PlotTsne(feature_data, labels, ...
        class_names, algorithm_names, knn_dimensions, config, result_dir);
    save(fullfile(result_dir, 'feature_data.mat'), ...
        'tsne_coordinates', 'tsne_success', '-append');
    disp(result_table);
    fprintf('完成：%s\n', result_dir);
end

function result = RunOneOriginal(original_path)
    % 原脚本的 clear/clc/close all 仅清理本函数工作区。
    % run 返回后直接收集它产生的变量，不复制或改写算法公式、特征索引或分类参数。
    run(original_path);
    result = struct('features', features, 'labels', labels, ...
        'train_idx', train_idx, 'X_train', X_train, 'X_test', X_test, ...
        'Y_train', Y_train, 'Y_test', Y_test, 'Y_pred', Y_pred, ...
        'confusion_matrix', confusion_matrix, 'class_names', {class_names});
end

function PlotAccuracy(t, dataset_name, result_dir)
    f = figure('Visible', 'off', 'Color', 'w', ...
        'Units', 'centimeters', 'Position', [2 2 19 11]);
    cleanup = onCleanup(@() close(f)); %#ok<NASGU>
    ax = axes(f);
    colors = [0.70 0.38 0.32; 0.55 0.71 0.65; ...
              0.57 0.67 0.77; 0.73 0.67 0.48];
    b = bar(ax, 1:height(t), t.MeanAccuracyPercent, 0.65, ...
        'FaceColor', 'flat', 'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.7);
    b.CData = colors;
    hold(ax, 'on');
    ax.XTick = 1:height(t);
    ax.XTickLabel = t.Algorithm;
    ax.XTickLabelRotation = 15;
    ax.FontSize = 10;
    ax.Color = [0.985 0.985 0.985];
    ax.GridColor = [0.78 0.78 0.78];
    ax.GridAlpha = 0.35;
    ax.YGrid = 'on';
    ax.Box = 'on';
    ax.TickDir = 'out';
    ylim(ax, [0 max(105, max(t.MeanAccuracyPercent) + 5)]);
    xlim(ax, [0.4 height(t) + 0.6]);
    ylabel(ax, 'Recognition Rate (%)');
    title(ax, {dataset_name, 'Original scripts: single run, seed = 42'}, ...
        'FontSize', 11, 'FontWeight', 'normal', 'Interpreter', 'none');
    for a = 1:height(t)
        text(ax, a, t.MeanAccuracyPercent(a) + 1.4, ...
            sprintf('%.2f%%', t.MeanAccuracyPercent(a)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
    end
    % 单次原实验没有重复间标准差，因此不绘制误差棒。
    exportgraphics(f, fullfile(result_dir, 'accuracy_comparison.png'), 'Resolution', 300);
    exportgraphics(f, fullfile(result_dir, 'accuracy_comparison.pdf'), 'ContentType', 'vector');
end

function [coordinates, success] = PlotTsne(feature_data, labels, ...
        class_names, algorithm_names, knn_dimensions, config, result_dir)
    n = numel(feature_data);
    coordinates = cell(1, n);
    success = false(1, n);
    for a = 1:n
        try
            rng(config.tsne_seed, 'twister');
            coordinates{a} = tsne(feature_data{a}, 'NumDimensions', 2, ...
                'Perplexity', config.tsne_perplexity, 'Standardize', false);
            if ~isequal(size(coordinates{a}), [numel(labels) 2]) || ...
                    ~isreal(coordinates{a}) || any(~isfinite(coordinates{a}(:)))
                error('t-SNE 返回了无效坐标。');
            end
            success(a) = true;
        catch ME
            coordinates{a} = [];
            warning('OriginalComparison:TsneFailed', '%s t-SNE 失败：%s', ...
                algorithm_names{a}, ME.message);
        end
    end
    if ~any(success), error('所有 t-SNE 均失败，未生成虚假散点图。'); end

    % 所有子图使用同一坐标范围，不旋转、平移或调整任何类别以改善聚集。
    valid_coordinates = coordinates(success);
    xy = vertcat(valid_coordinates{:});
    lo = min(xy, [], 1); hi = max(xy, [], 1);
    padding = max(0.06 * (hi - lo), [1 1]);
    first_success = find(success, 1);
    palette = [0.00 0.45 0.70; 0.85 0.33 0.10; ...
               0.95 0.65 0.10; 0.55 0.25 0.65; ...
               0.20 0.62 0.42; 0.34 0.70 0.86; ...
               0.75 0.22 0.40; 0.40 0.40 0.40];
    markers = {'o', '+', 'x', 's', '^', 'd', 'v', 'p'};
    f = figure('Visible', 'off', 'Color', 'w', ...
        'Units', 'centimeters', 'Position', [2 2 28 24]);
    cleanup = onCleanup(@() close(f)); %#ok<NASGU>
    layout = tiledlayout(f, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    legend_handles = gobjects(numel(class_names), 1);
    legend_ax = [];
    for a = 1:n
        ax = nexttile(layout);
        hold(ax, 'on');
        if success(a)
            for c = 1:numel(class_names)
                h = scatter(ax, coordinates{a}(labels == c, 1), ...
                    coordinates{a}(labels == c, 2), 12, ...
                    'Marker', markers{c}, 'MarkerEdgeColor', palette(c, :), ...
                    'MarkerFaceColor', 'none', 'LineWidth', 0.65);
                if a == first_success, legend_handles(c) = h; legend_ax = ax; end
            end
            xlim(ax, [lo(1)-padding(1) hi(1)+padding(1)]);
            ylim(ax, [lo(2)-padding(2) hi(2)+padding(2)]);
        else
            text(ax, 0.5, 0.5, 't-SNE failed; see warning', ...
                'Units', 'normalized', 'HorizontalAlignment', 'center');
        end
        ax.Color = 'w'; ax.FontSize = 9; ax.Box = 'on'; ax.TickDir = 'out';
        ax.GridColor = [0.85 0.85 0.85]; ax.GridAlpha = 0.3;
        grid(ax, 'on');
        xlabel(ax, 't-SNE Dimension 1'); ylabel(ax, 't-SNE Dimension 2');
        title(ax, sprintf('%s (KNN: %d-D)', algorithm_names{a}, knn_dimensions(a)), ...
            'FontSize', 11, 'FontWeight', 'normal');
    end
    lgd = legend(legend_ax, legend_handles, class_names, ...
        'NumColumns', min(2, numel(class_names)), 'Interpreter', 'none', ...
        'FontSize', 8, 'Box', 'off');
    lgd.Layout.Tile = 'south';
    title(layout, {config.dataset, ...
        't-SNE: all samples, 10 extracted features; visualization only'}, ...
        'FontSize', 11, 'FontWeight', 'normal', 'Interpreter', 'none');
    exportgraphics(f, fullfile(result_dir, 'tsne_comparison.png'), 'Resolution', 300);
    exportgraphics(f, fullfile(result_dir, 'tsne_comparison.pdf'), 'ContentType', 'vector');
end
