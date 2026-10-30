function run_flde_comparison(dataset_folder, mode)
% FLDE-style figures; original algorithms and classification columns retained.
% mode: full (fresh extraction), results (reclassify cache), plot (render only).
if nargin < 2, mode = 'full'; end
root = fileparts(mfilename('fullpath'));
folder = fullfile(root, dataset_folder);
out = fullfile(folder, 'results_flde_style');
if ~isfolder(out), mkdir(out); end
oldpath = path; cleanupPath = onCleanup(@() path(oldpath));
addpath(fullfile(root,'对比算法'), fullfile(root,'v2'));
cache = fullfile(out,'experiment.mat');
if strcmp(mode,'plot')
    s = load(cache); renderFigures(s,out); return;
end
if strcmp(dataset_folder,'v2')
    scripts = {'ci_fd_katz_multiscale10.m','ci_hfd_no_sr.m','ci_fd_fx.m','ci_fd_rs_multiscale10.m'};
    dataset = 'National Park';
    class_names = {'Ferry','Cargo Ship','Propeller Whine','Outboard Motor', ...
        'Cruise Ship','Outboard 60 hp, 10 kn','Outboard 60 hp, 20 kn','Small Diesel Engine'};
else
    assert(strcmp(dataset_folder,'v3_ds'),'Unknown dataset folder.');
    scripts = {'ds_katz.m','ds_higuchi.m','ds_fx.m','ds_rs.m'};
    dataset = 'DeepShip';
    class_names = {'Cargo','Passenger Ship','Tanker','Tug'};
end
if strcmp(mode,'full')
    diary(fullfile(out,'run.log')); diaryCleanup = onCleanup(@() diary('off'));
    % Original Higuchi scripts export legacy figures: preserve them byte-for-byte.
    legacy = fullfile(folder,'results_traditional_higuchi');
    backup = tempname; mkdir(backup);
    if isfolder(legacy), copyfile(legacy,fullfile(backup,'legacy')); end
    legacyCleanup = onCleanup(@() restoreLegacy(legacy,backup));
    feature_data = cell(1,4); source_results = cell(1,4);
    source_text = cell(1,4);
    for a=1:4
        fprintf('\nFresh extraction: %s / %s\n',dataset,scripts{a});
        source_text{a}=fileread(fullfile(folder,scripts{a}));
        r = runOriginal(fullfile(folder,scripts{a}));
        assert(all(isfinite(r.features(:))) && isreal(r.features));
        if a>1
            assert(isequal(r.labels,source_results{1}.labels));
            assert(isequal(r.train_idx,source_results{1}.train_idx));
            assert(isequal(r.wav_paths,source_results{1}.wav_paths),'Audio source mismatch.');
        end
        source_results{a}=r; feature_data{a}=r.features;
    end
    labels=source_results{1}.labels;
    knn_dimensions=cellfun(@(r) size(r.X_train,2),source_results);
    save(fullfile(out,'features.mat'),'feature_data','labels','knn_dimensions', ...
        'source_results','source_text','scripts','dataset','class_names');
else
    assert(strcmp(mode,'results'),'Expected full, results or plot.');
    load(fullfile(out,'features.mat'),'feature_data','labels','knn_dimensions','source_results');
end
algorithm_names={'Katz','Higuchi','FX','R/S'};
config=struct('dataset',dataset,'num_repeats',20,'seeds',42:61, ...
    'classifier','1-NN Euclidean; original columns; no standardization', ...
    'tsne_seed',1042,'tsne_perplexity',30,'tsne_standardize',false, ...
    'silhouette_space','Raw 10-dimensional extracted features, Euclidean', ...
    'split','Segment-level: 100 train / 100 test per class; shared across methods');
accuracy_all_runs=zeros(4,20); train_masks=false(numel(labels),20);
predictions=cell(4,20); confusion_matrices=cell(4,20);
for k=1:20
    rng(config.seeds(k),'twister'); mask=false(size(labels));
    for c=1:numel(class_names)
        ids=find(labels==c); assert(numel(ids)==200);
        mask(ids(randperm(200,100)))=true;
    end
    train_masks(:,k)=mask;
    for a=1:4
        x=feature_data{a}(:,1:knn_dimensions(a));
        mdl=fitcknn(x(mask,:),labels(mask),'NumNeighbors',1,'Distance','euclidean','Standardize',false);
        pred=predict(mdl,x(~mask,:)); predictions{a,k}=pred;
        cm=confusionmat(labels(~mask),pred); confusion_matrices{a,k}=cm;
        accuracy_all_runs(a,k)=100*mean(pred==labels(~mask));
        assert(abs(accuracy_all_runs(a,k)-100*trace(cm)/sum(cm(:)))<1e-10);
        if k==1
            assert(isequal(mask,source_results{a}.train_idx));
            assert(isequal(pred,source_results{a}.Y_pred),'Seed 42 must reproduce original predictions.');
        end
    end
end
mean_accuracy=mean(accuracy_all_runs,2); std_accuracy=std(accuracy_all_runs,0,2);
sc=zeros(4,1); tsne_coordinates=cell(1,4);
for a=1:4
    % SC in raw feature space, not the potentially distorted t-SNE embedding.
    sc(a)=mean(silhouette(feature_data{a},labels,'Euclidean'));
    rng(config.tsne_seed,'twister');
    tsne_coordinates{a}=tsne(feature_data{a},'NumDimensions',2, ...
        'Perplexity',config.tsne_perplexity,'Standardize',false);
    assert(isequal(size(tsne_coordinates{a}),[numel(labels),2]));
    assert(all(isfinite(tsne_coordinates{a}(:))));
end
result_table=table(string(algorithm_names(:)),mean_accuracy,std_accuracy,sc, ...
    knn_dimensions(:),repmat(20,4,1),'VariableNames', ...
    {'Algorithm','MeanAccuracyPercent','StdAccuracyPercent','SC_Raw10D','KNNDimensions','Repeats'});
writetable(result_table,fullfile(out,'accuracy_results.csv'));
writetable(array2table(accuracy_all_runs,'RowNames',algorithm_names), ...
    fullfile(out,'all_runs.csv'),'WriteRowNames',true);
save(cache,'feature_data','labels','class_names','algorithm_names','knn_dimensions', ...
    'config','accuracy_all_runs','train_masks','predictions','confusion_matrices', ...
    'mean_accuracy','std_accuracy','sc','tsne_coordinates','result_table');
s=load(cache); renderFigures(s,out); disp(result_table);
end

function r=runOriginal(original)
run(original);
r=struct('features',features,'labels',labels,'train_idx',train_idx, ...
    'X_train',X_train,'Y_test',Y_test,'Y_pred',Y_pred,'wav_paths',{wav_paths});
close all;
end

function restoreLegacy(legacy,backup)
% Copy back only pre-existing files; no recursive deletion of user material.
if isfolder(fullfile(backup,'legacy'))
    copyfile(fullfile(backup,'legacy','*'),legacy);
end
end

function renderFigures(s,out)
palette=[.70 .38 .32;.55 .71 .65;.57 .67 .77;.73 .67 .48];
f=figure('Visible','off','Color','w','Units','centimeters','Position',[2 2 25 12]);
ax=axes(f,'Position',[.085 .21 .90 .76]);
bar(ax,1:4,s.mean_accuracy,.72,'FaceColor','flat','CData',palette, ...
    'EdgeColor',[.3 .3 .3],'LineWidth',.6); hold(ax,'on');
errorbar(ax,1:4,s.mean_accuracy,s.std_accuracy,'k','LineStyle','none', ...
    'LineWidth',1,'CapSize',7);
set(ax,'FontName','Times New Roman','FontSize',16,'Box','on','TickDir','in', ...
    'XTick',1:4,'XTickLabel',s.algorithm_names,'XTickLabelRotation',30, ...
    'XGrid','on','YGrid','on','GridColor',[.8 .8 .8],'GridAlpha',.25,'Layer','bottom');
% Match reference lower limit when possible, expand if a real result is lower.
lower=min(70,5*floor(min(s.mean_accuracy-s.std_accuracy)/5));
ylim(ax,[lower 103]); xlim(ax,[.35 4.65]); yticks(ax,lower:5:100);
ylabel(ax,'Recognition rate (%)');
for a=1:4
    text(ax,a,s.mean_accuracy(a)+s.std_accuracy(a)+.6, ...
        sprintf('%.2f%s%.2f',s.mean_accuracy(a),char(177),s.std_accuracy(a)), ...
        'FontName','Times New Roman','FontSize',16, ...
        'HorizontalAlignment','center','VerticalAlignment','bottom');
end
exportFigure(f,out,'accuracy_comparison'); close(f);

colors=[0 .447 .741;.85 .325 .098;.929 .694 .125;.494 .184 .556; ...
    .466 .674 .188;.301 .745 .933;.635 .078 .184;.35 .35 .35];
markers={'o','+','x','s','^','d','v','p'};
f=figure('Visible','off','Color','w','Units','centimeters','Position',[2 2 25 20]);
% Four actual methods: 2x2 rather than eight reference panels.
layout=tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
handles=gobjects(numel(s.class_names),1);
for a=1:4
    ax=nexttile(layout); hold(ax,'on'); xy=s.tsne_coordinates{a};
    for c=1:numel(s.class_names)
        h=scatter(ax,xy(s.labels==c,1),xy(s.labels==c,2),22, ...
            'Marker',markers{c},'MarkerEdgeColor',colors(c,:), ...
            'MarkerFaceColor','none','LineWidth',.8);
        if a==1, handles(c)=h; end
    end
    set(ax,'FontName','Times New Roman','FontSize',13,'Box','on', ...
        'TickDir','in','XGrid','off','YGrid','off','LineWidth',.8);
    % Independent limits, as in reference; no class-wise manipulation.
    lo=min(xy,[],1); hi=max(xy,[],1); pad=max(.07*(hi-lo),[1 1]);
    xlim(ax,[lo(1)-pad(1),hi(1)+pad(1)]); ylim(ax,[lo(2)-pad(2),hi(2)+pad(2)]);
    title(ax,sprintf('%s (SC=%.3f)',s.algorithm_names{a},s.sc(a)), ...
        'FontName','Times New Roman','FontSize',17,'FontWeight','normal','Interpreter','none');
    if a==1, legendAx=ax; end
end
lg=legend(legendAx,handles,s.class_names,'NumColumns',min(4,numel(handles)), ...
    'FontName','Times New Roman','FontSize',11,'Box','on','Interpreter','none', ...
    'Orientation','horizontal');
lg.Layout.Tile='south';
exportFigure(f,out,'tsne_comparison'); close(f);
end

function exportFigure(f,out,name)
exportgraphics(f,fullfile(out,[name '.png']),'Resolution',300);
exportgraphics(f,fullfile(out,[name '.pdf']),'ContentType','vector');
savefig(f,fullfile(out,[name '.fig']));
print(f,fullfile(out,[name '.svg']),'-dsvg');
end
