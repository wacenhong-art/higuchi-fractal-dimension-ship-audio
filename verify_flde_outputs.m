% Numerical and export audit; run from repository root after full experiments.
root=fileparts(mfilename('fullpath'));
folders={'v2','v3_ds'};
for d=1:2
    out=fullfile(root,folders{d},'results_flde_style');
    s=load(fullfile(out,'experiment.mat'));
    f=load(fullfile(out,'features.mat'));
    legacy=fullfile(root,folders{d},'results_fd_comparison','feature_data.mat');
    if isfile(legacy)
        old=load(legacy);
        assert(isequal(s.feature_data,old.feature_data),'Fresh features differ from legacy extraction.');
        fprintf('Fresh features exactly equal legacy cache.\n');
    else
        fprintf('Legacy cache unavailable: legacy equality check skipped.\n');
    end
    assert(isequal(size(s.accuracy_all_runs),[4 20]));
    assert(isequal(s.mean_accuracy,mean(s.accuracy_all_runs,2)));
    assert(isequal(s.std_accuracy,std(s.accuracy_all_runs,0,2)));
    for k=1:20
        mask=s.train_masks(:,k);
        for c=1:numel(s.class_names)
            assert(sum(mask & s.labels==c)==100);
            assert(sum(~mask & s.labels==c)==100);
        end
        for a=1:4
            p=s.predictions{a,k};
            assert(abs(100*mean(p==s.labels(~mask))-s.accuracy_all_runs(a,k))<1e-10);
        end
    end
    for a=1:4
        assert(isequal(f.source_results{a}.Y_pred,s.predictions{a,1}));
        assert(abs(mean(silhouette(s.feature_data{a},s.labels,'Euclidean'))-s.sc(a))<1e-12);
        assert(all(isfinite(s.tsne_coordinates{a}(:))));
    end
    for base={'accuracy_comparison','tsne_comparison'}
        for ext={'.png','.pdf','.svg','.fig'}
            file=fullfile(out,[base{1} ext{1}]);
            info=dir(file); assert(~isempty(info) && info.bytes>0);
        end
    end
    fprintf('%s: 20 splits/predictions/statistics/SC/exports PASS\n',s.config.dataset);
    disp(s.result_table);
end
files={'run_flde_comparison.m','verify_flde_outputs.m', ...
    'v2/compare_all_fd_v2.m','v3_ds/compare_all_fd_v3_ds.m'};
for i=1:numel(files)
    messages=checkcode(fullfile(root,files{i}),'-id');
    fprintf('checkcode %s: %d messages\n',files{i},numel(messages));
    disp(messages);
end
