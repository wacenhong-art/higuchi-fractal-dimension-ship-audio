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
    assert(s.config.num_repeats==20 && isequal(s.config.seeds,42:61));
    assert(isequal(size(s.accuracy_all_runs),[4 20]));
    assert(isequal(s.mean_accuracy,mean(s.accuracy_all_runs,2)));
    assert(isequal(s.std_accuracy,std(s.accuracy_all_runs,0,2)));
    for k=1:s.config.num_repeats
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
        r=f.source_results{a};
        assert(isequal(r.Y_pred,s.predictions{a,1}));
        assert(isequal(r.Y_test,s.labels(~s.train_masks(:,1))));
        x=s.feature_data{a}(:,1:s.knn_dimensions(a));
        assert(isequal(r.X_train,x(s.train_masks(:,1),:)));
        originalAccuracy=100*mean(r.Y_pred==r.Y_test);
        assert(abs(s.accuracy_all_runs(a,1)-originalAccuracy)<1e-10, ...
            'First split must reproduce the original script.');
        cm=confusionmat(r.Y_test,r.Y_pred);
        assert(isequal(cm,s.confusion_matrices{a,1}));
        assert(abs(originalAccuracy-100*trace(cm)/sum(cm(:)))<1e-10);
        manualSC=independentSC(s.feature_data{a},s.labels);
        assert(abs(manualSC-s.sc(a))<1e-10);
        assert(abs(mean(silhouette(s.feature_data{a},s.labels,'Euclidean'))-s.sc(a))<1e-12);
        assert(all(isfinite(s.tsne_coordinates{a}(:))));
    end
    for base={'accuracy_comparison','tsne_comparison'}
        for ext={'.png','.pdf','.svg','.fig'}
            file=fullfile(out,[base{1} ext{1}]);
            info=dir(file); assert(~isempty(info) && info.bytes>0);
        end
    end
    fig=openfig(fullfile(out,'accuracy_comparison.fig'),'invisible');
    bars=findobj(fig,'Type','bar');
    assert(isscalar(bars) && isequal(bars.YData(:),s.mean_accuracy(:)));
    eb=findobj(fig,'Type','errorbar');
    assert(isscalar(eb));
    assert(isequal(eb.YData(:),s.mean_accuracy(:)));
    assert(isequal(eb.YPositiveDelta(:),s.std_accuracy(:)));
    assert(isequal(eb.YNegativeDelta(:),s.std_accuracy(:)));
    ax=findobj(fig,'Type','axes'); assert(isempty(ax.Title.String));
    labels=findobj(fig,'Type','text');
    strings=string(get(labels,'String'));
    for a=1:4
        assert(any(strings==sprintf('%.2f%s%.2f',s.mean_accuracy(a),char(177),s.std_accuracy(a))));
    end
    close(fig);
    fig=openfig(fullfile(out,'tsne_comparison.fig'),'invisible');
    lg=findobj(fig,'Type','legend');
    assert(isscalar(lg) && isequal(lg.String(:),s.class_names(:)));
    expectedMarkers={'o','+','x','square','^','diamond','v','pentagram'};
    axs=findobj(fig,'Type','axes');
    for a=1:4
        panel=[];
        expected=sprintf('%s (SC=%.3f)',s.algorithm_names{a},s.sc(a));
        for j=1:numel(axs)
            if strcmp(axs(j).Title.String,expected), panel=axs(j); end
        end
        assert(~isempty(panel)); points=findobj(panel,'Type','scatter');
        assert(numel(points)==numel(s.class_names));
        for c=1:numel(s.class_names)
            xy=s.tsne_coordinates{a}(s.labels==c,:); found=false;
            for j=1:numel(points)
                if isequal(points(j).XData(:),xy(:,1)) && isequal(points(j).YData(:),xy(:,2))
                    assert(strcmp(points(j).Marker,expectedMarkers{c}), ...
                        'Class %d marker mismatch: actual %s expected %s', ...
                        c,points(j).Marker,expectedMarkers{c});
                    found=true;
                end
            end
            assert(found,'Scatter coordinates do not match saved class samples.');
        end
    end
    close(fig);
    csv=readtable(fullfile(out,'accuracy_results.csv'));
    assert(max(abs(csv.MeanAccuracyPercent-s.mean_accuracy))<1e-10);
    fprintf('%s: 20 splits / original first split / mean / SD / error bars / annotations / SC / scatter / exports PASS\n',s.config.dataset);
    disp(s.result_table);
end
files={'run_flde_comparison.m','verify_flde_outputs.m', ...
    'v2/compare_all_fd_v2.m','v3_ds/compare_all_fd_v3_ds.m'};
for i=1:numel(files)
    messages=checkcode(fullfile(root,files{i}),'-id');
    fprintf('checkcode %s: %d messages\n',files{i},numel(messages));
    disp(messages);
end

function score=independentSC(x,y)
% Independently implement a(i), b(i) using the full pairwise distance matrix.
d=pdist2(x,x,'euclidean'); classes=unique(y); values=zeros(size(y));
for i=1:numel(y)
    own=(y==y(i)); own(i)=false;
    within=mean(d(i,own)); between=inf;
    for c=classes(:).'
        if c~=y(i), between=min(between,mean(d(i,y==c))); end
    end
    values(i)=(between-within)/max(within,between);
end
score=mean(values);
end
