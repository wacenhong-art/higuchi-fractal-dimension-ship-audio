% 多算法、10 尺度分形维数公平对比（v2 八类数据；不含随机共振）。
% 运行时才会读取音频、执行 KNN/t-SNE，并写入 results_fd_comparison。
clear; clc; close all;

%% 1. 固定实验配置：沿用现有 2048 点、200 段/类、100/100 划分与 1-NN
project_path = fileparts(mfilename('fullpath'));
class_names = {'国家渡轮','货轮','螺旋桨呜呜呜','舷外发动机发出呜呜声和加速','游轮','舷外发动机60马力10节','舷外发动机60马力20节','小型柴油发动机船只'};
config = struct('segment_length',2048,'samples_per_class',200,'scales',1:10,'train_per_class',100,'num_repeats',10,'seed_base',42,'tsne_seed',1042,'knn_neighbors',1,'knn_distance','euclidean','feature_standardization','none','dataset','v2 eight-class paths from existing scripts');
algorithm_names = {'Katz','CI-Higuchi','Traditional Higuchi','FX box-counting','R/S'};
result_dir = fullfile(project_path,'results_fd_comparison'); if exist(result_dir,'dir') ~= 7, mkdir(result_dir); end

%% 2. 根据现有 v2 脚本的实际路径读取音频、转单声道并取连续 2048 点片段
audio_dirs = {fullfile(project_path,'音频','国家渡轮'),fullfile(project_path,'音频','货轮'),fullfile(project_path,'音频','螺旋桨呜呜呜'),fullfile(project_path,'音频','舷外发动机发出呜呜声和加速'),fullfile(project_path,'音频','游轮'),fullfile(project_path,'音频','舷外发动机（60马力）在10节'),fullfile(project_path,'音频','舷外发动机（60马力）在20节'),fullfile(project_path,'音频','由小型柴油发动机驱动的过往船只的水下记录')};
wav_paths = cell(1,numel(class_names)); for c=1:5, wav_paths{c}=FindOneAudio(audio_dirs{c}); end
wav_paths{6}=fullfile(audio_dirs{6},'705da20f-b986-4e49-ae58-39f9f286e9ad.mp3'); wav_paths{7}=fullfile(audio_dirs{7},'39f7562b-1d7a-4fe4-a969-6be98e7d4886.mp3'); wav_paths{8}=fullfile(audio_dirs{8},'cb678cb6-5627-4544-9fbb-c8e3884f0827.mp3');
[segments,labels]=LoadSegments(wav_paths,config.segment_length,config.samples_per_class);

%% 3. 每一个片段独立进行 tau=1:10 非重叠均值粗粒化，得到各算法 10 维特征
feature_data=cell(1,numel(algorithm_names));
for a=1:numel(algorithm_names)
    feature_data{a}=ExtractFeatures(segments,config.scales,a);
    if ~isequal(size(feature_data{a}),[numel(labels),numel(config.scales)]) || ~isreal(feature_data{a}) || any(~isfinite(feature_data{a}(:))), error('%s 特征不是有限实数 N x 10 矩阵。',algorithm_names{a}); end
end

%% 4. 每次重复中所有算法严格共用同一分层划分；原有脚本未标准化，故此处保持 none
accuracy_all_runs=zeros(numel(algorithm_names),config.num_repeats);
for r=1:config.num_repeats
    train_idx=StratifiedSplit(labels,config.train_per_class,config.seed_base+r-1);
    for a=1:numel(algorithm_names)
        model=fitcknn(feature_data{a}(train_idx,:),labels(train_idx),'NumNeighbors',config.knn_neighbors,'Distance',config.knn_distance);
        prediction=predict(model,feature_data{a}(~train_idx,:)); accuracy_all_runs(a,r)=mean(prediction==labels(~train_idx))*100;
    end
end
mean_accuracy=mean(accuracy_all_runs,2); if config.num_repeats>1, std_accuracy=std(accuracy_all_runs,0,2); else, std_accuracy=nan(numel(algorithm_names),1); end

%% 5. 导出 KNN 结果和识别率柱状图（误差棒为重复实验的标准差）
result_table=table(string(algorithm_names(:)),mean_accuracy,std_accuracy,repmat(config.num_repeats,numel(algorithm_names),1),'VariableNames',{'Algorithm','MeanAccuracyPercent','StdAccuracyPercent','NumRepeats'});
writetable(result_table,fullfile(result_dir,'accuracy_results.csv')); PlotAccuracy(result_table,result_dir);

%% 6. t-SNE 仅用于所有真实 10 维特征的可视化，不输入 KNN
[tsne_coordinates,tsne_success]=PlotTsne(feature_data,labels,class_names,algorithm_names,config.tsne_seed,result_dir);
save(fullfile(result_dir,'feature_data.mat'),'feature_data','labels','class_names','algorithm_names','config','accuracy_all_runs','mean_accuracy','std_accuracy','tsne_coordinates','tsne_success','wav_paths');
fprintf('完成：结果写入 %s\n',result_dir);

%% ============================== 局部函数 ==============================
function [segments,labels]=LoadSegments(wav_paths,segment_length,samples_per_class)
n=numel(wav_paths); segments=zeros(n*samples_per_class,segment_length); labels=zeros(n*samples_per_class,1);
for c=1:n
    if exist(wav_paths{c},'file')~=2, error('找不到音频文件：%s',wav_paths{c}); end
    [x,~]=audioread(wav_paths{c}); if size(x,2)>1, x=mean(x,2); end
    if numel(x)<samples_per_class*segment_length, error('音频长度不足 %d 个 %d 点样本：%s',samples_per_class,segment_length,wav_paths{c}); end
    rows=(c-1)*samples_per_class+(1:samples_per_class); segments(rows,:)=reshape(x(1:samples_per_class*segment_length),segment_length,samples_per_class).'; labels(rows)=c;
end
end

function features=ExtractFeatures(segments,scales,algorithm_index)
features=zeros(size(segments,1),numel(scales));
for i=1:size(segments,1)
    for s=1:numel(scales)
        y=CoarseGrain(segments(i,:),scales(s));
        switch algorithm_index
            case 1, value=KatzFD(y);
            case 2, value=CIHiguchi(y,20,0.95);
            case 3, value=TraditionalHiguchi(y,20);
            case 4, value=FXFD(y);
            case 5, value=RSFD(y);
        end
        if ~isscalar(value) || ~isreal(value) || ~isfinite(value), error('无效特征：样本 %d、尺度 %d、算法 %d。',i,scales(s),algorithm_index); end
        features(i,s)=value;
    end
end
end

function y=CoarseGrain(x,tau)
x=x(:).'; n=floor(numel(x)/tau); if n<2, error('粗粒化后序列长度不足 2。'); end
y=mean(reshape(x(1:n*tau),tau,n),1);
end

function train_idx=StratifiedSplit(labels,train_per_class,seed)
rng(seed,'twister'); train_idx=false(size(labels));
for c=unique(labels(:)).'
    idx=find(labels==c); if numel(idx)<train_per_class, error('类别 %d 样本数不足。',c); end
    train_idx(idx(randperm(numel(idx),train_per_class)))=true;
end
end

function d=KatzFD(x)
x=x(:).'; n=numel(x)-1; L=sum(sqrt(1+diff(x).^2)); diameter=max(sqrt((1:n).^2+(x(1)-x(2:end)).^2)); d=log10(n)/(log10(n)+log10(diameter/L));
end

function d=CIHiguchi(x,kmax,alpha), L=HiguchiLengths(x,kmax,true,alpha); p=polyfit(log(1:kmax),log(L+eps),1); d=-p(1); end
function d=TraditionalHiguchi(x,kmax), L=HiguchiLengths(x,kmax,false,0); p=polyfit(log(1:kmax),log(L+eps),1); d=-p(1); end
function L=HiguchiLengths(x,kmax,use_ci,alpha)
N=numel(x); if N<2*kmax, error('Higuchi 输入长度不足 %d。',2*kmax); end; L=zeros(1,kmax);
for k=1:kmax
    Lk=zeros(1,k); for m=1:k, seq=x(m:k:N); n=numel(seq); if n>=2, Lk(m)=sum(abs(diff(seq)))*(N-1)/(n*k); end, end
    if use_ci
        keep=Lk(Lk>0); if numel(keep)>=3, t=tinv((1+alpha)/2,numel(keep)-1); keep=keep(keep>=mean(keep)-t*std(keep) & keep<=mean(keep)+t*std(keep)); end
        if isempty(keep), L(k)=mean(Lk); else, L(k)=mean(keep); end
    else, L(k)=mean(Lk); end
end
end

function d=FXFD(y), cellmax=2^floor(log2(numel(y))); d=FXOriginal(y(1:cellmax),cellmax); end
function d=FXOriginal(y,cellmax)
L=numel(y); y=y-min(y); yy=abs(interp1((0:L-1)/(L-1),y,(0:cellmax)/cellmax)*cellmax/max(y)); t=log2(cellmax)+1; N=zeros(1,t); S=zeros(1,t);
for e=1:t, cs=2^(e-1); S(e)=cellmax/cs; for j=1:S(e), q=cs*(j-1)+1:cs*j+1; N(e)=N(e)+ceil(max(yy(q))/cs)-floor(min(yy(q))/cs); end, end
r=-diff(log2(N)); id=find(r<=2 & r>=1); p=polyfit(log2(S(id)),log2(N(id)),1); d=p(1);
end

function d=RSFD(x)
x=double(x(:)); N=numel(x); if N<128, error('R/S 输入不足 128 点。'); end; windows=2.^(3:floor(log2(floor(N/4)))); if numel(windows)<3, error('R/S 可用窗口尺度不足。'); end; meanRS=zeros(size(windows));
for j=1:numel(windows)
    w=windows(j); ratios=[]; for b=1:floor(N/w), z=x((b-1)*w+1:b*w); z=z-mean(z); sd=sqrt(mean(z.^2)); if sd>0, cs=cumsum(z); ratios(end+1)=(max(cs)-min(cs))/sd; end, end %#ok<AGROW>
    if isempty(ratios), error('R/S 在窗口 %d 无有效块。',w); end; meanRS(j)=mean(ratios);
end
p=polyfit(log(windows),log(meanRS),1); d=2-p(1);
end

function PlotAccuracy(t,result_dir)
f=figure('Color','w','Position',[100 100 1000 560]); m=t.MeanAccuracyPercent; s=t.StdAccuracyPercent; colors=[.35 .64 .84;.95 .61 .45;.49 .72 .55;.72 .60 .82;.94 .77 .39]; b=bar(m,'FaceColor','flat'); b.CData=colors(1:numel(m),:); hold on; valid=isfinite(s); errorbar(find(valid),m(valid),s(valid),'k.','LineWidth',1.2,'CapSize',10); grid on; ax=gca; ax.Color=[.98 .98 .98]; ax.GridColor=[.78 .78 .78]; ax.GridAlpha=.55; ax.XTick=1:numel(m); ax.XTickLabel=t.Algorithm; ax.XTickLabelRotation=20; ylabel('Recognition Rate (%)'); title('Fractal-Dimension Classification Comparison');
for i=1:numel(m), if valid(i), label=sprintf('%.2f±%.2f',m(i),s(i)); y=m(i)+s(i); else, label=sprintf('%.2f (single run)',m(i)); y=m(i); end, text(i,y,label,'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',10); end
exportgraphics(f,fullfile(result_dir,'accuracy_comparison.png'),'Resolution',300); exportgraphics(f,fullfile(result_dir,'accuracy_comparison.pdf'),'ContentType','vector'); close(f);
end

function [coordinates,success]=PlotTsne(data,labels,class_names,algorithm_names,seed,result_dir)
n=numel(data); coordinates=cell(1,n); success=false(1,n);
for a=1:n, try, rng(seed+a-1,'twister'); coordinates{a}=tsne(data{a},'NumDimensions',2); success(a)=true; catch ME, warning('t-SNE failed for %s: %s',algorithm_names{a},ME.message); end, end
valid=coordinates(success); if isempty(valid), warning('所有 t-SNE 均失败；不导出虚假坐标。'); return; end
first_success=find(success,1); xy=vertcat(valid{:}); lo=min(xy,[],1); hi=max(xy,[],1); pad=max((hi-lo)*.05,[1 1]); rows=ceil(sqrt(n)); cols=ceil(n/rows); f=figure('Color','w','Position',[80 80 420*cols 330*rows]); tl=tiledlayout(rows,cols,'TileSpacing','compact','Padding','compact'); colors=lines(numel(class_names)); markers={'o','+','x','s','^','d','v','p'}; h=gobjects(numel(class_names),1);
for a=1:n
    nexttile; hold on; if success(a), for c=1:numel(class_names), q=scatter(coordinates{a}(labels==c,1),coordinates{a}(labels==c,2),26,'Marker',markers{c},'MarkerEdgeColor',colors(c,:),'LineWidth',1); if a==first_success,h(c)=q;else,q.HandleVisibility='off';end,end; xlim([lo(1)-pad(1) hi(1)+pad(1)]); ylim([lo(2)-pad(2) hi(2)+pad(2)]); else,text(.5,.5,'t-SNE failed; see warning','Units','normalized','HorizontalAlignment','center');end; grid on; set(gca,'Color','w','GridColor',[.82 .82 .82]); xlabel('t-SNE Dimension 1'); ylabel('t-SNE Dimension 2'); title(algorithm_names{a});
end
lgd=legend(h,class_names,'Orientation','horizontal'); lgd.Layout.Tile='south'; title(tl,'t-SNE of Full Feature Matrices (visualization only)'); exportgraphics(f,fullfile(result_dir,'tsne_comparison.png'),'Resolution',300); exportgraphics(f,fullfile(result_dir,'tsne_comparison.pdf'),'ContentType','vector'); close(f);
end

function file=FindOneAudio(folder)
ext={'.wav','.flac','.mp3','.m4a','.mp4','.ogg','.aif','.aiff','.au'}; files=[]; for i=1:numel(ext), q=dir(fullfile(folder,'**',['*' ext{i}])); files=[files;q(~[q.isdir])]; end %#ok<AGROW>
if numel(files)~=1, error('文件夹必须恰有一个受支持音频：%s',folder); end; file=fullfile(files(1).folder,files(1).name);
end
