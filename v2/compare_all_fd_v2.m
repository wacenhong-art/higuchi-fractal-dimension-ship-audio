% v2: FLDE-style comparison. Algorithms and original KNN columns retained.
% full: fresh extraction + single seed-42 run; results: cached features; plot: render only.
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(project_root);
run_flde_comparison('v2', 'full');
