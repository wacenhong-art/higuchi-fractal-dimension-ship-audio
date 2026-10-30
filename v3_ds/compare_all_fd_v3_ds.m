% v3_ds: FLDE-style comparison. Algorithms and original KNN columns retained.
% full: fresh extraction + 20 repeats; results: cached features; plot: render only.
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(project_root);
run_flde_comparison('v3_ds', 'full');
