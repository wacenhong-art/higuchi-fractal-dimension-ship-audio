# Parameters extracted from the MATLAB source

This table records literal settings found in the archived source. It does not infer values that are not set in code.

| Area | Setting | Value / behavior | Source |
|---|---|---|---|
| CI-Higuchi | Maximum scale `kmax` | `20`; scales are integers `1:kmax` | `ci_hfd.m`, `ci_hfd_no_sr.m`, `v2/ci_hfd.m`, `v2/ci_hfd_no_sr.m` |
| CI-Higuchi | Confidence level `alpha` | `0.95`; used in `tinv((1+alpha)/2, n-1)` to filter positive per-scale lengths | same CI-Higuchi scripts |
| Stochastic resonance / RK4 | `a`, `b`, `h` | `a=10`, `b=1`, `h=0.01` | `ci_hfd.m`, `v2/ci_hfd.m`, `v2/ci_fd_katz_sr.m` |
| RK4 | State dynamics | Four RK4 stages for `a*x - b*x^3 + input`; output starts at zero (`x=zeros(1,length(x1))`) | `Rung_kutta_JDSR_k.m`, `v2/Rung_kutta_JDSR_k.m` |
| Audio preprocessing | Reader / channels | `audioread`; if channels > 1, `mean(...,2)` produces mono | all pipeline scripts |
| Audio preprocessing | Sample rate | Captured as `~` and not resampled or used | all pipeline scripts |
| Segmentation | Fixed segments | first 200 non-overlapping segments, each 2048 samples; tail is unused | all pipeline scripts |
| Feature vector | Dimensions | One scalar fractal-dimension feature per segment; root scripts allocate 800x1 (4x200), `v2` scripts allocate `num_classes*200` = 1000x1 (5x200) | pipeline scripts |
| SR feature normalization | Standardization | `(sr_signal - mean(sr_signal)) / std(sr_signal)` before CI-Higuchi/Katz | `ci_hfd.m`, `v2/ci_hfd.m`, `v2/ci_fd_katz_sr.m` |
| Box-counting | `cellmax` | `2048`, equal to segment length; FX checks it is not smaller than the input | `ci_fd_fx.m`, `v2/ci_fd_fx.m`, `对比算法/FX.m` |
| Katz | Input requirement | Row vector (`x(:).'` in callers); algorithm has no active `Kmax` setting | `ci_fd_katz.m`, `v2/ci_fd_katz.m`, `v2/ci_fd_katz_sr.m`, `对比算法/Katz_FD.m` |
| Classifier | Random seed | `rng(42)` | all pipeline scripts |
| Classifier | Split | Per class, `randperm(200,100)` selects 100 training segments; remaining 100 are test segments | all pipeline scripts |
| Classifier | KNN | `fitcknn`, `NumNeighbors=1`, `Distance='euclidean'` | all pipeline scripts |

The scripts report per-class and overall accuracy and a confusion matrix when executed, but this archive publishes no result values.
