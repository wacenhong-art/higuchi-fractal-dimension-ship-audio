# CI-Higuchi ship-audio classification code archive

This repository archives MATLAB implementations for confidence-interval-improved Higuchi (CI-Higuchi) fractal-dimension feature extraction, optional stochastic-resonance preprocessing solved by fourth-order Runge--Kutta (RK4), Katz and box-counting comparison features, and 1-nearest-neighbor classification.

No raw audio, result files, metrics, or performance claims are included. The code is published as an archive of implementations; it is not a claim that every included script has been re-executed or validated in this release.

## Requirements

- MATLAB with `audioread`, Statistics and Machine Learning Toolbox (`fitcknn`, `confusionmat`), and a Statistics function providing `tinv`.
- Audio files that you are authorized to use. They are deliberately not included.

## Layout

- `ci_hfd.m` — four-class CI-Higuchi pipeline with stochastic resonance and RK4.
- `ci_hfd_no_sr.m` — four-class CI-Higuchi pipeline without stochastic resonance.
- `ci_fd_katz.m`, `ci_fd_fx.m` — four-class Katz and box-counting comparison pipelines.
- `Rung_kutta_JDSR_k.m` — RK4 solver used by stochastic-resonance variants.
- `v2/` — five-class versions: CI-Higuchi with/without stochastic resonance, Katz with/without stochastic resonance, box-counting, and an RK4 copy.
- `对比算法/` — Katz and box-counting implementations used by the comparison scripts. `Katz_FD.m` retains its in-file attribution to Jesús Monge Álvarez; no license is added by this archive.
- `PARAMETERS.md` — parameters extracted from source, with file locations.
- `EXPERIMENTS.md` — script/status boundary for this archive.

## Data placement and execution

Scripts locate data relative to their own directory. To use a four-class root-level script, provide the expected `DeepShip-main/<class>/<file>.wav` paths referenced in that script. To use a `v2` script, place one authorized supported audio file in each expected `v2/音频/<class>/` directory. See each script for its exact class names and filenames.

Static configuration note: the archived `v2/ci_fd_katz.m`, `v2/ci_fd_katz_sr.m`, and `v2/ci_fd_fx.m` construct `v2/对比算法` as their lookup folder, while the archived `Katz_FD.m` and `FX.m` reside at repository-root `对比算法/`. This archive does not alter that historical code path. Configure MATLAB's path or the local layout deliberately before attempting those scripts, then validate the configuration yourself.

The scripts read audio with `audioread`; multi-channel inputs are averaged to mono. Each class must contain at least 200 non-overlapping segments of 2048 samples. No sample-rate conversion is performed by the archived code, so the input sampling rate is preserved but not otherwise used as a parameter.

Run a script from MATLAB only after configuring authorized local data. This release did not run MATLAB, regenerate features, or create results.

## Reproducibility boundary

The random train/test selection is seeded with `rng(42)`, but it splits 200 segments within each source recording into 100 training and 100 test segments. Consequently, it is a script-level implementation detail, not a claim of source-file-level leakage-safe evaluation. Do not interpret absent results as validated performance.

No open-source license is supplied. Public visibility does not grant rights to third-party data or code beyond their own terms.
