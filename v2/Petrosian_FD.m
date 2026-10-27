function D = Petrosian_FD(x)
% PETROSIAN_FD Petrosian fractal dimension of a one-dimensional signal.
%   D = Petrosian_FD(x) returns one scalar feature for the signal x.
%
%   Definition:
%       N_delta = number of sign changes in the first difference of x
%       D = log10(N) / (log10(N) + log10(N/(N + 0.4*N_delta)))
%
%   Sign convention: a zero first difference is counted as nonnegative.
%   This matches the signbit convention used by the AntroPy implementation.
%   No stochastic resonance, filtering, resampling or coarse-graining is used.
%
%   Example:
%       x = sin(2*pi*(0:2047)'/100);
%       d = Petrosian_FD(x);
%
%   References:
%       Petrosian, A. (1995). Kolmogorov complexity of finite sequences
%       and recognition of different preictal EEG patterns.
%       https://raphaelvallat.com/antropy/_modules/antropy/fractal.html
%       https://in.mathworks.com/matlabcentral/answers/413573-petrosian-fractal-entropy-estimation
%
    validateattributes(x, {'numeric'}, ...
        {'vector', 'real', 'finite', 'nonempty'}, mfilename, 'x', 1);

    x = double(x(:));
    N = numel(x);
    if N < 3
        error('Petrosian_FD:SignalTooShort', ...
            'Input signal must contain at least 3 samples.');
    end

    dx = diff(x);
    is_negative = dx < 0;
    N_delta = sum(is_negative(1:end-1) ~= is_negative(2:end));

    D = log10(N) / (log10(N) + log10(N / (N + 0.4 * N_delta)));
end
