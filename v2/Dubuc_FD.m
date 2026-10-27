function [D, details] = Dubuc_FD(x, radii)
%DUBUC_FD Estimate the Dubuc fractal dimension of a 1-D signal.
%   D = DUBUC_FD(X) returns one fractal-dimension estimate for a real,
%   finite, uniformly sampled time series X (row or column vector).
%
%   D = DUBUC_FD(X, RADII) uses the specified positive integer neighborhood
%   radii (in sample intervals), e.g. [1 2 4 8 16 32 64]. At least three
%   distinct, strictly increasing radii are required.
%
%   [D, DETAILS] = DUBUC_FD(...) also returns the radii, epsilon-variation
%   values, fitted log-log points, slope and goodness of fit (R squared).
%
%   Method: Dubuc's variation method for profiles (NOT a variogram method).
%   For each radius r, form the local upper/lower envelopes:
%       u_r(i) = max{x(j): |j-i| <= r, 1 <= j <= N}
%       l_r(i) = min{x(j): |j-i| <= r, 1 <= j <= N}
%   The discrete variation is
%       V(r) = (1/r^2) * sum_i (u_r(i) - l_r(i)).
%   Fit a straight line to [log(1/r), log(V(r))]. Its slope is D.
%   Equivalently, D = 2 - slope(log(sum_i(u_r-l_r)) vs log(r)).
%
%   By default, radii are dyadic: 1,2,4,... up to floor(N/16).
%   This is a fixed numerical implementation choice, not a parameter
%   reported as mandatory by the original paper. Use the same radii rule
%   in all experiments to ensure fair comparisons.
%
%   No random resonance, filtering, normalization, or classification is
%   performed. Constant signals have a graph dimension of 1 (return D=1).
%   Finite-data estimates are not artificially clipped to [1,2].
%
%   Reference:
%   Dubuc, B., Quiniou, J. F., Roques-Carmes, C., Tricot, C., and
%   Zucker, S. W., "Evaluating the fractal dimension of profiles,"
%   Physical Review A, 39(3), 1500-1512, 1989.
%   DOI: 10.1103/PhysRevA.39.1500
%
%   Discrete variation and regression also described in:
%   Pilgrim, I. and Taylor, R. P., "Fractal Analysis of Time-Series Data
%   Sets: Methods and Challenges," IntechOpen, 2018.
%   DOI: 10.5772/intechopen.81958
%
%   Examples:
%       x = sin(2*pi*(0:2047)'/128);
%       D = Dubuc_FD(x);
%       [D, details] = Dubuc_FD(x, [1 2 4 8 16 32 64]);
%
%   See also MOVMAX, MOVMIN, POLYFIT.

    narginchk(1, 2);
    validateattributes(x, {'numeric'}, ...
        {'vector','nonempty','real','finite'}, mfilename, 'x', 1);
    x = double(x(:));
    N = numel(x);
    if N < 128
        error('Dubuc_FD:InsufficientLength', ...
            'The input must contain at least 128 samples (got %d).', N);
    end

    if nargin < 2 || isempty(radii)
        maxRadius = floor(N / 16);
        radii = 2 .^ (0:floor(log2(maxRadius)));
    end
    validateattributes(radii, {'numeric'}, ...
        {'vector','nonempty','real','finite','integer','positive'}, ...
        mfilename, 'radii', 2);
    radii = double(radii(:));
    if numel(radii) < 3 || any(diff(radii) <= 0)
        error('Dubuc_FD:InvalidRadii', ...
            'radii must contain at least 3 strictly increasing integers.');
    end
    if any(radii > floor(N / 4))
        error('Dubuc_FD:RadiusTooLarge', ...
            'All radii must be <= floor(N/4) = %d.', floor(N / 4));
    end

    % A constant sequence is the graph of a horizontal line.
    if all(x == x(1))
        D = 1;
        if nargout > 1
            details = struct('radii', radii, ...
                'variation', zeros(size(radii)), ...
                'logInvRadius', log(1 ./ radii), ...
                'logVariation', NaN(size(radii)), ...
                'fittedLogVariation', NaN(size(radii)), ...
                'slope', NaN, 'intercept', NaN, ...
                'R2', NaN, 'constantSignal', true);
        end
        return;
    end

    variation = zeros(size(radii));
    for k = 1:numel(radii)
        r = radii(k);
        % Truncated windows at both ends: [max(1,i-r), min(N,i+r)].
        upperEnvelope = movmax(x, [r r], 'Endpoints', 'shrink');
        lowerEnvelope = movmin(x, [r r], 'Endpoints', 'shrink');
        localOscillation = upperEnvelope - lowerEnvelope;
        variation(k) = sum(localOscillation) / (r^2);
    end

    if any(~isfinite(variation)) || any(variation <= 0)
        error('Dubuc_FD:DegenerateVariation', ...
            'Cannot fit a dimension because epsilon-variation is not positive and finite.');
    end

    logInvRadius = log(1 ./ radii);
    logVariation = log(variation);
    p = polyfit(logInvRadius, logVariation, 1);
    D = p(1);

    if nargout > 1
        fitted = polyval(p, logInvRadius);
        sse = sum((logVariation - fitted).^2);
        sst = sum((logVariation - mean(logVariation)).^2);
        if sst == 0
            R2 = NaN;
        else
            R2 = 1 - sse / sst;
        end
        details = struct('radii', radii, ...
            'variation', variation, ...
            'logInvRadius', logInvRadius, ...
            'logVariation', logVariation, ...
            'fittedLogVariation', fitted, ...
            'slope', p(1), 'intercept', p(2), ...
            'R2', R2, 'constantSignal', false);
    end
end
