function [D, H, windowSizes, meanRS] = RS_FD(x, windowSizes)
%RS_FD  R/S (rescaled-range) based fractal-dimension estimate for 1-D signals.
%
%   D = RS_FD(x) estimates the Hurst exponent H using nonoverlapping
%   blocks and a log-log R/S regression, then returns D = 2 - H.
%
%   [D, H, windowSizes, meanRS] = RS_FD(x, windowSizes) optionally uses
%   explicitly specified block lengths and returns the regression inputs.
%
%   Method reference:
%   Loo, C. K., Samraj, A., & Lee, G. C. (2011), "Evaluation of Methods
%   for Estimating Fractal Dimension in Motor Imagery-Based Brain Computer
%   Interface," Discrete Dynamics in Nature and Society, 2011, 724697.
%   https://doi.org/10.1155/2011/724697 , Section 2.3.3.
%
%   IMPORTANT:
%   * D = 2 - H is a self-affine graph interpretation, not a universally
%     valid relationship for arbitrary time series.
%   * Default block sizes are powers of two, from 8 to floor(N/4).
%     Every block size therefore uses at least four complete blocks.
%   * For fair experiments, document block sizes and keep other pipeline
%     settings unchanged. No random-resonance processing is used here.
%
%   Example:
%       x = randn(2048, 1);
%       d = RS_FD(x);
%
%   Input checks
    validateattributes(x, {'numeric'}, ...
        {'vector', 'real', 'finite', 'nonempty'}, mfilename, 'x', 1);
    x = double(x(:));
    N = numel(x);
    if N < 128
        error('RS_FD:TooShort', ...
            'At least 128 samples are required by the default scale policy.');
    end

    if nargin < 2 || isempty(windowSizes)
        maxPower = floor(log2(floor(N / 4)));
        windowSizes = 2 .^ (3:maxPower);
    else
        validateattributes(windowSizes, {'numeric'}, ...
            {'vector', 'real', 'finite', 'integer', 'positive'}, ...
            mfilename, 'windowSizes', 2);
        windowSizes = unique(double(windowSizes(:)'));
        if any(windowSizes < 8) || any(windowSizes > floor(N / 4))
            error('RS_FD:InvalidWindowSizes', ...
                'Block lengths must be >= 8 and <= floor(length(x)/4).');
        end
    end

    if numel(windowSizes) < 3
        error('RS_FD:InsufficientScales', ...
            'At least three different block lengths are needed for fitting.');
    end

    meanRS = zeros(size(windowSizes));
    for j = 1:numel(windowSizes)
        w = windowSizes(j);
        numBlocks = floor(N / w);
        ratios = zeros(1, numBlocks);
        validBlocks = false(1, numBlocks);

        for b = 1:numBlocks
            first = (b-1) * w + 1;
            block = x(first:first+w-1);
            centered = block - mean(block);
            S = sqrt(mean(centered .^ 2));
            if S > 0
                cumulative = cumsum(centered);
                R = max(cumulative) - min(cumulative);
                ratios(b) = R / S;
                validBlocks(b) = true;
            end
        end
        if ~any(validBlocks)
            error('RS_FD:NoValidBlocks', ...
                ['All nonoverlapping blocks of length %d have zero variance. ', ...
                 'R/S cannot be estimated at this window size.'], w);
        end
        % R/S is undefined for zero-variance blocks; average valid blocks only.
        meanRS(j) = mean(ratios(validBlocks));
    end

    if any(~isfinite(meanRS)) || any(meanRS <= 0)
        error('RS_FD:InvalidRS', 'Nonpositive or nonfinite R/S detected.');
    end

    p = polyfit(log(windowSizes), log(meanRS), 1);
    H = p(1);
    D = 2 - H;
    % Do not clamp D to [1,2]: clipping would conceal estimator failure.
end
