function [sample] = sampling_distributions(varargin)
% Samples from various probability distributions.
%
% Inputs (Required):
% distribution: (string) name of the distribution.
%               Supported: 'normal', 'multivariatenormal', 'uniform'.
% n_samples: (scalar) number of samples to generate.
%
% Inputs (Name-Value Pairs):
%       if distribution is 'normal'
%                'mu_normal' (scalar) is required, indicating the mean.
%                'sigma_normal' (scalar) is required, indicating the standard deviation.
%
%       if distribution is 'multivariatenormal'
%                'mu_multi' (vector) is required, indicating the mean vector.
%                'covariance_multi' (matrix) is required, indicating the covariance matrix.
%
%       if distribution is 'uniform'
%                'range_uniform' (vector, 1x2) is required, as [min, max].
%
% Outputs:
% sample: array with the samples
%       if 'normal' or 'uniform'
%               sample is a [n_samples x 1] column vector.
%       if 'multivariatenormal'
%               sample is a [n_samples x D] matrix, where D is the
%               number of variables (length of 'mu_multi').
%
%% Parse Inputs
p = inputParser;
p.CaseSensitive = false; % 'normal' and 'NORMAL' are treated the same
% Required inputs
addRequired(p, 'distribution', @(x) ischar(x) || isstring(x));
addRequired(p, 'n_samples', @(x) isnumeric(x) && isscalar(x) && x > 0 && floor(x) == x); % Ensure integer n_samples
% Parameters for 'normal'
addParameter(p, 'mu_normal', [], @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'sigma_normal', [], @(x) isnumeric(x) && isscalar(x) && x >= 0);
% Parameters for 'multivariatenormal'
addParameter(p, 'mu_multi', [], @(x) isnumeric(x) && isvector(x));
addParameter(p, 'covariance_multi', [], @(x) isnumeric(x) && ismatrix(x));
% Parameters for 'uniform'
addParameter(p, 'range_uniform', [], @(x) isnumeric(x) && isvector(x) && length(x) == 2);
% Parameters to limit the distributions (minimum and maximum values)
addParameter(p, 'min_value', [], @(x) isnumeric(x) && (isscalar(x) || isvector(x)));
addParameter(p, 'max_value', [], @(x) isnumeric(x) && (isscalar(x) || isvector(x)));
% Parse
parse(p, varargin{:});
% Retrieve common results
distribution = p.Results.distribution;
n_samples = p.Results.n_samples;
minVal = p.Results.min_value;
maxVal = p.Results.max_value;

%% Sample based on distribution
switch lower(distribution)
    case 'normal'
        mu = p.Results.mu_normal;
        sigma = p.Results.sigma_normal;
        if isempty(mu) || isempty(sigma)
            error('For ''normal'' distribution, ''mu_normal'' and ''sigma_normal'' parameters are required.');
        end
        sample = normrnd(mu, sigma, n_samples, 1);

        % Resample if any values are out of specified limits
        if ~isempty(minVal) || ~isempty(maxVal)
            if isempty(minVal), minVal = -Inf; end
            if isempty(maxVal), maxVal = Inf; end

            outOfBounds = (sample < minVal) | (sample > maxVal);
            while any(outOfBounds)
                n_resample = sum(outOfBounds);
                sample(outOfBounds) = normrnd(mu, sigma, n_resample, 1);
                % Re-check only the ones that were out of bounds
                outOfBounds(outOfBounds) = (sample(outOfBounds) < minVal) | (sample(outOfBounds) > maxVal);
            end
        end

    case 'multivariatenormal'
        mu_vec = p.Results.mu_multi;
        cov_mat = p.Results.covariance_multi;
        if isempty(mu_vec) || isempty(cov_mat)
            error('For ''multivariatenormal'', ''mu_multi'' and ''covariance_multi'' are required.');
        end

        % Ensure mu_vec is a row vector for consistency
        mu_vec = reshape(mu_vec, 1, []);
        D = length(mu_vec); % Number of dimensions

        % Basic dimension check
        if D ~= size(cov_mat, 1) || D ~= size(cov_mat, 2)
            error('Mean vector length (%d) must match covariance matrix dimensions (%d x %d).', ...
                D, size(cov_mat, 1), size(cov_mat, 2));
        end

        % Generate initial samples
        sample = mvnrnd(mu_vec, cov_mat, n_samples);

        % Handle truncated sampling
        if ~isempty(minVal) || ~isempty(maxVal)     % if needed only

            % minVal must be 1xD row vector
            if isempty(minVal)
                minVal = repmat(-Inf, 1, D);
            elseif isscalar(minVal)
                minVal = repmat(minVal, 1, D); % Expand scalar to vector
            elseif length(minVal) ~= D
                error('min_value must be a scalar or a vector of length %d (number of dimensions).', D);
            end

            % maxVal must be 1xD row vector
            if isempty(maxVal)
                maxVal = repmat(Inf, 1, D);
            elseif isscalar(maxVal)
                maxVal = repmat(maxVal, 1, D); % Expand scalar to vector
            elseif length(maxVal) ~= D
                error('max_value must be a scalar or a vector of length %d (number of dimensions).', D);
            end

            % Ensure they are row vectors for broadcasting
            minVal = reshape(minVal, 1, D);
            maxVal = reshape(maxVal, 1, D);

            % Find and resample out-of-bounds rows
            outOfBoundsMatrix = (sample < minVal) | (sample > maxVal);

            % `outOfBoundsRows` is a [n_samples x 1] logical vector.
            % It's True for any row that has at least *one* out-of-bounds value.
            outOfBoundsRows = any(outOfBoundsMatrix, 2);

            % Resampling loop until no row is needed to be resampled
            while any(outOfBoundsRows)
                % Count how many rows need resampling
                n_to_resample = sum(outOfBoundsRows);

                % Generate new correlated samples only for the bad rows
                sample(outOfBoundsRows, :) = mvnrnd(mu_vec, cov_mat, n_to_resample);

                % Re-check only the rows we just replaced
                samples_to_recheck = sample(outOfBoundsRows, :);

                % Check this subset against the bounds
                recheck_matrix = (samples_to_recheck < minVal) | (samples_to_recheck > maxVal);
                recheck_results = any(recheck_matrix, 2); % [n_to_resample x 1] logical

                outOfBoundsRows(outOfBoundsRows) = recheck_results;
            end
        end

    case 'uniform'
        % Retrieve and validate specific parameters
        range = p.Results.range_uniform;
        if isempty(range)
            error('For ''uniform'' distribution, ''range_uniform'' [min, max] is required.');
        end
        minU = range(1);
        maxU = range(2);
        if minU >= maxU
            error('In ''range_uniform'', min must be less than max.');
        end

        % Generate samples
        sample = minU + (maxU - minU) * rand(n_samples, 1);

        % minVal and maxVal are not needed for uniform
        if ~isempty(minVal) || ~isempty(maxVal)
            warning('min_value/max_value are ignored for ''uniform'' distribution. Use ''range_uniform'' instead.');
        end

    otherwise
        error('Unknown distribution: %s. Supported are ''normal'', ''multivariatenormal'', ''uniform''.', distribution);
end

end