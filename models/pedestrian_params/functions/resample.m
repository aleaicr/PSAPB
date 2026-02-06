function [sample_fixed] = resample(varargin)
% RESAMPLE Resamples data based on a specified model and required parameters.
%
%   Requires specific parameters based on the model chosen.
%   Example: resample('normal', 'mu', 0.5, 'sigma', 0.57)

    % 1. Create the inputParser object
    p = inputParser;

    % 2. Define the required positional argument 'model'
    addRequired(p, 'distribution', @(x) ischar(x) || isstring(x));

    % 3. Define the optional Name-Value parameters (No defaults needed)
    % We define them as parameters, but will check if they were actually supplied
    addParameter(p, 'mu', [], @isnumeric); 
    addParameter(p, 'sigma', [], @isnumeric); 
    addParameter(p, 'min', [], @isnumeric); 
    addParameter(p, 'max', [], @isnumeric); 

    % 4. Parse the input arguments
    parse(p, varargin{:});

    % 5. Access the parsed results
    distribution_name = p.Results.model;
    
    % --- CONDITIONAL VALIDATION (No default values) ---

    if strcmpi(distribution_name, 'normal')
        % Check if 'mu' and 'sigma' were supplied (are not empty)
        if isempty(p.Results.mu)
            error('resample:MissingParameter', ...
                  'Error: For the "normal" distribution, the parameter ''mu'' is required.');
        end
        if isempty(p.Results.sigma)
            error('resample:MissingParameter', ...
                  'Error: For the "normal" distribution the parameter ''sigma'' is required.');
        end
        
        param_mu = p.Results.mu;
        param_sigma = p.Results.sigma;
        
        % Core Logic for Normal
        fprintf('Using Normal Distribution: mu = %.2f, sigma = %.2f\n', param_mu, param_sigma);
        sample_fixed = param_mu + param_sigma * randn(10, 1);

    elseif strcmpi(distribution_name, 'uniform')
        % Check if 'min' and 'max' were supplied
        if isempty(p.Results.min)
            error('resample:MissingParameter', ...
                  'Error: For the "uniform" distribution, the parameter ''min'' is required.');
        end
        if isempty(p.Results.max)
            error('resample:MissingParameter', ...
                  'Error: For the "uniform" distribution, the parameter ''max'' is required.');
        end
        
        param_min = p.Results.min;
        param_max = p.Results.max;
        
        % Core Logic for Uniform
        fprintf('Using Uniform Distribution: min = %.2f, max = %.2f\n', param_min, param_max);
        sample_fixed = param_min + (param_max - param_min) * rand(10, 1);

    else
        error('resample:UnsupportedModel', 'Unsupported distribution type: %s.', distribution_name);
    end
end