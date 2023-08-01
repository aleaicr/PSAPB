function [distParams] = getStripeParams(fileFolder,fileName,stripe,distribution)
% Obtain the parameters of the distribution and the comparison of the histogram
% with the requested distribution curveda
%
% INPUTS:
% fileFolder:           Folder where the results are stored
% fileName:             File where the results are stored
% stripe:               Stripe (pedestrian quantity) to which the distribution of the results is to be observed
% distribution:         Distribution to fit the results
%
% OUTPUTS
% distParams:          Distribution parameters
%

% Load Data 
results = sort(getStripe(fileFolder,fileName,stripe));
n_sims = length(results);

% Get the distribution parameters
if isequal(distribution,'normal')                                                   % If the desired distribution is normal
    distParams = fitdist(results,'Normal');                                         % Fit the distribution
    distParams.n_sims = n_sims;                                                     % Add the number of simulations to the distribution parameters

elseif isequal(distribution,'lognormal')                                            % If the desired distribution is lognormal
    distParams = fitdist(results,'Lognormal');                                      % Fit the distribution
    distParams.n_sims = n_sims;                                                     % Add the number of simulations to the distribution parameters
else
    error('Parameter "distribution" must be "normal" or "lognormal" (as a char)')   % If the desired distribution is not normal or lognormal, throw an error
end
end