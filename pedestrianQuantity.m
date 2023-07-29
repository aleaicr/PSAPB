function pq_vect = pedestrianQuantity(n_min,n_max,n_step,np_step)
% This function, creates a vector of the pedestrian quantity to be used in the simulations,
% these will be the only stripes that will be performed.
% INPUTS
%   n_min   -->     Minimum (first) pedestrian quantity
%   n_max   -->     Maximum (last) pedestrian quantity
%   n_step  -->     Maximum size for a group of pedestrians
%   np_step -->     Step size for the number of pedestrians vector
% OUTPUTS:
%   pq_vect      -->     Pedestrian quantity vector
%
% NOTES:
%
%

    % Check inputs
    % n_step must be 1 at the moment
    if or(n_step ~= 1, n_min > n_max, np_step > n_max-n_min+1)
        fprintf('One of the following errors occurred:\n')
        fprintf('n_step must be 1\n')
        fprintf('n_min must be smaller than n_max\n')
        fprintf('np_step must be smaller than n_max-n_min+1\n')
        error('Wrong inputs')
    end

    auxVect = (n_min:n_step:n_max).';             % Create a vector with the number of pedestrians        
    auxMat = vec2mat(auxVect,np_step);            % Create a matrix with the number of pedestrians where each row is the pedestrian IDs that will be used in the simulation  
    pq_vect = auxMat(:,end);                           % The last column is the one we want
    if isnan(pq_vect(end))                             % If the last column is NaN, remove it
        pq_vect = pq_vect(1:end-1);
    end

end