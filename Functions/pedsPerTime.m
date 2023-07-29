function [ppt] = pedsPerTime(tac,t_vect)
% Author: Alexis Contreras R.
% Creates a vector that indicates the number of pedestrians for each time
% INPUTS:
% tac:          double vector, Incorporation pattern
% t_vect:       double vector, Simulation time
% OUTPUTS:
% ppt:          double vector, Number of pedestrians for each time of t_vect
%%

% Initialice variables
np = length(tac);                                                           % Maximum amount of pedestrians in the simulation
ppt = zeros(length(t_vect),1);                                              % Initialice vector 

% For each amount of pedestrians in the simulation loop
for ped = 1:np-1
    % The amount of pedestrians is the same for each time between the
    % incorporation of the current pedestrian and the next one.
    ppt(t_vect >= tac(ped) & t_vect < tac(ped + 1)) = ped;
end

% Before the first pedestrian is incorporated, there are no pedestrians
ppt(t_vect <= tac(1)) = 0;

% After the last pedestrian is incorporated, there are np pedestrians
ppt(t_vect >= tac(np)) = np;

end

