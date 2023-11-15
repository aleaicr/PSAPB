function [psi_pos] = sinModalShapes_psiposit(n_modos,L,posi)
% Creates the matrix psi_posi of the pedestrian i that contains the values of the modal shapes
% evaluated at the positions where the pedestrian is, as follows:
% psi_posi = [psi1_posi  psi2_posi  psi3_posi ... psin_posi] (matrix)
%
% INPUTS
% n_modos:      (double) Number of modes for the analysis
% L:            (double) Length of the equivalent beam (meters)
% posi:         (Vector)(t_length x 1) of the position of the pedestrian along the beam for pedestrian "i" where each row is for each time t-th time step
%
% OUTPUTS
% psi_pos:      (Matrix)(t_length x n_modos) Matrix of the modal shapes evaluated at the position for each time for each mode
%   with the modal shapes evaluated at the position for each time for each mode, 3 Dimensions --> time, #mode, #pedestrian
% 
% Notes
% - lenght(posi) = t_length -> Number of time steps
% - Line 28: Alternative version, probably is more efficient but I'm not sure, n_modos is a low-dimensional vector, so it's not a problem.

% Initialize the matrix psi_posi
psi_pos = zeros(length(posi),n_modos);                                      % length(posi) = t_length -> Cantidades de tiempos

% Fill the matrix psi_posi with the values of the modal shapes evaluated at the position for each time for each mode
for n = 1:n_modos
    psi_pos(:,n) = sin(pi*n/L*posi);
end

% Alternative
% % Create matrices of positions and mode numbers using meshgrid
% [PosiMat,ModeMat] = meshgrid(posi,1:1:n_modos); % posi is a column vector and 1:1:n_modos is a row vector
%
% % Compute the modal shapes using vectorized operations
% psi_pos = sin(pi*ModeMat/L.*PosiMat);

end