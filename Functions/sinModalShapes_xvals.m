function [psi_xvals] = sinModalShapes_xvals(n_modos,L,x_vals)
% Creates a matrix psi_xvals that contains the values of the modal shapes
% evaluated at the positions 0 < x_vals < L as follows:
%
% psi_xvals = [ psi1_xvals  psi2_xvals  psi3_xvals ... psin_xvals] (size: x_vals_length x n_modos)
% where psin_xvsls = [psin_xval1; psin_xval2; psin_xval3; ... psin_xvalm] (size: x_vals_length x 1)
% 
% Inputs
% n_modos               number of modes to be considered (all are sinusoidal)
% L                     length of the equivalent beam
% x_vals                vector of x values where the modal shapes are to be evaluated
% 
% Outputs
% psi_xvals             matrix of modal shapes evaluated at x_vals
%
% 

%% Efficent alternative
x_vals_length = length(x_vals);                         % Amount of x values
M = pi/L*ones(x_vals_length,1)*(1:1:n_modos);           % M matrix
psi_xvals = sin(M.*x_vals);                             % Modal shapes matrix for any x value

%% Unefficient alternative
% psi_xvals = zeros(length(x_vals),n_modos);
% for n = 1:n_modos
%     psi_xvals(:,n) = sin(pi*n/L*x_vals);
% end
end