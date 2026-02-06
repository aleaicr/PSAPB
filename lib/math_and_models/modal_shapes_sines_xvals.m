function [psi_xvals] = modal_shapes_sines_xvals(n_modos,L,x_vals)
% Creates a matrix with the values of the modal shapes evaluated at x_vals
% Input
% n_modos: Number of modes
% L: Length of the bridge
% x_vals: Vector of x values
%
% Output
% psi_xvals: Matrix of modal shapes evaluated at x_vals
% psi_xvals = [ psi1_xvals  psi2_xvals  psi3_xvals ... psin_xvals] (matrix)
% psi_xvals = [psi1_xval1 psi2_xval1 psi3_ xval1  ... psin_xval1
%              psi1_xval2 psi2_xval2 psi3_xval2  ... psin_xval2
%              ...
%              psi1_xvalm pis2_xvalm psi3_xvalm ... psin_xvalm]

Mmatrix = x_vals * (pi/L * (1:n_modos));

% Apply the sine function element-wise
psi_xvals = sin(Mmatrix);
psi_xvals = psi_xvals.';
end