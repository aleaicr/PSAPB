function [psi] = sinModalShapes(n_modos,L)
% Author: Alexis Contreras R.
% Generates modal shapes for a beam with two simple supports at its ends.
% Note that the modal shapes are sinusoidal functions that satisfy
% orthogonality and boundary conditions.
%
% Inputs
% n_modos           -> Number of modes to be generated
% L                 -> Length of the equivalent beam
%
% Outputs
% psi               -> Symbolic vector containing the modal shapes
%

% Define symbolic variable
syms x

% Initialize vector
psi = sym(zeros(1,n_modos));

% Generate modal shapes functions (symbolic)
for n = 1:n_modos
    psi(n) = sin(n*pi*x/L);
end

end

