function [psi] = modal_shapes_sines(n_modos,L)
% Alexis Contreras R.
% Generate sinusoidal functions with a symbolic variable, each mode is the
% subsequent sinusoidal mode. For instance:
% if n_modos = 3, then -->  psi = [sin(1*pi*x/L) sin(2*pi*x/L) sin(3*pi*x/L)]
%
% Inputs
% n_modos           -> Number of modes to be used
% L                 -> Length of the bridge (equivalent beam)
%
% Outputs
% psi               -> Symbolic vector with the modal shapes of the assumed sinusoidal modes
%% Computation
syms x
n = 1:n_modos;
psi = sin(n*pi*x/L);

end

