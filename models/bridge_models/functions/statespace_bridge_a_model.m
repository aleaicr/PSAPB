function [modelo,As,Bs,Cs,Ds] = statespace_bridge_a_model(n_modos,Me,Ke,Ce,Ge)
% Author: Alexis Contreras R.
% Returns the state space representation model and the matrices of state
% and response
%
%
% Inputs:
% n_modos           -> Number of modes to use
% Me                -> Equivalent Mass Matrix
% Ke                -> Equivalent Stiffness Matrix
% Ce                -> Equivalent Damping Matrix
% Ge                -> Equivalent Participation Matrix
%
% Outputs:
% modelo            -> State-Space model as system variable
% As                -> Matrix As from the state-space model
% Bs                -> Matrix Bs from the state-space model
% Cs                -> Matrix Cs from the state-space model
% Ds                -> Matrix Ds from the state-space model

%% Mathematical Explaination
% Equation of motion:
% [Me]{ddq} + [Ce]{dq} + [Ke]{q} = [Ge]{Pe}
% Where {q} are the assumed modes, and {Pe} are the equivalent forces
%
% The state-space representation:
% {dx} = As{x} + Bs{u}
% {y} = Cs{x} + Ds{u}
%
% Where, the states are {x} = {q; dq} and {dx} = {dq; ddq}.
% and the inputs are {u} = {Pe}
% and the outputs will be {ddq} (equivalent acceleration (?))

%% Computation
% State matrices
As = [zeros(n_modos) eye(n_modos); -Me\Ke -Me\Ce];                          % Matriz de estados, nxn
Bs = [zeros(n_modos); Me\Ge];                                               % Matriz de influencia de excitacion pe1(t), nx1

% Response matrices  
Cs = [-Me\Ke -Me\Ce];
Ds = Me\Ge;

% Variable type system
modelo = ss(As,Bs,Cs,Ds);
end

