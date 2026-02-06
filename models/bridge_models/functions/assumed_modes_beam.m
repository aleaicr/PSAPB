function [Me,Ke,Ce,Ge,wn] = assumed_modes_beam(psi,EI,rho_lin,L,zrmodal)
% Author: Alexis Contreras R.
% This function generates mass, stiffness, and damping matrices using the
% ASSUMED MODES method for a beam.
%
% Inputs
% psi           -> (sym) Vector of functions with the modal shapes (Use "x"
% as the symbolic variable because this is the variable we integrate in
% this function)
% EI            -> (scalar) Flexural Rigidity of the equivalent beam (Elasticity modulus times the moment of inertia in the direction of the analysis)
% rho           -> (scalar) Linear density of the equivalent beam
% A             -> (scalar) Cross-sectional area of the equivalent beam
% L             -> (scalar) Length of the beam
% zrmodal       -> (vector) Damping ratio of each lateral vibration mode
%
% Outputs
% Me            -> (Matrix) Equivalent mass matrix
% Ke            -> (Matrix) Equivalent stiffness matrix
% Ce            -> (Matrix) Equivalent damping matrix (proportional)
% Ge            -> (Matrix) Equivalent participation matrix (actually always an identity matrix) 
% wn            -> (Vector) Natural frequency of each mode.
%%
syms x 
ddpsi = diff(psi,x,x);
cant_modos = length(psi);

% Equivalent mass matrix
% Me_{i,j} = \int_0^L (rho*A*psi_i*psi_j) dx
% as the variable psi is a vector of symbolic functions, we can multiply them as psi.'*psi to get the matrix
% then, each element is integrated over x from 0 to L
Me = double(int(rho_lin*(psi).'*psi,x,0,L));

% Equivalent stiffness matrix
Ke = double(int(EI*(ddpsi).'*ddpsi,x,0,L));

% Equivalent participation matrix
Ge = eye(cant_modos,cant_modos);  % As force is Pe, Ge must be the identity matrix

% Modal analysis
[Phi_eig, lambda] = eig(Ke,Me); % Eigenvalue problem (eigenvektoren and eigenwerte)
wn = sqrt(diag(lambda)); % Natural frequencies (vector)

% Proportional modal damping
Me_n= diag(Phi_eig.'*Me*Phi_eig); % Diagonal mass matrix in modal coordinates
Ce_n = 2*zrmodal.*wn.*Me_n;

% Obtain Ce
% As Ce_n_diag = Phi_eig.' * Ce * Phi_eig;  --> Entails that Ce = inv(Phi_eig') * diag(Ce_n_diag) 
Ce = (Phi_eig') \ diag(Ce_n) / Phi_eig; % This is the same than writting inv(Phi_eig') * diag(Ce_n_diag) * inv(Phi_eig)
end

