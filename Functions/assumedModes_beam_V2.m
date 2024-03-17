function [Me,Ke,Ce,Ge] = assumedModes_beam_V2(psi,E,I,rho,A,L,zrmodal)
% Author: Alexis Contreras R.
% Generate the equivalent matrices of mass, stiffness, damping and (participación modal, G)
% using the assumed modes method in a simply supported beam and the proportional modal damping.
% Inputs
% psi           -> (sym vector) Vector of modal shape functions
% E             -> (double) Modulus of elasticity of the equivalent beam
% I             -> (double) Moment of inertia of the equivalent beam in the direction of analysis
% rho           -> (double) Linear density of the equivalent beam
% A             -> (double) Cross-sectional area of the equivalent beam
% L             -> (double) Length of the beam
% zrmodal       -> (double vector) Damping of each mode
%
% Outputs
% Me            -> (double Matrix) Equivalent mass matrix
% Ke            -> (double Matrix) Equivalent stiffness matrix
% Ce            -> (double Matrix) Equivalent proportional damping matrix
% Ge            -> (double Matrix) Equivalent modal participation matrix (always remains an identity matrix)
%%

syms x
ddpsi = diff(psi,x,x);
nModos = length(psi);
Me = double(int(rho*A*(psi).'*psi,x,0,L));
Ke = double(int(E*I*(ddpsi).'*ddpsi,x,0,L));
Ge = eye(nModos,nModos); 

% Proportional modal damping
[~, lambda] = eig(Ke,Me);                                                
wn = sqrt(diag(lambda));  
Phi = eye(nModos);
Ce = (Me*Phi)*diag(2*zrmodal.*wn./Me).*(Me*Phi).';                       
end

