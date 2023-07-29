function [PBM] = BridgeModel(EI,L,rho_lin,xi,n_modos,alternative)
% Author: Alexis Contreras R.
% BridgeModel creates a struct with all the properties of the bridge
%
% INPUTS
% EI:               double, Distributed stiffness of the equivalent beam
% L:                double, Length of the equivalent beam
% rho_lin:          double, Linear density of the equivalent beam
% n_modos:          double, Number of vibration modes to be considered (must be an integer number)
% xi:               double, Damping ratio of the equivalent beam (same for all vibration modes)
% alternative:      char, Para la representación state-space, si la ecuación de respuesta se quiere desplazamiento, velocidad y/o aceleración (solo existe: 'd', 'v', 'a', 'dv', 'da', 'av', da lo mismo el orden)
%
% OUTPUTS
% PBM:             Struct of the Pedestrian Bridge Model
%
% Notes
% * Until now, all the modes have the same damping ratio
%

% Previous calculations
zrmodal = xi*ones(n_modos,1);                                               % Vector of damping ratios for each mode

% Modal shapes functions (symbolic, for integration in Assumed Modes method
psi = sinModalShapes(n_modos,L);                                            % (sym) Vector of modal shapes functions

% Assumed Modes method resulting matrices
[Me,Ke,Ce,Ge,wn] = assumedModes_beam(psi,EI,rho_lin,L,zrmodal);             % Resulting matrices of the Assumed Modes method (for the Equation of Motion) 

% State Space representation
[modelo,As,Bs,Cs,Ds] = SpaceState_bridge(n_modos,Me,Ke,Ce,Ge,alternative); % State Space representation matrices of the bridge
damp(modelo)

% Save Data
PBM = struct();                                                             % Pedestrian Bridge Model
PBM.EI = EI;
PBM.L = L;
PBM.rho = rho_lin;
PBM.n_modos = n_modos;
PBM.xi = xi;
PBM.alternative = alternative;
PBM.psi = psi;
PBM.Me = Me;
PBM.Ke = Ke;
PBM.Ce = Ce;
PBM.Ge = Ge;
PBM.wn = wn;
PBM.As = As;
PBM.Bs = Bs;
PBM.Cs = Cs;
PBM.Ds = Ds;
PBM.modelo = modelo;
end