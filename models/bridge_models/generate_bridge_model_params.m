%% Create a bridge model
% Authors: Alexis Contreras V., Gastón Fermandois C.
% This script creates a model of the bridge, the output is a struct saved
% in a .mat file. This struct will be loaded in the script for the
% simulation (main.m)

%% Init
clear variables
close all
clc

%% Inputs
function_path = 'functions';
bridge_number = 5;

%% Output
bridge_model_output_path = ['bridge_model_', num2str(bridge_number), '.mat'];

%% Add path
addpath(function_path);

%% Bridge Parameters
rho_lin = 150;                                      % (kg/m) Linear density
L = 144;                                            % (m) Length of the bridge (Same as the mid-span of the Millennium Bridge according to Dallard et al 2001)
n_modos = 3;                                        % Number of modes

if bridge_number == 1
    % Bridge 1 parameters       % PAPER = PUENTE 1
    f = 0.5;                                               % hz
    EI = 4*f^2 * rho_lin * L^4 / pi^2;                     % (N*m^2) Flexural stiffness for fixed bridge's lateral frequency
    xi = 0.7/100;                                          % Modal damping ratio

elseif bridge_number == 2
    % Bridge 2 parameters       % PAPER = PUENTE 2
    f = 0.5; % hz
    EI = 4*f^2 * rho_lin * L^4 / pi^2;
    xi = 2/100;

elseif bridge_number == 3
    % Bridge 3 parameters       % PAPER = PUENTE 3
    f = 0.5; % hz
    EI = 4*f^2 * rho_lin * L^4 / pi^2;
    xi = 5/100;

elseif bridge_number == 4
    % Bridge 4 parameters       % PAPER = PUENTE 4
    f = 0.9; % hz
    EI = 4*f^2 * rho_lin * L^4 / pi^2;
    xi = 2.0/100;

elseif bridge_number == 5
    % Bridge 5 parameters       % PAPER = PUENTE 5
    f = 1.3; % hz
    EI = 4*f^2 * rho_lin * L^4 / pi^2;
    xi = 2.0/100;
% elseif bridge_number == 6
%     % Bridge 6 parameters       % PAPER = PUENTE 6
%     f = 1.3; % hz
%     EI = 4*f^2 * rho_lin * L^4 / pi^2;
%     xi = 1/100;
end

%% Create data
% Damping ratio vector for each mode
zrmodal = xi*ones(n_modos,1);                                               % Damping ratio vector for each mode

% Modal shapes
psi = modal_shapes_sines(n_modos,L);                                        % Modal shapes

% Assumed modes
[Me,Ke,Ce,Ge,wn] = assumed_modes_beam(psi,EI,rho_lin,L,zrmodal);            % Equivalent matrices

% State-space
[modelo,As,Bs,Cs,Ds] = statespace_bridge_a_model(n_modos,Me,Ke,Ce,Ge);      % State-space model

%% Create Struct
PBM = struct();                                                             % Pedestrian Bridge Model
PBM.EI = EI;
PBM.L = L;
PBM.rho = rho_lin;
PBM.n_modos = n_modos;
PBM.xi = xi;
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

%% Save struct to a .mat
save(bridge_model_output_path, 'EI', 'rho_lin', 'xi', 'L', 'n_modos', ...
    'zrmodal', 'psi', 'Me', 'Ke', 'Ce', 'Ge', 'wn', ...
    'modelo', 'As', 'Bs', 'Cs', 'Ds', 'PBM');
