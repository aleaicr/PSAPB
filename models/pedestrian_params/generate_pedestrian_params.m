%% Generate Pedestrian Parameters File
% Use this script to create a pedestrian_bridge_parameters.mat file
%
% In folder `functions` there is the function `sampling_destributions.m`
% 
%
%% Init
clear variables
close all
clc

%% Inputs
% Paths
output_path = 'pedestrian_parameters.mat';

% Mass
mu_m = 71.91; % kg                                                         % Media de la distribución de masa
sigma_m = 14.89; % kg                                                       % Desviación estándar de la distribución de masa
m_min = 40; % kg
m_max = 120; % kg

% Walking speed
mu_v = 1.3; % m/s                                                           % Media
sigma_v = 0.13; % m/s                                                       % Desviación estándar
v_min = 0.8; % m/s                                                          % Velocidad mínima
v_max = 2; % m/s (buscar fuentes)

% Frequency vertical gait
mu_freq = 1.8;      % hz                                                         % Media                                                       
sigma_freq = 0.11;  % hz                                                    % Desviación estándar
freq_min = 1.2;     % hz (Un poco más que los resultados de Pachi&Ji 2005)
freq_max = 2.4;     % hz

% Correlation between speed and frequency
rhofv = 0.51;                                                            
mu_fv = [mu_freq; mu_v];
sigma_fv = [sigma_freq; sigma_v];
rho_matrix_fv = [1 rhofv; rhofv 1];
min_fv = [freq_min, v_min];
max_fv = [freq_max, v_max];
covariance_fv = diag(sigma_fv)*rho_matrix_fv*diag(sigma_fv);

% Van der Pol oscillator parameters
% a
mu_ai = 1.0;
ai_min = 0.8;
ai_max = 2*mu_ai - ai_min; % ai_max = mu_ai + (mu_ai - ai_min)
sigma_ai = (ai_max - mu_ai)/3;

% lambdai
mu_lambdai = 10;
lambdai_min = 8;
lambdai_max = 2*mu_lambdai - lambdai_min;
sigma_lambdai = (lambdai_max - mu_lambdai)/3;

% b
b = 2*pi;

% Scalling factors
k1 = 1.5;
k2 = 3.5;
k3 = 0.8;
alpha_s = 1/2;
A = 2*1/b^2;

% Initial conditions for pedestrians
x0_range = [-0.03; 0.03];  % m
x0p_range = [-0.3; 0.3];   % m/s

%% Save all workspace variables to the specified MAT-file
save(output_path);