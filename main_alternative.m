%% Footbridge and pedestrians lateral interaction simulation
% Author: Alexis Contreras R
%
% Comments:
% This alternative main file to execute the simulations perform the same pedestrian quantity
% for all the simulation time length, instead of using a pedestrian incorporation rate range.

%% Initialization
clear variables
close all
clc

%% Addpath functions
currentDir = pwd();
functionsDir = fullfile(currentDir, 'Functions');
addpath(functionsDir)

%%  Inputs
% Bridge model parameters
EI  = 43*10^9;                       % N*m2                                 % Bridge distributed stiffness
L = 144;                             % m                                    % Bridge length
rho_lin = 100;                       % kg/m                                 % Bridge linear density
n_modos = 3;                                                                % Number of desired modes
xi = 0.7/100;                                                               % Damping ratio
x_parts = 100;                                                              % Number of bridge's divisions to evaluate the response
alternative = 'a';                                                          % Alternative is a value for the state-space representation of the bridge mode, 'a' is acceleration, 'v' is velocity, 'd' is displacement, 'ad' is acceleration and displacement, 'vd' is velocity and displacement, 'avd' is acceleration, velocity and displacement

% Simulation parameters
n_sim = 100;                                                                % Number simulations to perform for each stripe (for each pedestrian quantity)                                                     
t_inicial = 0;                      % sec                                   % Initial time of the simulation
t_step = 1/200;                     % sec                                   % Time step of the simulation
tpq = 200;                          % sec                                   % Time to simulate the specific pedestrian quantity

% Number of pedestrians parameters
n_min = 1;                                                                  % First pedestrian (do not change)
n_max = 200;                                                                % Maximum number of pedestrians
n_step = 1;                        % Must be 1                              % Maximum size of the group of pedestrians (e.g.: 2 pedestrians walk in the same position), at the moment n_step must be 1
np_step = 10;                                                               % Step of pedestrian quantity (1by1, 10by10)

% Filenames
if isequal(sort(alternative),'a')                                           % If the SS response vector y is acceleration only
    simName = 'Simulink/main_sim_acc.slx';
elseif isequal(sort(alternative),'ad')                                      % If the SS response vector y is acceleration and displacement
    simName = 'Simulink/main_sim.slx';
end
fileFolder = 'results/bridge1';                                             % Direction (folder) where the results will be saved
fileName_yn = 'yN.txt';                                                     % File name for the displacement results to be saved
fileName_ypn = 'ypN.txt';                                                   % File name for the velocity results to be saved
fileName_yppn = 'yppN.txt';                                                 % File name for the acceleration results to be saved

% Maximum accepted acceleration (if there is any numerical instability)
acc_max = 0.9;      % m/s2

% Pedestrian mass normal distribution, year 1999 (Johnson et al 2008)
mu_m = 71.91;       % kg                                                    % Mean
sigma_m = 14.89;    % kg                                                    % Standar deviation
m_min = 40;         % kg                                                    % Minimum mass
m_max = 150;        % kg                                                    % Maximum mass

% Pedestrian Walking speed normal distribution, year 2005 (Pachi & Ji, 2005)
mu_v = 1.3;         % m/s                                                   % Mean
sigma_v = 0.13;     % m/s                                                   % Standard deviation
v_min = 0.1;        % m/s                                                   % Minimum walking speed (before correlation)
v_max = 10;         % m/s                                                   % Maximum walking speed (before correlation)

% Lateral gait frequency normal distribution, year 2005 (Pachi & Ji 2005)
mu_freq = 1.8;      % hz                                                    % Mean                                                       
sigma_freq = 0.11;  % hz                                                    % Standard deviation
freq_min = 1.2;     % hz                                                    % Minimum frequency
freq_max = 2.4;     % hz                                                    % Maximum frequency

% Waling speed vs frequency correlation (Pachi & Ji 2005)
rhofv = 0.51;                                                            

% Van der Pol oscillator parameters
% a
mu_ai = 1.5;                                                                % Mean
ai_min = 1;                                                                 % Minimum
ai_max = 2*mu_ai - ai_min;                                                  % Maximum
sigma_ai = (ai_max - mu_ai)/3;                                              % Standard deviation
% lambda
mu_lambdai = 10;                                                            % Mean
lambdai_min = 8;                                                            % Minimum
lambdai_max = 2*mu_lambdai - lambdai_min;                                   % Maximum
sigma_lambdai = (lambdai_max - mu_lambdai)/3;                               % Standard deviation
% b
b = 2*pi;                                                                   % Constant

% Incorporation rate parameters (Incorporation pattern)
Tadd_min = 1;           % sec                                               % Minumum incorporation time
Tadd_max_min = 7;       % sec                                               % Minimum maximum incorporation time
Tadd_max_max = 16;      % sec                                               % Maximum maximum incorporation time
Tadd_primero = 10;      % sec                                               % As the Van der Pol oscillator first seconds are not stable, this is the time to wait before starting to incorporate pedestrians

% Pedestrian model initial conditions ranges (these values work well for the Van der Pol oscillator)
v0_range = 20;                                                              % Range of initial conditions for the velocity (1 to v0_range)
v0p_range = 1;                                                              % Range of initial conditions for the velocity derivative (1_v0p_range)

%% Previous calculations
PBM = BridgeModel(EI,L,rho_lin,xi,n_modos,alternative);                     % Generate the bridge model
L = PBM.L;                                                                  % Bridge length
x_vals = (0:L/x_parts:L).';                                                 % Vector of positions to evaluate the bridge response
psi_xvals = sinModalShapes_xvals(n_modos,L,x_vals);                         % Transformation matrix from equivalent to physical domain.

% State-space representation of the bridge matrices
As = PBM.As;
Bs = PBM.Bs;
Cs = PBM.Cs;
Ds = PBM.Ds;

% Save all input in two structs
simParams = struct();
pedParams = struct();

% simParams
simParams.Tadd_min = Tadd_min;
simParams.Tadd_max_min = Tadd_max_min;
simParams.Tadd_max_max = Tadd_max_max;
simParams.Tadd_primero = Tadd_primero;
simParams.t_inicial = t_inicial;
simParams.t_step = t_step;
simParams.tpq = tpq;

% PedParams
pedParams.n_max = n_max;
pedParams.n_min = n_min;
pedParams.n_step = n_step;
pedParams.np_step = np_step;
pedParams.mu_m = mu_m;
pedParams.sigma_m = sigma_m;
pedParams.m_min = m_min;
pedParams.m_max = m_max;
pedParams.mu_v = mu_v;
pedParams.sigma_v = sigma_v;
pedParams.v_min = v_min;
pedParams.v_max = v_max;
pedParams.mu_freq= mu_freq;
pedParams.sigma_freq = sigma_freq;
pedParams.freq_min = freq_min;
pedParams.freq_max = freq_max;
pedParams.rhofv = rhofv;
pedParams.mu_ai = mu_ai;
pedParams.sigma_ai = sigma_ai;
pedParams.ai_min = ai_min;
pedParams.ai_max = ai_max;
pedParams.mu_lambdai = mu_lambdai;
pedParams.sigma_lambdai = sigma_lambdai;
pedParams.lambdai_min = lambdai_min;
pedParams.lambdai_max = lambdai_max;
pedParams.b = b;
pedParams.v0_range = v0_range;
pedParams.v0p_range = v0p_range;

% State-space initial conditions
x0 = zeros(2*n_modos,1);                                                    % Initial conditions (rest)

% Pedestrian quantities
pq_vect = pedestrianQuantity(n_min,n_max,n_step,np_step);                   % Vector of the Pedestrian quantities

% Pedestrian quantities loop
for pq_index = 1:length(pq_vect)
    pq = pq(pq_index);                                                      % Pedestrian quantity

    % Simulations loop
    for s = 1:nsims
        [mi,vi,fi,wi,ai,lambdai,bi,side,tac,v0,v0p] = NewPedestrianProperties(simParams,pedParams,pq); % Generate pedestrian properties
        Ai = 1/(bi.^2);                                                     % Amplitude ammplification factor of the Van der Pol oscillator
        
        % Simulation times
        t_final = tac(end) + tpq;                                           % Final time of the simulation
        t_vect = (t_inicial:t_step:t_final).';                              % Vector with all the time steps
        t_length = length(t_vect);                                          % Amount of time steps
        
        % Pedestrain quantity in function of time
        ppt = pedsPerTime(tac,t_vect);                                      % Pedestrian quantity in function of time
        
        % Execute simulation
        tic
        out = sim(simName);                                                 % Execute simulation 's'
        toc

        % Compose the structural response of the bridge
        if isequal(alternative,'ad') 
            % Compose maximum bridge response for pq pedestrians
            [~, y_max] = genToPhys(out.q.Data(ppt == pq),psi_xvals.');              % [t x p, t x 1] % [y_xvals, y_max]
            [~, ypp_max] = genToPhys(out.qpp.Data(ppt == pq),psi_xvals.');          % [t x p, t x 1] % [ypp_xvals, y_max]
            
            %%% I NEED TO CREATE ANOTHER FUNCTION SAVERESULTS TO STORE THE DATA
        elseif isequal(alternative,'a')
            % Compose maximum bridge response for pq pedestrians
            [~, ypp_max] = genToPhys(out.qpp.Data(ppt == pq),psi_xvals.');          % [t x p, t x 1] % [ypp_xvals, y_max]
  
            %%% I NEED TO CREATE ANOTHER FUNCTION SAVERESULTS TO STORE THE DATA
        end
        clear out tout q qpp py v vp vpp ypp y_xvals y_max ypp_xvals ypp_max y_n ypp_n t_vect

    end

end

%% Remove path and return to original path
rmpath(functionsDir); % Remove functions directory from path
cd(currentDir); % Return to original path





