%% Main Simulation Script
% Authors: Alexis Contreras V., Gastón Fermandois C.
% Simulates the bridge response to a range of crowd sizes.
% Generates or fill the output file with the results.
%
% Marks
%#ok<*UNRCH>

%% Init
clear variables
close all
clc

%% Add Paths
addpath(genpath('../lib'))

%% Save File
save_file = true;

%% Inputs
% Bridge number
bridge_num = 1;

% Bridge Model
BRIDGE_MODEL_PARAMS_PATH = ['../models/bridge_models/bridge_model_', num2str(bridge_num), '.mat'];

% Pedestrian model
PEDESTRIAN_MODEL_PARAMS_PATH = '../models/pedestrian_params/pedestrian_parameters.mat';

% Simulink folder path
SIMULINK_FOLDER_PATH = '../simulink/';    % Path of the simulink to use

% Output folder path
OUTPUT_PATH = ['../results/main_analysis/bridge', num2str(bridge_num), '/results_2000s_euler_100125150.mat'];    % Path to store all the simulations.

% Simulations parameters
N_sim = 50;                 % Number of simulations to realize
t_init = 0;                 % seconds
t_step = 1/200;             % seconds
t_endsimu = 2000;           % seconds, t_vect = (t_init:t_step:t_end).'
t_to_start_measure = 100;   % seconds, from this time start to save (if there is a transient time span)
t_vect_oder_Tppt = 1;       % 1: use t_vect for the simulation, 0: use only tac (and the t_extra for the last pedestrian)

% Number of Pedestrians
P_ped = [100, 120, 125, 150];    % This can be a single integer or a list (vector) of multiple integers

% 100: 7 days     (50 simulations)
% 125: ~10 days
% 150: 14 days-
% -> Paralelizado (31/4 ~~ 7 days)
%          -> Crear un archivo de resultados para cada núcleo, luego unir,
%           para que no escriban al mismo tiempo el un archivo.

% Chequear
% Figura 12 y 14 del paper

% Paper
% Queda en evidencia que con las pocas simulaciones el puente 1 es caótico
% 

% Hacer
% - Hacer puente 6, equivalente al bridge 1 pero con más amortiguamiento, utilizar 1%
% - Una simulación suficientemente larga para el puente 3 con 10, 50, 100 y 150 peatones
% con un t_end de 4.000s, 20 ventanas de 200s (4000/20). En cada una de
% esas ventanas calcular el máximo y eso hacer un histograma y compararlo
% con el raw 2data (y obtener estadísticas lognormal, mu_ln y sigma_ln)

% Martes 30 de diciembre.


% How many points will be used to measure
x_points_n = 100;           % Number of equidistant points to evaluate the lateral acceleration

% Time to wait to the foot-force model stabilizes in the VdP cycle, after
% this time, the first pedestrian will be incorporated to the bridge.
Twaitadd = 30;              % seconds,

% One by One incorporation pattern
% alternative = 'initialloc';   % 'onebyone': if an incorporation pattern will be used; 'initloc': for all pedestrians to start from a random location
Tped_min = 0;               % Minimum time to a pedestrian to be incorporated to the bridge after the previous one
Tped_max = 1;               % Maximum time to a pedestrian to be incorporated to the bridge after the previous one.

% Side
p_side = 0.5;               % Probability of the pedestrian to get on the bridge from the right side  of the bridge (from x = L)

%% Load data
load(BRIDGE_MODEL_PARAMS_PATH);
load(PEDESTRIAN_MODEL_PARAMS_PATH);

% Show bridge
damp(modelo)

%% Needed variables
% To compute acceleration in multiple points given the response in
% generalized coordinates
x_points = (1:x_points_n) * (L/(x_points_n+1)); % x_points_n equidistant points from 0 to L

% Vector of the time of the simulation
t_end = t_endsimu + Twaitadd + t_to_start_measure;
t_vect = (t_init:t_step:t_end).';

% Range of incorporation times
range_ti = [Tped_min, Tped_max];

% Location range to simulate the initial location of the pedestrians.
loc_range = [0; L];

% Initial conditions for the bridge
x0 = zeros(2*n_modos,1);

% Create structs simParams and pedParams
run('../utils/generate_sim_and_ped_params_struct.m');

% Modal shapes in the predefined points
x_vals = x_points.';
psi_xvals = modal_shapes_sines_xvals(n_modos, L, x_vals);

%% Setup Output File

if save_file
    if ~isfile(OUTPUT_PATH)  % if filename doesn't exist in the directory
        resultsMatrix = NaN(N_sim, length(P_ped));  % Create new resultsMatrix (for each P, we have N_simulations)
        save(OUTPUT_PATH, 'resultsMatrix', 'P_ped');  % Create new .mat file
        disp('.mat file created');
        m = matfile(OUTPUT_PATH, 'Writable', true); % Open the .mat file for writing
    else
        m = matfile(OUTPUT_PATH, 'Writable', true); % Open the .mat file for writing
        % Check if the P_ped values match the existing results
        existingP_ped = m.P_ped;
        if ~isequal(P_ped, existingP_ped) % In the case the file is not equal.
            % Load existing results
            existingResultsMatrix = m.resultsMatrix;

            % Create new resultsMatrix with NaNs
            newResultsMatrix = NaN(N_sim, length(P_ped));

            % Find indices of current P_ped in the existing file
            % Lia: Logical 1 where new P_ped exists in old set
            % Locb: Index in old set corresponding to new P_ped
            [Lia, Locb] = ismember(P_ped, existingP_ped);

            % Copy data from existing matrix to the new positions
            if any(Lia)
                newResultsMatrix(:, Lia) = existingResultsMatrix(:, Locb(Lia));
            end
            % Update the file
            m.P_ped = P_ped;
            m.resultsMatrix = newResultsMatrix;
            disp('Updated .mat file with new P_ped values and reorganized resultsMatrix.');
        end

        % Add more rows to the matrix if N_sim > the dimension of the existingMatrix
        currentN_sim = size(m.resultsMatrix, 1);
        if N_sim > currentN_sim % The number o
            m.resultsMatrix = [m.resultsMatrix; NaN(N_sim - currentN_sim, length(P_ped))];
            disp(['Added ', num2str(N_sim - currentN_sim), ' rows to the resultsMatrix.']);
        end
    end
end


%% Simulation
% Print init hour
currentTime = datetime('now', 'Format', 'HH:mm:ss');
fprintf('Iteration %d ended at: %s\n', i, char(currentTime));

% For loop for the simulation index
for s = 1:N_sim

    % For loop for the number of pedestrians index
    for i = 1:length(P_ped)

        % Number of pedestrians
        P = P_ped(i); % Number of pedestrians
        fprintf("Simulation s = %.0f, Crowd Size P = %.0f\n", s, P);

        % If this simulation was already performed, then continue to the
        % next simulation
        if save_file
            valueInMatrix = m.resultsMatrix(s, i);
            if ~isnan(valueInMatrix)
                continue;
            end
        end

        % Wait time vector
        Twaitaddi = Twaitadd*ones(P, 1);

        % Create new sample of pedestrian properties
        [mi, vi, fi, wi, ai, lambdai, Ai, k1i, k2i, k3i, alpha_si, bi, side, ti, v0, v0p, loci] = sampling_all_pedestrians(pedParams, P);

        % Simulation
        % Choose simulink file name
        simName = [SIMULINK_FOLDER_PATH, 'euler_initialloc_acc_', num2str(P), '.slx'];

        tic
        out = sim(simName, 'SimulationMode', 'accelerator');
        toc

        % Print current hour
        currentTime = datetime('now', 'Format', 'HH:mm:ss');
        fprintf('Iteration %d ended at: %s\n', i, char(currentTime));

        % Extract data:
        qpp = out.qpp.Data;
        tout = out.tout;

        % Get idx_start
        idx_start = find(tout >= (t_to_start_measure + Twaitadd), 1, 'first'); % Find the index for the start time of the simulation

        % Only consider qpp from t_to_start_measure + Twaitadd
        qpp_filtered = qpp(:, :, idx_start:end);
        ypp_t = genToPhys(qpp_filtered, psi_xvals);

        % Obtain maximum in this simulation
        ypp_max = get_ypp_max(ypp_t);

        % Save
        if save_file
            m.resultsMatrix(s, i) = ypp_max;
        end
    end
    Simulink.sdi.clear
end
  