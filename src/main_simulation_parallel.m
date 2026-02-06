%% Main script for parallel simulation
% Authors: Alexis Contreras V., Gastón Fermandois C.

% Marks
%#ok<*SPEVB>
%#ok<*NASGU>

%% Init
clear variables
close all
clc

% EI_2 = 6.6*10^9;
% EI_4 = 21*10^9;   (21/6.6 = 3.1818)
% EI_5 = 41.1*10^9; (41.1/6.6 = 6.2273)

% Get the Root Directory (Absolute Path)
ROOT_DIR = pwd;

% Add Paths (Using Absolute Paths)
addpath(genpath(fullfile(ROOT_DIR, '../lib')));

%% Save Output Files
save_file = true;

%% Inputs
% Bridge number (Matches main_simulation.m)
bridge_num = 1;

% Bridge Model
BRIDGE_MODEL_PARAMS_PATH = fullfile(ROOT_DIR, ['../models/bridge_models/bridge_model_', num2str(bridge_num), '.mat']);

% Pedestrian model
PEDESTRIAN_MODEL_PARAMS_PATH = fullfile(ROOT_DIR, '../models/pedestrian_params/pedestrian_parameters.mat');

% Simulink folder path
SIMULINK_FOLDER_PATH = fullfile(ROOT_DIR, '../simulink/');

% Output folder configuration (User requested "parallelized_results" folder)
base_results_folder = fullfile(ROOT_DIR, ['../results/main_analysis/bridge', num2str(bridge_num), '/parallelized_results/']);

% Simulations parameters (Matches main_simulation.m)
N_sim = 9;                 % Number of simulations to realize
t_init = 0;                 % seconds
t_step = 1/200;             % seconds
t_endsimu = 2000;           % seconds
t_to_start_measure = 100;   % seconds
t_vect_oder_Tppt = 1;

% Number of Pedestrians (Matches main_simulation.m)
P_ped = [120, 125];
% Original: P_ped = [5, 10, 20, 30, 40, 50, 75, 100, 110, 120, 125, 130, 140, 150, 175, 200, 250, 300];
% B1: 70, 75, 80, 90, 110

% Parameters for Simulation Logic
x_points_n = 100;           % Number of equidistant points
Twaitadd = 30;              % seconds
Tped_min = 0;               % Minimum incorporation time
Tped_max = 1;               % Maximum incorporation time
p_side = 0.5;               % Probability of the pedestrian starting from the right side

%% Parallel Pool Setup
poolObj = gcp('nocreate');
if isempty(poolObj)
    poolObj = parpool();
end
numWorkers = poolObj.NumWorkers;
fprintf('Running on %d cores.\n', numWorkers);

%% Load Data (Global Load)
load(BRIDGE_MODEL_PARAMS_PATH);
load(PEDESTRIAN_MODEL_PARAMS_PATH);
damp(modelo)

%% Pre-calculation of Constant Variables
t_end = t_endsimu + Twaitadd + t_to_start_measure;
t_vect = (t_init:t_step:t_end).';
range_ti = [Tped_min, Tped_max];
loc_range = [0; L];
x0 = zeros(2*n_modos, 1);
x_points = (1:x_points_n) * (L/(x_points_n+1));
x_vals = x_points.';
psi_xvals = modal_shapes_sines_xvals(n_modos, L, x_vals);

% Create structs simParams and pedParams
run('../utils/generate_sim_and_ped_params_struct.m');

%% Parallel Execution (SPMD)
spmd
    % Determine Identity
    workerID = spmdIndex;

    % Create a unique temp folder for this core to avoid 'slprj' collisions
    workerTempDir = fullfile(ROOT_DIR, 'parallel_temp', sprintf('worker_%d', workerID));

    if ~exist(workerTempDir, 'dir')
        mkdir(workerTempDir);
    end

    % Move the worker into this folder
    cd(workerTempDir);
    fprintf('Core %d: Working in %s\n', workerID, workerTempDir);

    % Define Worker-Specific Output Path (Using Absolute Base Path)
    workerFileName = sprintf('results_parallel_core_%d.mat', workerID);
    workerFullPath = fullfile(base_results_folder, workerFileName);

    % Initialize/Check File
    if save_file
        % Ensure directory exists (only worker 1 checks to avoid conflicts)
        if workerID == 1 && ~exist(base_results_folder, 'dir')
            mkdir(base_results_folder);
        end
        pause(0.5);

        if ~isfile(workerFullPath)
            % Create new file
            dataToSave = struct();
            dataToSave.resultsMatrix = NaN(N_sim, length(P_ped));
            dataToSave.P_ped = P_ped;

            % Save using helper with -v7.3 flag
            save_worker_data(workerFullPath, dataToSave);

            m = matfile(workerFullPath, 'Writable', true);
            fprintf('Core %d: Created new .mat file at %s.\n', workerID, workerFullPath);
        else
            % Update existing file
            m = matfile(workerFullPath, 'Writable', true);
            existingP_ped = m.P_ped;

            if ~isequal(P_ped, existingP_ped)
                existingResultsMatrix = m.resultsMatrix;
                newResultsMatrix = NaN(N_sim, length(P_ped));
                [Lia, Locb] = ismember(P_ped, existingP_ped);
                if any(Lia)
                    newResultsMatrix(:, Lia) = existingResultsMatrix(:, Locb(Lia));
                end
                m.P_ped = P_ped;
                m.resultsMatrix = newResultsMatrix;
                fprintf('Core %d: Updated P_ped values.\n', workerID);
            end

            currentN_sim = size(m.resultsMatrix, 1);
            if N_sim > currentN_sim
                m.resultsMatrix = [m.resultsMatrix; NaN(N_sim - currentN_sim, length(P_ped))];
                fprintf('Core %d: Extended matrix rows.\n', workerID);
            end
        end
    end

    % Run Simulations
    for s = workerID : numWorkers : N_sim
        for i = 1:length(P_ped)

            % Check if done
            if save_file
                currentVal = m.resultsMatrix(s, i);
                if ~isnan(currentVal)
                    continue;
                end
            end

            P = P_ped(i);
            % print hour
            currentTime = datetime('now', 'Format', 'HH:mm:ss');
            fprintf('Core %d: Sim s=%d Ped=%d started at: %s\n', workerID, s, P, char(currentTime));


            % Parameters
            Twaitaddi = Twaitadd * ones(P, 1);

            % Sampling
            [mi, vi, fi, wi, ai, lambdai, Ai, k1i, k2i, k3i, alpha_si, bi, side, ti, v0, v0p, loci] = sampling_all_pedestrians(pedParams, P);

            % Assign Variables to Base Workspace (Required for Simulink in parpool)
            assignin('base', 't_init', t_init);
            assignin('base', 't_step', t_step);
            assignin('base', 't_end', t_end);
            assignin('base', 'P', P);
            assignin('base', 'Twaitaddi', Twaitaddi);
            assignin('base', 'mi', mi);
            assignin('base', 'vi', vi);
            assignin('base', 'fi', fi);
            assignin('base', 'wi', wi);
            assignin('base', 'ai', ai);
            assignin('base', 'lambdai', lambdai);
            assignin('base', 'Ai', Ai);
            assignin('base', 'k1i', k1i);
            assignin('base', 'k2i', k2i);
            assignin('base', 'k3i', k3i);
            assignin('base', 'alpha_si', alpha_si);
            assignin('base', 'bi', bi);
            assignin('base', 'side', side);
            assignin('base', 'ti', ti);
            assignin('base', 'v0', v0);
            assignin('base', 'v0p', v0p);
            assignin('base', 'loci', loci);
            assignin('base', 'L', L);
            assignin('base', 'n_modos', n_modos);
            assignin('base', 'Twaitadd', Twaitadd);
            assignin('base', 'x0', x0);
            assignin('base', 'As', As);
            assignin('base', 'Bs', Bs);
            assignin('base', 'Cs', Cs);
            assignin('base', 'Ds', Ds);

            % Simulink Path Selection Logic
            sim_filename = '';
            if bridge_num == 1
                if P < 60
                    sim_filename = ['initialloc_acc_', num2str(P), '.slx'];
                elseif P >= 60 &&  P <= 75
                    t_step_min = 1/10000;
                    t_step_max = 1/1000;
                    t_step_init = 1/1000;
                    assignin('base', 't_step_min', t_step_min)
                    assignin('base', 't_step_max', t_step_max)
                    assignin('base', 't_step_init', t_step_init)
                    sim_filename = ['dormandprince_initialloc_acc_', num2str(P), '.slx'];
                else
                    sim_filename = ['euler_initialloc_acc_', num2str(P), '.slx'];
                end
            elseif bridge_num == 2
                if P < 175
                    sim_filename = ['initialloc_acc_', num2str(P), '.slx'];
                else
                    sim_filename = ['euler_initialloc_acc_', num2str(P), '.slx'];
                end
            else
                % Default case (includes bridge 5)
                sim_filename = ['initialloc_acc_', num2str(P), '.slx'];
            end

            % Construct Absolute Path
            simName = fullfile(SIMULINK_FOLDER_PATH, sim_filename);
            fprintf('Core %d: Simulating in %s\n', workerID, sim_filename);

            % Run Simulation
            tic;
            try
                load_system(simName);
                simOut = sim(simName, 'SimulationMode', 'accelerator');

                % Post-Processing
                simTime = toc;
                qpp = simOut.qpp.Data;
                tout = simOut.tout;

                idx_start = find(tout >= (t_to_start_measure + Twaitadd), 1, 'first');

                if isempty(idx_start)
                    warning('Core %d: Simulation time too short.', workerID);
                    ypp_max = NaN;
                else
                    qpp_filtered = qpp(:, :, idx_start:end);
                    ypp_t = genToPhys(qpp_filtered, psi_xvals);
                    ypp_max = get_ypp_max(ypp_t);
                end
            catch ME
                warning('Core %d: Simulation failed for P=%d, s=%d. Error: %s', workerID, P, s, ME.message);
                ypp_max = NaN;
            end

            % Save Result
            if save_file
                m.resultsMatrix(s, i) = ypp_max;
            end

            Simulink.sdi.clear;
        end
    end
end
fprintf('All parallel simulations completed.\n');

% Optional: Cleanup temp folders
rmdir('parallel_temp');

%% Helper Functions
function save_worker_data(fname, data)
% Helper to save data from within spmd blocks with -v7.3 flag
save(fname, '-struct', 'data', '-v7.3');
end

