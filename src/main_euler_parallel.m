%% Main script for parallel processing of the simulation
% Authors: Alexis Contreras V., Gastón Fermandois C.

%% Init
clear variables
close all
clc

% Get the Root Directory (Absolute Path) to fix relative path issues
% We assume this script is running from the 'src' folder (or wherever it is located)
ROOT_DIR = pwd; 

% Add Paths (Using Absolute Paths)
addpath(genpath(fullfile(ROOT_DIR, '../lib')));

%% Save Output Files
save_file = true;

%% Inputs
% Bridge number
bridge_num = 1;

% --- CRITICAL FIX: CONVERT ALL PATHS TO ABSOLUTE ---
% If we don't do this, the workers won't find the files when we move them to temp folders.

% Bridge Model
BRIDGE_MODEL_PARAMS_PATH = fullfile(ROOT_DIR, ['../models/bridge_models/bridge_model_', num2str(bridge_num), '.mat']);

% Pedestrian model
PEDESTRIAN_MODEL_PARAMS_PATH = fullfile(ROOT_DIR, '../models/pedestrian_params/pedestrian_parameters.mat');

% Simulink folder path
SIMULINK_FOLDER_PATH = fullfile(ROOT_DIR, '../simulink/'); 

% Output folder configuration
base_results_folder = fullfile(ROOT_DIR, ['../results/main_analysis/bridge', num2str(bridge_num), '/euler_paralell/']);
 
% Simulations parameters
N_sim = 50;                 % Number of simulations to realize
t_init = 0;                 % seconds
t_step = 1/200;             % seconds
t_endsimu = 2000;           % seconds
t_to_start_measure = 100;   % seconds
t_vect_oder_Tppt = 1;

% Number of Pedestrians
P_ped = [100, 120, 125, 150];    

% Parameters for Simulation Logic
x_points_n = 100;           % Number of equidistant points to evaluate the lateral acceleration
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
% Note: using run() with absolute path logic is safer, but if it works relatively, keep it. 
% Better to use absolute if possible, but run() handles relative to pwd well.
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
    workerFileName = sprintf('results_euler_paralell_core_%d.mat', workerID);
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
            fprintf('Core %d: Created new .mat file.\n', workerID);
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
            fprintf("Core %d: Sim s=%d, Ped=%d\n", workerID, s, P);
            % print hour
            currentTime = datetime('now', 'Format', 'HH:mm:ss');
            fprintf('Core %d: Sim s=%d Ped=%d started at: %s\n', workerID, s, P, char(currentTime));

            
            % Parameters
            Twaitaddi = Twaitadd * ones(P, 1);
            
            % Sampling
            [mi, vi, fi, wi, ai, lambdai, Ai, k1i, k2i, k3i, alpha_si, bi, side, ti, v0, v0p, loci] = sampling_all_pedestrians(pedParams, P);
            
            % Simulation Time Parameters
            assignin('base', 't_init', t_init); %#ok<SPEVB>
            assignin('base', 't_step', t_step); %#ok<SPEVB>
            assignin('base', 't_end', t_end);   %#ok<SPEVB>
            assignin('base', 'P', P);           %#ok<SPEVB>
            assignin('base', 'Twaitaddi', Twaitaddi); %#ok<SPEVB>

            % Pedestrian Parameters
            assignin('base', 'mi', mi);         %#ok<SPEVB>
            assignin('base', 'vi', vi);         %#ok<SPEVB>
            assignin('base', 'fi', fi);         %#ok<SPEVB>
            assignin('base', 'wi', wi);         %#ok<SPEVB>
            assignin('base', 'ai', ai);         %#ok<SPEVB>
            assignin('base', 'lambdai', lambdai); %#ok<SPEVB>
            assignin('base', 'Ai', Ai);         %#ok<SPEVB>
            assignin('base', 'k1i', k1i);       %#ok<SPEVB>
            assignin('base', 'k2i', k2i);       %#ok<SPEVB>
            assignin('base', 'k3i', k3i);       %#ok<SPEVB>
            assignin('base', 'alpha_si', alpha_si); %#ok<SPEVB>
            assignin('base', 'bi', bi);         %#ok<SPEVB>
            assignin('base', 'side', side);     %#ok<SPEVB>
            assignin('base', 'ti', ti);         %#ok<SPEVB>
            assignin('base', 'v0', v0);         %#ok<SPEVB>
            assignin('base', 'v0p', v0p);       %#ok<SPEVB>
            assignin('base', 'loci', loci);     %#ok<SPEVB>

            % Bridge physical constants
            assignin('base', 'L', L);               %#ok<SPEVB>
            assignin('base', 'n_modos', n_modos);   %#ok<SPEVB>
            assignin('base', 'Twaitadd', Twaitadd); %#ok<SPEVB>
            assignin('base', 'x0', x0);             %#ok<SPEVB>
            
            % State-Space matrices
            assignin('base', 'As', As);             %#ok<SPEVB>
            assignin('base', 'Bs', Bs);             %#ok<SPEVB>
            assignin('base', 'Cs', Cs);             %#ok<SPEVB>
            assignin('base', 'Ds', Ds);             %#ok<SPEVB>
            
            % Simulink Setup (Using Absolute Path)
            % simName is constructed using SIMULINK_FOLDER_PATH which is now Absolute
            simName = [SIMULINK_FOLDER_PATH, 'euler_initialloc_acc_', num2str(P), '.slx'];
            
            % Run Simulation
            tic;
            load_system(simName);
            
            simOut = sim(simName, 'SimulationMode', 'accelerator');
            
            % print hour
            currentTime = datetime('now', 'Format', 'HH:mm:ss');
            fprintf('Core %d: Sim s=%d started at: %s\n', workerID, s, char(currentTime));

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
            
            % Save Result
            if save_file
                m.resultsMatrix(s, i) = ypp_max;
            end
            
            Simulink.sdi.clear;
        end
    end
end
fprintf('All parallel simulations completed.\n');

% Optional: Cleanup temp folders (uncomment if desired)
rmdir('parallel_temp', 's');

%% Helper Functions
function save_worker_data(fname, data)
    % Helper to save data from within spmd blocks with -v7.3 flag
    save(fname, '-struct', 'data', '-v7.3');
end