% poisson_incorporation_rate.m
% Script to generate .fig file using data from the Python script

%% Init
clear; close all; clc;

%% Load Data
% Assumes the .mat file is in the same directory as this script
data_file = 'poisson_data.mat';
if ~exist(data_file, 'file')
    error('File %s not found. Please run the Python script first.', data_file);
end
load(data_file);

%% Setup Output
script_path = mfilename('fullpath');
script_dir = fileparts(script_path);
% Assuming same structure as python script:
% repo_root/scripts/plots/others -> up 3 levels
repo_root = fileparts(fileparts(fileparts(script_dir)));
output_fig_dir = fullfile(repo_root, 'assets', 'figures', 'fig');

if ~exist(output_fig_dir, 'dir'), mkdir(output_fig_dir); end

%% Plot
h = figure('Name', 'Poisson Incorporation Rate', 'Color', 'w', 'Position', [100, 100, 1200, 600]);
hold on;

% Background simulations (Gray)
% t_back and c_back are cell arrays (1D object arrays from Python)
num_back = length(t_back);
for i = 1:num_back
    % Check if data is cellular or array formatted (depends on savemat version usually)
    % Python object array -> cell array in matlab
    if iscell(t_back)
        t = double(t_back{i});
        c = double(c_back{i});
    else
        % In case it loaded differently
        t = double(t_back(:, i)); % Unlikely for var length
    end

    stairs(t, c, 'Color', [0.6, 0.6, 0.6, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');
end

% Main simulation (Black)
stairs(double(t_main), double(c_main), 'k', 'LineWidth', 2.5, 'DisplayName', 'Single Simulation');

% Target Line (Red)
yline(double(target_val), '--r', 'LineWidth', 2, 'DisplayName', 'Target Crowd Density');

% Styling
grid on;
xlabel('Time [s]');
ylabel('Number of pedestrians');
legend('show', 'Location', 'best');
set(gca, 'FontSize', 12);
xlim([0, max(t_main)]); % Adjust limits if necessary

%% Save .fig
filename_fig = fullfile(output_fig_dir, 'poisson_incorporation_rate.fig');
savefig(h, filename_fig);

fprintf('Figure saved to: %s\n', filename_fig);
