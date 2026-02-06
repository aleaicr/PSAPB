function results_mat2xlsx(matFilename, xlsxFilename)
% RESULTS_MAT2XLSX - Extracts specific variables from a .mat file and stores them in an Excel file.
%
%   Usage:
%       results_mat2xlsx('data.mat', 'data_editable.xlsx')
%
%   Inputs:
%       matFilename  - String containing the name of the source .mat file.
%       xlsxFilename - String containing the name of the destination .xlsx file.

% Check if input file exists
if ~isfile(matFilename)
    error('The file %s does not exist.', matFilename);
end

% Load the data structure from the .mat file
data = load(matFilename);

% Check if the required variables exist in the loaded file
if isfield(data, 'resultsMatrix') && isfield(data, 'P_ped')

    % Write 'resultsMatrix' to the Excel file
    try
        writematrix(data.resultsMatrix, xlsxFilename, 'Sheet', 'resultsMatrix');
        fprintf('Successfully wrote "resultsMatrix" to sheet "resultsMatrix".\n');
    catch ME
        warning(ME.identifier, 'Could not write resultsMatrix. Error: %s', ME.message);
    end

    % Write 'P_ped' to the Excel file
    try
        writematrix(data.P_ped, xlsxFilename, 'Sheet', 'P_ped');
        fprintf('Successfully wrote "P_ped" to sheet "P_ped".\n');
    catch ME
        warning(ME.identifier, 'Could not write P_ped. Error: %s', ME.message);
    end

    fprintf('Conversion complete: %s created.\n', xlsxFilename);

else
    error('The .mat file must contain both "resultsMatrix" and "P_ped" variables.');
end