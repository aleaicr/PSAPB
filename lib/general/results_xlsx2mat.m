function results_xlsx2mat(xlsxFilename, matFilename)
% RESULTS_XLSX2MAT Reads an edited Excel file and saves it back to .mat.
%
%   Usage:
%       results_xlsx2mat('data_editable.xlsx', 'data_new.mat')
%
%   Inputs:
%       xlsxFilename - String containing the name of the source .xlsx file.
%       matFilename  - String containing the name of the destination .mat file.

% Check if input file exists
if ~isfile(xlsxFilename)
    error('The file %s does not exist.', xlsxFilename);
end

fprintf('Reading data from %s...\n', xlsxFilename);

try
    % Read the data back from specific sheets
    % Note: If your excel file has headers, use readtable, otherwise readmatrix is cleaner
    resultsMatrix = readmatrix(xlsxFilename, 'Sheet', 'resultsMatrix');
    P_ped = readmatrix(xlsxFilename, 'Sheet', 'P_ped');
    % Save the variables into a new .mat file
    save(matFilename, 'resultsMatrix', 'P_ped');

    fprintf('Success! Data saved to %s\n', matFilename);

catch ME
    error('Failed to read Excel file or save .mat file. Ensure sheet names are "resultsMatrix" and "P_ped".\nError details: %s', ME.message);
end
end