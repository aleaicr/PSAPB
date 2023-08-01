function stripe = getStripe(fileFolder,fileName,pq)
% This returns the data from a single stripe (a specific pedestrian quantity)
% that is stored in a .txt file.
%
% INPUTS:
%   fileFolder: (char) the folder where the .txt file is stored
%   fileName: (char) the name of the .txt file
%   pq: (char) the pedestrian quantity of the data to be extracted
% 
% OUTPUTS:
%   stripe: (double vector) the data from the .txt file for the specified pedestrian quantity
%
% Notes:
%
%

% Create the file path
fileDir = [fileFolder '\' fileName];

% Read the data from the file
data = readmatrix(fileDir);

% Find the index of the desired pedestrian quantity
pqRow = data(data(:,1) == pq,:);    % data(:,1) is the pedestrian quantity column

% Extract the data for the desired pedestrian quantity
stripe = pqRow(:,2:end);

% Quit NaN values in the vector
stripe = stripe(~isnan(stripe));
end
