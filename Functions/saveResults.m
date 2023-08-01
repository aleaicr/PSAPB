function [] = saveResults(fileFolder,fileName,sCond,pq)
% Author: Alexis Contreras R.
% Save the obtained serviceability condition from the simulation for
% the respective pedestrian quantity (pq) in a .txt file
% 
% Inputs
% fileFolder: char with the name of the folder where the file will be saved
% fileName: char with the name of the file where the
% sCond: serviceability condition
% pq: pedestrian quantity
%
% Outputs
% A text file with the results is created in the specified folder
%
% Notes:
% The data is saved in the following format:
%  pq1 sCond1(pq1) sCond2(pq1) ... sCondN(pq1)
%  pq2 sCond1(pq2) sCond2(pq2) ... sCondN2(pq2)
%  ...
%  pq_final sCond1(pq_final) sCond2(pq_final) ... sCondN_final(pq_final)
%
% Note that the first column is the pedestrian quantity and the other
% columns are the serviceability conditions for the respective pedestrian
% quantity. Also, Note that the Serviceability Conditions depends on the number
% of simulations performed for that pedestrian quantity, so is not necessary that
% all the pedestrian quantities have the same number of serviceability conditions.

% Create the file name
fileDir = [fileFolder '\' fileName];
fileTempDir = [fileFolder '\temp.txt'];

% Check if the file exists
if exists(fileDir)
    % Open the file for reading
    fid = fopen(fileDir,'r');
    
    % Open the temporary file for writing
    fidTemp = fopen(fileTempDir,'w');
    
    % While loop to read lines
    found = 0;                  % The pedestrian quantity was not found yet
    while ~feof(fid)
        % Original line into a variable
        originalLine = strplitfgetl(fid); % char vector
        originalLine_vect = str2double(strsplit(originalLine,' ')); % double vector
        
        % If this line contain the pedestrian quantity
        if pq == originalLine_vect(1)
            % Write this line in the temporary file
            newLine = [originalLine ' ' num2str(sCond)];
            fprintf(fidTemp,'%s\n',newLine);
            found = 1;
        % If this line does not contain the pedestrian quantity
        elseif pq ~= originalLine_vect(1)
            % Write the original line in the temporary file
            fprintf(fidTemp,'%s\n',originalLine);
            % Continue to the next line
            continue
        end
    end

    % If the pedestrian quantity was not found
    if found == 0
        % Write it in the last line
        fprintf(fidTemp,'%d %d\n',pq,sCond);
    end

    % Close the files
    fclose(fid);        % Original file
    fclose(fidTemp);    % Temporary file

    % Rename temporary file to the original file and delete the temporary file
    movefile(fileTempDir,fileDir,'f'); % 'f' to overwrite the original file

% If the file does not exist
else
    % Create the file
    fid = fopen(fileDir,'w');
    
    % Write the first result
    fprintf(fid,'%d %d\n',pq,sCond);
    
    % Close the file
    fclose(fid);
end
end