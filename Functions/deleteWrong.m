function [] = deleteWrong(fileFolder,fileName,max_value)
% Author: Alexis Contreras R.
% This function deleltes/discard the failed Monte-Carlo simulation results
% 
% INPUTS:
% fileFolder:
% fileName:
% max_value:
%
% OUTPUTS:
% - : No outputs, but writes a file
%
% NOTES:
% * This function check if the largest number of the i-th column is larger than max_value, if the condition
% is true, then deletes the column. There is no need to verify one by one.
%%

% File direction
fileDir = [fileFolder '\' fileName];
fileDirTemp = [fileFolder '\temp.txt']; % Temporal file

% Load data
datos = load(fileDir);

% Number of columns (number of simulations) in the file
nSimus = size(datos, 2);

% Loop in columns
column = 1;                                                                 % Inicializar columna en la que voy observando
while column <= nSimus
    maxInColumn = max(datos(:,column));                                     % Máximo valor de columna
    % Máximo valor es mayor a max_value, eliminar la columna
    if maxInColumn > max_value
        datos(:,column) = [];                                               % Borrar columna
        nSimus = nSimus - 1;                                                % Disminuye el número de columnas
        % Si borra no hay que aumentar en i ya que hay una nueva columna en
        % esa posición que debe ser revisada
     else
        column = column + 1;                                                % Si no borra, entonces sigo a la siguiente columna
    end
end
% Guardar los datos modificados en un archivo temporal
save(fileDirTemp, 'datos', '-ascii');

% Eliminar el archivo original y renombrar el archivo temporal
fileDirAllSimus = [fileFolder '\' fileName(1,1:(end-4)) '_all_Simus.txt'];  % Nombre del archivo que contiene todas las simulacines (sin borrar)
if exist(fileDirAllSimus, 'file') == 0                                      % Si no existe tal archivo
    copyfile(fileDir,fileDirAllSimus);                                      % Crearlo
    delete(fileDir);                                                        % Eliminar el yppN antiguo
elseif exist(fileDirAllSimus, 'file') == 2                                  % Si existe el archivo
    % Si la cantidad de simulaciones que tiene guardado yppN_all_simus.txt
    % es menor a la cantidad de yppN.txt, quiere decir que se hicieron más
    % simulaciones nuevas por lo que yppN_all_simus debe ser
    datosAllSimus = load(fileDirAllSimus);
    nsimsAllSimus = size(datosAllSimus,2);
    if nsimsAllSimus < nSimus
        delete(fileDirAllSimus)                                             % Eliminar el archivo que contiene todas las simulaciones (porque ya no tiene todas)
        copyfile(fileDir,fileDirAllSimus);                                  % yppN.txt es el que tiene todas las simulaciones (la mayor cantidad)
        delete(fileDir);                                                    % borrar yppN.txt
    else
        delete(fileDir);                                                    % Si existe y tiene más simulaciones que yppN.txt, solo borrar yppN.txt
    end
end

% Archivo temporal a nuevo yppN corregido
movefile(fileDirTemp, fileDir);                                             % Dejar el archivo temporal (el corregido/reducido) como el nuevo

end


