function [] = joinResults(fileFolder1,fileName1,fileFolder2,fileName2)
% Esta función une los resultados de las simulaciones de dos carpetas
% distintas
%
% INPUTS
% fileFolder1: Nombre de la carpeta de los resultados 1
% fileName1: Nombre del archivo de los resultados 1
% fileFolder2: Nombre de la carpeta de los resultados 2
% fileName2: Nombre del archivo de los resultados 2
%
% OUTPUTS
% * 
% 
% COMENTARIOS
% * Deja todos los resultados en los resultados de fileFolder1 y 2 archivos
% nuevos llamados old_fileName1_fileFolder2 y old_fileName2_fileFolder2 que contiene los antiguos 
% resultados (los guarda en fileFolder1)
% * Esta función la cree para unir los resultados de Results4\m1_w2_z1 y
% Results5\m1_w2_z1 ya que simulé dos veces en carpetas distintas porque
% quería probar un cambio mínimo que no tenía efecto en los resultados.
% * Para futuro sería bueno parametrizar con un "vector de fileFolders" y
% "vector de fileNames" para que uno pueda unir varias carpetas y archivos
% a la vez, pero no debería ser necesario hasta el momento
%

% Direcciones de los resultados 1 y 2
fileDir1 = [fileFolder1 '\' fileName1];
fileDir2 = [fileFolder2 '\' fileName2];

% Load datos archivo 1 y 2
results1 = load(fileDir1);                                                  % Matriz [200 x nsim1]
results2 = load(fileDir2);                                                  % Matriz [200 x nsim2]

% Unir matrices
resultsNew = [results1 results2];                                           % Matriz [200 x (nsim1 + nsim2)]

% Guardar en archivo temporal
save([fileFolder1 '\temp.txt'],'resultsNew','-ascii')

end