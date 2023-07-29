function [resultsN] = getSimResults(fileFolder,fileName,nsim)
% Obtener los resultados en función de N desde los archivos 
%
% INPUTS
% fileDir: Carpeta donde están guardados los resultados
% fileName: Nombre del archivo donde están guardado los resultados
% nsim: Número de simulaciones a la que se le quiere extraer los resultados
%
% OUTPUTS
% resultsN: El vector con los resultados yN o ypN o yppN de la simulación consultada
% (figure): Se crea una figura con los resultados si es que se deseó
% 
% COMENTARIO
% * Primero lee el archivo readme para obtener la cantidad de peatones 'np'
% y la cantidad de simulaciones acumuladas 'nsim'
% * Si nsim < nsims entonces puedo seguir ya que existe esa simulación
% * Obtengo la línea y extraigo el valor de respuestaN(n) para esa
% simulación específica y sigo así para todas las cantidaddes N = n

% Previous 
fid_readme = fopen([fileFolder '\' 'readme.txt'],'r');                         % Abrir readme
linea1 = fgetl(fid_readme);                                                 % La primera línea contiene la cantidad de simulaciones acumuladas
linea2 = fgetl(fid_readme);                                                 % La segunda linea contiene la cantidad de peatones máxima en cada simulación
eval(linea1)                                                                % nsims = ...
eval(linea2)                                                                % np = ...
fclose(fid_readme);                                                         % cerrar readme

if nsim > nsims % Si no existe esa simulación
    error('nsim debe ser menor o igual a la cantidad de simulaciones acumuladas en el archivo')
else % Si existe la simulación
    % Inicializar
    resultsN = zeros(np,1);                                                 % Inicializar vector de los resultados
    n = 1;                                                                  % Cantidad de peatones
    % Abrir el archivo
    fid = fopen([fileFolder '\' fileName],'r');
    
    % Recorrer archivo y extraer datos de la simulación nsim
    while ~feof(fid)
        if n == np + 1 
            break                                                           % Break si llegué a n > np
        end
        % get Line N = n
        line = fgetline(fid);
        
        % Separate line
        line = strplit(line);

        % Convertir a double
        line = str2double(line(:));

        % extraer nsim-th data
        resultsN(n,1) = line(nsim,1);

        % siguiente N = n
        n = n + 1;
    end
    
    % Cerrar archivo
    fclose(fid);
end
end