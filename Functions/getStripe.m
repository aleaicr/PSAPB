function [results] = getStripe(fileFolder, fileName, stripe)
% Lee una línea específica de un archivo .txt que contiene números float
% y devuelve los valores de la línea en un vector columna
%
% INPUTS:
% fileName: char vector, con el nombre (y carpeta incluida) del archivo
% stripe:   double, franja (número de línea) para análisis
%
% OUTPUTS
% results:  double vector, todos los valores de la franja (de la línea de texto) 
%
% COMENTARIOS
% * Recorro el filas hasta llegar a la que quiero
%

% Abrir el archivo
fid = fopen([fileFolder '\' fileName],'r');

% Leer líneas hasta la anterior
for i = 1:stripe-1
    fgetl(fid);
end

% Leer línea que quiero
linea = strsplit(fgetl(fid));

% Transformar string a vector de doubles
results = str2double(linea(:));

% Cerrar archivo
fclose(fid);

end