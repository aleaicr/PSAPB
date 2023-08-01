function [distParams,nSims] = getStripeParams_2(fileFolder,fileName,stripe,nSims,distribution)
% Obtener los parámetros de la distribución y la comparación del histograma
% con la curva de distribución solicitada
%
% INPUTS:
% fileFolder:           Carpeta donde se encuentran los resultados
% fileName:             Archivo en el que se encuentran los resultados
% stripe:               Franja (cantidad de peatones) a la que se quiere observar la distribución de los resultados
% nBins:                Cantidad de Bins para el histograma
% distribution:         Distribución para graficar (normal o lognormal)
% f:                    f = 'pdf','cdf', para ver pdf, cdf o pdf y cdf respectivamente, '' para no graficar nada
% 
% OUTPUTS
% distParams:           Parámetros de la distribución
% nPDF,nCDF,lnPDF,lnCDF (as requested): Histograma
%
% COMENTARIOS:
% * Es lo mismo que getStripeParams() pero solo extrae nSims simulaciones y
% no todas.

% Load Data 
results = getStripe(fileFolder,fileName,stripe);
if isequal(nSims,'all')
    nSims =  length(results);
end
results = sort(results(1:nSims,1));                                         % get all stripe results
n_sims = length(results);                                                   % number of stripe results

if isequal(distribution,'normal')    % Si quiero la distribución normal
    distParams = fitdist(results,'Normal');                                 % obtener distribución normal
    distParams.n_sims = n_sims;
elseif isequal(distribution,'lognormal')    % Si quiero la distribución lognormal
    distParams = fitdist(results,'Lognormal');                              % Genero los parámetros
    distParams = n_sims;
else
    error('Parámetro "distribución" tiene que ser "normal" o "lognormal" en char')
end
end