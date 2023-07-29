function [distParams,nSims] = getStripeParams_2(fileFolder,fileName,stripe,nSims,nBins,distribution,f)
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
    
    % Figure if requested
    if isequal(f,'pdf')                                                     % pdf normal
        [N,EDGES] = hist(results,nBins);
        figure
        h = bar(EDGES,N/trapz(EDGES,N));
        h.FaceColor = '#606060';
        hold on
        plot(results,pdf(distParams,results),'color','r','LineWidth',2)
        hold off
        grid on
        xlabel('\delta')
        ylabel('PDF')
        title('Histograma PDF y Ajuste Distribución Normal',['\mu = ' convertStringsToChars(string(distParams.mu)) '; \sigma = ' convertStringsToChars(string(distParams.sigma))])
        legend(convertStringsToChars("Stripe = " + string(stripe) + "; nsims = " + string(n_sims) + "; nBins = " + string(nBins)))
    elseif isequal(f,'cdf')                                                 % cdf normal
        figure
        h = histogram(results,nBins,'Normalization','cdf');
        h.FaceColor = '#606060';
        hold on
        plot(results,cdf(distParams,results),'color','r','LineWidth',2)
        hold off
        grid on
        xlabel('\delta')
        ylabel('CDF')
        title('Histograma CDF y ajuste Distribución Normal',['\mu = ' convertStringsToChars(string(distParams.mu)) '; \sigma = ' convertStringsToChars(string(distParams.sigma))])
        legend(convertStringsToChars("Stripe = " + string(stripe) + "; nsims = " + string(n_sims)))
    elseif isequal(f,'')
        
    else
        error('Parámetro "f" tiene que ser "pdf" o "cdf" en char')
    end

elseif isequal(distribution,'lognormal')    % Si quiero la distribución lognormal
    distParams = fitdist(results,'Lognormal');                              % Genero los parámetros

    % Figure if requested
    if isequal(f,'pdf')                                                     % pdf lognormal
        [f,x] = hist(results,nBins);
        figure
        h = bar(x,f/trapz(x,f));
        h.FaceColor = '#606060';
        hold on
        plot(results,pdf(distParams,results),'color','r','LineWidth',2)
        hold off
        grid on
        xlabel('\delta')
        ylabel('PDF')
        title('Histograma PDF y Ajuste Distribución Lognormal',['\mu_{ln} = ' convertStringsToChars(string(distParams.mu)) '; \sigma_{ln} = ' convertStringsToChars(string(distParams.sigma))])
        legend(convertStringsToChars("Stripe = " + string(stripe) + "; nsims = " + string(n_sims)))
    
    elseif isequal(f,'cdf')                                                 % cdf lognormal
        figure
        h = histogram(results,nBins,'Normalization','cdf');
        h.FaceColor = '#606060';
        hold on
        plot(results,cdf(distParams,results),'color','r','LineWidth',2)
        hold off
        grid on
        xlabel('\delta')
        ylabel('CDF')
        title('Histograma CDF y Ajuste distribución Lognormal',['\mu_{ln} = ' convertStringsToChars(string(distParams.mu)) '; \sigma_{ln} = ' convertStringsToChars(string(distParams.sigma))])
        legend(convertStringsToChars("Stripe = " + string(stripe) + "; nsims = " + string(n_sims)))
    elseif isequal(f,'')
        % no hacer nada
    else
        error('Parámetro "f" tiene que ser "pdf" o "cdf" en char')
    end
else
    error('Parámetro "distribución" tiene que ser "normal" o "lognormal" en char')
end
end