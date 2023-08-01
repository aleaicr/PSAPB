function [] = getHistogramFit_2(fileFolder,fileName,stripe,nBins,distribution,f)
% This function creates an histogram of the data for the desired pedestrian quantity
%
% INPUTS
% fileFolder:           (char) name of the folder where the data is stored
% fileName:             (char) name of the file where the data is stored
% stripe:               (double) number of the stripe to be considered (the pedestrian quantity)
% nBins:                (double) number of bins for the histogram
% distribution:         (char) distribution to be fitted
                            % 'normal' for normal distribution
                            % 'lognormal' for lognormal distribution
% f:                    (char) 'pdf' or 'cdf' for the distribution to be fitted
%
% OUTPUTS
% none
%
% Notes:
% - The function creates a figure with the histogram and the fitted distribution
% - The histogram can be a pdf or a cdf.

% load data
results = sort(getStripe(fileFolder,fileName,stripe));                      % get all stripe results
n_sims = length(results);                                                   % number of stripe results

if isequal(distribution,'normal')
    distParams = fitdist(results,'Normal');
    
    % Figure if requested
    if isequal(f,'pdf')
        [N,EDGES] = hist(results,nBins);
        getHistogramFit(results,nBins,'Normal','pdf')
        figure
        h = bar(EDGES,N/trapz(EDGES,N));
        h.FaceColor = '#606060';
        hold on
        plot(results,pdf(distParams,results),'color','r','LineWidth',2.2)
        hold off
        grid on
        xlabel('\delta')
        ylabel('PDF')
        title('PDF Histogram & Normal Distribution Fit',['\mu = ' convertStringsToChars(string(distParams.mu)) '; \sigma = ' convertStringsToChars(string(distParams.sigma))])
        legend(convertStringsToChars("Stripe = " + string(stripe) + "; nsims = " + string(n_sims) + "; nBins = " + string(nBins)))
    
    elseif isequal(f,'cdf')
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
        % Do not plot
    else
        error('Parámetro "f" tiene que ser "pdf" o "cdf" en char')
    end

elseif isequal(distribution,'lognormal') 
    distParams = fitdist(results,'Lognormal'); 

    if isequal(f,'pdf')
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
    
    elseif isequal(f,'cdf')
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
        % Do not plot
    else
        error('Parámetro "f" tiene que ser "pdf" o "cdf" en char')
    end
else
    error('Parámetro "distribución" tiene que ser "normal" o "lognormal" en char')
end
end